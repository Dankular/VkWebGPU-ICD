//! SPIR-V to WGSL shader translation using Naga

use crate::error::{Result, VkError};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Shader cache: SPIR-V hash → compiled WGSL
pub struct ShaderCache {
    cache: RwLock<FxHashMap<u64, Arc<String>>>,
}

impl ShaderCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn get_or_translate(&self, spirv: &[u32]) -> Result<Arc<String>> {
        let hash = Self::hash_spirv(spirv);

        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(wgsl) = cache.get(&hash) {
                return Ok(wgsl.clone());
            }
        }

        // Translate SPIR-V to WGSL
        let wgsl = Self::translate_spirv_to_wgsl(spirv)?;
        let wgsl = Arc::new(wgsl);

        // Store in cache
        {
            let mut cache = self.cache.write();
            cache.insert(hash, wgsl.clone());
        }

        Ok(wgsl)
    }

    fn hash_spirv(spirv: &[u32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        spirv.hash(&mut hasher);
        hasher.finish()
    }

    pub fn translate_spirv_to_wgsl(spirv: &[u32]) -> Result<String> {
        log::debug!("Translating SPIR-V to WGSL ({} words)", spirv.len());

        // Pre-process SPIR-V: split any combined image samplers into separate
        // image + sampler variables so that naga 0.20's SPIR-V frontend can
        // handle them (it requires OpSampledImage in function bodies).
        let preprocessed;
        let spirv = if spv_has_combined_image_samplers(spirv) {
            preprocessed = split_combined_image_samplers(spirv);
            log::debug!(
                "Combined-image-sampler split: {} -> {} words",
                spirv.len(),
                preprocessed.len()
            );
            preprocessed.as_slice()
        } else {
            spirv
        };

        // Parse SPIR-V using Naga
        let options = naga::front::spv::Options {
            adjust_coordinate_space: true,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        };
        let mut module = naga::front::spv::Frontend::new(spirv.iter().cloned(), &options)
            .parse()
            .map_err(|e| {
                VkError::ShaderTranslationFailed(format!("SPIR-V parse error: {:?}", e))
            })?;

        // CRITICAL: Transform push constants to uniform buffers.
        // WebGPU doesn't support push constants. When push constants are present we:
        //   1. Shift all existing user resource bindings +1 group (to match the pipeline layout
        //      which inserts the push constant bind group at index 0).
        //   2. Map the push constant block to group=0, binding=0.
        Self::transform_push_constants_to_uniform(&mut module)?;

        // Validate the module
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| VkError::ShaderTranslationFailed(format!("Validation error: {:?}", e)))?;

        // Generate WGSL
        let wgsl = naga::back::wgsl::write_string(
            &module,
            &info,
            naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
        )
        .map_err(|e| VkError::ShaderTranslationFailed(format!("WGSL generation error: {:?}", e)))?;

        log::debug!(
            "Translated SPIR-V shader ({} words) to WGSL ({} bytes)",
            spirv.len(),
            wgsl.len()
        );

        Ok(wgsl)
    }

    /// Transform push constants to uniform buffers, correcting binding group numbers.
    ///
    /// Pipeline layout inserts the push constant bind group at WebGPU group 0, which shifts
    /// all user descriptor sets up by one. This function applies the same shift to WGSL
    /// resource bindings so that the shader agrees with the pipeline layout:
    ///
    ///   - All non-push-constant resource bindings have their `group` incremented by 1.
    ///   - The push constant block is remapped to group=0, binding=0.
    ///
    /// When no push constants are present the function is a no-op.
    fn transform_push_constants_to_uniform(module: &mut naga::Module) -> Result<()> {
        use naga::{AddressSpace, ResourceBinding};

        // First pass: check if any push constants exist.
        let has_push_constants = module
            .global_variables
            .iter()
            .any(|(_, var)| var.space == AddressSpace::PushConstant);

        if !has_push_constants {
            return Ok(());
        }

        // Second pass: shift all existing non-push-constant resource bindings by +1 group.
        // This keeps user descriptors aligned with the pipeline layout which will have
        // user bind groups starting at index 1 (index 0 reserved for push constants).
        let mut shifted_count = 0usize;
        for (_, var) in module.global_variables.iter_mut() {
            if var.space != AddressSpace::PushConstant {
                if let Some(ref mut binding) = var.binding {
                    binding.group += 1;
                    shifted_count += 1;
                }
            }
        }

        // Third pass: remap push constants to the emulated uniform buffer at group=0, binding=0.
        let mut transformed_count = 0usize;
        for (handle, var) in module.global_variables.iter_mut() {
            if var.space == AddressSpace::PushConstant {
                if let Some(ref name) = var.name {
                    log::debug!(
                        "Transforming push constant variable '{}' to uniform buffer at group=0, binding=0",
                        name
                    );
                } else {
                    log::debug!(
                        "Transforming push constant variable {:?} to uniform buffer at group=0, binding=0",
                        handle
                    );
                }

                var.space = AddressSpace::Uniform;
                var.binding = Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                });
                transformed_count += 1;
            }
        }

        log::info!(
            "Push constant transform: {} block(s) → uniform at (0,0); {} user binding(s) shifted +1 group",
            transformed_count,
            shifted_count
        );

        Ok(())
    }

    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }

    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── SPIR-V combined-image-sampler splitting ───────────────────────────────────
//
// naga 0.20's SPIR-V frontend only supports image sampling when the sampled-image
// expression comes from an explicit OpSampledImage instruction (which pairs a
// separate image variable with a separate sampler variable).  It cannot process
// SPIR-V that loads a TypeSampledImage variable and feeds the result directly to
// OpImageSampleImplicitLod/etc. — the pattern that Mesa/Zink and many OpenGL-style
// SPIR-V compilers produce for COMBINED_IMAGE_SAMPLER descriptors.
//
// `split_combined_image_samplers` rewrites such SPIR-V so that every
// COMBINED_IMAGE_SAMPLER variable at (set S, binding B) gains a companion
// TypeSampler variable at (set S, binding B_samp), and every
//   %R = OpLoad %TypeSampledImage %V
// in a function body is expanded to:
//   %R_img  = OpLoad %TypeSampledImage %V
//   %R_samp = OpLoad %TypeSampler      %V_samp
//   %R      = OpSampledImage %TypeSampledImage %R_img %R_samp
//
// The synthetic sampler binding B_samp is assigned compactly:
//   B_samp = (max_binding_in_group + 1) + rank
// where rank is the 0-indexed position of B among sorted combined-sampler bindings
// within that descriptor set group. This keeps all binding numbers small (< 1000),
// matching WebGPU's maxBindingsPerBindGroup limit.
//
// descriptor.rs uses the same formula (see `compute_sampler_binding_map`), so the
// WGSL bindings always match the BindGroupLayout/BindGroup entries.

const SPV_MAGIC: u32 = 0x07230203;

// SPIR-V opcodes
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_SAMPLER: u32 = 26;
const OP_TYPE_POINTER: u32 = 32;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_LOAD: u32 = 61;
const OP_SAMPLED_IMAGE: u32 = 86;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;

// SPIR-V storage classes / decorations
const SC_UNIFORM_CONSTANT: u32 = 0;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;

/// Returns true if the SPIR-V module contains any OpVariable of TypeSampledImage
/// type in UniformConstant storage — i.e. if `split_combined_image_samplers` has
/// any work to do.
fn spv_has_combined_image_samplers(words: &[u32]) -> bool {
    if words.len() < 5 || words[0] != SPV_MAGIC {
        return false;
    }

    use std::collections::HashSet;
    let mut sampled_image_types = HashSet::<u32>::new();
    let mut combined_ptr_types = HashSet::<u32>::new();

    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = words[i] & 0xFFFF;
        if wc == 0 || i + wc > words.len() {
            break;
        }

        match op {
            OP_TYPE_SAMPLED_IMAGE if wc == 3 => {
                sampled_image_types.insert(words[i + 1]);
            }
            OP_TYPE_POINTER if wc == 4 => {
                let sc = words[i + 2];
                let base = words[i + 3];
                if sc == SC_UNIFORM_CONSTANT && sampled_image_types.contains(&base) {
                    combined_ptr_types.insert(words[i + 1]);
                }
            }
            OP_VARIABLE if wc >= 4 => {
                let rt = words[i + 1];
                let sc = words[i + 3];
                if sc == SC_UNIFORM_CONSTANT && combined_ptr_types.contains(&rt) {
                    return true;
                }
            }
            OP_FUNCTION => break, // past global section — no combined var found
            _ => {}
        }

        i += wc;
    }
    false
}

/// Compute the synthetic sampler binding number for a COMBINED_IMAGE_SAMPLER
/// at `texture_binding` within a descriptor set whose combined-sampler bindings
/// (sorted ascending) are `sorted_combined` and whose overall maximum binding is
/// `max_binding`.
///
/// Formula: `max_binding + 1 + rank` where rank is the 0-indexed position of
/// `texture_binding` in `sorted_combined`.
///
/// This function is also mirrored in `descriptor.rs::compute_sampler_binding_map`.
pub fn compact_sampler_binding(
    texture_binding: u32,
    max_binding: u32,
    sorted_combined: &[u32],
) -> u32 {
    let rank = sorted_combined
        .iter()
        .position(|&b| b == texture_binding)
        .unwrap_or(0);
    max_binding + 1 + rank as u32
}

/// Rewrite SPIR-V to split combined-image-sampler variables into separate image
/// and sampler variables, inserting `OpSampledImage` at each use site.
fn split_combined_image_samplers(words: &[u32]) -> Vec<u32> {
    use std::collections::HashMap;

    if words.len() < 5 || words[0] != SPV_MAGIC {
        return words.to_vec();
    }

    // ── Phase 1: collect type / variable / decoration info ────────────────────

    // image_types: set of OpTypeImage result IDs
    let mut image_types = std::collections::HashSet::<u32>::new();
    // sampled_image_types: SI_id → underlying image_id
    let mut sampled_image_types = HashMap::<u32, u32>::new();
    // existing TypeSampler ID (if already present in the module)
    let mut existing_sampler_type: Option<u32> = None;
    // combined_ptr_types: ptr_type_id → SI_type_id
    let mut combined_ptr_types = HashMap::<u32, u32>::new();
    // combined_vars: var_id → SI_type_id
    let mut combined_vars = HashMap::<u32, u32>::new();
    // all UniformConstant vars, for computing max_binding_per_group
    let mut uc_vars = std::collections::HashSet::<u32>::new();
    // binding/set for combined vars
    let mut var_ds = HashMap::<u32, u32>::new();
    let mut var_binding = HashMap::<u32, u32>::new();
    // binding/set for ALL UniformConstant vars (to compute max_binding)
    let mut all_uc_ds = HashMap::<u32, u32>::new();
    let mut all_uc_binding = HashMap::<u32, u32>::new();

    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = words[i] & 0xFFFF;
        if wc == 0 || i + wc > words.len() {
            break;
        }
        let instr = &words[i..i + wc];

        match op {
            OP_TYPE_IMAGE if wc >= 2 => {
                image_types.insert(instr[1]);
            }
            OP_TYPE_SAMPLED_IMAGE if wc == 3 => {
                let si_id = instr[1];
                let img_id = instr[2];
                if image_types.contains(&img_id) {
                    sampled_image_types.insert(si_id, img_id);
                }
            }
            OP_TYPE_SAMPLER if wc == 2 => {
                existing_sampler_type = Some(instr[1]);
            }
            OP_TYPE_POINTER if wc == 4 => {
                let ptr_id = instr[1];
                let sc = instr[2];
                let base = instr[3];
                if sc == SC_UNIFORM_CONSTANT && sampled_image_types.contains_key(&base) {
                    combined_ptr_types.insert(ptr_id, base);
                }
            }
            OP_VARIABLE if wc >= 4 => {
                let rt = instr[1];
                let var_id = instr[2];
                let sc = instr[3];
                if sc == SC_UNIFORM_CONSTANT {
                    uc_vars.insert(var_id);
                    if let Some(&si_type) = combined_ptr_types.get(&rt) {
                        combined_vars.insert(var_id, si_type);
                    }
                }
            }
            OP_DECORATE if wc >= 4 => {
                let target = instr[1];
                let dec = instr[2];
                let val = instr[3];
                if uc_vars.contains(&target) {
                    match dec {
                        DEC_DESCRIPTOR_SET => {
                            all_uc_ds.insert(target, val);
                            if combined_vars.contains_key(&target) {
                                var_ds.insert(target, val);
                            }
                        }
                        DEC_BINDING => {
                            all_uc_binding.insert(target, val);
                            if combined_vars.contains_key(&target) {
                                var_binding.insert(target, val);
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        i += wc;
    }

    if combined_vars.is_empty() {
        return words.to_vec();
    }

    // ── Compute compact sampler bindings ─────────────────────────────────────
    // For each descriptor set group G:
    //   max_binding = max over all UC vars in G
    //   sorted combined = sorted list of combined-sampler bindings in G
    //   sampler binding for B_i = max_binding + 1 + rank_i

    // max_binding per group (over ALL UC vars, not just combined)
    let mut max_binding_per_group: HashMap<u32, u32> = HashMap::new();
    for (&var_id, &b) in &all_uc_binding {
        let g = all_uc_ds.get(&var_id).copied().unwrap_or(0);
        max_binding_per_group
            .entry(g)
            .and_modify(|m| *m = (*m).max(b))
            .or_insert(b);
    }

    // Sorted combined-sampler bindings per group
    let mut combined_bindings_per_group: HashMap<u32, Vec<u32>> = HashMap::new();
    for (&var_id, _) in &combined_vars {
        let g = var_ds.get(&var_id).copied().unwrap_or(0);
        let b = var_binding.get(&var_id).copied().unwrap_or(0);
        combined_bindings_per_group.entry(g).or_default().push(b);
    }
    for v in combined_bindings_per_group.values_mut() {
        v.sort_unstable();
        v.dedup();
    }

    // sampler_binding_for: var_id → compact sampler binding number
    let mut sampler_binding_for: HashMap<u32, u32> = HashMap::new();
    for (&var_id, _) in &combined_vars {
        let g = var_ds.get(&var_id).copied().unwrap_or(0);
        let b = var_binding.get(&var_id).copied().unwrap_or(0);
        let max_b = max_binding_per_group.get(&g).copied().unwrap_or(b);
        let sorted = combined_bindings_per_group
            .get(&g)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let samp_b = compact_sampler_binding(b, max_b, sorted);
        sampler_binding_for.insert(var_id, samp_b);
        log::debug!(
            "CIS split: group={} tex_binding={} → sampler_binding={}",
            g, b, samp_b
        );
    }

    // ── Phase 2: find OpLoad of combined vars inside function bodies ──────────

    // combined_loads: load_result_id → var_id
    let mut combined_loads = HashMap::<u32, u32>::new();
    let mut in_fn = false;
    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = words[i] & 0xFFFF;
        if wc == 0 || i + wc > words.len() {
            break;
        }
        match op {
            OP_FUNCTION => {
                in_fn = true;
            }
            OP_FUNCTION_END => {
                in_fn = false;
            }
            OP_LOAD if in_fn && wc >= 4 => {
                let result_id = words[i + 2];
                let pointer_id = words[i + 3];
                if combined_vars.contains_key(&pointer_id) {
                    combined_loads.insert(result_id, pointer_id);
                }
            }
            _ => {}
        }
        i += wc;
    }

    if combined_loads.is_empty() {
        return words.to_vec();
    }

    // ── Phase 3: allocate new SPIR-V IDs ──────────────────────────────────────

    let old_bound = words[3];
    let mut next_id = old_bound;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    // New TypeSampler (if not already present)
    let sampler_type_id = existing_sampler_type.unwrap_or_else(&mut alloc);
    let need_sampler_type = existing_sampler_type.is_none();

    // Existing TypePointer(UniformConstant, sampler_type_id), if any
    let mut existing_ptr_to_sampler: Option<u32> = None;
    {
        let mut j = 5usize;
        while j < words.len() {
            let wc = (words[j] >> 16) as usize;
            let op = words[j] & 0xFFFF;
            if wc == 0 || j + wc > words.len() {
                break;
            }
            if op == OP_TYPE_POINTER
                && wc == 4
                && words[j + 2] == SC_UNIFORM_CONSTANT
                && words[j + 3] == sampler_type_id
            {
                existing_ptr_to_sampler = Some(words[j + 1]);
                break;
            }
            j += wc;
        }
    }
    let ptr_sampler_id = existing_ptr_to_sampler.unwrap_or_else(&mut alloc);
    let need_ptr_sampler = existing_ptr_to_sampler.is_none();

    // Companion sampler OpVariable for each combined image sampler variable
    let mut companion_samp_var: HashMap<u32, u32> = HashMap::new();
    for &var_id in combined_vars.keys() {
        companion_samp_var.insert(var_id, alloc());
    }

    // For each combined load: separate image-load result and sampler-load result
    let mut img_load_id: HashMap<u32, u32> = HashMap::new();
    let mut samp_load_id: HashMap<u32, u32> = HashMap::new();
    for &load_result in combined_loads.keys() {
        img_load_id.insert(load_result, alloc());
        samp_load_id.insert(load_result, alloc());
    }

    // ── Phase 4: build rewritten binary ──────────────────────────────────────

    let extra = (need_sampler_type as usize) * 2
        + (need_ptr_sampler as usize) * 4
        + combined_vars.len() * 4
        + combined_loads.len() * 9;

    let mut out: Vec<u32> = Vec::with_capacity(words.len() + extra);

    // Updated header (new bound)
    out.extend_from_slice(&[words[0], words[1], words[2], next_id, words[4]]);

    let mut types_inserted = false;
    let mut in_fn = false;

    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = words[i] & 0xFFFF;
        if wc == 0 || i + wc > words.len() {
            out.extend_from_slice(&words[i..]);
            break;
        }
        let instr = &words[i..i + wc];

        match op {
            // ── Global variable declaration ───────────────────────────────────
            OP_VARIABLE if !in_fn => {
                // On the first global variable, emit any new type declarations.
                if !types_inserted {
                    if need_sampler_type {
                        out.extend_from_slice(&[(2 << 16) | OP_TYPE_SAMPLER, sampler_type_id]);
                    }
                    if need_ptr_sampler {
                        out.extend_from_slice(&[
                            (4 << 16) | OP_TYPE_POINTER,
                            ptr_sampler_id,
                            SC_UNIFORM_CONSTANT,
                            sampler_type_id,
                        ]);
                    }
                    types_inserted = true;
                }

                out.extend_from_slice(instr);

                // If this is a combined image sampler var, append its companion sampler.
                let var_id = instr[2];
                if let Some(&samp_var) = companion_samp_var.get(&var_id) {
                    out.extend_from_slice(&[
                        (4 << 16) | OP_VARIABLE,
                        ptr_sampler_id,
                        samp_var,
                        SC_UNIFORM_CONSTANT,
                    ]);
                }
            }

            // ── Decoration: mirror DescriptorSet/Binding to companion vars ────
            OP_DECORATE if wc >= 4 => {
                out.extend_from_slice(instr);
                let target = instr[1];
                let dec = instr[2];
                let val = instr[3];
                if let Some(&samp_var) = companion_samp_var.get(&target) {
                    match dec {
                        DEC_DESCRIPTOR_SET => {
                            out.extend_from_slice(&[
                                (4 << 16) | OP_DECORATE,
                                samp_var,
                                DEC_DESCRIPTOR_SET,
                                val,
                            ]);
                        }
                        DEC_BINDING => {
                            // Use compact sampler binding (stays within WebGPU's 1000 limit).
                            let samp_b = sampler_binding_for
                                .get(&target)
                                .copied()
                                .unwrap_or(val + 1);
                            out.extend_from_slice(&[
                                (4 << 16) | OP_DECORATE,
                                samp_var,
                                DEC_BINDING,
                                samp_b,
                            ]);
                        }
                        _ => {}
                    }
                }
            }

            // ── Function boundaries ───────────────────────────────────────────
            OP_FUNCTION => {
                in_fn = true;
                out.extend_from_slice(instr);
            }
            OP_FUNCTION_END => {
                in_fn = false;
                out.extend_from_slice(instr);
            }

            // ── Expand combined-image-sampler loads inside functions ──────────
            OP_LOAD if in_fn && wc >= 4 => {
                let result_id = instr[2];
                let pointer_id = instr[3];

                if let Some(&var_id) = combined_loads.get(&result_id) {
                    // Expand into three instructions:
                    //   %R_img  = OpLoad %SI_type %pointer_id [mem_access?]
                    //   %R_samp = OpLoad %sampler_type_id %samp_var
                    //   %R      = OpSampledImage %SI_type %R_img %R_samp
                    let r_img = img_load_id[&result_id];
                    let r_samp = samp_load_id[&result_id];
                    let samp_var = companion_samp_var[&var_id];
                    let si_type = instr[1];
                    let extra_wc = wc - 4;

                    // (1) Image load (same pointer, new result ID)
                    out.push(((4 + extra_wc as u32) << 16) | OP_LOAD);
                    out.push(si_type);
                    out.push(r_img);
                    out.push(pointer_id);
                    out.extend_from_slice(&instr[4..]);

                    // (2) Sampler load (no memory-access operand needed for Handle types)
                    out.extend_from_slice(&[
                        (4 << 16) | OP_LOAD,
                        sampler_type_id,
                        r_samp,
                        samp_var,
                    ]);

                    // (3) Combine into OpSampledImage — result keeps original ID %R
                    out.extend_from_slice(&[
                        (5 << 16) | OP_SAMPLED_IMAGE,
                        si_type,
                        result_id,
                        r_img,
                        r_samp,
                    ]);
                } else {
                    out.extend_from_slice(instr);
                }
            }

            // ── Everything else passes through unchanged ───────────────────────
            _ => {
                out.extend_from_slice(instr);
            }
        }

        i += wc;
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_cache_basic() {
        let cache = ShaderCache::new();
        assert_eq!(cache.cache_size(), 0);
        cache.clear();
        assert_eq!(cache.cache_size(), 0);
    }

    #[test]
    fn test_push_constant_transform() {
        let mut module = naga::Module::default();
        let result = ShaderCache::transform_push_constants_to_uniform(&mut module);
        assert!(result.is_ok(), "Transform should succeed on empty module");
    }

    #[test]
    fn test_no_combined_samplers_passthrough() {
        let not_spirv = vec![0u32, 1, 2, 3, 4];
        assert!(!spv_has_combined_image_samplers(&not_spirv));

        let minimal_header = vec![SPV_MAGIC, 0x00010000u32, 0, 1, 0];
        assert!(!spv_has_combined_image_samplers(&minimal_header));
    }

    #[test]
    fn test_compact_sampler_binding() {
        // Single combined sampler at binding 0, max_binding=2 → sampler at 3
        assert_eq!(compact_sampler_binding(0, 2, &[0]), 3);
        // Two combined samplers at 0, 2 with max_binding=5
        assert_eq!(compact_sampler_binding(0, 5, &[0, 2]), 6);
        assert_eq!(compact_sampler_binding(2, 5, &[0, 2]), 7);
    }
}
