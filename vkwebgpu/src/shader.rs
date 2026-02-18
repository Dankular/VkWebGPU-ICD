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

    fn translate_spirv_to_wgsl(spirv: &[u32]) -> Result<String> {
        log::debug!("Translating SPIR-V to WGSL ({} words)", spirv.len());

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_cache_basic() {
        let cache = ShaderCache::new();

        // Test cache initialization
        assert_eq!(cache.cache_size(), 0);

        // Test cache clear
        cache.clear();
        assert_eq!(cache.cache_size(), 0);
    }

    #[test]
    fn test_push_constant_transform() {
        // Test the transformation logic without complex module creation
        // The actual transformation will be tested when real shaders are processed

        let mut module = naga::Module::default();

        // Test with empty module - should succeed without errors
        let result = ShaderCache::transform_push_constants_to_uniform(&mut module);
        assert!(result.is_ok(), "Transform should succeed on empty module");
    }
}
