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

        // CRITICAL: Transform push constants to uniform buffers
        // This is required because WebGPU doesn't support push constants directly
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

    /// Transform push constants to uniform buffers
    ///
    /// WebGPU doesn't support push constants, so we transform them to uniform buffers.
    /// Push constant blocks are converted to uniform buffers at group=0, binding=0.
    /// This matches the binding used by our push constant emulation in the pipeline.
    fn transform_push_constants_to_uniform(module: &mut naga::Module) -> Result<()> {
        use naga::{AddressSpace, ResourceBinding};

        let mut transformed_count = 0;

        // Iterate through all global variables to find push constants
        for (handle, var) in module.global_variables.iter_mut() {
            if var.space == AddressSpace::PushConstant {
                // Log the transformation
                if let Some(ref name) = var.name {
                    log::debug!(
                        "Transforming push constant variable '{}' to uniform buffer",
                        name
                    );
                } else {
                    log::debug!(
                        "Transforming push constant variable {:?} to uniform buffer",
                        handle
                    );
                }

                // Change address space from PushConstant to Uniform
                var.space = AddressSpace::Uniform;

                // Set binding to group=0, binding=0
                // This is where we'll bind our emulated push constant buffer
                var.binding = Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                });

                transformed_count += 1;
            }
        }

        if transformed_count > 0 {
            log::info!(
                "Transformed {} push constant block(s) to uniform buffer(s)",
                transformed_count
            );
        }

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
