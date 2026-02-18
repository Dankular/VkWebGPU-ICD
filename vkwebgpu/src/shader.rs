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
        // Parse SPIR-V using Naga
        let module = naga::front::spv::parse_u32_slice(
            spirv,
            &naga::front::spv::Options {
                adjust_coordinate_space: true,
                strict_capabilities: false,
                block_ctx_dump_prefix: None,
            },
        )
        .map_err(|e| VkError::ShaderTranslationFailed(format!("SPIR-V parse error: {:?}", e)))?;

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
    fn test_shader_cache() {
        let cache = ShaderCache::new();

        // Simple SPIR-V shader (vertex shader that passes through position)
        let spirv = include_bytes!("../test_data/simple.spv");
        let spirv_words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        if !spirv_words.is_empty() {
            let result = cache.get_or_translate(&spirv_words);
            assert!(result.is_ok() || spirv_words.len() < 5); // Basic validation
        }
    }
}
