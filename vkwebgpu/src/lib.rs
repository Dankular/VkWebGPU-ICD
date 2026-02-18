//! VkWebGPU ICD - Vulkan to WebGPU Translation Layer
//!
//! This is a Vulkan Installable Client Driver (ICD) that translates Vulkan API calls
//! to WebGPU, enabling Vulkan applications (including DXVK-translated DirectX games)
//! to run in web browsers.
//!
//! Architecture:
//! ```text
//! Game → DirectX → DXVK → Vulkan API → VkWebGPU ICD → WebGPU → Browser GPU
//! ```

pub mod backend;
pub mod error;
pub mod handle;
pub mod format;
pub mod shader;
pub mod instance;
pub mod device;
pub mod queue;
pub mod memory;
pub mod buffer;
pub mod image;
pub mod sampler;
pub mod descriptor;
pub mod pipeline;
pub mod render_pass;
pub mod framebuffer;
pub mod command_buffer;
pub mod command_pool;
pub mod sync;
pub mod swapchain;
pub mod push_constants;
pub mod icd;

use log::info;

/// Initialize the VkWebGPU ICD
pub fn init() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .try_init();
    }
    
    info!("VkWebGPU ICD v0.1.0 initialized");
    info!("Vulkan API → WebGPU translation layer active");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
    }
}
