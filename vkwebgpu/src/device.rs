//! Vulkan Device implementation
//!
//! Maps VkDevice to WebGPU GPUDevice

use ash::vk::{self, Handle};
use log::info;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::ffi::CStr;
use std::sync::Arc;

use crate::backend::WebGPUBackend;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;
use crate::instance;
use crate::push_constants::PushConstantRingBuffer;
use crate::shader::ShaderCache;

/// Global device allocator
pub static DEVICE_ALLOCATOR: Lazy<HandleAllocator<VkDeviceData>> =
    Lazy::new(|| HandleAllocator::new());

/// Device data
pub struct VkDeviceData {
    pub physical_device: vk::PhysicalDevice,
    pub enabled_features: vk::PhysicalDeviceFeatures,
    pub enabled_extensions: Vec<String>,
    pub queues: RwLock<Vec<vk::Queue>>,
    pub backend: Arc<WebGPUBackend>,
    pub shader_cache: ShaderCache,
    pub push_constant_buffer: Arc<PushConstantRingBuffer>,
}

pub unsafe fn create_device(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_device: *mut vk::Device,
) -> Result<()> {
    let create_info = &*p_create_info;

    info!("Creating logical device");

    // Parse enabled features
    let enabled_features = if !create_info.p_enabled_features.is_null() {
        *create_info.p_enabled_features
    } else {
        vk::PhysicalDeviceFeatures::default()
    };

    // Parse enabled extensions
    let enabled_extensions = if create_info.enabled_extension_count > 0
        && !create_info.pp_enabled_extension_names.is_null()
    {
        let ext_slice = std::slice::from_raw_parts(
            create_info.pp_enabled_extension_names,
            create_info.enabled_extension_count as usize,
        );
        ext_slice
            .iter()
            .map(|&ext| CStr::from_ptr(ext).to_string_lossy().into_owned())
            .collect()
    } else {
        Vec::new()
    };

    info!("Enabled extensions: {:?}", enabled_extensions);

    // Get the WebGPU backend from the physical device
    let pd_data = instance::get_physical_device_data(physical_device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid physical device".to_string()))?;

    let backend = if let Some(backend) = &pd_data.backend {
        backend.clone()
    } else {
        // For WASM, we need to create the backend here
        #[cfg(target_arch = "wasm32")]
        {
            // This would be async in real WASM context
            return Err(VkError::DeviceCreationFailed(
                "WASM backend creation not yet implemented".to_string(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            return Err(VkError::DeviceCreationFailed(
                "No backend available".to_string(),
            ));
        }
    };

    // Initialize push constant ring buffer for DXVK compatibility
    let push_constant_buffer = Arc::new(PushConstantRingBuffer::new(
        &backend.device,
        PushConstantRingBuffer::DEFAULT_CAPACITY,
    ));

    let device_data = VkDeviceData {
        physical_device,
        enabled_features,
        enabled_extensions,
        queues: RwLock::new(Vec::new()),
        backend,
        shader_cache: ShaderCache::new(),
        push_constant_buffer,
    };

    let device_handle = DEVICE_ALLOCATOR.allocate(device_data);
    *p_device = Handle::from_raw(device_handle);

    info!("Logical device created successfully");
    Ok(())
}

pub unsafe fn destroy_device(device: vk::Device, _p_allocator: *const vk::AllocationCallbacks) {
    let device_handle = device.as_raw();
    if let Some(device_data) = DEVICE_ALLOCATOR.get(device_handle) {
        // Clear shader cache
        device_data.shader_cache.clear();
    }
    DEVICE_ALLOCATOR.remove(device_handle);
    info!("Device destroyed");
}

pub fn get_device_data(device: vk::Device) -> Option<Arc<VkDeviceData>> {
    DEVICE_ALLOCATOR.get(device.as_raw())
}

pub unsafe fn device_wait_idle(device: vk::Device) -> Result<()> {
    let device_data = get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    #[cfg(not(target_arch = "wasm32"))]
    {
        device_data.backend.device.poll(wgpu::Maintain::Wait);
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM: would need to use promises
    }

    Ok(())
}

/// Enumerate device extensions
pub unsafe fn enumerate_device_extension_properties(
    _physical_device: vk::PhysicalDevice,
    _p_layer_name: *const std::os::raw::c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> Result<()> {
    // Extensions our device/ICD supports
    const DEVICE_EXTENSIONS: &[(&str, u32)] = &[
        ("VK_KHR_swapchain", 70),   // Swapchain support
        ("VK_KHR_maintenance1", 2), // DXVK uses this
        ("VK_KHR_maintenance2", 1),
        ("VK_KHR_maintenance3", 1),
        ("VK_KHR_dedicated_allocation", 3),
        ("VK_KHR_get_memory_requirements2", 1),
        ("VK_EXT_descriptor_indexing", 2), // DXVK may need this
        ("VK_KHR_bind_memory2", 1),
    ];

    if p_properties.is_null() {
        *p_property_count = DEVICE_EXTENSIONS.len() as u32;
    } else {
        let count = (*p_property_count as usize).min(DEVICE_EXTENSIONS.len());
        let props = std::slice::from_raw_parts_mut(p_properties, count);

        for (i, (name, version)) in DEVICE_EXTENSIONS.iter().enumerate().take(count) {
            props[i] = vk::ExtensionProperties {
                extension_name: {
                    let mut arr = [0i8; 256];
                    for (j, &c) in name.as_bytes().iter().enumerate() {
                        arr[j] = c as i8;
                    }
                    arr
                },
                spec_version: *version,
            };
        }

        *p_property_count = count as u32;
    }

    Ok(())
}
