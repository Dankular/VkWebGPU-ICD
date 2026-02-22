//! Vulkan Device implementation
//!
//! Maps VkDevice to WebGPU GPUDevice

use ash::vk::{self, Handle};
use log::info;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::ffi::CStr;
use std::sync::Arc;

use std::collections::HashMap;

use crate::backend::WebGPUBackend;
use crate::error::{Result, VkError};
use crate::handle::{self, HandleAllocator};
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
    #[cfg(feature = "webx")]
    let push_constant_buffer = Arc::new(PushConstantRingBuffer::new_stub(
        PushConstantRingBuffer::DEFAULT_CAPACITY,
    ));
    #[cfg(not(feature = "webx"))]
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

    let device_index = DEVICE_ALLOCATOR.allocate(device_data);
    // Dispatchable handle: heap-alloc a slot with ICD_LOADER_MAGIC at offset 0
    let device_ptr = handle::alloc_dispatchable(device_index);
    *p_device = Handle::from_raw(device_ptr);

    info!("Logical device created successfully");
    Ok(())
}

// ===========================================================================
// VK_EXT_private_data
// ===========================================================================

/// Global private-data store: (object_handle, slot_handle) → u64
static PRIVATE_DATA: Lazy<RwLock<HashMap<(u64, u64), u64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Counter for private data slot handles
static PRIVATE_DATA_SLOT_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

pub unsafe fn create_private_data_slot(
    _device: vk::Device,
    _p_create_info: *const vk::PrivateDataSlotCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_private_data_slot: *mut vk::PrivateDataSlot,
) -> Result<()> {
    let handle = PRIVATE_DATA_SLOT_COUNTER
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    *p_private_data_slot = Handle::from_raw(handle);
    Ok(())
}

pub unsafe fn destroy_private_data_slot(
    _device: vk::Device,
    private_data_slot: vk::PrivateDataSlot,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if private_data_slot == vk::PrivateDataSlot::null() {
        return;
    }
    // Remove all entries for this slot
    let slot_raw = private_data_slot.as_raw();
    PRIVATE_DATA.write().retain(|k, _| k.1 != slot_raw);
}

pub unsafe fn set_private_data(
    _device: vk::Device,
    _object_type: vk::ObjectType,
    object_handle: u64,
    private_data_slot: vk::PrivateDataSlot,
    data: u64,
) -> Result<()> {
    PRIVATE_DATA
        .write()
        .insert((object_handle, private_data_slot.as_raw()), data);
    Ok(())
}

pub unsafe fn get_private_data(
    _device: vk::Device,
    _object_type: vk::ObjectType,
    object_handle: u64,
    private_data_slot: vk::PrivateDataSlot,
    p_data: *mut u64,
) {
    let val = PRIVATE_DATA
        .read()
        .get(&(object_handle, private_data_slot.as_raw()))
        .copied()
        .unwrap_or(0);
    *p_data = val;
}

/// vkGetDescriptorSetLayoutSupport / vkGetDescriptorSetLayoutSupportKHR
/// We always report the layout as supported.
pub unsafe fn get_descriptor_set_layout_support(
    _device: vk::Device,
    _p_create_info: *const vk::DescriptorSetLayoutCreateInfo,
    p_support: *mut vk::DescriptorSetLayoutSupport,
) {
    if !p_support.is_null() {
        (*p_support).supported = vk::TRUE;
    }
}

pub unsafe fn destroy_device(device: vk::Device, _p_allocator: *const vk::AllocationCallbacks) {
    let raw = device.as_raw();
    if let Some(device_data) = DEVICE_ALLOCATOR.get_dispatchable(raw) {
        device_data.shader_cache.clear();
    }
    DEVICE_ALLOCATOR.remove_dispatchable(raw);
    handle::free_dispatchable(raw);
    info!("Device destroyed");
}

pub fn get_device_data(device: vk::Device) -> Option<Arc<VkDeviceData>> {
    DEVICE_ALLOCATOR.get_dispatchable(device.as_raw())
}

pub unsafe fn device_wait_idle(device: vk::Device) -> Result<()> {
    let device_data = get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    #[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
    {
        device_data.backend.device.poll(wgpu::Maintain::Wait);
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM: would need to use promises
    }

    #[cfg(feature = "webx")]
    {
        // WebX: host handles GPU sync; no local device to poll
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
        // Swapchain
        ("VK_KHR_swapchain", 70),
        // Maintenance
        ("VK_KHR_maintenance1", 2),
        ("VK_KHR_maintenance2", 1),
        ("VK_KHR_maintenance3", 1),
        ("VK_KHR_maintenance4", 2),
        // Memory
        ("VK_KHR_dedicated_allocation", 3),
        ("VK_KHR_get_memory_requirements2", 1),
        ("VK_KHR_bind_memory2", 1),
        // Descriptors
        ("VK_EXT_descriptor_indexing", 2),
        ("VK_EXT_scalar_block_layout", 1),
        // Rendering / render pass
        ("VK_KHR_dynamic_rendering", 1),
        ("VK_KHR_create_renderpass2", 1),
        ("VK_KHR_imageless_framebuffer", 1),
        // Synchronization
        ("VK_KHR_synchronization2", 1),
        ("VK_KHR_timeline_semaphore", 2),
        // Shader features
        ("VK_KHR_shader_draw_parameters", 1),
        ("VK_KHR_separate_depth_stencil_layouts", 1),
        ("VK_KHR_uniform_buffer_standard_layout", 1),
        ("VK_KHR_buffer_device_address", 1),
        // Extended dynamic state
        ("VK_EXT_extended_dynamic_state", 1),
        ("VK_EXT_extended_dynamic_state2", 1),
        ("VK_EXT_depth_clip_enable", 1),
        // Misc
        ("VK_EXT_host_query_reset", 1),
        ("VK_KHR_image_format_list", 1),
        ("VK_KHR_copy_commands2", 1),
        ("VK_KHR_format_feature_flags2", 1),
        ("VK_EXT_private_data", 1),
        ("VK_KHR_pipeline_library", 1),
        ("VK_KHR_driver_properties", 1),
        ("VK_KHR_depth_stencil_resolve", 1),
        ("VK_EXT_sampler_filter_minmax", 2),
        ("VK_KHR_multiview", 1),
        ("VK_KHR_device_group", 3),
        // vkd3d-proton / DX12 compatibility
        ("VK_EXT_robustness2", 1),                  // null descriptor support (DX12 tier)
        ("VK_KHR_16bit_storage", 1),                // 16-bit types in shaders
        ("VK_KHR_8bit_storage", 1),                 // 8-bit types in shaders
        ("VK_KHR_storage_buffer_storage_class", 1), // required by 16bit/8bit storage
        ("VK_EXT_shader_viewport_index_layer", 1),  // SV_ViewportArrayIndex / SV_RenderTargetArrayIndex
        ("VK_KHR_draw_indirect_count", 1),          // ExecuteIndirect
        ("VK_EXT_mutable_descriptor_type", 1),      // vkd3d-proton descriptor heap model
        ("VK_VALVE_mutable_descriptor_type", 1),    // alias used by older vkd3d-proton
        ("VK_EXT_vertex_attribute_divisor", 1),     // D3D instance data step rate
        ("VK_KHR_shader_float16_int8", 1),          // 16/8-bit arithmetic in SPIR-V
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
