//! Vulkan Instance implementation
//!
//! Maps VkInstance to WebGPU adapter enumeration

use ash::vk::{self, Handle};
use log::{debug, info};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::ffi::CStr;
use std::sync::Arc;

use crate::backend::WebGPUBackend;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

/// Global instance allocator
pub static INSTANCE_ALLOCATOR: Lazy<HandleAllocator<VkInstanceData>> =
    Lazy::new(|| HandleAllocator::new());

/// Global physical device allocator
pub static PHYSICAL_DEVICE_ALLOCATOR: Lazy<HandleAllocator<VkPhysicalDeviceData>> =
    Lazy::new(|| HandleAllocator::new());

/// Instance data
pub struct VkInstanceData {
    pub application_info: Option<ApplicationInfo>,
    pub enabled_extensions: Vec<String>,
    pub enabled_layers: Vec<String>,
    pub physical_devices: RwLock<Vec<vk::PhysicalDevice>>,
    pub backend: Option<Arc<WebGPUBackend>>,
}

#[derive(Clone, Debug)]
pub struct ApplicationInfo {
    pub application_name: String,
    pub application_version: u32,
    pub engine_name: String,
    pub engine_version: u32,
    pub api_version: u32,
}

/// Physical device data
pub struct VkPhysicalDeviceData {
    pub instance: vk::Instance,
    pub adapter_info: AdapterInfo,
    pub backend: Option<Arc<WebGPUBackend>>,
}

#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
}

pub unsafe fn create_instance(
    p_create_info: *const vk::InstanceCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_instance: *mut vk::Instance,
) -> Result<()> {
    let create_info = &*p_create_info;

    // Parse application info
    let app_info = if !create_info.p_application_info.is_null() {
        let info = &*create_info.p_application_info;
        Some(ApplicationInfo {
            application_name: if !info.p_application_name.is_null() {
                CStr::from_ptr(info.p_application_name)
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::from("Unknown")
            },
            application_version: info.application_version,
            engine_name: if !info.p_engine_name.is_null() {
                CStr::from_ptr(info.p_engine_name)
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::from("Unknown")
            },
            engine_version: info.engine_version,
            api_version: info.api_version,
        })
    } else {
        None
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

    // Parse enabled layers
    let enabled_layers =
        if create_info.enabled_layer_count > 0 && !create_info.pp_enabled_layer_names.is_null() {
            let layer_slice = std::slice::from_raw_parts(
                create_info.pp_enabled_layer_names,
                create_info.enabled_layer_count as usize,
            );
            layer_slice
                .iter()
                .map(|&layer| CStr::from_ptr(layer).to_string_lossy().into_owned())
                .collect()
        } else {
            Vec::new()
        };

    info!(
        "Creating instance for application: {:?}, extensions: {:?}, layers: {:?}",
        app_info.as_ref().map(|a| &a.application_name),
        enabled_extensions,
        enabled_layers
    );

    // Create WebGPU backend once for the instance
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
    let backend = Some(Arc::new(WebGPUBackend::new()?));

    // WebX: stub backend (no local wgpu device; all GPU work is IPC to the browser host)
    #[cfg(feature = "webx")]
    let backend = Some(Arc::new(WebGPUBackend::new()?));

    #[cfg(target_arch = "wasm32")]
    let backend = None;

    let instance_data = VkInstanceData {
        application_info: app_info,
        enabled_extensions,
        enabled_layers,
        physical_devices: RwLock::new(Vec::new()),
        backend,
    };

    let instance_handle = INSTANCE_ALLOCATOR.allocate(instance_data);
    *p_instance = Handle::from_raw(instance_handle);

    // Enumerate physical devices immediately
    enumerate_physical_devices_internal(Handle::from_raw(instance_handle))?;

    Ok(())
}

pub unsafe fn destroy_instance(
    instance: vk::Instance,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    let instance_handle = instance.as_raw();

    // Clean up physical devices
    if let Some(instance_data) = INSTANCE_ALLOCATOR.get(instance_handle) {
        let physical_devices = instance_data.physical_devices.read().clone();
        for pd in physical_devices {
            PHYSICAL_DEVICE_ALLOCATOR.remove(pd.as_raw());
        }
    }

    INSTANCE_ALLOCATOR.remove(instance_handle);
    info!("Instance destroyed");
}

fn enumerate_physical_devices_internal(instance: vk::Instance) -> Result<()> {
    let instance_handle = instance.as_raw();
    let instance_data = INSTANCE_ALLOCATOR
        .get(instance_handle)
        .ok_or_else(|| VkError::InvalidHandle("Invalid instance".to_string()))?;

    // Reuse the backend from instance (created during instance creation)
    let backend = instance_data.backend.clone();

    // Get adapter info
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
    let adapter_info = {
        let info = backend.as_ref().unwrap().adapter.get_info();
        AdapterInfo {
            name: info.name.clone(),
            vendor_id: info.vendor,
            device_id: info.device,
            device_type: match info.device_type {
                wgpu::DeviceType::DiscreteGpu => vk::PhysicalDeviceType::DISCRETE_GPU,
                wgpu::DeviceType::IntegratedGpu => vk::PhysicalDeviceType::INTEGRATED_GPU,
                wgpu::DeviceType::VirtualGpu => vk::PhysicalDeviceType::VIRTUAL_GPU,
                wgpu::DeviceType::Cpu => vk::PhysicalDeviceType::CPU,
                wgpu::DeviceType::Other => vk::PhysicalDeviceType::OTHER,
            },
        }
    };

    // WebX: report a virtual browser GPU adapter (real GPU info comes from the host)
    #[cfg(feature = "webx")]
    let adapter_info = AdapterInfo {
        name: String::from("WebGPU Browser Adapter"),
        vendor_id: 0x1414, // Microsoft virtual device ID (placeholder)
        device_id: 0x0001,
        device_type: vk::PhysicalDeviceType::VIRTUAL_GPU,
    };

    #[cfg(target_arch = "wasm32")]
    let adapter_info = AdapterInfo {
        name: String::from("WebGPU Virtual Adapter"),
        vendor_id: 0x10DE,
        device_id: 0x0001,
        device_type: vk::PhysicalDeviceType::VIRTUAL_GPU,
    };

    let physical_device_data = VkPhysicalDeviceData {
        instance,
        adapter_info,
        backend,
    };

    let physical_device_handle = PHYSICAL_DEVICE_ALLOCATOR.allocate(physical_device_data);
    let physical_device = Handle::from_raw(physical_device_handle);

    let mut devices = instance_data.physical_devices.write();
    devices.push(physical_device);

    debug!("Enumerated 1 physical device");
    Ok(())
}

pub unsafe fn enumerate_physical_devices(
    instance: vk::Instance,
    p_physical_device_count: *mut u32,
    p_physical_devices: *mut vk::PhysicalDevice,
) -> Result<()> {
    let instance_handle = instance.as_raw();
    let instance_data = INSTANCE_ALLOCATOR
        .get(instance_handle)
        .ok_or_else(|| VkError::InvalidHandle("Invalid instance".to_string()))?;

    let devices = instance_data.physical_devices.read();
    let device_count = devices.len() as u32;

    if p_physical_devices.is_null() {
        *p_physical_device_count = device_count;
        return Ok(());
    }

    let requested_count = *p_physical_device_count as usize;
    let copy_count = requested_count.min(devices.len());

    let dest_slice = std::slice::from_raw_parts_mut(p_physical_devices, copy_count);
    dest_slice.copy_from_slice(&devices[..copy_count]);

    *p_physical_device_count = copy_count as u32;

    Ok(())
}

pub unsafe fn get_physical_device_properties(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties,
) {
    let pd_handle = physical_device.as_raw();
    let pd_data = match PHYSICAL_DEVICE_ALLOCATOR.get(pd_handle) {
        Some(data) => data,
        None => return,
    };

    let properties = &mut *p_properties;

    properties.api_version = vk::make_api_version(0, 1, 3, 0);
    properties.driver_version = vk::make_api_version(0, 0, 1, 0);
    properties.vendor_id = pd_data.adapter_info.vendor_id;
    properties.device_id = pd_data.adapter_info.device_id;
    properties.device_type = pd_data.adapter_info.device_type;

    // Copy device name
    let name_bytes = pd_data.adapter_info.name.as_bytes();
    let copy_len = name_bytes.len().min(vk::MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
    for (i, &byte) in name_bytes.iter().take(copy_len).enumerate() {
        properties.device_name[i] = byte as i8;
    }
    properties.device_name[copy_len] = 0;

    properties.limits = get_device_limits();
    properties.sparse_properties = vk::PhysicalDeviceSparseProperties::default();
}

pub unsafe fn get_physical_device_features(
    _physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures,
) {
    let features = &mut *p_features;
    *features = vk::PhysicalDeviceFeatures::default();

    features.robust_buffer_access = vk::TRUE;
    features.shader_clip_distance = vk::TRUE;
    features.shader_cull_distance = vk::TRUE;
    features.sampler_anisotropy = vk::TRUE;
    features.fill_mode_non_solid = vk::TRUE;
    features.depth_clamp = vk::TRUE;
    features.depth_bias_clamp = vk::TRUE;
    features.fragment_stores_and_atomics = vk::TRUE;
    features.vertex_pipeline_stores_and_atomics = vk::TRUE;
    features.shader_storage_image_extended_formats = vk::TRUE;
    features.shader_uniform_buffer_array_dynamic_indexing = vk::TRUE;
    features.shader_storage_buffer_array_dynamic_indexing = vk::TRUE;
}

pub unsafe fn get_physical_device_queue_family_properties(
    _physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties,
) {
    let queue_family_count = 1u32;

    if p_queue_family_properties.is_null() {
        *p_queue_family_property_count = queue_family_count;
        return;
    }

    let properties = &mut *p_queue_family_properties;
    properties.queue_flags =
        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
    properties.queue_count = 1;
    properties.timestamp_valid_bits = 64;
    properties.min_image_transfer_granularity = vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    };

    *p_queue_family_property_count = 1;
}

fn get_device_limits() -> vk::PhysicalDeviceLimits {
    vk::PhysicalDeviceLimits {
        max_image_dimension1_d: 16384,
        max_image_dimension2_d: 16384,
        max_image_dimension3_d: 2048,
        max_image_dimension_cube: 16384,
        max_image_array_layers: 2048,
        max_texel_buffer_elements: 134217728,
        max_uniform_buffer_range: 65536,
        max_storage_buffer_range: 134217728,
        max_push_constants_size: 128,
        max_memory_allocation_count: 4096,
        max_sampler_allocation_count: 4000,
        buffer_image_granularity: 1,
        sparse_address_space_size: 0,
        max_bound_descriptor_sets: 4,
        max_per_stage_descriptor_samplers: 16,
        max_per_stage_descriptor_uniform_buffers: 12,
        max_per_stage_descriptor_storage_buffers: 8,
        max_per_stage_descriptor_sampled_images: 16,
        max_per_stage_descriptor_storage_images: 8,
        max_per_stage_descriptor_input_attachments: 8,
        max_per_stage_resources: 128,
        max_descriptor_set_samplers: 80,
        max_descriptor_set_uniform_buffers: 90,
        max_descriptor_set_uniform_buffers_dynamic: 8,
        max_descriptor_set_storage_buffers: 40,
        max_descriptor_set_storage_buffers_dynamic: 8,
        max_descriptor_set_sampled_images: 80,
        max_descriptor_set_storage_images: 40,
        max_descriptor_set_input_attachments: 8,
        max_vertex_input_attributes: 16,
        max_vertex_input_bindings: 16,
        max_vertex_input_attribute_offset: 2047,
        max_vertex_input_binding_stride: 2048,
        max_vertex_output_components: 64,
        max_tessellation_generation_level: 0,
        max_tessellation_patch_size: 0,
        max_tessellation_control_per_vertex_input_components: 0,
        max_tessellation_control_per_vertex_output_components: 0,
        max_tessellation_control_per_patch_output_components: 0,
        max_tessellation_control_total_output_components: 0,
        max_tessellation_evaluation_input_components: 0,
        max_tessellation_evaluation_output_components: 0,
        max_geometry_shader_invocations: 0,
        max_geometry_input_components: 0,
        max_geometry_output_components: 0,
        max_geometry_output_vertices: 0,
        max_geometry_total_output_components: 0,
        max_fragment_input_components: 64,
        max_fragment_output_attachments: 8,
        max_fragment_dual_src_attachments: 1,
        max_fragment_combined_output_resources: 16,
        max_compute_shared_memory_size: 32768,
        max_compute_work_group_count: [65535, 65535, 65535],
        max_compute_work_group_invocations: 256,
        max_compute_work_group_size: [256, 256, 64],
        sub_pixel_precision_bits: 8,
        sub_texel_precision_bits: 8,
        mipmap_precision_bits: 8,
        max_draw_indexed_index_value: 0xFFFFFFFF,
        max_draw_indirect_count: 0xFFFFFFFF,
        max_sampler_lod_bias: 16.0,
        max_sampler_anisotropy: 16.0,
        max_viewports: 1,
        max_viewport_dimensions: [16384, 16384],
        viewport_bounds_range: [-32768.0, 32767.0],
        viewport_sub_pixel_bits: 0,
        min_memory_map_alignment: 64,
        min_texel_buffer_offset_alignment: 256,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 256,
        min_texel_offset: -8,
        max_texel_offset: 7,
        min_texel_gather_offset: -8,
        max_texel_gather_offset: 7,
        min_interpolation_offset: -0.5,
        max_interpolation_offset: 0.5,
        sub_pixel_interpolation_offset_bits: 4,
        max_framebuffer_width: 16384,
        max_framebuffer_height: 16384,
        max_framebuffer_layers: 256,
        framebuffer_color_sample_counts: vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4,
        framebuffer_depth_sample_counts: vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4,
        framebuffer_stencil_sample_counts: vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4,
        framebuffer_no_attachments_sample_counts: vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4,
        max_color_attachments: 8,
        sampled_image_color_sample_counts: vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4,
        sampled_image_integer_sample_counts: vk::SampleCountFlags::TYPE_1,
        sampled_image_depth_sample_counts: vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4,
        sampled_image_stencil_sample_counts: vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4,
        storage_image_sample_counts: vk::SampleCountFlags::TYPE_1,
        max_sample_mask_words: 1,
        timestamp_compute_and_graphics: vk::TRUE,
        timestamp_period: 1.0,
        max_clip_distances: 8,
        max_cull_distances: 8,
        max_combined_clip_and_cull_distances: 8,
        discrete_queue_priorities: 2,
        point_size_range: [1.0, 64.0],
        line_width_range: [1.0, 1.0],
        point_size_granularity: 1.0,
        line_width_granularity: 1.0,
        strict_lines: vk::FALSE,
        standard_sample_locations: vk::TRUE,
        optimal_buffer_copy_offset_alignment: 1,
        optimal_buffer_copy_row_pitch_alignment: 1,
        non_coherent_atom_size: 256,
    }
}

pub fn get_instance_data(instance: vk::Instance) -> Option<Arc<VkInstanceData>> {
    INSTANCE_ALLOCATOR.get(instance.as_raw())
}

pub fn get_physical_device_data(
    physical_device: vk::PhysicalDevice,
) -> Option<Arc<VkPhysicalDeviceData>> {
    PHYSICAL_DEVICE_ALLOCATOR.get(physical_device.as_raw())
}

/// Enumerate instance extensions
pub unsafe fn enumerate_instance_extension_properties(
    _p_layer_name: *const std::os::raw::c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> Result<()> {
    // List of extensions our ICD supports
    const INSTANCE_EXTENSIONS: &[(&str, u32)] = &[
        ("VK_KHR_surface", 25),
        ("VK_KHR_win32_surface", 6),
        ("VK_KHR_get_physical_device_properties2", 2),
        ("VK_KHR_external_memory_capabilities", 1),
        ("VK_KHR_external_semaphore_capabilities", 1),
        ("VK_KHR_external_fence_capabilities", 1),
        ("VK_KHR_device_group_creation", 1),
        ("VK_KHR_portability_enumeration", 1),
        ("VK_EXT_debug_utils", 2),
    ];

    if p_properties.is_null() {
        *p_property_count = INSTANCE_EXTENSIONS.len() as u32;
    } else {
        let count = (*p_property_count as usize).min(INSTANCE_EXTENSIONS.len());
        let props = std::slice::from_raw_parts_mut(p_properties, count);

        for (i, (name, version)) in INSTANCE_EXTENSIONS.iter().enumerate().take(count) {
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

/// Get format properties
pub unsafe fn get_physical_device_format_properties(
    _physical_device: vk::PhysicalDevice,
    format: vk::Format,
    p_format_properties: *mut vk::FormatProperties,
) {
    let mut props = vk::FormatProperties::default();

    // Check if we support this format
    #[cfg(not(target_arch = "wasm32"))]
    let supported = crate::format::vk_to_wgpu_format(format).is_some();

    #[cfg(target_arch = "wasm32")]
    let supported = false; // Conservative for WASM

    if supported {
        // WebGPU supports most operations on supported formats
        let features = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;

        // Add color attachment features for color formats
        let color_features = if !matches!(
            format,
            vk::Format::D16_UNORM
                | vk::Format::D32_SFLOAT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::S8_UINT
        ) {
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
        } else {
            vk::FormatFeatureFlags::empty()
        };

        props.linear_tiling_features = features | color_features;
        props.optimal_tiling_features = features | color_features;
        props.buffer_features =
            vk::FormatFeatureFlags::VERTEX_BUFFER | vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER;

        // Add depth/stencil features for depth formats
        if matches!(
            format,
            vk::Format::D16_UNORM
                | vk::Format::D32_SFLOAT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
        ) {
            props.optimal_tiling_features |= vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
            props.linear_tiling_features |= vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
        }
    }

    *p_format_properties = props;
}

// ─── helper ──────────────────────────────────────────────────────────────────

/// Copy a Rust string into a fixed-size C-string array (null-terminated).
fn copy_str_to_c_array<const N: usize>(arr: &mut [i8; N], s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(N - 1);
    for i in 0..len {
        arr[i] = bytes[i] as i8;
    }
    arr[len] = 0;
}

// ─── Properties2 / Features2 ─────────────────────────────────────────────────

/// vkGetPhysicalDeviceProperties2 – walks the pNext chain and fills every
/// recognised extended-properties structure.
pub unsafe fn get_physical_device_properties2(
    physical_device: vk::PhysicalDevice,
    p_properties2: *mut vk::PhysicalDeviceProperties2,
) {
    if p_properties2.is_null() {
        return;
    }
    let properties2 = &mut *p_properties2;

    // Fill the base VkPhysicalDeviceProperties
    get_physical_device_properties(physical_device, &mut properties2.properties);

    debug!("GetPhysicalDeviceProperties2");

    let mut p_next = properties2.p_next as *mut vk::BaseOutStructure;
    while !p_next.is_null() {
        match (*p_next).s_type {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceVulkan11Properties);
                s.subgroup_size = 32;
                s.subgroup_supported_stages = vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT;
                s.subgroup_supported_operations = vk::SubgroupFeatureFlags::BASIC
                    | vk::SubgroupFeatureFlags::VOTE
                    | vk::SubgroupFeatureFlags::BALLOT;
                s.subgroup_quad_operations_in_all_stages = vk::FALSE;
                s.point_clipping_behavior = vk::PointClippingBehavior::ALL_CLIP_PLANES;
                s.max_multiview_view_count = 0;
                s.max_multiview_instance_index = 0;
                s.protected_no_fault = vk::FALSE;
                s.max_per_set_descriptors = 1024;
                s.max_memory_allocation_size = 2u64 * 1024 * 1024 * 1024;
                s.device_node_mask = 1;
                s.device_luid_valid = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceVulkan12Properties);
                s.driver_id = vk::DriverId::GOOGLE_SWIFTSHADER;
                copy_str_to_c_array(&mut s.driver_name, "VkWebGPU");
                copy_str_to_c_array(&mut s.driver_info, "Vulkan->WebGPU ICD");
                s.conformance_version = vk::ConformanceVersion {
                    major: 1,
                    minor: 3,
                    subminor: 0,
                    patch: 0,
                };
                s.denorm_behavior_independence = vk::ShaderFloatControlsIndependence::NONE;
                s.rounding_mode_independence = vk::ShaderFloatControlsIndependence::NONE;
                // All float-control flags unsupported
                s.shader_signed_zero_inf_nan_preserve_float16 = vk::FALSE;
                s.shader_signed_zero_inf_nan_preserve_float32 = vk::FALSE;
                s.shader_signed_zero_inf_nan_preserve_float64 = vk::FALSE;
                s.shader_denorm_preserve_float16 = vk::FALSE;
                s.shader_denorm_preserve_float32 = vk::FALSE;
                s.shader_denorm_preserve_float64 = vk::FALSE;
                s.shader_denorm_flush_to_zero_float16 = vk::FALSE;
                s.shader_denorm_flush_to_zero_float32 = vk::FALSE;
                s.shader_denorm_flush_to_zero_float64 = vk::FALSE;
                s.shader_rounding_mode_rte_float16 = vk::FALSE;
                s.shader_rounding_mode_rte_float32 = vk::FALSE;
                s.shader_rounding_mode_rte_float64 = vk::FALSE;
                s.shader_rounding_mode_rtz_float16 = vk::FALSE;
                s.shader_rounding_mode_rtz_float32 = vk::FALSE;
                s.shader_rounding_mode_rtz_float64 = vk::FALSE;
                // Descriptor-indexing properties
                s.max_update_after_bind_descriptors_in_all_pools = 65536;
                s.shader_uniform_buffer_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_sampled_image_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_storage_buffer_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_storage_image_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_input_attachment_array_non_uniform_indexing_native = vk::FALSE;
                s.robust_buffer_access_update_after_bind = vk::FALSE;
                s.quad_divergent_implicit_lod = vk::FALSE;
                s.max_per_stage_descriptor_update_after_bind_samplers = 16;
                s.max_per_stage_descriptor_update_after_bind_uniform_buffers = 12;
                s.max_per_stage_descriptor_update_after_bind_storage_buffers = 8;
                s.max_per_stage_descriptor_update_after_bind_sampled_images = 16;
                s.max_per_stage_descriptor_update_after_bind_storage_images = 8;
                s.max_per_stage_descriptor_update_after_bind_input_attachments = 8;
                s.max_per_stage_update_after_bind_resources = 128;
                s.max_descriptor_set_update_after_bind_samplers = 80;
                s.max_descriptor_set_update_after_bind_uniform_buffers = 72;
                s.max_descriptor_set_update_after_bind_uniform_buffers_dynamic = 8;
                s.max_descriptor_set_update_after_bind_storage_buffers = 40;
                s.max_descriptor_set_update_after_bind_storage_buffers_dynamic = 8;
                s.max_descriptor_set_update_after_bind_sampled_images = 80;
                s.max_descriptor_set_update_after_bind_storage_images = 40;
                s.max_descriptor_set_update_after_bind_input_attachments = 8;
                // Misc
                s.filter_minmax_single_component_formats = vk::TRUE;
                s.filter_minmax_image_component_mapping = vk::FALSE;
                s.max_timeline_semaphore_value_difference = u64::MAX / 2;
                s.framebuffer_integer_color_sample_counts = vk::SampleCountFlags::TYPE_1;
            }
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceVulkan13Properties);
                s.min_subgroup_size = 32;
                s.max_subgroup_size = 32;
                s.max_compute_workgroup_subgroups = 8;
                s.required_subgroup_size_stages = vk::ShaderStageFlags::empty();
                s.max_inline_uniform_block_size = 256;
                s.max_per_stage_descriptor_inline_uniform_blocks = 4;
                s.max_per_stage_descriptor_update_after_bind_inline_uniform_blocks = 4;
                s.max_descriptor_set_inline_uniform_blocks = 4;
                s.max_descriptor_set_update_after_bind_inline_uniform_blocks = 4;
                s.max_inline_uniform_total_size = 4096;
                s.storage_texel_buffer_offset_alignment_bytes = 256;
                s.storage_texel_buffer_offset_single_texel_alignment = vk::FALSE;
                s.uniform_texel_buffer_offset_alignment_bytes = 256;
                s.uniform_texel_buffer_offset_single_texel_alignment = vk::FALSE;
                s.max_buffer_size = 2u64 * 1024 * 1024 * 1024;
            }
            vk::StructureType::PHYSICAL_DEVICE_DRIVER_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceDriverProperties);
                s.driver_id = vk::DriverId::GOOGLE_SWIFTSHADER;
                copy_str_to_c_array(&mut s.driver_name, "VkWebGPU");
                copy_str_to_c_array(&mut s.driver_info, "Vulkan->WebGPU translation layer");
                s.conformance_version = vk::ConformanceVersion {
                    major: 1,
                    minor: 3,
                    subminor: 0,
                    patch: 0,
                };
            }
            vk::StructureType::PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceMaintenance3Properties);
                s.max_per_set_descriptors = 1024;
                s.max_memory_allocation_size = 2u64 * 1024 * 1024 * 1024;
            }
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceDescriptorIndexingProperties);
                s.max_update_after_bind_descriptors_in_all_pools = 65536;
                s.shader_uniform_buffer_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_sampled_image_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_storage_buffer_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_storage_image_array_non_uniform_indexing_native = vk::FALSE;
                s.shader_input_attachment_array_non_uniform_indexing_native = vk::FALSE;
                s.robust_buffer_access_update_after_bind = vk::FALSE;
                s.quad_divergent_implicit_lod = vk::FALSE;
                s.max_per_stage_descriptor_update_after_bind_samplers = 16;
                s.max_per_stage_descriptor_update_after_bind_uniform_buffers = 12;
                s.max_per_stage_descriptor_update_after_bind_storage_buffers = 8;
                s.max_per_stage_descriptor_update_after_bind_sampled_images = 16;
                s.max_per_stage_descriptor_update_after_bind_storage_images = 8;
                s.max_per_stage_descriptor_update_after_bind_input_attachments = 8;
                s.max_per_stage_update_after_bind_resources = 128;
                s.max_descriptor_set_update_after_bind_samplers = 80;
                s.max_descriptor_set_update_after_bind_uniform_buffers = 72;
                s.max_descriptor_set_update_after_bind_uniform_buffers_dynamic = 8;
                s.max_descriptor_set_update_after_bind_storage_buffers = 40;
                s.max_descriptor_set_update_after_bind_storage_buffers_dynamic = 8;
                s.max_descriptor_set_update_after_bind_sampled_images = 80;
                s.max_descriptor_set_update_after_bind_storage_images = 40;
                s.max_descriptor_set_update_after_bind_input_attachments = 8;
            }
            vk::StructureType::PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceDepthStencilResolveProperties);
                s.supported_depth_resolve_modes =
                    vk::ResolveModeFlags::SAMPLE_ZERO | vk::ResolveModeFlags::AVERAGE;
                s.supported_stencil_resolve_modes = vk::ResolveModeFlags::SAMPLE_ZERO;
                s.independent_resolve_none = vk::TRUE;
                s.independent_resolve = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceTimelineSemaphoreProperties);
                s.max_timeline_semaphore_value_difference = u64::MAX / 2;
            }
            vk::StructureType::PHYSICAL_DEVICE_SUBGROUP_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceSubgroupProperties);
                s.subgroup_size = 32;
                s.supported_stages = vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT;
                s.supported_operations = vk::SubgroupFeatureFlags::BASIC
                    | vk::SubgroupFeatureFlags::VOTE
                    | vk::SubgroupFeatureFlags::BALLOT;
                s.quad_operations_in_all_stages = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceFloatControlsProperties);
                s.denorm_behavior_independence = vk::ShaderFloatControlsIndependence::NONE;
                s.rounding_mode_independence = vk::ShaderFloatControlsIndependence::NONE;
                s.shader_signed_zero_inf_nan_preserve_float16 = vk::FALSE;
                s.shader_signed_zero_inf_nan_preserve_float32 = vk::FALSE;
                s.shader_signed_zero_inf_nan_preserve_float64 = vk::FALSE;
                s.shader_denorm_preserve_float16 = vk::FALSE;
                s.shader_denorm_preserve_float32 = vk::FALSE;
                s.shader_denorm_preserve_float64 = vk::FALSE;
                s.shader_denorm_flush_to_zero_float16 = vk::FALSE;
                s.shader_denorm_flush_to_zero_float32 = vk::FALSE;
                s.shader_denorm_flush_to_zero_float64 = vk::FALSE;
                s.shader_rounding_mode_rte_float16 = vk::FALSE;
                s.shader_rounding_mode_rte_float32 = vk::FALSE;
                s.shader_rounding_mode_rte_float64 = vk::FALSE;
                s.shader_rounding_mode_rtz_float16 = vk::FALSE;
                s.shader_rounding_mode_rtz_float32 = vk::FALSE;
                s.shader_rounding_mode_rtz_float64 = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceSamplerFilterMinmaxProperties);
                s.filter_minmax_single_component_formats = vk::TRUE;
                s.filter_minmax_image_component_mapping = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceMultiviewProperties);
                s.max_multiview_view_count = 0;
                s.max_multiview_instance_index = 0;
            }
            _ => {
                debug!(
                    "GetPhysicalDeviceProperties2: unhandled sType {:?}",
                    (*p_next).s_type
                );
            }
        }
        p_next = (*p_next).p_next as *mut vk::BaseOutStructure;
    }
}

/// vkGetPhysicalDeviceFeatures2 – walks the pNext chain and fills every
/// recognised extended-features structure.
pub unsafe fn get_physical_device_features2(
    physical_device: vk::PhysicalDevice,
    p_features2: *mut vk::PhysicalDeviceFeatures2,
) {
    if p_features2.is_null() {
        return;
    }
    let features2 = &mut *p_features2;

    // Fill the base VkPhysicalDeviceFeatures
    get_physical_device_features(physical_device, &mut features2.features);

    debug!("GetPhysicalDeviceFeatures2");

    let mut p_next = features2.p_next as *mut vk::BaseOutStructure;
    while !p_next.is_null() {
        match (*p_next).s_type {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceVulkan11Features);
                s.shader_draw_parameters = vk::TRUE;
                s.storage_buffer16_bit_access = vk::FALSE;
                s.uniform_and_storage_buffer16_bit_access = vk::FALSE;
                s.storage_push_constant16 = vk::FALSE;
                s.storage_input_output16 = vk::FALSE;
                s.multiview = vk::FALSE;
                s.multiview_geometry_shader = vk::FALSE;
                s.multiview_tessellation_shader = vk::FALSE;
                s.variable_pointers_storage_buffer = vk::FALSE;
                s.variable_pointers = vk::FALSE;
                s.protected_memory = vk::FALSE;
                s.sampler_ycbcr_conversion = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceVulkan12Features);
                s.sampler_mirror_clamp_to_edge = vk::TRUE;
                s.draw_indirect_count = vk::FALSE;
                s.storage_buffer8_bit_access = vk::FALSE;
                s.uniform_and_storage_buffer8_bit_access = vk::FALSE;
                s.storage_push_constant8 = vk::FALSE;
                s.shader_buffer_int64_atomics = vk::FALSE;
                s.shader_shared_int64_atomics = vk::FALSE;
                s.shader_float16 = vk::FALSE;
                s.shader_int8 = vk::FALSE;
                s.descriptor_indexing = vk::TRUE;
                s.shader_input_attachment_array_dynamic_indexing = vk::FALSE;
                s.shader_uniform_texel_buffer_array_dynamic_indexing = vk::FALSE;
                s.shader_storage_texel_buffer_array_dynamic_indexing = vk::FALSE;
                s.shader_uniform_buffer_array_non_uniform_indexing = vk::FALSE;
                s.shader_sampled_image_array_non_uniform_indexing = vk::FALSE;
                s.shader_storage_buffer_array_non_uniform_indexing = vk::FALSE;
                s.shader_storage_image_array_non_uniform_indexing = vk::FALSE;
                s.shader_input_attachment_array_non_uniform_indexing = vk::FALSE;
                s.shader_uniform_texel_buffer_array_non_uniform_indexing = vk::FALSE;
                s.shader_storage_texel_buffer_array_non_uniform_indexing = vk::FALSE;
                s.descriptor_binding_uniform_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_sampled_image_update_after_bind = vk::TRUE;
                s.descriptor_binding_storage_image_update_after_bind = vk::TRUE;
                s.descriptor_binding_storage_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_uniform_texel_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_storage_texel_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_update_unused_while_pending = vk::TRUE;
                s.descriptor_binding_partially_bound = vk::TRUE;
                s.descriptor_binding_variable_descriptor_count = vk::FALSE;
                s.runtime_descriptor_array = vk::TRUE;
                s.sampler_filter_minmax = vk::TRUE;
                s.scalar_block_layout = vk::TRUE;
                s.imageless_framebuffer = vk::TRUE;
                s.uniform_buffer_standard_layout = vk::TRUE;
                s.shader_subgroup_extended_types = vk::FALSE;
                s.separate_depth_stencil_layouts = vk::TRUE;
                s.host_query_reset = vk::TRUE;
                s.timeline_semaphore = vk::TRUE;
                s.buffer_device_address = vk::TRUE;
                s.buffer_device_address_capture_replay = vk::FALSE;
                s.buffer_device_address_multi_device = vk::FALSE;
                s.vulkan_memory_model = vk::FALSE;
                s.vulkan_memory_model_device_scope = vk::FALSE;
                s.vulkan_memory_model_availability_visibility_chains = vk::FALSE;
                s.shader_output_viewport_index = vk::TRUE;
                s.shader_output_layer = vk::TRUE;
                s.subgroup_broadcast_dynamic_id = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceVulkan13Features);
                s.robust_image_access = vk::FALSE;
                s.inline_uniform_block = vk::TRUE;
                s.descriptor_binding_inline_uniform_block_update_after_bind = vk::TRUE;
                s.pipeline_creation_cache_control = vk::TRUE;
                s.private_data = vk::TRUE;
                s.shader_demote_to_helper_invocation = vk::FALSE;
                s.shader_terminate_invocation = vk::FALSE;
                s.subgroup_size_control = vk::FALSE;
                s.compute_full_subgroups = vk::FALSE;
                s.synchronization2 = vk::TRUE;
                s.texture_compression_astc_hdr = vk::FALSE;
                s.shader_zero_initialize_workgroup_memory = vk::FALSE;
                s.dynamic_rendering = vk::TRUE;
                s.shader_integer_dot_product = vk::FALSE;
                s.maintenance4 = vk::TRUE;
            }
            // Note: KHR aliases share the same sType value as core Vulkan 1.3 variants,
            // so they are handled by the PHYSICAL_DEVICE_VULKAN_1_3_FEATURES arm above.
            vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceDynamicRenderingFeatures);
                s.dynamic_rendering = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceSynchronization2Features);
                s.synchronization2 = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceTimelineSemaphoreFeatures);
                s.timeline_semaphore = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceDescriptorIndexingFeatures);
                s.shader_input_attachment_array_dynamic_indexing = vk::FALSE;
                s.shader_uniform_texel_buffer_array_dynamic_indexing = vk::FALSE;
                s.shader_storage_texel_buffer_array_dynamic_indexing = vk::FALSE;
                s.shader_uniform_buffer_array_non_uniform_indexing = vk::FALSE;
                s.shader_sampled_image_array_non_uniform_indexing = vk::FALSE;
                s.shader_storage_buffer_array_non_uniform_indexing = vk::FALSE;
                s.shader_storage_image_array_non_uniform_indexing = vk::FALSE;
                s.shader_input_attachment_array_non_uniform_indexing = vk::FALSE;
                s.shader_uniform_texel_buffer_array_non_uniform_indexing = vk::FALSE;
                s.shader_storage_texel_buffer_array_non_uniform_indexing = vk::FALSE;
                s.descriptor_binding_uniform_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_sampled_image_update_after_bind = vk::TRUE;
                s.descriptor_binding_storage_image_update_after_bind = vk::TRUE;
                s.descriptor_binding_storage_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_uniform_texel_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_storage_texel_buffer_update_after_bind = vk::TRUE;
                s.descriptor_binding_update_unused_while_pending = vk::TRUE;
                s.descriptor_binding_partially_bound = vk::TRUE;
                s.descriptor_binding_variable_descriptor_count = vk::FALSE;
                s.runtime_descriptor_array = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceHostQueryResetFeatures);
                s.host_query_reset = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceImagelessFramebufferFeatures);
                s.imageless_framebuffer = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceBufferDeviceAddressFeatures);
                s.buffer_device_address = vk::TRUE;
                s.buffer_device_address_capture_replay = vk::FALSE;
                s.buffer_device_address_multi_device = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceScalarBlockLayoutFeatures);
                s.scalar_block_layout = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceShaderDrawParametersFeatures);
                s.shader_draw_parameters = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES => {
                let s =
                    &mut *(p_next as *mut vk::PhysicalDeviceUniformBufferStandardLayoutFeatures);
                s.uniform_buffer_standard_layout = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES => {
                let s =
                    &mut *(p_next as *mut vk::PhysicalDeviceSeparateDepthStencilLayoutsFeatures);
                s.separate_depth_stencil_layouts = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceMaintenance4Features);
                s.maintenance4 = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDevicePrivateDataFeatures);
                s.private_data = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceInlineUniformBlockFeatures);
                s.inline_uniform_block = vk::TRUE;
                s.descriptor_binding_inline_uniform_block_update_after_bind = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES => {
                let s = &mut *(p_next
                    as *mut vk::PhysicalDevicePipelineCreationCacheControlFeatures);
                s.pipeline_creation_cache_control = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT => {
                let s =
                    &mut *(p_next as *mut vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT);
                s.extended_dynamic_state = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT => {
                let s =
                    &mut *(p_next as *mut vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT);
                s.extended_dynamic_state2 = vk::TRUE;
                s.extended_dynamic_state2_logic_op = vk::FALSE;
                s.extended_dynamic_state2_patch_control_points = vk::FALSE;
            }
            // vkd3d-proton / DX12 extension features
            vk::StructureType::PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceRobustness2FeaturesEXT);
                s.robust_buffer_access2 = vk::FALSE;
                s.robust_image_access2 = vk::FALSE;
                s.null_descriptor = vk::TRUE; // needed for DX12 null descriptors
            }
            vk::StructureType::PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDevice16BitStorageFeatures);
                s.storage_buffer16_bit_access = vk::TRUE;
                s.uniform_and_storage_buffer16_bit_access = vk::TRUE;
                s.storage_push_constant16 = vk::FALSE;
                s.storage_input_output16 = vk::FALSE;
            }
            vk::StructureType::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES => {
                let s = &mut *(p_next as *mut vk::PhysicalDeviceShaderFloat16Int8Features);
                s.shader_float16 = vk::TRUE;
                s.shader_int8 = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT => {
                let s =
                    &mut *(p_next as *mut vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT);
                s.mutable_descriptor_type = vk::TRUE;
            }
            vk::StructureType::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT => {
                let s =
                    &mut *(p_next as *mut vk::PhysicalDeviceVertexAttributeDivisorFeaturesEXT);
                s.vertex_attribute_instance_rate_divisor = vk::TRUE;
                s.vertex_attribute_instance_rate_zero_divisor = vk::TRUE;
            }
            _ => {
                debug!(
                    "GetPhysicalDeviceFeatures2: unhandled sType {:?}",
                    (*p_next).s_type
                );
            }
        }
        p_next = (*p_next).p_next as *mut vk::BaseOutStructure;
    }
}

pub unsafe fn get_physical_device_memory_properties2(
    physical_device: vk::PhysicalDevice,
    p_memory_properties2: *mut vk::PhysicalDeviceMemoryProperties2,
) {
    if p_memory_properties2.is_null() {
        return;
    }
    crate::memory::get_physical_device_memory_properties(
        physical_device,
        &mut (*p_memory_properties2).memory_properties,
    );
}

pub unsafe fn get_physical_device_queue_family_properties2(
    physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties2,
) {
    let queue_family_count = 1u32;
    if p_queue_family_properties.is_null() {
        *p_queue_family_property_count = queue_family_count;
        return;
    }
    // Fill the first (and only) queue family
    let qfp = &mut *p_queue_family_properties;
    get_physical_device_queue_family_properties(
        physical_device,
        p_queue_family_property_count,
        &mut qfp.queue_family_properties,
    );
}

pub unsafe fn get_physical_device_format_properties2(
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    p_format_properties2: *mut vk::FormatProperties2,
) {
    if p_format_properties2.is_null() {
        return;
    }
    get_physical_device_format_properties(
        physical_device,
        format,
        &mut (*p_format_properties2).format_properties,
    );
}

pub unsafe fn get_physical_device_image_format_properties2(
    physical_device: vk::PhysicalDevice,
    p_image_format_info: *const vk::PhysicalDeviceImageFormatInfo2,
    p_image_format_properties2: *mut vk::ImageFormatProperties2,
) -> Result<()> {
    if p_image_format_info.is_null() || p_image_format_properties2.is_null() {
        return Err(VkError::InvalidHandle("Null pointer".to_string()));
    }
    let info = &*p_image_format_info;
    get_physical_device_image_format_properties(
        physical_device,
        info.format,
        info.ty,
        info.tiling,
        info.usage,
        info.flags,
        &mut (*p_image_format_properties2).image_format_properties,
    )
}

pub unsafe fn get_physical_device_sparse_image_format_properties2(
    _physical_device: vk::PhysicalDevice,
    _p_format_info: *const vk::PhysicalDeviceSparseImageFormatInfo2,
    p_property_count: *mut u32,
    _p_properties: *mut vk::SparseImageFormatProperties2,
) {
    // WebGPU does not support sparse resources – return zero properties.
    if !p_property_count.is_null() {
        *p_property_count = 0;
    }
}

pub unsafe fn get_physical_device_external_buffer_properties(
    _physical_device: vk::PhysicalDevice,
    _p_external_buffer_info: *const vk::PhysicalDeviceExternalBufferInfo,
    p_external_buffer_properties: *mut vk::ExternalBufferProperties,
) {
    if p_external_buffer_properties.is_null() {
        return;
    }
    // We do not support external memory handles.
    let props = &mut (*p_external_buffer_properties).external_memory_properties;
    props.external_memory_features = vk::ExternalMemoryFeatureFlags::empty();
    props.export_from_imported_handle_types = vk::ExternalMemoryHandleTypeFlags::empty();
    props.compatible_handle_types = vk::ExternalMemoryHandleTypeFlags::empty();
}

pub unsafe fn get_physical_device_external_fence_properties(
    _physical_device: vk::PhysicalDevice,
    _p_external_fence_info: *const vk::PhysicalDeviceExternalFenceInfo,
    p_external_fence_properties: *mut vk::ExternalFenceProperties,
) {
    if p_external_fence_properties.is_null() {
        return;
    }
    let props = &mut *p_external_fence_properties;
    props.export_from_imported_handle_types = vk::ExternalFenceHandleTypeFlags::empty();
    props.compatible_handle_types = vk::ExternalFenceHandleTypeFlags::empty();
    props.external_fence_features = vk::ExternalFenceFeatureFlags::empty();
}

pub unsafe fn get_physical_device_external_semaphore_properties(
    _physical_device: vk::PhysicalDevice,
    _p_external_semaphore_info: *const vk::PhysicalDeviceExternalSemaphoreInfo,
    p_external_semaphore_properties: *mut vk::ExternalSemaphoreProperties,
) {
    if p_external_semaphore_properties.is_null() {
        return;
    }
    let props = &mut *p_external_semaphore_properties;
    props.export_from_imported_handle_types = vk::ExternalSemaphoreHandleTypeFlags::empty();
    props.compatible_handle_types = vk::ExternalSemaphoreHandleTypeFlags::empty();
    props.external_semaphore_features = vk::ExternalSemaphoreFeatureFlags::empty();
}

/// Get image format properties
pub unsafe fn get_physical_device_image_format_properties(
    _physical_device: vk::PhysicalDevice,
    format: vk::Format,
    image_type: vk::ImageType,
    _tiling: vk::ImageTiling,
    _usage: vk::ImageUsageFlags,
    _flags: vk::ImageCreateFlags,
    p_image_format_properties: *mut vk::ImageFormatProperties,
) -> Result<()> {
    // Check if format is supported
    #[cfg(not(target_arch = "wasm32"))]
    let supported = crate::format::vk_to_wgpu_format(format).is_some();

    #[cfg(target_arch = "wasm32")]
    let supported = false;

    if !supported {
        return Err(VkError::FormatNotSupported);
    }

    // WebGPU limits (conservative defaults)
    let mut props = vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: 16384, // Max texture size
            height: 16384,
            depth: 2048,
        },
        max_mip_levels: 14, // log2(16384) + 1
        max_array_layers: 2048,
        sample_counts: vk::SampleCountFlags::TYPE_1 | vk::SampleCountFlags::TYPE_4,
        max_resource_size: 4 * 1024 * 1024 * 1024, // 4GB
    };

    // Adjust based on image type
    match image_type {
        vk::ImageType::TYPE_1D => {
            props.max_extent.height = 1;
            props.max_extent.depth = 1;
            props.max_array_layers = 2048;
        }
        vk::ImageType::TYPE_2D => {
            props.max_extent.depth = 1;
        }
        vk::ImageType::TYPE_3D => {
            props.max_array_layers = 1;
        }
        _ => {}
    }

    *p_image_format_properties = props;
    Ok(())
}
