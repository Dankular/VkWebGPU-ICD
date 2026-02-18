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

    let instance_data = VkInstanceData {
        application_info: app_info,
        enabled_extensions,
        enabled_layers,
        physical_devices: RwLock::new(Vec::new()),
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

    // Create WebGPU backend for native
    #[cfg(not(target_arch = "wasm32"))]
    let backend = Some(Arc::new(WebGPUBackend::new()?));

    #[cfg(target_arch = "wasm32")]
    let backend = None;

    // Get adapter info
    #[cfg(not(target_arch = "wasm32"))]
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

    properties.api_version = vk::make_api_version(0, 1, 2, 0);
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
        ("VK_KHR_surface", 25),                        // Required for swapchain
        ("VK_KHR_get_physical_device_properties2", 2), // DXVK uses this
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
