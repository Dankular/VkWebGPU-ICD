//! Vulkan Image implementation
//! Maps VkImage to WebGPU GPUTexture

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::format;
use crate::handle::HandleAllocator;

pub static IMAGE_ALLOCATOR: Lazy<HandleAllocator<VkImageData>> =
    Lazy::new(|| HandleAllocator::new());
pub static IMAGE_VIEW_ALLOCATOR: Lazy<HandleAllocator<VkImageViewData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkImageData {
    pub device: vk::Device,
    pub image_type: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub initial_layout: vk::ImageLayout,
    pub memory: RwLock<Option<vk::DeviceMemory>>,
    pub memory_offset: RwLock<vk::DeviceSize>,
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_texture: RwLock<Option<Arc<wgpu::Texture>>>,
}

pub struct VkImageViewData {
    pub device: vk::Device,
    pub image: vk::Image,
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub subresource_range: vk::ImageSubresourceRange,
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_view: RwLock<Option<Arc<wgpu::TextureView>>>,
}

pub unsafe fn create_image(
    device: vk::Device,
    p_create_info: *const vk::ImageCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_image: *mut vk::Image,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating image: type={:?}, format={:?}, extent={:?}, mips={}, layers={}",
        create_info.image_type,
        create_info.format,
        create_info.extent,
        create_info.mip_levels,
        create_info.array_layers
    );

    let image_data = VkImageData {
        device,
        image_type: create_info.image_type,
        format: create_info.format,
        extent: create_info.extent,
        mip_levels: create_info.mip_levels,
        array_layers: create_info.array_layers,
        samples: create_info.samples,
        tiling: create_info.tiling,
        usage: create_info.usage,
        sharing_mode: create_info.sharing_mode,
        initial_layout: create_info.initial_layout,
        memory: RwLock::new(None),
        memory_offset: RwLock::new(0),
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_texture: RwLock::new(None),
    };

    let image_handle = IMAGE_ALLOCATOR.allocate(image_data);
    *p_image = Handle::from_raw(image_handle);

    Ok(())
}

pub unsafe fn destroy_image(
    _device: vk::Device,
    image: vk::Image,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if image == vk::Image::null() {
        return;
    }

    IMAGE_ALLOCATOR.remove(image.as_raw());
    debug!("Destroyed image");
}

pub unsafe fn bind_image_memory(
    _device: vk::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    memory_offset: vk::DeviceSize,
) -> Result<()> {
    let image_data = IMAGE_ALLOCATOR
        .get(image.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid image".to_string()))?;

    let device_data = device::get_device_data(image_data.device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    *image_data.memory.write() = Some(memory);
    *image_data.memory_offset.write() = memory_offset;

    #[cfg(not(target_arch = "wasm32"))]
    {
        let format =
            format::vk_to_wgpu_format(image_data.format).ok_or(VkError::FormatNotSupported)?;

        let dimension = match image_data.image_type {
            vk::ImageType::TYPE_1D => wgpu::TextureDimension::D1,
            vk::ImageType::TYPE_2D => wgpu::TextureDimension::D2,
            vk::ImageType::TYPE_3D => wgpu::TextureDimension::D3,
            _ => {
                return Err(VkError::FeatureNotSupported(
                    "Invalid image type".to_string(),
                ))
            }
        };

        let size = wgpu::Extent3d {
            width: image_data.extent.width,
            height: image_data.extent.height,
            depth_or_array_layers: if dimension == wgpu::TextureDimension::D3 {
                image_data.extent.depth
            } else {
                image_data.array_layers
            },
        };

        let usage = vk_to_wgpu_texture_usage(image_data.usage);

        let wgpu_texture = device_data
            .backend
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VkImage"),
                size,
                mip_level_count: image_data.mip_levels,
                sample_count: vk_to_wgpu_sample_count(image_data.samples),
                dimension,
                format,
                usage,
                view_formats: &[],
            });

        *image_data.wgpu_texture.write() = Some(Arc::new(wgpu_texture));
    }

    debug!("Bound image to memory at offset {}", memory_offset);

    Ok(())
}

pub unsafe fn get_image_memory_requirements(
    _device: vk::Device,
    image: vk::Image,
    p_memory_requirements: *mut vk::MemoryRequirements,
) {
    let image_data = match IMAGE_ALLOCATOR.get(image.as_raw()) {
        Some(data) => data,
        None => return,
    };

    let requirements = &mut *p_memory_requirements;

    // Calculate size based on format and extent
    let format_size = format::format_size(image_data.format).unwrap_or(4);
    let size = image_data.extent.width as u64
        * image_data.extent.height as u64
        * image_data.extent.depth as u64
        * image_data.array_layers as u64
        * format_size as u64;

    requirements.size = size;
    requirements.alignment = 256;
    requirements.memory_type_bits = 0b111;
}

pub unsafe fn create_image_view(
    device: vk::Device,
    p_create_info: *const vk::ImageViewCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_view: *mut vk::ImageView,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating image view: image={:?}, type={:?}, format={:?}",
        create_info.image, create_info.view_type, create_info.format
    );

    let image_data = IMAGE_ALLOCATOR
        .get(create_info.image.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid image".to_string()))?;

    let view_data = VkImageViewData {
        device,
        image: create_info.image,
        view_type: create_info.view_type,
        format: create_info.format,
        components: create_info.components,
        subresource_range: create_info.subresource_range,
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_view: RwLock::new(None),
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(texture) = image_data.wgpu_texture.read().as_ref() {
            let aspect = if create_info
                .subresource_range
                .aspect_mask
                .contains(vk::ImageAspectFlags::DEPTH)
            {
                wgpu::TextureAspect::DepthOnly
            } else if create_info
                .subresource_range
                .aspect_mask
                .contains(vk::ImageAspectFlags::STENCIL)
            {
                wgpu::TextureAspect::StencilOnly
            } else {
                wgpu::TextureAspect::All
            };

            let dimension = match create_info.view_type {
                vk::ImageViewType::TYPE_1D => Some(wgpu::TextureViewDimension::D1),
                vk::ImageViewType::TYPE_2D => Some(wgpu::TextureViewDimension::D2),
                vk::ImageViewType::TYPE_3D => Some(wgpu::TextureViewDimension::D3),
                vk::ImageViewType::CUBE => Some(wgpu::TextureViewDimension::Cube),
                vk::ImageViewType::TYPE_1D_ARRAY => Some(wgpu::TextureViewDimension::D1),
                vk::ImageViewType::TYPE_2D_ARRAY => Some(wgpu::TextureViewDimension::D2Array),
                vk::ImageViewType::CUBE_ARRAY => Some(wgpu::TextureViewDimension::CubeArray),
                _ => None,
            };

            let wgpu_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("VkImageView"),
                format: format::vk_to_wgpu_format(create_info.format),
                dimension,
                aspect,
                base_mip_level: create_info.subresource_range.base_mip_level,
                mip_level_count: if create_info.subresource_range.level_count
                    == vk::REMAINING_MIP_LEVELS
                {
                    None
                } else {
                    Some(create_info.subresource_range.level_count)
                },
                base_array_layer: create_info.subresource_range.base_array_layer,
                array_layer_count: if create_info.subresource_range.layer_count
                    == vk::REMAINING_ARRAY_LAYERS
                {
                    None
                } else {
                    Some(create_info.subresource_range.layer_count)
                },
            });

            *view_data.wgpu_view.write() = Some(Arc::new(wgpu_view));
        }
    }

    let view_handle = IMAGE_VIEW_ALLOCATOR.allocate(view_data);
    *p_view = Handle::from_raw(view_handle);

    Ok(())
}

pub unsafe fn destroy_image_view(
    _device: vk::Device,
    image_view: vk::ImageView,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if image_view == vk::ImageView::null() {
        return;
    }

    IMAGE_VIEW_ALLOCATOR.remove(image_view.as_raw());
    debug!("Destroyed image view");
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_texture_usage(vk_usage: vk::ImageUsageFlags) -> wgpu::TextureUsages {
    let mut usage = wgpu::TextureUsages::empty();

    if vk_usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    if vk_usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        usage |= wgpu::TextureUsages::COPY_DST;
    }
    if vk_usage.contains(vk::ImageUsageFlags::SAMPLED) {
        usage |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if vk_usage.contains(vk::ImageUsageFlags::STORAGE) {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if vk_usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if vk_usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }

    if usage.is_empty() {
        usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    }

    usage
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_sample_count(samples: vk::SampleCountFlags) -> u32 {
    if samples.contains(vk::SampleCountFlags::TYPE_1) {
        1
    } else if samples.contains(vk::SampleCountFlags::TYPE_2) {
        2
    } else if samples.contains(vk::SampleCountFlags::TYPE_4) {
        4
    } else if samples.contains(vk::SampleCountFlags::TYPE_8) {
        8
    } else if samples.contains(vk::SampleCountFlags::TYPE_16) {
        16
    } else if samples.contains(vk::SampleCountFlags::TYPE_32) {
        32
    } else if samples.contains(vk::SampleCountFlags::TYPE_64) {
        64
    } else {
        1
    }
}

pub fn get_image_data(image: vk::Image) -> Option<Arc<VkImageData>> {
    IMAGE_ALLOCATOR.get(image.as_raw())
}

pub fn get_image_view_data(image_view: vk::ImageView) -> Option<Arc<VkImageViewData>> {
    IMAGE_VIEW_ALLOCATOR.get(image_view.as_raw())
}
