//! Vulkan Swapchain implementation (KHR extension)
//!
//! Swapchain images are backed by real wgpu Textures with RENDER_ATTACHMENT usage so that
//! vkCreateImageView, framebuffer creation, and render passes all work correctly.
//! Presentation is not yet connected to a real display surface; frames are rendered into
//! offscreen GPU textures.  A future step will wire up wgpu::Surface from the Win32 HWND
//! stored in the VkSurfaceKHR for actual display.

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::format;
use crate::handle::HandleAllocator;
use crate::image;

pub static SWAPCHAIN_ALLOCATOR: Lazy<HandleAllocator<VkSwapchainData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkSwapchainData {
    pub device: vk::Device,
    pub surface: vk::SurfaceKHR,
    pub image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,

    /// Real VkImage handles backed by wgpu Textures (registered in IMAGE_ALLOCATOR).
    pub images: Vec<vk::Image>,

    /// Tracks which image index was most recently acquired.
    pub current_image_index: AtomicU32,
}

/// Helper function to get swapchain data from handle
unsafe fn get_swapchain_data(swapchain: vk::SwapchainKHR) -> Result<Arc<VkSwapchainData>> {
    SWAPCHAIN_ALLOCATOR
        .get(swapchain.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid swapchain handle".to_string()))
}

pub unsafe fn create_swapchain_khr(
    device: vk::Device,
    p_create_info: *const vk::SwapchainCreateInfoKHR,
    _p_allocator: *const vk::AllocationCallbacks,
    p_swapchain: *mut vk::SwapchainKHR,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating swapchain: {}x{}, format={:?}, min_image_count={}",
        create_info.image_extent.width,
        create_info.image_extent.height,
        create_info.image_format,
        create_info.min_image_count
    );

    // Clamp image count to a sensible range (triple buffering default).
    let image_count = create_info.min_image_count.clamp(2, 8).max(3);

    let device_data = device::get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    // Map the Vulkan format to a wgpu TextureFormat for the render targets.
    // Fall back to Bgra8Unorm if the format is unknown (rare for swapchain formats).
    let wgpu_format = format::vk_to_wgpu_format(create_info.image_format)
        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);

    // Build wgpu texture usage flags from what the application requests.
    let wgpu_usage = {
        let mut u = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING;
        if create_info
            .image_usage
            .contains(vk::ImageUsageFlags::STORAGE)
        {
            u |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        u
    };

    // Create one real wgpu Texture per swapchain image and register each as a VkImage.
    let mut images: Vec<vk::Image> = Vec::with_capacity(image_count as usize);
    for idx in 0..image_count {
        let texture = device_data
            .backend
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("SwapchainImage[{}]", idx)),
                size: wgpu::Extent3d {
                    width: create_info.image_extent.width,
                    height: create_info.image_extent.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage: wgpu_usage,
                view_formats: &[],
            });

        let vk_image = image::create_swapchain_image(
            device,
            create_info.image_format,
            create_info.image_extent,
            texture,
        );
        images.push(vk_image);
        debug!("Created swapchain image[{}]: {:?}", idx, vk_image);
    }

    let swapchain_data = VkSwapchainData {
        device,
        surface: create_info.surface,
        image_count,
        image_format: create_info.image_format,
        image_color_space: create_info.image_color_space,
        image_extent: create_info.image_extent,
        image_array_layers: create_info.image_array_layers,
        image_usage: create_info.image_usage,
        pre_transform: create_info.pre_transform,
        composite_alpha: create_info.composite_alpha,
        present_mode: create_info.present_mode,
        images,
        current_image_index: AtomicU32::new(0),
    };

    let swapchain_handle = SWAPCHAIN_ALLOCATOR.allocate(swapchain_data);
    *p_swapchain = Handle::from_raw(swapchain_handle);

    debug!("Created swapchain with {} wgpu-backed images", image_count);
    Ok(())
}

pub unsafe fn destroy_swapchain_khr(
    swapchain: vk::SwapchainKHR,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if swapchain == vk::SwapchainKHR::null() {
        return;
    }

    // Unregister all backing VkImages before dropping the swapchain.
    if let Some(data) = SWAPCHAIN_ALLOCATOR.get(swapchain.as_raw()) {
        for &img in &data.images {
            image::destroy_swapchain_image(img);
        }
    }

    debug!("Destroying swapchain");
    SWAPCHAIN_ALLOCATOR.remove(swapchain.as_raw());
}

pub unsafe fn get_swapchain_images_khr(
    swapchain: vk::SwapchainKHR,
    p_swapchain_image_count: *mut u32,
    p_swapchain_images: *mut vk::Image,
) -> Result<()> {
    let swapchain_data = get_swapchain_data(swapchain)?;

    if p_swapchain_images.is_null() {
        // Query: return the image count
        *p_swapchain_image_count = swapchain_data.image_count;
        debug!(
            "Querying swapchain image count: {}",
            swapchain_data.image_count
        );
    } else {
        // Retrieve: copy image handles to the output array
        let count = (*p_swapchain_image_count).min(swapchain_data.image_count);
        let images = std::slice::from_raw_parts_mut(p_swapchain_images, count as usize);
        images.copy_from_slice(&swapchain_data.images[..count as usize]);
        *p_swapchain_image_count = count;
        debug!("Retrieved {} swapchain images", count);
    }

    Ok(())
}

pub unsafe fn acquire_next_image_khr(
    swapchain: vk::SwapchainKHR,
    _timeout: u64,
    _semaphore: vk::Semaphore,
    _fence: vk::Fence,
    p_image_index: *mut u32,
) -> Result<()> {
    let swapchain_data = get_swapchain_data(swapchain)?;

    // Cycle through the available images in order.
    // In a real implementation with a wgpu Surface this would call surface.get_current_texture().
    let current = swapchain_data.current_image_index.load(Ordering::Relaxed);
    let next = (current + 1) % swapchain_data.image_count;
    swapchain_data
        .current_image_index
        .store(next, Ordering::Relaxed);

    *p_image_index = next;

    debug!(
        "Acquired swapchain image index: {} ({}/{})",
        next,
        next + 1,
        swapchain_data.image_count
    );

    Ok(())
}

pub unsafe fn queue_present_khr(
    _queue: vk::Queue,
    p_present_info: *const vk::PresentInfoKHR,
) -> Result<()> {
    let present_info = &*p_present_info;

    if present_info.swapchain_count == 0 {
        debug!("Queue present called with 0 swapchains");
        return Ok(());
    }

    let swapchains = std::slice::from_raw_parts(
        present_info.p_swapchains,
        present_info.swapchain_count as usize,
    );
    let image_indices = std::slice::from_raw_parts(
        present_info.p_image_indices,
        present_info.swapchain_count as usize,
    );

    for (i, (&swapchain, &image_index)) in swapchains.iter().zip(image_indices.iter()).enumerate()
    {
        let swapchain_data = get_swapchain_data(swapchain)?;

        debug!(
            "Presenting swapchain {} image[{}] ({}x{} {:?})",
            i,
            image_index,
            swapchain_data.image_extent.width,
            swapchain_data.image_extent.height,
            swapchain_data.image_format
        );

        if image_index >= swapchain_data.image_count {
            return Err(VkError::InvalidHandle(format!(
                "Invalid image index: {} (max: {})",
                image_index, swapchain_data.image_count
            )));
        }

        // The frame has already been rendered into the offscreen wgpu texture during
        // vkQueueSubmit.  A future implementation will blit to an actual wgpu::Surface
        // acquired from the Win32 HWND stored in VkSurfaceKHR.

        if !present_info.p_results.is_null() {
            let results = std::slice::from_raw_parts_mut(
                present_info.p_results as *mut vk::Result,
                present_info.swapchain_count as usize,
            );
            results[i] = vk::Result::SUCCESS;
        }
    }

    debug!(
        "Queue present completed for {} swapchain(s)",
        present_info.swapchain_count
    );
    Ok(())
}
