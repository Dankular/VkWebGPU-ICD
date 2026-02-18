//! Vulkan Swapchain implementation (KHR extension)

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

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

    // Swapchain images (dummy VkImage handles for API compatibility)
    pub images: Vec<vk::Image>,

    // Track which image is currently acquired
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

    // Determine actual image count (typically triple buffering)
    let image_count = create_info.min_image_count.max(3);

    // Create dummy VkImage handles for the swapchain images
    // These are placeholders for API compatibility
    let mut images = Vec::with_capacity(image_count as usize);
    for i in 0..image_count {
        // Create a unique handle by combining swapchain info with image index
        // Using a high bit pattern to avoid conflicts with regular images
        let image_handle =
            0xDEAD_0000_0000_0000u64 | ((device.as_raw() & 0xFFFFFF) << 32) as u64 | i as u64;
        images.push(Handle::from_raw(image_handle));
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

    debug!("Created swapchain with {} images", image_count);

    Ok(())
}

pub unsafe fn destroy_swapchain_khr(
    swapchain: vk::SwapchainKHR,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if swapchain == vk::SwapchainKHR::null() {
        return;
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

    // For WebGPU, we don't actually acquire until present time
    // Just return the next index in sequence (cycle through 0, 1, 2, ...)
    let current = swapchain_data.current_image_index.load(Ordering::Relaxed);
    let next = (current + 1) % swapchain_data.image_count;
    swapchain_data
        .current_image_index
        .store(next, Ordering::Relaxed);

    *p_image_index = next;

    debug!(
        "Acquired swapchain image index: {} (cycling {}/{})",
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

    // Process wait semaphores (for API compatibility, but we don't actually wait)
    if present_info.wait_semaphore_count > 0 {
        debug!(
            "Queue present waiting on {} semaphores (ignored in WebGPU)",
            present_info.wait_semaphore_count
        );
    }

    // Present each swapchain
    for (i, (&swapchain, &image_index)) in swapchains.iter().zip(image_indices.iter()).enumerate() {
        let swapchain_data = get_swapchain_data(swapchain)?;

        debug!(
            "Presenting swapchain {} image index: {} ({}x{} {:?})",
            i,
            image_index,
            swapchain_data.image_extent.width,
            swapchain_data.image_extent.height,
            swapchain_data.image_format
        );

        // Validate image index
        if image_index >= swapchain_data.image_count {
            return Err(VkError::InvalidHandle(format!(
                "Invalid image index: {} (max: {})",
                image_index, swapchain_data.image_count
            )));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // For native: WebGPU surface presentation is automatic after queue.submit()
            // The actual rendering was done during vkQueueSubmit
            // This call just confirms the frame is ready to be displayed
            // In a full implementation, we would call surface.present() here
            debug!("Native presentation complete for image {}", image_index);
        }

        #[cfg(target_arch = "wasm32")]
        {
            // For WASM: Presentation happens automatically in requestAnimationFrame
            // WebGPU handles this internally via the canvas context
            debug!("WASM presentation queued for image {}", image_index);
        }

        // Write result if requested
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
