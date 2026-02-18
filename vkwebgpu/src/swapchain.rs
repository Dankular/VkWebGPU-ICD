//! Vulkan Swapchain implementation (KHR extension)

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::error::Result;
use crate::handle::HandleAllocator;

pub static SWAPCHAIN_ALLOCATOR: Lazy<HandleAllocator<VkSwapchainData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkSwapchainData {
    pub device: vk::Device,
    pub surface: vk::SurfaceKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<vk::Image>,
}

pub unsafe fn create_swapchain_khr(
    device: vk::Device,
    p_create_info: *const vk::SwapchainCreateInfoKHR,
    _p_allocator: *const vk::AllocationCallbacks,
    p_swapchain: *mut vk::SwapchainKHR,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating swapchain: {}x{}, format={:?}",
        create_info.image_extent.width, create_info.image_extent.height, create_info.image_format
    );

    let swapchain_data = VkSwapchainData {
        device,
        surface: create_info.surface,
        min_image_count: create_info.min_image_count,
        image_format: create_info.image_format,
        image_color_space: create_info.image_color_space,
        image_extent: create_info.image_extent,
        image_array_layers: create_info.image_array_layers,
        image_usage: create_info.image_usage,
        pre_transform: create_info.pre_transform,
        composite_alpha: create_info.composite_alpha,
        present_mode: create_info.present_mode,
        images: Vec::new(),
    };

    let swapchain_handle = SWAPCHAIN_ALLOCATOR.allocate(swapchain_data);
    *p_swapchain = Handle::from_raw(swapchain_handle);

    Ok(())
}

pub unsafe fn destroy_swapchain_khr(
    swapchain: vk::SwapchainKHR,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if swapchain == vk::SwapchainKHR::null() {
        return;
    }

    SWAPCHAIN_ALLOCATOR.remove(swapchain.as_raw());
}

pub unsafe fn get_swapchain_images_khr(
    swapchain: vk::SwapchainKHR,
    p_swapchain_image_count: *mut u32,
    p_swapchain_images: *mut vk::Image,
) -> Result<()> {
    // Simplified: return empty for now
    if p_swapchain_images.is_null() {
        *p_swapchain_image_count = 3; // Typical triple buffering
    } else {
        *p_swapchain_image_count = 0;
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
    *p_image_index = 0;
    debug!("Acquired swapchain image 0");
    Ok(())
}

pub unsafe fn queue_present_khr(
    _queue: vk::Queue,
    p_present_info: *const vk::PresentInfoKHR,
) -> Result<()> {
    let present_info = &*p_present_info;
    debug!("Presenting {} swapchains", present_info.swapchain_count);
    Ok(())
}
