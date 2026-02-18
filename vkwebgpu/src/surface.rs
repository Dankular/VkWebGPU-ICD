//! Vulkan Surface implementation (KHR extension)

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static SURFACE_ALLOCATOR: Lazy<HandleAllocator<VkSurfaceData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkSurfaceData {
    pub instance: vk::Instance,
    pub hwnd: usize,
    pub hinstance: usize,
}

pub unsafe fn create_win32_surface_khr(
    instance: vk::Instance,
    p_create_info: *const vk::Win32SurfaceCreateInfoKHR,
    _p_allocator: *const vk::AllocationCallbacks,
    p_surface: *mut vk::SurfaceKHR,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating Win32 surface: hwnd={:?}, hinstance={:?}",
        create_info.hwnd, create_info.hinstance
    );

    let surface_data = VkSurfaceData {
        instance,
        hwnd: create_info.hwnd as usize,
        hinstance: create_info.hinstance as usize,
    };

    let surface_handle = SURFACE_ALLOCATOR.allocate(surface_data);
    *p_surface = Handle::from_raw(surface_handle);

    debug!("Win32 surface created: {:?}", *p_surface);
    Ok(())
}

pub unsafe fn destroy_surface_khr(
    _instance: vk::Instance,
    surface: vk::SurfaceKHR,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if surface == vk::SurfaceKHR::null() {
        return;
    }

    debug!("Destroying surface: {:?}", surface);
    SURFACE_ALLOCATOR.remove(surface.as_raw());
}

pub unsafe fn get_physical_device_surface_support_khr(
    _physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    _surface: vk::SurfaceKHR,
    p_supported: *mut vk::Bool32,
) -> Result<()> {
    debug!(
        "Querying surface support for queue family: {}",
        queue_family_index
    );

    *p_supported = vk::TRUE;
    Ok(())
}

pub unsafe fn get_physical_device_surface_capabilities_khr(
    _physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    p_surface_capabilities: *mut vk::SurfaceCapabilitiesKHR,
) -> Result<()> {
    debug!("Querying surface capabilities for: {:?}", surface);

    let _surface_data = SURFACE_ALLOCATOR
        .get(surface.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid surface handle".to_string()))?;

    let caps = &mut *p_surface_capabilities;

    caps.min_image_count = 2;
    caps.max_image_count = 8;
    caps.current_extent = vk::Extent2D {
        width: 0xFFFFFFFF,
        height: 0xFFFFFFFF,
    };
    caps.min_image_extent = vk::Extent2D { width: 1, height: 1 };
    caps.max_image_extent = vk::Extent2D {
        width: 16384,
        height: 16384,
    };
    caps.max_image_array_layers = 1;
    caps.supported_transforms = vk::SurfaceTransformFlagsKHR::IDENTITY;
    caps.current_transform = vk::SurfaceTransformFlagsKHR::IDENTITY;
    caps.supported_composite_alpha = vk::CompositeAlphaFlagsKHR::OPAQUE
        | vk::CompositeAlphaFlagsKHR::INHERIT;
    caps.supported_usage_flags = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::SAMPLED;

    Ok(())
}

pub unsafe fn get_physical_device_surface_formats_khr(
    _physical_device: vk::PhysicalDevice,
    _surface: vk::SurfaceKHR,
    p_surface_format_count: *mut u32,
    p_surface_formats: *mut vk::SurfaceFormatKHR,
) -> Result<()> {
    const FORMATS: &[vk::SurfaceFormatKHR] = &[
        vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
    ];

    if p_surface_formats.is_null() {
        *p_surface_format_count = FORMATS.len() as u32;
        debug!("Querying surface format count: {}", FORMATS.len());
    } else {
        let count = (*p_surface_format_count as usize).min(FORMATS.len());
        let formats = std::slice::from_raw_parts_mut(p_surface_formats, count);
        formats.copy_from_slice(&FORMATS[..count]);
        *p_surface_format_count = count as u32;
        debug!("Returning {} surface formats", count);
    }

    Ok(())
}

pub unsafe fn get_physical_device_surface_present_modes_khr(
    _physical_device: vk::PhysicalDevice,
    _surface: vk::SurfaceKHR,
    p_present_mode_count: *mut u32,
    p_present_modes: *mut vk::PresentModeKHR,
) -> Result<()> {
    const MODES: &[vk::PresentModeKHR] = &[
        vk::PresentModeKHR::FIFO,
        vk::PresentModeKHR::FIFO_RELAXED,
        vk::PresentModeKHR::IMMEDIATE,
        vk::PresentModeKHR::MAILBOX,
    ];

    if p_present_modes.is_null() {
        *p_present_mode_count = MODES.len() as u32;
        debug!("Querying present mode count: {}", MODES.len());
    } else {
        let count = (*p_present_mode_count as usize).min(MODES.len());
        let modes = std::slice::from_raw_parts_mut(p_present_modes, count);
        modes.copy_from_slice(&MODES[..count]);
        *p_present_mode_count = count as u32;
        debug!("Returning {} present modes", count);
    }

    Ok(())
}

pub fn get_surface_data(surface: vk::SurfaceKHR) -> Option<Arc<VkSurfaceData>> {
    SURFACE_ALLOCATOR.get(surface.as_raw())
}
