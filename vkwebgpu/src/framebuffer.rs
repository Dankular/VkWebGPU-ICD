//! Vulkan Framebuffer implementation

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::error::Result;
use crate::handle::HandleAllocator;

pub static FRAMEBUFFER_ALLOCATOR: Lazy<HandleAllocator<VkFramebufferData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkFramebufferData {
    pub device: vk::Device,
    pub render_pass: vk::RenderPass,
    pub attachments: Vec<vk::ImageView>,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

pub unsafe fn create_framebuffer(
    device: vk::Device,
    p_create_info: *const vk::FramebufferCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_framebuffer: *mut vk::Framebuffer,
) -> Result<()> {
    let create_info = &*p_create_info;

    let attachments = if create_info.attachment_count > 0 {
        std::slice::from_raw_parts(
            create_info.p_attachments,
            create_info.attachment_count as usize,
        )
        .to_vec()
    } else {
        Vec::new()
    };

    debug!(
        "Creating framebuffer: {}x{} with {} attachments",
        create_info.width,
        create_info.height,
        attachments.len()
    );

    let framebuffer_data = VkFramebufferData {
        device,
        render_pass: create_info.render_pass,
        attachments,
        width: create_info.width,
        height: create_info.height,
        layers: create_info.layers,
    };

    let framebuffer_handle = FRAMEBUFFER_ALLOCATOR.allocate(framebuffer_data);
    *p_framebuffer = Handle::from_raw(framebuffer_handle);

    Ok(())
}

pub unsafe fn destroy_framebuffer(
    _device: vk::Device,
    framebuffer: vk::Framebuffer,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if framebuffer == vk::Framebuffer::null() {
        return;
    }

    FRAMEBUFFER_ALLOCATOR.remove(framebuffer.as_raw());
}

pub fn get_framebuffer_data(framebuffer: vk::Framebuffer) -> Option<Arc<VkFramebufferData>> {
    FRAMEBUFFER_ALLOCATOR.get(framebuffer.as_raw())
}
