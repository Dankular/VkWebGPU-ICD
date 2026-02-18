//! Vulkan Render Pass implementation

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::error::Result;
use crate::handle::HandleAllocator;

pub static RENDER_PASS_ALLOCATOR: Lazy<HandleAllocator<VkRenderPassData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkRenderPassData {
    pub device: vk::Device,
    pub attachments: Vec<vk::AttachmentDescription>,
    pub subpasses: Vec<vk::SubpassDescription<'static>>,
    pub dependencies: Vec<vk::SubpassDependency>,
}

pub unsafe fn create_render_pass(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_render_pass: *mut vk::RenderPass,
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
        "Creating render pass with {} attachments",
        attachments.len()
    );

    let render_pass_data = VkRenderPassData {
        device,
        attachments,
        subpasses: Vec::new(),
        dependencies: Vec::new(),
    };

    let render_pass_handle = RENDER_PASS_ALLOCATOR.allocate(render_pass_data);
    *p_render_pass = Handle::from_raw(render_pass_handle);

    Ok(())
}

pub unsafe fn destroy_render_pass(
    _device: vk::Device,
    render_pass: vk::RenderPass,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if render_pass == vk::RenderPass::null() {
        return;
    }

    RENDER_PASS_ALLOCATOR.remove(render_pass.as_raw());
}

pub fn get_render_pass_data(render_pass: vk::RenderPass) -> Option<Arc<VkRenderPassData>> {
    RENDER_PASS_ALLOCATOR.get(render_pass.as_raw())
}
