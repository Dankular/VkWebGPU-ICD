//! Vulkan Render Pass implementation

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::error::Result;
use crate::handle::HandleAllocator;

pub static RENDER_PASS_ALLOCATOR: Lazy<HandleAllocator<VkRenderPassData>> =
    Lazy::new(|| HandleAllocator::new());

/// Owned subpass information extracted from VkSubpassDescription / VkSubpassDescription2.
/// Used by pipeline creation to determine which attachment formats are needed.
pub struct OwnedSubpassInfo {
    /// Indices into the render pass attachment array for color attachments.
    pub color_attachment_indices: Vec<u32>,
    /// Index into the render pass attachment array for the depth/stencil attachment (if any).
    pub depth_stencil_attachment_index: Option<u32>,
}

pub struct VkRenderPassData {
    pub device: vk::Device,
    pub attachments: Vec<vk::AttachmentDescription>,
    pub subpasses: Vec<OwnedSubpassInfo>,
    pub dependencies: Vec<vk::SubpassDependency>,
}

pub unsafe fn create_render_pass(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_render_pass: *mut vk::RenderPass,
) -> Result<()> {
    let create_info = &*p_create_info;

    let attachments = if create_info.attachment_count > 0 && !create_info.p_attachments.is_null() {
        std::slice::from_raw_parts(
            create_info.p_attachments,
            create_info.attachment_count as usize,
        )
        .to_vec()
    } else {
        Vec::new()
    };

    // Parse subpass attachment usage so pipeline creation can look up formats per subpass.
    let subpasses: Vec<OwnedSubpassInfo> =
        if create_info.subpass_count > 0 && !create_info.p_subpasses.is_null() {
            std::slice::from_raw_parts(create_info.p_subpasses, create_info.subpass_count as usize)
                .iter()
                .map(|sp| {
                    let color_attachment_indices: Vec<u32> =
                        if sp.color_attachment_count > 0 && !sp.p_color_attachments.is_null() {
                            std::slice::from_raw_parts(
                                sp.p_color_attachments,
                                sp.color_attachment_count as usize,
                            )
                            .iter()
                            .filter(|ar| ar.attachment != vk::ATTACHMENT_UNUSED)
                            .map(|ar| ar.attachment)
                            .collect()
                        } else {
                            Vec::new()
                        };

                    let depth_stencil_attachment_index =
                        if !sp.p_depth_stencil_attachment.is_null() {
                            let ar = &*sp.p_depth_stencil_attachment;
                            if ar.attachment != vk::ATTACHMENT_UNUSED {
                                Some(ar.attachment)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                    OwnedSubpassInfo {
                        color_attachment_indices,
                        depth_stencil_attachment_index,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

    debug!(
        "Creating render pass with {} attachments, {} subpasses",
        attachments.len(),
        subpasses.len()
    );

    let render_pass_data = VkRenderPassData {
        device,
        attachments,
        subpasses,
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

/// vkCreateRenderPass2 / vkCreateRenderPass2KHR
pub unsafe fn create_render_pass2(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo2,
    _p_allocator: *const vk::AllocationCallbacks,
    p_render_pass: *mut vk::RenderPass,
) -> Result<()> {
    let create_info = &*p_create_info;

    // Convert RenderPassCreateInfo2 attachments to AttachmentDescription
    let attachments: Vec<vk::AttachmentDescription> = if create_info.attachment_count > 0
        && !create_info.p_attachments.is_null()
    {
        std::slice::from_raw_parts(create_info.p_attachments, create_info.attachment_count as usize)
            .iter()
            .map(|a| vk::AttachmentDescription {
                flags: a.flags,
                format: a.format,
                samples: a.samples,
                load_op: a.load_op,
                store_op: a.store_op,
                stencil_load_op: a.stencil_load_op,
                stencil_store_op: a.stencil_store_op,
                initial_layout: a.initial_layout,
                final_layout: a.final_layout,
            })
            .collect()
    } else {
        Vec::new()
    };

    // Parse subpass attachment usage from VkSubpassDescription2.
    let subpasses: Vec<OwnedSubpassInfo> =
        if create_info.subpass_count > 0 && !create_info.p_subpasses.is_null() {
            std::slice::from_raw_parts(create_info.p_subpasses, create_info.subpass_count as usize)
                .iter()
                .map(|sp| {
                    let color_attachment_indices: Vec<u32> =
                        if sp.color_attachment_count > 0 && !sp.p_color_attachments.is_null() {
                            std::slice::from_raw_parts(
                                sp.p_color_attachments,
                                sp.color_attachment_count as usize,
                            )
                            .iter()
                            .filter(|ar| ar.attachment != vk::ATTACHMENT_UNUSED)
                            .map(|ar| ar.attachment)
                            .collect()
                        } else {
                            Vec::new()
                        };

                    let depth_stencil_attachment_index =
                        if !sp.p_depth_stencil_attachment.is_null() {
                            let ar = &*sp.p_depth_stencil_attachment;
                            if ar.attachment != vk::ATTACHMENT_UNUSED {
                                Some(ar.attachment)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                    OwnedSubpassInfo {
                        color_attachment_indices,
                        depth_stencil_attachment_index,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

    debug!(
        "Creating render pass2 with {} attachments, {} subpasses",
        attachments.len(),
        subpasses.len()
    );

    let render_pass_data = VkRenderPassData {
        device,
        attachments,
        subpasses,
        dependencies: Vec::new(),
    };

    let render_pass_handle = RENDER_PASS_ALLOCATOR.allocate(render_pass_data);
    *p_render_pass = Handle::from_raw(render_pass_handle);

    Ok(())
}
