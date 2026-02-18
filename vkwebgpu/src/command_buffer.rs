//! Vulkan Command Buffer implementation
//! Maps VkCommandBuffer to WebGPU GPUCommandEncoder
//!
//! Uses deferred command recording: Vulkan commands are recorded into a command list
//! and then replayed at vkQueueSubmit time to create the actual WebGPU command buffer.
//! This approach correctly handles WebGPU's scoped RenderPass lifetime while maintaining
//! Vulkan's thread-safe command buffer recording semantics.

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::backend::{CommandBuffer, WebGPUBackend};
use crate::command_pool;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static COMMAND_BUFFER_ALLOCATOR: Lazy<HandleAllocator<VkCommandBufferData>> =
    Lazy::new(|| HandleAllocator::new());

/// Attachment info for dynamic rendering (vkCmdBeginRendering)
#[derive(Clone)]
pub struct RenderingAttachment {
    pub image_view: vk::ImageView,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: vk::ClearValue,
}

pub struct VkCommandBufferData {
    pub device: vk::Device,
    pub command_pool: vk::CommandPool,
    pub level: vk::CommandBufferLevel,
    pub state: RwLock<CommandBufferState>,
    pub commands: RwLock<Vec<RecordedCommand>>,
}

pub enum CommandBufferState {
    Initial,
    Recording,
    Executable,
    Invalid,
}

/// Recorded command - stores all data needed to replay the command at submit time
#[derive(Clone)]
pub enum RecordedCommand {
    BeginRenderPass {
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        render_area: vk::Rect2D,
        clear_values: Vec<vk::ClearValue>,
    },
    EndRenderPass,
    BindPipeline {
        bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    },
    BindVertexBuffers {
        first_binding: u32,
        buffers: Vec<vk::Buffer>,
        offsets: Vec<vk::DeviceSize>,
    },
    BindIndexBuffer {
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    },
    BindDescriptorSets {
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: Vec<vk::DescriptorSet>,
        dynamic_offsets: Vec<u32>,
    },
    PushConstants {
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        size: u32,
        data: Vec<u8>,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    Dispatch {
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
    CopyBuffer {
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    CopyBufferToImage {
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: Vec<vk::BufferImageCopy>,
    },
    PipelineBarrier {
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
    },
    SetViewport {
        first_viewport: u32,
        viewports: Vec<vk::Viewport>,
    },
    SetScissor {
        first_scissor: u32,
        scissors: Vec<vk::Rect2D>,
    },
    SetBlendConstants {
        blend_constants: [f32; 4],
    },
    SetStencilReference {
        face_mask: vk::StencilFaceFlags,
        reference: u32,
    },
    ClearColorImage {
        image: vk::Image,
        image_layout: vk::ImageLayout,
        color: vk::ClearColorValue,
        ranges: Vec<vk::ImageSubresourceRange>,
    },
    ClearDepthStencilImage {
        image: vk::Image,
        image_layout: vk::ImageLayout,
        depth_stencil: vk::ClearDepthStencilValue,
        ranges: Vec<vk::ImageSubresourceRange>,
    },
    ClearAttachments {
        attachments: Vec<vk::ClearAttachment>,
        rects: Vec<vk::ClearRect>,
    },
    CopyImageToBuffer {
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_buffer: vk::Buffer,
        regions: Vec<vk::BufferImageCopy>,
    },
    CopyImage {
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: Vec<vk::ImageCopy>,
    },
    BlitImage {
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: Vec<vk::ImageBlit>,
        filter: vk::Filter,
    },
    // Dynamic rendering (Vulkan 1.3 / VK_KHR_dynamic_rendering)
    BeginRendering {
        render_area: vk::Rect2D,
        layer_count: u32,
        color_attachments: Vec<RenderingAttachment>,
        depth_attachment: Option<RenderingAttachment>,
        stencil_attachment: Option<RenderingAttachment>,
    },
    EndRendering,
    // Indirect draw commands
    DrawIndirect {
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    },
    DrawIndexedIndirect {
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    },
    // Buffer fill / update
    FillBuffer {
        dst_buffer: vk::Buffer,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
        data: u32,
    },
    UpdateBuffer {
        dst_buffer: vk::Buffer,
        dst_offset: vk::DeviceSize,
        data: Vec<u8>,
    },
    // Synchronization2 barrier (no-op like PipelineBarrier)
    PipelineBarrier2,
    // Copy2 variants (VK_KHR_copy_commands2)
    CopyBuffer2 {
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    CopyImage2 {
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: Vec<vk::ImageCopy>,
    },
    CopyBufferToImage2 {
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: Vec<vk::BufferImageCopy>,
    },
    CopyImageToBuffer2 {
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_buffer: vk::Buffer,
        regions: Vec<vk::BufferImageCopy>,
    },
    BlitImage2 {
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: Vec<vk::ImageBlit>,
        filter: vk::Filter,
    },
    ResolveImage {
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: Vec<vk::ImageResolve>,
    },
    // Extended dynamic state (VK_EXT_extended_dynamic_state)
    SetCullMode { cull_mode: vk::CullModeFlags },
    SetFrontFace { front_face: vk::FrontFace },
    SetPrimitiveTopology { primitive_topology: vk::PrimitiveTopology },
    SetDepthTestEnable { depth_test_enable: vk::Bool32 },
    SetDepthWriteEnable { depth_write_enable: vk::Bool32 },
    SetDepthCompareOp { depth_compare_op: vk::CompareOp },
    SetDepthBiasEnable { depth_bias_enable: vk::Bool32 },
    SetStencilTestEnable { stencil_test_enable: vk::Bool32 },
    SetStencilOp {
        face_mask: vk::StencilFaceFlags,
        fail_op: vk::StencilOp,
        pass_op: vk::StencilOp,
        depth_fail_op: vk::StencilOp,
        compare_op: vk::CompareOp,
    },
    SetDepthBounds {
        min_depth_bounds: f32,
        max_depth_bounds: f32,
    },
    SetLineWidth { line_width: f32 },
    SetDepthBias {
        depth_bias_constant_factor: f32,
        depth_bias_clamp: f32,
        depth_bias_slope_factor: f32,
    },
    // RenderPass2 variants (VK_KHR_create_renderpass2)
    BeginRenderPass2 {
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        render_area: vk::Rect2D,
        clear_values: Vec<vk::ClearValue>,
    },
    NextSubpass,
    NextSubpass2,
    EndRenderPass2,
    // Secondary command buffers
    ExecuteCommands {
        command_buffers: Vec<vk::CommandBuffer>,
    },
    // Dispatch with base offset
    DispatchBase {
        base_group_x: u32,
        base_group_y: u32,
        base_group_z: u32,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
}

pub unsafe fn allocate_command_buffers(
    device: vk::Device,
    p_allocate_info: *const vk::CommandBufferAllocateInfo,
    p_command_buffers: *mut vk::CommandBuffer,
) -> Result<()> {
    let allocate_info = &*p_allocate_info;

    let pool_data = command_pool::get_command_pool_data(allocate_info.command_pool)
        .ok_or_else(|| VkError::InvalidHandle("Invalid command pool".to_string()))?;

    let command_buffers = std::slice::from_raw_parts_mut(
        p_command_buffers,
        allocate_info.command_buffer_count as usize,
    );

    debug!("Allocating {} command buffers", command_buffers.len());

    for cmd_buffer in command_buffers.iter_mut() {
        let cmd_data = VkCommandBufferData {
            device,
            command_pool: allocate_info.command_pool,
            level: allocate_info.level,
            state: RwLock::new(CommandBufferState::Initial),
            commands: RwLock::new(Vec::new()),
        };

        let cmd_handle = COMMAND_BUFFER_ALLOCATOR.allocate(cmd_data);
        *cmd_buffer = Handle::from_raw(cmd_handle);

        pool_data.allocated_buffers.write().push(*cmd_buffer);
    }

    Ok(())
}

pub unsafe fn begin_command_buffer(
    command_buffer: vk::CommandBuffer,
    _p_begin_info: *const vk::CommandBufferBeginInfo,
) -> Result<()> {
    let cmd_data = COMMAND_BUFFER_ALLOCATOR
        .get(command_buffer.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid command buffer".to_string()))?;

    debug!("Beginning command buffer recording");

    // Clear previous commands
    cmd_data.commands.write().clear();

    *cmd_data.state.write() = CommandBufferState::Recording;

    Ok(())
}

pub unsafe fn end_command_buffer(command_buffer: vk::CommandBuffer) -> Result<()> {
    let cmd_data = COMMAND_BUFFER_ALLOCATOR
        .get(command_buffer.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid command buffer".to_string()))?;

    debug!("Ending command buffer recording");

    *cmd_data.state.write() = CommandBufferState::Executable;

    Ok(())
}

pub unsafe fn free_command_buffers(
    _device: vk::Device,
    command_pool: vk::CommandPool,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    if command_buffer_count == 0 || p_command_buffers.is_null() {
        return;
    }
    let buffers = std::slice::from_raw_parts(p_command_buffers, command_buffer_count as usize);

    for &cmd_buffer in buffers {
        if cmd_buffer != vk::CommandBuffer::null() {
            COMMAND_BUFFER_ALLOCATOR.remove(cmd_buffer.as_raw());
        }
    }

    // Also remove from pool's tracking list
    if let Some(pool_data) = command_pool::get_command_pool_data(command_pool) {
        let mut allocated = pool_data.allocated_buffers.write();
        for &cmd_buffer in buffers {
            allocated.retain(|&b| b != cmd_buffer);
        }
    }

    debug!("Freed {} command buffers", command_buffer_count);
}

pub unsafe fn reset_command_buffer(
    command_buffer: vk::CommandBuffer,
    _flags: vk::CommandBufferResetFlags,
) -> Result<()> {
    let cmd_data = COMMAND_BUFFER_ALLOCATOR
        .get(command_buffer.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid command buffer".to_string()))?;

    cmd_data.commands.write().clear();
    *cmd_data.state.write() = CommandBufferState::Initial;

    debug!("Reset command buffer");
    Ok(())
}

pub unsafe fn reset_command_pool(
    _device: vk::Device,
    command_pool: vk::CommandPool,
    _flags: vk::CommandPoolResetFlags,
) -> Result<()> {
    if let Some(pool_data) = command_pool::get_command_pool_data(command_pool) {
        let allocated = pool_data.allocated_buffers.read().clone();
        for &cmd_buffer in &allocated {
            if let Some(cmd_data) = COMMAND_BUFFER_ALLOCATOR.get(cmd_buffer.as_raw()) {
                cmd_data.commands.write().clear();
                *cmd_data.state.write() = CommandBufferState::Initial;
            }
        }
    }
    debug!("Reset command pool");
    Ok(())
}

pub unsafe fn cmd_begin_render_pass(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo,
    _contents: vk::SubpassContents,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    let render_pass_begin = &*p_render_pass_begin;

    debug!("Recording BeginRenderPass");

    let clear_values = if render_pass_begin.clear_value_count > 0 {
        std::slice::from_raw_parts(
            render_pass_begin.p_clear_values,
            render_pass_begin.clear_value_count as usize,
        )
        .to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::BeginRenderPass {
            render_pass: render_pass_begin.render_pass,
            framebuffer: render_pass_begin.framebuffer,
            render_area: render_pass_begin.render_area,
            clear_values,
        });
}

pub unsafe fn cmd_end_render_pass(command_buffer: vk::CommandBuffer) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording EndRenderPass");

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::EndRenderPass);
}

pub unsafe fn cmd_bind_pipeline(
    command_buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording BindPipeline: {:?}", pipeline);

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::BindPipeline {
            bind_point: pipeline_bind_point,
            pipeline,
        });
}

pub unsafe fn cmd_bind_descriptor_sets(
    command_buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    first_set: u32,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
    dynamic_offset_count: u32,
    p_dynamic_offsets: *const u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording BindDescriptorSets: {} sets",
        descriptor_set_count
    );

    let descriptor_sets = if descriptor_set_count > 0 {
        std::slice::from_raw_parts(p_descriptor_sets, descriptor_set_count as usize).to_vec()
    } else {
        Vec::new()
    };

    let dynamic_offsets = if dynamic_offset_count > 0 {
        std::slice::from_raw_parts(p_dynamic_offsets, dynamic_offset_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::BindDescriptorSets {
            bind_point: pipeline_bind_point,
            layout,
            first_set,
            descriptor_sets,
            dynamic_offsets,
        });
}

pub unsafe fn cmd_draw(
    command_buffer: vk::CommandBuffer,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording Draw: vertices={}, instances={}, first_vertex={}, first_instance={}",
        vertex_count, instance_count, first_vertex, first_instance
    );

    cmd_data.commands.write().push(RecordedCommand::Draw {
        vertex_count,
        instance_count,
        first_vertex,
        first_instance,
    });
}

pub unsafe fn cmd_draw_indexed(
    command_buffer: vk::CommandBuffer,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording DrawIndexed: indices={}, instances={}, first_index={}, vertex_offset={}, first_instance={}",
        index_count, instance_count, first_index, vertex_offset, first_instance
    );

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        });
}

pub unsafe fn cmd_bind_vertex_buffers(
    command_buffer: vk::CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_buffers: *const vk::Buffer,
    p_offsets: *const vk::DeviceSize,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording BindVertexBuffers: {} buffers starting at binding {}",
        binding_count, first_binding
    );

    let buffers = if binding_count > 0 {
        std::slice::from_raw_parts(p_buffers, binding_count as usize).to_vec()
    } else {
        Vec::new()
    };

    let offsets = if binding_count > 0 {
        std::slice::from_raw_parts(p_offsets, binding_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::BindVertexBuffers {
            first_binding,
            buffers,
            offsets,
        });
}

pub unsafe fn cmd_bind_index_buffer(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording BindIndexBuffer: offset={}, type={:?}",
        offset, index_type
    );

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::BindIndexBuffer {
            buffer,
            offset,
            index_type,
        });
}

pub unsafe fn cmd_copy_buffer(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    region_count: u32,
    p_regions: *const vk::BufferCopy,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording CopyBuffer: {} regions", region_count);

    let regions = if region_count > 0 {
        std::slice::from_raw_parts(p_regions, region_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data.commands.write().push(RecordedCommand::CopyBuffer {
        src_buffer,
        dst_buffer,
        regions,
    });
}

pub unsafe fn cmd_copy_buffer_to_image(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::BufferImageCopy,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording CopyBufferToImage: {} regions", region_count);

    let regions = if region_count > 0 {
        std::slice::from_raw_parts(p_regions, region_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::CopyBufferToImage {
            src_buffer,
            dst_image,
            dst_image_layout,
            regions,
        });
}

pub unsafe fn cmd_pipeline_barrier(
    command_buffer: vk::CommandBuffer,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barrier_count: u32,
    _p_memory_barriers: *const vk::MemoryBarrier,
    buffer_memory_barrier_count: u32,
    _p_buffer_memory_barriers: *const vk::BufferMemoryBarrier,
    image_memory_barrier_count: u32,
    _p_image_memory_barriers: *const vk::ImageMemoryBarrier,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording PipelineBarrier: mem={}, buf={}, img={}",
        memory_barrier_count, buffer_memory_barrier_count, image_memory_barrier_count
    );

    // WebGPU handles barriers implicitly, but we record it for completeness
    cmd_data
        .commands
        .write()
        .push(RecordedCommand::PipelineBarrier {
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
        });
}

pub unsafe fn cmd_dispatch(
    command_buffer: vk::CommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording Dispatch: {}x{}x{}",
        group_count_x, group_count_y, group_count_z
    );

    cmd_data.commands.write().push(RecordedCommand::Dispatch {
        group_count_x,
        group_count_y,
        group_count_z,
    });
}

pub unsafe fn cmd_push_constants(
    command_buffer: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    stage_flags: vk::ShaderStageFlags,
    offset: u32,
    size: u32,
    p_values: *const std::ffi::c_void,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording PushConstants: layout={:?}, stages={:?}, offset={}, size={}",
        layout, stage_flags, offset, size
    );

    // Copy the data from the pointer
    let data = if size > 0 && !p_values.is_null() {
        let src_slice = std::slice::from_raw_parts(p_values as *const u8, size as usize);
        src_slice.to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::PushConstants {
            layout,
            stage_flags,
            offset,
            size,
            data,
        });
}

pub unsafe fn cmd_set_viewport(
    command_buffer: vk::CommandBuffer,
    first_viewport: u32,
    viewport_count: u32,
    p_viewports: *const vk::Viewport,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording SetViewport: first={}, count={}",
        first_viewport, viewport_count
    );

    let viewports = if viewport_count > 0 {
        std::slice::from_raw_parts(p_viewports, viewport_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::SetViewport {
            first_viewport,
            viewports,
        });
}

pub unsafe fn cmd_set_scissor(
    command_buffer: vk::CommandBuffer,
    first_scissor: u32,
    scissor_count: u32,
    p_scissors: *const vk::Rect2D,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording SetScissor: first={}, count={}",
        first_scissor, scissor_count
    );

    let scissors = if scissor_count > 0 {
        std::slice::from_raw_parts(p_scissors, scissor_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data.commands.write().push(RecordedCommand::SetScissor {
        first_scissor,
        scissors,
    });
}

pub unsafe fn cmd_set_blend_constants(
    command_buffer: vk::CommandBuffer,
    blend_constants: *const [f32; 4],
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording SetBlendConstants");

    let blend_constants_array = if !blend_constants.is_null() {
        *blend_constants
    } else {
        [0.0, 0.0, 0.0, 0.0]
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::SetBlendConstants {
            blend_constants: blend_constants_array,
        });
}

pub unsafe fn cmd_set_stencil_reference(
    command_buffer: vk::CommandBuffer,
    face_mask: vk::StencilFaceFlags,
    reference: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording SetStencilReference: mask={:?}, ref={}",
        face_mask, reference
    );

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::SetStencilReference {
            face_mask,
            reference,
        });
}

pub unsafe fn cmd_clear_color_image(
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_layout: vk::ImageLayout,
    p_color: *const vk::ClearColorValue,
    range_count: u32,
    p_ranges: *const vk::ImageSubresourceRange,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording ClearColorImage: {} ranges", range_count);

    let color = if !p_color.is_null() {
        *p_color
    } else {
        vk::ClearColorValue::default()
    };

    let ranges = if range_count > 0 {
        std::slice::from_raw_parts(p_ranges, range_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::ClearColorImage {
            image,
            image_layout,
            color,
            ranges,
        });
}

pub unsafe fn cmd_clear_depth_stencil_image(
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_layout: vk::ImageLayout,
    p_depth_stencil: *const vk::ClearDepthStencilValue,
    range_count: u32,
    p_ranges: *const vk::ImageSubresourceRange,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording ClearDepthStencilImage: {} ranges", range_count);

    let depth_stencil = if !p_depth_stencil.is_null() {
        *p_depth_stencil
    } else {
        vk::ClearDepthStencilValue::default()
    };

    let ranges = if range_count > 0 {
        std::slice::from_raw_parts(p_ranges, range_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::ClearDepthStencilImage {
            image,
            image_layout,
            depth_stencil,
            ranges,
        });
}

pub unsafe fn cmd_clear_attachments(
    command_buffer: vk::CommandBuffer,
    attachment_count: u32,
    p_attachments: *const vk::ClearAttachment,
    rect_count: u32,
    p_rects: *const vk::ClearRect,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!(
        "Recording ClearAttachments: {} attachments, {} rects",
        attachment_count, rect_count
    );

    let attachments = if attachment_count > 0 {
        std::slice::from_raw_parts(p_attachments, attachment_count as usize).to_vec()
    } else {
        Vec::new()
    };

    let rects = if rect_count > 0 {
        std::slice::from_raw_parts(p_rects, rect_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::ClearAttachments { attachments, rects });
}

pub unsafe fn cmd_copy_image_to_buffer(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_buffer: vk::Buffer,
    region_count: u32,
    p_regions: *const vk::BufferImageCopy,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording CopyImageToBuffer: {} regions", region_count);

    let regions = if region_count > 0 {
        std::slice::from_raw_parts(p_regions, region_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::CopyImageToBuffer {
            src_image,
            src_image_layout,
            dst_buffer,
            regions,
        });
}

pub unsafe fn cmd_copy_image(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::ImageCopy,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording CopyImage: {} regions", region_count);

    let regions = if region_count > 0 {
        std::slice::from_raw_parts(p_regions, region_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data.commands.write().push(RecordedCommand::CopyImage {
        src_image,
        src_image_layout,
        dst_image,
        dst_image_layout,
        regions,
    });
}

pub unsafe fn cmd_blit_image(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::ImageBlit,
    filter: vk::Filter,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    debug!("Recording BlitImage: {} regions", region_count);

    let regions = if region_count > 0 {
        std::slice::from_raw_parts(p_regions, region_count as usize).to_vec()
    } else {
        Vec::new()
    };

    cmd_data.commands.write().push(RecordedCommand::BlitImage {
        src_image,
        src_image_layout,
        dst_image,
        dst_image_layout,
        regions,
        filter,
    });
}

pub fn get_command_buffer_data(
    command_buffer: vk::CommandBuffer,
) -> Option<Arc<VkCommandBufferData>> {
    COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw())
}

// ─── Dynamic rendering ────────────────────────────────────────────────────────

pub unsafe fn cmd_begin_rendering(
    command_buffer: vk::CommandBuffer,
    p_rendering_info: *const vk::RenderingInfo,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    let ri = &*p_rendering_info;
    debug!(
        "Recording BeginRendering: {} color attachments",
        ri.color_attachment_count
    );

    let color_attachments: Vec<RenderingAttachment> = if ri.color_attachment_count > 0
        && !ri.p_color_attachments.is_null()
    {
        std::slice::from_raw_parts(ri.p_color_attachments, ri.color_attachment_count as usize)
            .iter()
            .map(|a| RenderingAttachment {
                image_view: a.image_view,
                load_op: a.load_op,
                store_op: a.store_op,
                clear_value: a.clear_value,
            })
            .collect()
    } else {
        Vec::new()
    };

    let depth_attachment = if !ri.p_depth_attachment.is_null() {
        let a = &*ri.p_depth_attachment;
        if a.image_view != vk::ImageView::null() {
            Some(RenderingAttachment {
                image_view: a.image_view,
                load_op: a.load_op,
                store_op: a.store_op,
                clear_value: a.clear_value,
            })
        } else {
            None
        }
    } else {
        None
    };

    let stencil_attachment = if !ri.p_stencil_attachment.is_null() {
        let a = &*ri.p_stencil_attachment;
        if a.image_view != vk::ImageView::null() {
            Some(RenderingAttachment {
                image_view: a.image_view,
                load_op: a.load_op,
                store_op: a.store_op,
                clear_value: a.clear_value,
            })
        } else {
            None
        }
    } else {
        None
    };

    cmd_data
        .commands
        .write()
        .push(RecordedCommand::BeginRendering {
            render_area: ri.render_area,
            layer_count: ri.layer_count,
            color_attachments,
            depth_attachment,
            stencil_attachment,
        });
}

pub unsafe fn cmd_end_rendering(command_buffer: vk::CommandBuffer) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    debug!("Recording EndRendering");
    cmd_data
        .commands
        .write()
        .push(RecordedCommand::EndRendering);
}

// ─── Indirect draw ────────────────────────────────────────────────────────────

pub unsafe fn cmd_draw_indirect(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    debug!("Recording DrawIndirect: count={}", draw_count);
    cmd_data
        .commands
        .write()
        .push(RecordedCommand::DrawIndirect {
            buffer,
            offset,
            draw_count,
            stride,
        });
}

pub unsafe fn cmd_draw_indexed_indirect(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    debug!("Recording DrawIndexedIndirect: count={}", draw_count);
    cmd_data
        .commands
        .write()
        .push(RecordedCommand::DrawIndexedIndirect {
            buffer,
            offset,
            draw_count,
            stride,
        });
}

// ─── Buffer fill / update ─────────────────────────────────────────────────────

pub unsafe fn cmd_fill_buffer(
    command_buffer: vk::CommandBuffer,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    data: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(d) => d,
        None => return,
    };
    debug!("Recording FillBuffer: offset={}, size={}, data={:#x}", dst_offset, size, data);
    cmd_data
        .commands
        .write()
        .push(RecordedCommand::FillBuffer { dst_buffer, dst_offset, size, data });
}

pub unsafe fn cmd_update_buffer(
    command_buffer: vk::CommandBuffer,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    data_size: vk::DeviceSize,
    p_data: *const std::ffi::c_void,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(d) => d,
        None => return,
    };
    debug!("Recording UpdateBuffer: offset={}, size={}", dst_offset, data_size);
    let data = if data_size > 0 && !p_data.is_null() {
        std::slice::from_raw_parts(p_data as *const u8, data_size as usize).to_vec()
    } else {
        Vec::new()
    };
    cmd_data
        .commands
        .write()
        .push(RecordedCommand::UpdateBuffer { dst_buffer, dst_offset, data });
}

// ─── Synchronization2 barrier ─────────────────────────────────────────────────

pub unsafe fn cmd_pipeline_barrier2(
    command_buffer: vk::CommandBuffer,
    _p_dependency_info: *const vk::DependencyInfo,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    debug!("Recording PipelineBarrier2 (no-op)");
    cmd_data
        .commands
        .write()
        .push(RecordedCommand::PipelineBarrier2);
}

// ─── Copy2 variants ───────────────────────────────────────────────────────────

pub unsafe fn cmd_copy_buffer2(
    command_buffer: vk::CommandBuffer,
    p_copy_buffer_info: *const vk::CopyBufferInfo2,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let info = &*p_copy_buffer_info;
    let regions: Vec<vk::BufferCopy> = if info.region_count > 0 {
        std::slice::from_raw_parts(info.p_regions, info.region_count as usize)
            .iter()
            .map(|r| vk::BufferCopy { src_offset: r.src_offset, dst_offset: r.dst_offset, size: r.size })
            .collect()
    } else {
        Vec::new()
    };
    debug!("Recording CopyBuffer2: {} regions", regions.len());
    cmd_data.commands.write().push(RecordedCommand::CopyBuffer2 {
        src_buffer: info.src_buffer,
        dst_buffer: info.dst_buffer,
        regions,
    });
}

pub unsafe fn cmd_copy_image2(
    command_buffer: vk::CommandBuffer,
    p_copy_image_info: *const vk::CopyImageInfo2,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let info = &*p_copy_image_info;
    let regions: Vec<vk::ImageCopy> = if info.region_count > 0 {
        std::slice::from_raw_parts(info.p_regions, info.region_count as usize)
            .iter()
            .map(|r| vk::ImageCopy {
                src_subresource: r.src_subresource,
                src_offset: r.src_offset,
                dst_subresource: r.dst_subresource,
                dst_offset: r.dst_offset,
                extent: r.extent,
            })
            .collect()
    } else {
        Vec::new()
    };
    debug!("Recording CopyImage2: {} regions", regions.len());
    cmd_data.commands.write().push(RecordedCommand::CopyImage2 {
        src_image: info.src_image,
        src_image_layout: info.src_image_layout,
        dst_image: info.dst_image,
        dst_image_layout: info.dst_image_layout,
        regions,
    });
}

pub unsafe fn cmd_copy_buffer_to_image2(
    command_buffer: vk::CommandBuffer,
    p_copy_buffer_to_image_info: *const vk::CopyBufferToImageInfo2,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let info = &*p_copy_buffer_to_image_info;
    let regions: Vec<vk::BufferImageCopy> = if info.region_count > 0 {
        std::slice::from_raw_parts(info.p_regions, info.region_count as usize)
            .iter()
            .map(|r| vk::BufferImageCopy {
                buffer_offset: r.buffer_offset,
                buffer_row_length: r.buffer_row_length,
                buffer_image_height: r.buffer_image_height,
                image_subresource: r.image_subresource,
                image_offset: r.image_offset,
                image_extent: r.image_extent,
            })
            .collect()
    } else {
        Vec::new()
    };
    debug!("Recording CopyBufferToImage2: {} regions", regions.len());
    cmd_data.commands.write().push(RecordedCommand::CopyBufferToImage2 {
        src_buffer: info.src_buffer,
        dst_image: info.dst_image,
        dst_image_layout: info.dst_image_layout,
        regions,
    });
}

pub unsafe fn cmd_copy_image_to_buffer2(
    command_buffer: vk::CommandBuffer,
    p_copy_image_to_buffer_info: *const vk::CopyImageToBufferInfo2,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let info = &*p_copy_image_to_buffer_info;
    let regions: Vec<vk::BufferImageCopy> = if info.region_count > 0 {
        std::slice::from_raw_parts(info.p_regions, info.region_count as usize)
            .iter()
            .map(|r| vk::BufferImageCopy {
                buffer_offset: r.buffer_offset,
                buffer_row_length: r.buffer_row_length,
                buffer_image_height: r.buffer_image_height,
                image_subresource: r.image_subresource,
                image_offset: r.image_offset,
                image_extent: r.image_extent,
            })
            .collect()
    } else {
        Vec::new()
    };
    debug!("Recording CopyImageToBuffer2: {} regions", regions.len());
    cmd_data.commands.write().push(RecordedCommand::CopyImageToBuffer2 {
        src_image: info.src_image,
        src_image_layout: info.src_image_layout,
        dst_buffer: info.dst_buffer,
        regions,
    });
}

pub unsafe fn cmd_blit_image2(
    command_buffer: vk::CommandBuffer,
    p_blit_image_info: *const vk::BlitImageInfo2,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let info = &*p_blit_image_info;
    let regions: Vec<vk::ImageBlit> = if info.region_count > 0 {
        std::slice::from_raw_parts(info.p_regions, info.region_count as usize)
            .iter()
            .map(|r| vk::ImageBlit {
                src_subresource: r.src_subresource,
                src_offsets: [
                    vk::Offset3D { x: r.src_offsets[0].x, y: r.src_offsets[0].y, z: r.src_offsets[0].z },
                    vk::Offset3D { x: r.src_offsets[1].x, y: r.src_offsets[1].y, z: r.src_offsets[1].z },
                ],
                dst_subresource: r.dst_subresource,
                dst_offsets: [
                    vk::Offset3D { x: r.dst_offsets[0].x, y: r.dst_offsets[0].y, z: r.dst_offsets[0].z },
                    vk::Offset3D { x: r.dst_offsets[1].x, y: r.dst_offsets[1].y, z: r.dst_offsets[1].z },
                ],
            })
            .collect()
    } else {
        Vec::new()
    };
    debug!("Recording BlitImage2: {} regions", regions.len());
    cmd_data.commands.write().push(RecordedCommand::BlitImage2 {
        src_image: info.src_image,
        src_image_layout: info.src_image_layout,
        dst_image: info.dst_image,
        dst_image_layout: info.dst_image_layout,
        regions,
        filter: info.filter,
    });
}

pub unsafe fn cmd_resolve_image(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::ImageResolve,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let regions = if region_count > 0 {
        std::slice::from_raw_parts(p_regions, region_count as usize).to_vec()
    } else {
        Vec::new()
    };
    debug!("Recording ResolveImage: {} regions", region_count);
    cmd_data.commands.write().push(RecordedCommand::ResolveImage {
        src_image, src_image_layout, dst_image, dst_image_layout, regions,
    });
}

// ─── Extended dynamic state ───────────────────────────────────────────────────

pub unsafe fn cmd_set_cull_mode(command_buffer: vk::CommandBuffer, cull_mode: vk::CullModeFlags) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetCullMode { cull_mode });
    }
}

pub unsafe fn cmd_set_front_face(command_buffer: vk::CommandBuffer, front_face: vk::FrontFace) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetFrontFace { front_face });
    }
}

pub unsafe fn cmd_set_primitive_topology(
    command_buffer: vk::CommandBuffer,
    primitive_topology: vk::PrimitiveTopology,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetPrimitiveTopology { primitive_topology });
    }
}

pub unsafe fn cmd_set_depth_test_enable(
    command_buffer: vk::CommandBuffer,
    depth_test_enable: vk::Bool32,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetDepthTestEnable { depth_test_enable });
    }
}

pub unsafe fn cmd_set_depth_write_enable(
    command_buffer: vk::CommandBuffer,
    depth_write_enable: vk::Bool32,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetDepthWriteEnable { depth_write_enable });
    }
}

pub unsafe fn cmd_set_depth_compare_op(
    command_buffer: vk::CommandBuffer,
    depth_compare_op: vk::CompareOp,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetDepthCompareOp { depth_compare_op });
    }
}

pub unsafe fn cmd_set_depth_bias_enable(
    command_buffer: vk::CommandBuffer,
    depth_bias_enable: vk::Bool32,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetDepthBiasEnable { depth_bias_enable });
    }
}

pub unsafe fn cmd_set_stencil_test_enable(
    command_buffer: vk::CommandBuffer,
    stencil_test_enable: vk::Bool32,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetStencilTestEnable { stencil_test_enable });
    }
}

pub unsafe fn cmd_set_stencil_op(
    command_buffer: vk::CommandBuffer,
    face_mask: vk::StencilFaceFlags,
    fail_op: vk::StencilOp,
    pass_op: vk::StencilOp,
    depth_fail_op: vk::StencilOp,
    compare_op: vk::CompareOp,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetStencilOp {
            face_mask, fail_op, pass_op, depth_fail_op, compare_op,
        });
    }
}

pub unsafe fn cmd_set_depth_bounds(
    command_buffer: vk::CommandBuffer,
    min_depth_bounds: f32,
    max_depth_bounds: f32,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetDepthBounds { min_depth_bounds, max_depth_bounds });
    }
}

pub unsafe fn cmd_set_line_width(command_buffer: vk::CommandBuffer, line_width: f32) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetLineWidth { line_width });
    }
}

pub unsafe fn cmd_set_depth_bias(
    command_buffer: vk::CommandBuffer,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        d.commands.write().push(RecordedCommand::SetDepthBias {
            depth_bias_constant_factor, depth_bias_clamp, depth_bias_slope_factor,
        });
    }
}

// ─── RenderPass2 ─────────────────────────────────────────────────────────────

pub unsafe fn cmd_begin_render_pass2(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo,
    _p_subpass_begin_info: *const vk::SubpassBeginInfo,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let render_pass_begin = &*p_render_pass_begin;
    let clear_values = if render_pass_begin.clear_value_count > 0 {
        std::slice::from_raw_parts(
            render_pass_begin.p_clear_values,
            render_pass_begin.clear_value_count as usize,
        )
        .to_vec()
    } else {
        Vec::new()
    };
    debug!("Recording BeginRenderPass2");
    cmd_data.commands.write().push(RecordedCommand::BeginRenderPass2 {
        render_pass: render_pass_begin.render_pass,
        framebuffer: render_pass_begin.framebuffer,
        render_area: render_pass_begin.render_area,
        clear_values,
    });
}

pub unsafe fn cmd_next_subpass2(
    command_buffer: vk::CommandBuffer,
    _p_subpass_begin_info: *const vk::SubpassBeginInfo,
    _p_subpass_end_info: *const vk::SubpassEndInfo,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        debug!("Recording NextSubpass2 (no-op)");
        d.commands.write().push(RecordedCommand::NextSubpass2);
    }
}

pub unsafe fn cmd_end_render_pass2(
    command_buffer: vk::CommandBuffer,
    _p_subpass_end_info: *const vk::SubpassEndInfo,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        debug!("Recording EndRenderPass2");
        d.commands.write().push(RecordedCommand::EndRenderPass2);
    }
}

pub unsafe fn cmd_next_subpass(
    command_buffer: vk::CommandBuffer,
    _contents: vk::SubpassContents,
) {
    if let Some(d) = COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        debug!("Recording NextSubpass (no-op)");
        d.commands.write().push(RecordedCommand::NextSubpass);
    }
}

// ─── Secondary command buffers ────────────────────────────────────────────────

pub unsafe fn cmd_execute_commands(
    command_buffer: vk::CommandBuffer,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    let command_buffers = if command_buffer_count > 0 && !p_command_buffers.is_null() {
        std::slice::from_raw_parts(p_command_buffers, command_buffer_count as usize).to_vec()
    } else {
        Vec::new()
    };
    debug!("Recording ExecuteCommands: {} buffers", command_buffer_count);
    cmd_data.commands.write().push(RecordedCommand::ExecuteCommands { command_buffers });
}

// ─── Dispatch with base ───────────────────────────────────────────────────────

pub unsafe fn cmd_dispatch_base(
    command_buffer: vk::CommandBuffer,
    base_group_x: u32,
    base_group_y: u32,
    base_group_z: u32,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) {
    let cmd_data = match COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };
    debug!("Recording DispatchBase: base=({},{},{}), count=({},{},{})",
        base_group_x, base_group_y, base_group_z,
        group_count_x, group_count_y, group_count_z);
    cmd_data.commands.write().push(RecordedCommand::DispatchBase {
        base_group_x, base_group_y, base_group_z,
        group_count_x, group_count_y, group_count_z,
    });
}

/// Replay recorded commands to create a WebGPU command buffer
///
/// This function handles the complex lifetime management required to bridge Vulkan's
/// deferred recording model with WebGPU's scoped render passes. We use unsafe lifetime
/// extension to keep render/compute passes alive for the duration of their commands,
/// which is safe because we control the scope and ensure passes are dropped before
/// encoder.finish().
#[cfg(not(target_arch = "wasm32"))]
pub fn replay_commands(
    cmd_data: &VkCommandBufferData,
    backend: &WebGPUBackend,
) -> Result<CommandBuffer> {
    use crate::{buffer, descriptor, framebuffer, image, pipeline, render_pass};

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("VkWebGPU Command Encoder"),
        });

    let commands = cmd_data.commands.read();

    debug!("Replaying {} recorded commands", commands.len());

    // Active pass state - only one can be active at a time
    // We use Option<Box<_>> to allow taking ownership when we need to drop
    let mut active_render_pass: Option<Box<wgpu::RenderPass>> = None;
    let mut active_compute_pass: Option<Box<wgpu::ComputePass>> = None;

    // Keep Arc references alive for the duration of command replay
    // This ensures WebGPU resources aren't dropped while passes reference them
    let mut _resource_refs: Vec<Arc<dyn std::any::Any + Send + Sync>> = Vec::new();

    // Track the active compute pipeline for dispatch commands
    let mut active_compute_pipeline: Option<Arc<wgpu::ComputePipeline>> = None;

    // Track pending push constants - these need to be bound before the next draw/dispatch
    let mut pending_push_constant_offset: Option<u32> = None;

    // Push constant bind group (created once, used for all draws with dynamic offset)
    let mut push_constant_bind_group: Option<Arc<wgpu::BindGroup>> = None;

    // Dynamic state tracking
    let mut current_viewports: Vec<vk::Viewport> = Vec::new();
    let mut current_scissors: Vec<vk::Rect2D> = Vec::new();
    let mut current_blend_constants: Option<[f32; 4]> = None;
    let mut current_stencil_reference: Option<u32> = None;

    // Get device data for push constant buffer access
    let device_data = crate::device::get_device_data(cmd_data.device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    for command in commands.iter() {
        match command {
            RecordedCommand::BeginRenderPass {
                render_pass,
                framebuffer,
                render_area: _,
                clear_values,
            } => {
                debug!("Replay: BeginRenderPass");

                // Drop any active passes before starting a new one
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                // Get framebuffer data
                let fb_data = framebuffer::get_framebuffer_data(*framebuffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid framebuffer".to_string()))?;

                // Get render pass data for attachment info
                let rp_data = render_pass::get_render_pass_data(*render_pass)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid render pass".to_string()))?;

                // Build color attachments from framebuffer
                // We need to collect Arc<TextureView> references first, then build descriptors
                let mut view_arcs: Vec<Arc<wgpu::TextureView>> = Vec::new();
                let mut attachment_info: Vec<(bool, wgpu::Color, f32)> = Vec::new(); // (is_depth, clear_color, clear_depth)

                for (i, &image_view_handle) in fb_data.attachments.iter().enumerate() {
                    let view_data = image::get_image_view_data(image_view_handle)
                        .ok_or_else(|| VkError::InvalidHandle("Invalid image view".to_string()))?;

                    // Get the WebGPU texture view Arc and clone it
                    let wgpu_view_arc = {
                        let guard = view_data.wgpu_view.read();
                        guard
                            .as_ref()
                            .ok_or_else(|| {
                                VkError::InvalidHandle("Image view not bound".to_string())
                            })?
                            .clone()
                    };

                    // Determine if this is a color or depth/stencil attachment
                    let is_depth_stencil = if i < rp_data.attachments.len() {
                        let format = rp_data.attachments[i].format;
                        format == vk::Format::D16_UNORM
                            || format == vk::Format::D32_SFLOAT
                            || format == vk::Format::D24_UNORM_S8_UINT
                            || format == vk::Format::D32_SFLOAT_S8_UINT
                    } else {
                        false
                    };

                    let (clear_color, clear_depth) = if is_depth_stencil {
                        let depth = if i < clear_values.len() {
                            unsafe { clear_values[i].depth_stencil.depth }
                        } else {
                            1.0
                        };
                        (wgpu::Color::BLACK, depth)
                    } else {
                        let color = if i < clear_values.len() {
                            let cv = unsafe { clear_values[i].color };
                            wgpu::Color {
                                r: unsafe { cv.float32[0] } as f64,
                                g: unsafe { cv.float32[1] } as f64,
                                b: unsafe { cv.float32[2] } as f64,
                                a: unsafe { cv.float32[3] } as f64,
                            }
                        } else {
                            wgpu::Color::BLACK
                        };
                        (color, 1.0)
                    };

                    view_arcs.push(wgpu_view_arc);
                    attachment_info.push((is_depth_stencil, clear_color, clear_depth));
                }

                // Now build the attachment descriptors using unsafe lifetime extension
                // SAFETY: We keep view_arcs alive until the render pass is dropped
                let mut color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> =
                    Vec::new();
                let mut depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment> =
                    None;

                for (_i, (view_arc, (is_depth, clear_color, clear_depth))) in
                    view_arcs.iter().zip(attachment_info.iter()).enumerate()
                {
                    let view_ref: &'static wgpu::TextureView =
                        unsafe { std::mem::transmute(view_arc.as_ref()) };

                    if *is_depth {
                        depth_stencil_attachment = Some(wgpu::RenderPassDepthStencilAttachment {
                            view: view_ref,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(*clear_depth),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0),
                                store: wgpu::StoreOp::Store,
                            }),
                        });
                    } else {
                        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                            view: view_ref,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(*clear_color),
                                store: wgpu::StoreOp::Store,
                            },
                        }));
                    }
                }

                // Keep the Arc references alive for the duration of the render pass
                _resource_refs.extend(
                    view_arcs
                        .into_iter()
                        .map(|v| v as Arc<dyn std::any::Any + Send + Sync>),
                );

                // SAFETY: We extend the lifetime of the encoder reference to create a render pass
                // This is safe because:
                // 1. We control the scope - the render pass is dropped before encoder.finish()
                // 2. We store it in active_render_pass which is dropped before the function returns
                // 3. The encoder lives for the entire function scope
                let encoder_ptr = &mut encoder as *mut wgpu::CommandEncoder;
                let encoder_ref: &'static mut wgpu::CommandEncoder = unsafe { &mut *encoder_ptr };

                let render_pass = encoder_ref.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("VkRenderPass"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                active_render_pass = Some(Box::new(render_pass));
            }

            RecordedCommand::EndRenderPass => {
                debug!("Replay: EndRenderPass");
                // Drop the active render pass
                drop(active_render_pass.take());
            }

            RecordedCommand::BindPipeline {
                bind_point,
                pipeline,
            } => {
                debug!("Replay: BindPipeline");

                match *bind_point {
                    vk::PipelineBindPoint::GRAPHICS => {
                        if let Some(ref mut pass) = active_render_pass {
                            let pipeline_data =
                                pipeline::get_pipeline_data(*pipeline).ok_or_else(|| {
                                    VkError::InvalidHandle("Invalid pipeline".to_string())
                                })?;

                            match pipeline_data.as_ref() {
                                pipeline::VkPipelineData::Graphics { wgpu_pipeline, .. } => {
                                    let pipeline_arc = wgpu_pipeline.clone();
                                    // SAFETY: We extend the lifetime to static. This is safe because
                                    // we keep pipeline_arc alive in _resource_refs until the pass is dropped.
                                    let pipeline_ref: &'static wgpu::RenderPipeline =
                                        unsafe { std::mem::transmute(pipeline_arc.as_ref()) };
                                    _resource_refs
                                        .push(pipeline_arc as Arc<dyn std::any::Any + Send + Sync>);
                                    pass.set_pipeline(pipeline_ref);
                                }
                                _ => {
                                    return Err(VkError::InvalidHandle(
                                        "Graphics bind point requires graphics pipeline"
                                            .to_string(),
                                    ));
                                }
                            }
                        }
                    }
                    vk::PipelineBindPoint::COMPUTE => {
                        // Compute pass will be created on-demand in Dispatch
                        // Store the pipeline for later use
                        let pipeline_data =
                            pipeline::get_pipeline_data(*pipeline).ok_or_else(|| {
                                VkError::InvalidHandle("Invalid pipeline".to_string())
                            })?;

                        match pipeline_data.as_ref() {
                            pipeline::VkPipelineData::Compute { wgpu_pipeline, .. } => {
                                active_compute_pipeline = Some(wgpu_pipeline.clone());
                            }
                            _ => {
                                return Err(VkError::InvalidHandle(
                                    "Compute bind point requires compute pipeline".to_string(),
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }

            RecordedCommand::BindVertexBuffers {
                first_binding,
                buffers,
                offsets,
            } => {
                debug!("Replay: BindVertexBuffers");

                if let Some(ref mut pass) = active_render_pass {
                    for (i, (&buffer_handle, &offset)) in
                        buffers.iter().zip(offsets.iter()).enumerate()
                    {
                        let buffer_data = buffer::get_buffer_data(buffer_handle)
                            .ok_or_else(|| VkError::InvalidHandle("Invalid buffer".to_string()))?;

                        let wgpu_buffer_arc = {
                            let guard = buffer_data.wgpu_buffer.read();
                            guard
                                .as_ref()
                                .ok_or_else(|| {
                                    VkError::InvalidHandle("Buffer not bound".to_string())
                                })?
                                .clone()
                        };

                        // SAFETY: Extend lifetime - buffer is kept alive in _resource_refs
                        let buffer_ref: &'static wgpu::Buffer =
                            unsafe { std::mem::transmute(wgpu_buffer_arc.as_ref()) };

                        _resource_refs
                            .push(wgpu_buffer_arc as Arc<dyn std::any::Any + Send + Sync>);

                        let slot = first_binding + i as u32;
                        pass.set_vertex_buffer(slot, buffer_ref.slice(offset..));
                    }
                }
            }

            RecordedCommand::BindIndexBuffer {
                buffer,
                offset,
                index_type,
            } => {
                debug!("Replay: BindIndexBuffer");

                if let Some(ref mut pass) = active_render_pass {
                    let buffer_data = buffer::get_buffer_data(*buffer)
                        .ok_or_else(|| VkError::InvalidHandle("Invalid buffer".to_string()))?;

                    let wgpu_buffer_arc = {
                        let guard = buffer_data.wgpu_buffer.read();
                        guard
                            .as_ref()
                            .ok_or_else(|| VkError::InvalidHandle("Buffer not bound".to_string()))?
                            .clone()
                    };

                    // SAFETY: Extend lifetime - buffer is kept alive in _resource_refs
                    let buffer_ref: &'static wgpu::Buffer =
                        unsafe { std::mem::transmute(wgpu_buffer_arc.as_ref()) };

                    _resource_refs.push(wgpu_buffer_arc as Arc<dyn std::any::Any + Send + Sync>);

                    let format = match *index_type {
                        vk::IndexType::UINT16 => wgpu::IndexFormat::Uint16,
                        vk::IndexType::UINT32 => wgpu::IndexFormat::Uint32,
                        _ => wgpu::IndexFormat::Uint32,
                    };

                    pass.set_index_buffer(buffer_ref.slice(*offset..), format);
                }
            }

            RecordedCommand::BindDescriptorSets {
                bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            } => {
                debug!("Replay: BindDescriptorSets");

                // Check if this pipeline layout has push constants
                // If so, user descriptor sets are shifted by +1 (set 0 is reserved for push constants)
                let has_push_constants =
                    if let Some(layout_data) = pipeline::get_pipeline_layout_data(*layout) {
                        !layout_data.push_constant_ranges.is_empty()
                    } else {
                        false
                    };

                for (i, &desc_set_handle) in descriptor_sets.iter().enumerate() {
                    let desc_data = descriptor::get_descriptor_set_data(desc_set_handle)
                        .ok_or_else(|| {
                            VkError::InvalidHandle("Invalid descriptor set".to_string())
                        })?;

                    let wgpu_bind_group_arc = {
                        let guard = desc_data.wgpu_bind_group.read();
                        guard
                            .as_ref()
                            .ok_or_else(|| {
                                VkError::InvalidHandle("Descriptor set not updated".to_string())
                            })?
                            .clone()
                    };

                    // SAFETY: Extend lifetime - bind group is kept alive in _resource_refs
                    let bind_group_ref: &'static wgpu::BindGroup =
                        unsafe { std::mem::transmute(wgpu_bind_group_arc.as_ref()) };

                    _resource_refs
                        .push(wgpu_bind_group_arc as Arc<dyn std::any::Any + Send + Sync>);

                    // Calculate actual set index
                    // If pipeline has push constants, shift user sets by +1
                    let set_index = if has_push_constants {
                        first_set + i as u32 + 1
                    } else {
                        first_set + i as u32
                    };

                    // Extract dynamic offsets for this set (if any)
                    // FIXME: This is a simplified implementation that only handles single descriptor sets correctly.
                    // For multiple descriptor sets with dynamic offsets, proper offset distribution requires
                    // tracking the number of dynamic descriptors per set from the pipeline layout.
                    let offsets_slice = if !dynamic_offsets.is_empty() {
                        if descriptor_sets.len() == 1 {
                            // Single descriptor set: pass all dynamic offsets to it
                            dynamic_offsets.as_slice()
                        } else {
                            // Multiple descriptor sets: cannot distribute offsets correctly without
                            // tracking dynamic descriptor counts per set from pipeline layout.
                            // TODO: Implement proper offset distribution by tracking dynamic counts.
                            if i == 0 {
                                debug!("Warning: Multiple descriptor sets with dynamic offsets not fully supported yet");
                            }
                            &[]
                        }
                    } else {
                        &[]
                    };

                    match *bind_point {
                        vk::PipelineBindPoint::GRAPHICS => {
                            if let Some(ref mut pass) = active_render_pass {
                                pass.set_bind_group(set_index, bind_group_ref, offsets_slice);
                            }
                        }
                        vk::PipelineBindPoint::COMPUTE => {
                            if let Some(ref mut pass) = active_compute_pass {
                                pass.set_bind_group(set_index, bind_group_ref, offsets_slice);
                            }
                        }
                        _ => {}
                    }
                }
            }

            RecordedCommand::PushConstants {
                layout: _,
                stage_flags: _,
                offset,
                size,
                data,
            } => {
                debug!("Replay: PushConstants(offset={}, size={})", offset, size);

                // Write push constant data to the ring buffer
                let buffer_offset = device_data.push_constant_buffer.push(&backend.queue, data);

                // Create bind group if not already created
                if push_constant_bind_group.is_none() {
                    let bind_group = device_data
                        .push_constant_buffer
                        .create_bind_group(&backend.device);
                    push_constant_bind_group = Some(Arc::new(bind_group));
                }

                // Store the offset for binding with the next draw/dispatch
                pending_push_constant_offset = Some(buffer_offset);

                debug!(
                    "Push constants written to ring buffer at offset {}",
                    buffer_offset
                );
            }

            RecordedCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                debug!(
                    "Replay: Draw({} vertices, {} instances)",
                    vertex_count, instance_count
                );

                if let Some(ref mut pass) = active_render_pass {
                    // Apply dynamic state before draw
                    if !current_viewports.is_empty() {
                        let vp = &current_viewports[0];
                        pass.set_viewport(
                            vp.x,
                            vp.y,
                            vp.width,
                            vp.height,
                            vp.min_depth,
                            vp.max_depth,
                        );
                    }

                    if !current_scissors.is_empty() {
                        let sc = &current_scissors[0];
                        pass.set_scissor_rect(
                            sc.offset.x as u32,
                            sc.offset.y as u32,
                            sc.extent.width,
                            sc.extent.height,
                        );
                    }

                    if let Some(bc) = current_blend_constants {
                        pass.set_blend_constant(wgpu::Color {
                            r: bc[0] as f64,
                            g: bc[1] as f64,
                            b: bc[2] as f64,
                            a: bc[3] as f64,
                        });
                    }

                    if let Some(stencil_ref) = current_stencil_reference {
                        pass.set_stencil_reference(stencil_ref);
                    }

                    // Bind push constants if pending
                    if let (Some(ref bg), Some(offset)) =
                        (&push_constant_bind_group, pending_push_constant_offset)
                    {
                        // SAFETY: Extend lifetime - bind group is kept alive in _resource_refs
                        let bind_group_ref: &'static wgpu::BindGroup =
                            unsafe { std::mem::transmute(bg.as_ref()) };
                        _resource_refs.push(bg.clone() as Arc<dyn std::any::Any + Send + Sync>);

                        // Bind at set 0 with dynamic offset
                        pass.set_bind_group(0, bind_group_ref, &[offset]);
                        debug!(
                            "Bound push constants at set 0 with dynamic offset {}",
                            offset
                        );
                    }

                    pass.draw(
                        *first_vertex..*first_vertex + *vertex_count,
                        *first_instance..*first_instance + *instance_count,
                    );
                }
            }

            RecordedCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            } => {
                debug!(
                    "Replay: DrawIndexed({} indices, {} instances)",
                    index_count, instance_count
                );

                if let Some(ref mut pass) = active_render_pass {
                    // Apply dynamic state before draw
                    if !current_viewports.is_empty() {
                        let vp = &current_viewports[0];
                        pass.set_viewport(
                            vp.x,
                            vp.y,
                            vp.width,
                            vp.height,
                            vp.min_depth,
                            vp.max_depth,
                        );
                    }

                    if !current_scissors.is_empty() {
                        let sc = &current_scissors[0];
                        pass.set_scissor_rect(
                            sc.offset.x as u32,
                            sc.offset.y as u32,
                            sc.extent.width,
                            sc.extent.height,
                        );
                    }

                    if let Some(bc) = current_blend_constants {
                        pass.set_blend_constant(wgpu::Color {
                            r: bc[0] as f64,
                            g: bc[1] as f64,
                            b: bc[2] as f64,
                            a: bc[3] as f64,
                        });
                    }

                    if let Some(stencil_ref) = current_stencil_reference {
                        pass.set_stencil_reference(stencil_ref);
                    }

                    // Bind push constants if pending
                    if let (Some(ref bg), Some(offset)) =
                        (&push_constant_bind_group, pending_push_constant_offset)
                    {
                        // SAFETY: Extend lifetime - bind group is kept alive in _resource_refs
                        let bind_group_ref: &'static wgpu::BindGroup =
                            unsafe { std::mem::transmute(bg.as_ref()) };
                        _resource_refs.push(bg.clone() as Arc<dyn std::any::Any + Send + Sync>);

                        // Bind at set 0 with dynamic offset
                        pass.set_bind_group(0, bind_group_ref, &[offset]);
                        debug!(
                            "Bound push constants at set 0 with dynamic offset {}",
                            offset
                        );
                    }

                    pass.draw_indexed(
                        *first_index..*first_index + *index_count,
                        *vertex_offset,
                        *first_instance..*first_instance + *instance_count,
                    );
                }
            }

            RecordedCommand::Dispatch {
                group_count_x,
                group_count_y,
                group_count_z,
            } => {
                debug!(
                    "Replay: Dispatch({}x{}x{})",
                    group_count_x, group_count_y, group_count_z
                );

                // Drop any active render pass before compute
                drop(active_render_pass.take());

                // Create compute pass if needed
                if active_compute_pass.is_none() {
                    // SAFETY: Same lifetime extension as render pass
                    let encoder_ptr = &mut encoder as *mut wgpu::CommandEncoder;
                    let encoder_ref: &'static mut wgpu::CommandEncoder =
                        unsafe { &mut *encoder_ptr };

                    let compute_pass =
                        encoder_ref.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("VkComputePass"),
                            timestamp_writes: None,
                        });

                    active_compute_pass = Some(Box::new(compute_pass));
                }

                if let Some(ref mut pass) = active_compute_pass {
                    // Bind the compute pipeline if one is active
                    if let Some(ref pipeline_arc) = active_compute_pipeline {
                        // SAFETY: We extend the lifetime to static. This is safe because
                        // we keep pipeline_arc alive in _resource_refs until the pass is dropped.
                        let pipeline_ref: &'static wgpu::ComputePipeline =
                            unsafe { std::mem::transmute(pipeline_arc.as_ref()) };
                        _resource_refs
                            .push(pipeline_arc.clone() as Arc<dyn std::any::Any + Send + Sync>);
                        pass.set_pipeline(pipeline_ref);
                    }

                    // Bind push constants if pending
                    if let (Some(ref bg), Some(offset)) =
                        (&push_constant_bind_group, pending_push_constant_offset)
                    {
                        // SAFETY: Extend lifetime - bind group is kept alive in _resource_refs
                        let bind_group_ref: &'static wgpu::BindGroup =
                            unsafe { std::mem::transmute(bg.as_ref()) };
                        _resource_refs.push(bg.clone() as Arc<dyn std::any::Any + Send + Sync>);

                        // Bind at set 0 with dynamic offset
                        pass.set_bind_group(0, bind_group_ref, &[offset]);
                        debug!(
                            "Bound push constants at set 0 with dynamic offset {}",
                            offset
                        );
                    }

                    pass.dispatch_workgroups(*group_count_x, *group_count_y, *group_count_z);
                }

                // End compute pass after dispatch
                drop(active_compute_pass.take());
            }

            RecordedCommand::CopyBuffer {
                src_buffer,
                dst_buffer,
                regions,
            } => {
                debug!("Replay: CopyBuffer");

                // Drop any active passes before copy operations
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = buffer::get_buffer_data(*src_buffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source buffer".to_string()))?;
                let dst_data = buffer::get_buffer_data(*dst_buffer).ok_or_else(|| {
                    VkError::InvalidHandle("Invalid destination buffer".to_string())
                })?;

                let src_wgpu_guard = src_data.wgpu_buffer.read();
                let dst_wgpu_guard = dst_data.wgpu_buffer.read();

                let src_wgpu = src_wgpu_guard
                    .as_ref()
                    .ok_or_else(|| VkError::InvalidHandle("Source buffer not bound".to_string()))?;
                let dst_wgpu = dst_wgpu_guard.as_ref().ok_or_else(|| {
                    VkError::InvalidHandle("Destination buffer not bound".to_string())
                })?;

                for region in regions {
                    encoder.copy_buffer_to_buffer(
                        src_wgpu.as_ref(),
                        region.src_offset,
                        dst_wgpu.as_ref(),
                        region.dst_offset,
                        region.size,
                    );
                }
            }

            RecordedCommand::CopyBufferToImage {
                src_buffer,
                dst_image,
                dst_image_layout: _,
                regions,
            } => {
                debug!("Replay: CopyBufferToImage");
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = buffer::get_buffer_data(*src_buffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source buffer".to_string()))?;
                let dst_data = image::get_image_data(*dst_image).ok_or_else(|| {
                    VkError::InvalidHandle("Invalid destination image".to_string())
                })?;

                let dst_wgpu_guard = dst_data.wgpu_texture.read();
                let dst_wgpu = dst_wgpu_guard.as_ref().ok_or_else(|| {
                    VkError::InvalidHandle("Destination image not bound".to_string())
                })?;

                for region in regions {
                    let bytes_per_pixel = crate::format::format_size(dst_data.format)
                        .ok_or_else(|| VkError::FormatNotSupported)?;

                    // buffer_row_length==0 means tightly packed (use image width).
                    let row_px = if region.buffer_row_length == 0 {
                        region.image_extent.width
                    } else {
                        region.buffer_row_length
                    };
                    let actual_bpr = row_px * bytes_per_pixel;
                    // wgpu requires bytes_per_row to be a multiple of 256.
                    let aligned_bpr = (actual_bpr + 255) & !255;

                    let img_height = if region.buffer_image_height == 0 {
                        region.image_extent.height
                    } else {
                        region.buffer_image_height
                    };
                    let depth = region.image_extent.depth.max(1);

                    let dst_tex = wgpu::ImageCopyTexture {
                        texture: dst_wgpu.as_ref(),
                        mip_level: region.image_subresource.mip_level,
                        origin: wgpu::Origin3d {
                            x: region.image_offset.x as u32,
                            y: region.image_offset.y as u32,
                            z: region.image_offset.z as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    };
                    let copy_size = wgpu::Extent3d {
                        width: region.image_extent.width,
                        height: region.image_extent.height,
                        depth_or_array_layers: depth,
                    };

                    if aligned_bpr == actual_bpr {
                        // Already 256-aligned; use the source wgpu buffer directly.
                        let src_guard = src_data.wgpu_buffer.read();
                        let src_wgpu = src_guard.as_ref().ok_or_else(|| {
                            VkError::InvalidHandle("Source buffer not bound".to_string())
                        })?;
                        encoder.copy_buffer_to_texture(
                            wgpu::ImageCopyBuffer {
                                buffer: src_wgpu.as_ref(),
                                layout: wgpu::ImageDataLayout {
                                    offset: region.buffer_offset,
                                    bytes_per_row: Some(aligned_bpr),
                                    rows_per_image: Some(img_height),
                                },
                            },
                            dst_tex,
                            copy_size,
                        );
                    } else {
                        // Row pitch is not 256-aligned: re-pack from CPU-side memory.
                        let total_rows = (img_height * depth) as usize;
                        let aligned_total = aligned_bpr as usize * total_rows;
                        let copy_bpr = region.image_extent.width * bytes_per_pixel;
                        let mut packed = vec![0u8; aligned_total];

                        let mem_guard = src_data.memory.read();
                        if let Some(mem_h) = mem_guard.as_ref() {
                            if let Some(mem_data) = crate::memory::get_memory_data(*mem_h) {
                                let cpu = mem_data.data.read();
                                let base = *src_data.memory_offset.read() as usize
                                    + region.buffer_offset as usize;
                                for row in 0..total_rows {
                                    let src_off = base + row * actual_bpr as usize;
                                    let dst_off = row * aligned_bpr as usize;
                                    let len = copy_bpr as usize;
                                    if src_off + len <= cpu.len() {
                                        packed[dst_off..dst_off + len]
                                            .copy_from_slice(&cpu[src_off..src_off + len]);
                                    }
                                }
                            }
                        }

                        let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("CopyBufferToImage_aligned"),
                            size: aligned_total as u64,
                            usage: wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        });
                        backend.queue.write_buffer(&staging, 0, &packed);
                        let staging_arc = Arc::new(staging);
                        // SAFETY: staging_arc is stored in _resource_refs, keeping
                        // it alive until after queue.submit().
                        let staging_ref: &'static wgpu::Buffer =
                            unsafe { std::mem::transmute(staging_arc.as_ref()) };
                        _resource_refs
                            .push(staging_arc as Arc<dyn std::any::Any + Send + Sync>);

                        encoder.copy_buffer_to_texture(
                            wgpu::ImageCopyBuffer {
                                buffer: staging_ref,
                                layout: wgpu::ImageDataLayout {
                                    offset: 0,
                                    bytes_per_row: Some(aligned_bpr),
                                    rows_per_image: Some(img_height),
                                },
                            },
                            dst_tex,
                            copy_size,
                        );
                        debug!(
                            "CopyBufferToImage: re-packed {}→{} bytes/row for {}x{} texture",
                            actual_bpr, aligned_bpr,
                            region.image_extent.width, region.image_extent.height
                        );
                    }
                }
            }

            RecordedCommand::SetViewport {
                first_viewport,
                viewports,
            } => {
                debug!("Replay: SetViewport");
                // Store viewport state for application before next draw
                if *first_viewport == 0 {
                    current_viewports = viewports.clone();
                } else {
                    // Extend viewports array if needed
                    while current_viewports.len() < *first_viewport as usize {
                        current_viewports.push(vk::Viewport::default());
                    }
                    for (i, vp) in viewports.iter().enumerate() {
                        let idx = *first_viewport as usize + i;
                        if idx >= current_viewports.len() {
                            current_viewports.push(*vp);
                        } else {
                            current_viewports[idx] = *vp;
                        }
                    }
                }
            }

            RecordedCommand::SetScissor {
                first_scissor,
                scissors,
            } => {
                debug!("Replay: SetScissor");
                // Store scissor state for application before next draw
                if *first_scissor == 0 {
                    current_scissors = scissors.clone();
                } else {
                    // Extend scissors array if needed
                    while current_scissors.len() < *first_scissor as usize {
                        current_scissors.push(vk::Rect2D::default());
                    }
                    for (i, sc) in scissors.iter().enumerate() {
                        let idx = *first_scissor as usize + i;
                        if idx >= current_scissors.len() {
                            current_scissors.push(*sc);
                        } else {
                            current_scissors[idx] = *sc;
                        }
                    }
                }
            }

            RecordedCommand::SetBlendConstants { blend_constants } => {
                debug!("Replay: SetBlendConstants");
                current_blend_constants = Some(*blend_constants);
            }

            RecordedCommand::SetStencilReference {
                face_mask: _,
                reference,
            } => {
                debug!("Replay: SetStencilReference");
                current_stencil_reference = Some(*reference);
            }

            RecordedCommand::ClearColorImage {
                image,
                image_layout: _,
                color,
                ranges,
            } => {
                debug!("Replay: ClearColorImage");

                // Drop any active passes
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let img_data = match image::get_image_data(*image) {
                    Some(data) => data,
                    None => {
                        debug!("Warning: Invalid image in ClearColorImage");
                        continue;
                    }
                };

                let texture_guard = img_data.wgpu_texture.read();
                let texture = match texture_guard.as_ref() {
                    Some(tex) => tex,
                    None => {
                        debug!("Warning: Image not bound in ClearColorImage");
                        continue;
                    }
                };

                // Use clear_texture for each range
                for range in ranges {
                    for mip in range.base_mip_level..range.base_mip_level + range.level_count {
                        for layer in
                            range.base_array_layer..range.base_array_layer + range.layer_count
                        {
                            // Extract color values
                            let clear_color = unsafe { color.float32 };
                            let clear_value = wgpu::Color {
                                r: clear_color[0] as f64,
                                g: clear_color[1] as f64,
                                b: clear_color[2] as f64,
                                a: clear_color[3] as f64,
                            };

                            // Create temporary texture view for this mip/layer
                            let view = texture.as_ref().create_view(&wgpu::TextureViewDescriptor {
                                label: Some("Clear temp view"),
                                format: None,
                                dimension: None,
                                aspect: wgpu::TextureAspect::All,
                                base_mip_level: mip,
                                mip_level_count: Some(1),
                                base_array_layer: layer,
                                array_layer_count: Some(1),
                            });

                            // Use a render pass with LoadOp::Clear
                            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Clear color image"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(clear_value),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            drop(pass); // End pass immediately
                        }
                    }
                }
            }

            RecordedCommand::ClearDepthStencilImage {
                image,
                image_layout: _,
                depth_stencil,
                ranges,
            } => {
                debug!("Replay: ClearDepthStencilImage");

                // Drop any active passes
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let img_data = match image::get_image_data(*image) {
                    Some(data) => data,
                    None => {
                        debug!("Warning: Invalid image in ClearDepthStencilImage");
                        continue;
                    }
                };

                let texture_guard = img_data.wgpu_texture.read();
                let texture = match texture_guard.as_ref() {
                    Some(tex) => tex,
                    None => {
                        debug!("Warning: Image not bound in ClearDepthStencilImage");
                        continue;
                    }
                };

                // Use clear_texture for each range
                for range in ranges {
                    for mip in range.base_mip_level..range.base_mip_level + range.level_count {
                        for layer in
                            range.base_array_layer..range.base_array_layer + range.layer_count
                        {
                            let view = texture.as_ref().create_view(&wgpu::TextureViewDescriptor {
                                label: Some("Clear temp view"),
                                format: None,
                                dimension: None,
                                aspect: wgpu::TextureAspect::All,
                                base_mip_level: mip,
                                mip_level_count: Some(1),
                                base_array_layer: layer,
                                array_layer_count: Some(1),
                            });

                            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Clear depth stencil image"),
                                color_attachments: &[],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: &view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(depth_stencil.depth),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(depth_stencil.stencil),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            drop(pass);
                        }
                    }
                }
            }

            RecordedCommand::ClearAttachments {
                attachments: _,
                rects: _,
            } => {
                debug!("Replay: ClearAttachments (not supported - use LoadOp::Clear instead)");
                // WebGPU doesn't support clearing attachments inside a render pass
                // This should be handled by LoadOp::Clear in BeginRenderPass
                // Log warning and skip
            }

            RecordedCommand::CopyImageToBuffer {
                src_image,
                src_image_layout: _,
                dst_buffer,
                regions,
            } => {
                debug!("Replay: CopyImageToBuffer");

                // Drop any active passes
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = image::get_image_data(*src_image)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source image".to_string()))?;
                let dst_data = buffer::get_buffer_data(*dst_buffer).ok_or_else(|| {
                    VkError::InvalidHandle("Invalid destination buffer".to_string())
                })?;

                let src_wgpu_guard = src_data.wgpu_texture.read();
                let dst_wgpu_guard = dst_data.wgpu_buffer.read();

                let src_wgpu = src_wgpu_guard
                    .as_ref()
                    .ok_or_else(|| VkError::InvalidHandle("Source image not bound".to_string()))?;
                let dst_wgpu = dst_wgpu_guard.as_ref().ok_or_else(|| {
                    VkError::InvalidHandle("Destination buffer not bound".to_string())
                })?;

                for region in regions {
                    let bytes_per_pixel = crate::format::format_size(src_data.format)
                        .ok_or_else(|| VkError::FormatNotSupported)?;

                    // buffer_row_length==0 means tightly packed (use image width).
                    let row_px = if region.buffer_row_length == 0 {
                        region.image_extent.width
                    } else {
                        region.buffer_row_length
                    };
                    let actual_bpr = row_px * bytes_per_pixel;
                    // wgpu requires bytes_per_row to be a multiple of 256.
                    let aligned_bpr = (actual_bpr + 255) & !255;

                    let img_height = if region.buffer_image_height == 0 {
                        region.image_extent.height
                    } else {
                        region.buffer_image_height
                    };

                    if aligned_bpr != actual_bpr {
                        debug!(
                            "CopyImageToBuffer: actual bpr {} not 256-aligned, using {} (output layout has alignment padding)",
                            actual_bpr, aligned_bpr
                        );
                    }

                    encoder.copy_texture_to_buffer(
                        wgpu::ImageCopyTexture {
                            texture: src_wgpu.as_ref(),
                            mip_level: region.image_subresource.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.image_offset.x as u32,
                                y: region.image_offset.y as u32,
                                z: region.image_offset.z as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyBuffer {
                            buffer: dst_wgpu.as_ref(),
                            layout: wgpu::ImageDataLayout {
                                offset: region.buffer_offset,
                                bytes_per_row: Some(aligned_bpr),
                                rows_per_image: Some(img_height),
                            },
                        },
                        wgpu::Extent3d {
                            width: region.image_extent.width,
                            height: region.image_extent.height,
                            depth_or_array_layers: region.image_extent.depth.max(1),
                        },
                    );
                }
            }

            RecordedCommand::CopyImage {
                src_image,
                src_image_layout: _,
                dst_image,
                dst_image_layout: _,
                regions,
            } => {
                debug!("Replay: CopyImage");

                // Drop any active passes
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = image::get_image_data(*src_image)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source image".to_string()))?;
                let dst_data = image::get_image_data(*dst_image).ok_or_else(|| {
                    VkError::InvalidHandle("Invalid destination image".to_string())
                })?;

                let src_wgpu_guard = src_data.wgpu_texture.read();
                let dst_wgpu_guard = dst_data.wgpu_texture.read();

                let src_wgpu = src_wgpu_guard
                    .as_ref()
                    .ok_or_else(|| VkError::InvalidHandle("Source image not bound".to_string()))?;
                let dst_wgpu = dst_wgpu_guard.as_ref().ok_or_else(|| {
                    VkError::InvalidHandle("Destination image not bound".to_string())
                })?;

                for region in regions {
                    encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture {
                            texture: src_wgpu.as_ref(),
                            mip_level: region.src_subresource.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.src_offset.x as u32,
                                y: region.src_offset.y as u32,
                                z: region.src_offset.z as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyTexture {
                            texture: dst_wgpu.as_ref(),
                            mip_level: region.dst_subresource.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.dst_offset.x as u32,
                                y: region.dst_offset.y as u32,
                                z: region.dst_offset.z as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: region.extent.width,
                            height: region.extent.height,
                            depth_or_array_layers: region.extent.depth,
                        },
                    );
                }
            }

            RecordedCommand::BlitImage {
                src_image: _,
                src_image_layout: _,
                dst_image: _,
                dst_image_layout: _,
                regions: _,
                filter: _,
            } => {
                debug!("Replay: BlitImage (not fully implemented - would require compute shader)");
                // WebGPU doesn't have direct blit support
                // This would need to be implemented via a compute shader or render pass
                // For now, log and skip - most uses are covered by CopyImage
            }

            RecordedCommand::PipelineBarrier { .. } => {
                debug!("Replay: PipelineBarrier (no-op)");
                // WebGPU handles synchronization implicitly
            }

            RecordedCommand::PipelineBarrier2 => {
                debug!("Replay: PipelineBarrier2 (no-op)");
            }

            RecordedCommand::BeginRendering {
                render_area: _,
                layer_count: _,
                color_attachments,
                depth_attachment,
                stencil_attachment,
            } => {
                debug!(
                    "Replay: BeginRendering: {} color attachments",
                    color_attachments.len()
                );

                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let mut color_view_arcs: Vec<Arc<wgpu::TextureView>> = Vec::new();
                let mut color_attach_info: Vec<(wgpu::Color, vk::AttachmentLoadOp, vk::AttachmentStoreOp)> = Vec::new();

                for att in color_attachments.iter() {
                    let view_data = image::get_image_view_data(att.image_view)
                        .ok_or_else(|| VkError::InvalidHandle("Invalid color attachment image view".to_string()))?;
                    let wgpu_view_arc = {
                        let guard = view_data.wgpu_view.read();
                        guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Color attachment not bound".to_string()))?.clone()
                    };
                    let clear_color = unsafe { att.clear_value.color.float32 };
                    let color = wgpu::Color {
                        r: clear_color[0] as f64,
                        g: clear_color[1] as f64,
                        b: clear_color[2] as f64,
                        a: clear_color[3] as f64,
                    };
                    color_view_arcs.push(wgpu_view_arc);
                    color_attach_info.push((color, att.load_op, att.store_op));
                }

                let mut depth_view_arc: Option<Arc<wgpu::TextureView>> = None;
                let mut depth_att_info: Option<(f32, vk::AttachmentLoadOp, vk::AttachmentStoreOp)> = None;
                if let Some(att) = depth_attachment {
                    let view_data = image::get_image_view_data(att.image_view)
                        .ok_or_else(|| VkError::InvalidHandle("Invalid depth attachment image view".to_string()))?;
                    let wgpu_view_arc = {
                        let guard = view_data.wgpu_view.read();
                        guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Depth attachment not bound".to_string()))?.clone()
                    };
                    let clear_depth = unsafe { att.clear_value.depth_stencil.depth };
                    depth_att_info = Some((clear_depth, att.load_op, att.store_op));
                    depth_view_arc = Some(wgpu_view_arc);
                }

                // Build color attachment descriptors with lifetime extension
                let mut wgpu_color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = Vec::new();
                for (i, (view_arc, (clear_color, load_op, store_op))) in
                    color_view_arcs.iter().zip(color_attach_info.iter()).enumerate()
                {
                    let _ = i;
                    let view_ref: &'static wgpu::TextureView =
                        unsafe { std::mem::transmute(view_arc.as_ref()) };
                    let wgpu_load = match *load_op {
                        vk::AttachmentLoadOp::CLEAR => wgpu::LoadOp::Clear(*clear_color),
                        vk::AttachmentLoadOp::DONT_CARE => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        _ => wgpu::LoadOp::Load,
                    };
                    let wgpu_store = match *store_op {
                        vk::AttachmentStoreOp::DONT_CARE => wgpu::StoreOp::Discard,
                        _ => wgpu::StoreOp::Store,
                    };
                    wgpu_color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                        view: view_ref,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu_load, store: wgpu_store },
                    }));
                }
                _resource_refs.extend(color_view_arcs.into_iter().map(|v| v as Arc<dyn std::any::Any + Send + Sync>));

                let mut wgpu_depth_stencil: Option<wgpu::RenderPassDepthStencilAttachment> = None;
                if let (Some(view_arc), Some((clear_depth, load_op, store_op))) = (depth_view_arc, depth_att_info) {
                    let view_ref: &'static wgpu::TextureView =
                        unsafe { std::mem::transmute(view_arc.as_ref()) };
                    let wgpu_load = match load_op {
                        vk::AttachmentLoadOp::CLEAR => wgpu::LoadOp::Clear(clear_depth),
                        _ => wgpu::LoadOp::Load,
                    };
                    let wgpu_store = match store_op {
                        vk::AttachmentStoreOp::DONT_CARE => wgpu::StoreOp::Discard,
                        _ => wgpu::StoreOp::Store,
                    };
                    wgpu_depth_stencil = Some(wgpu::RenderPassDepthStencilAttachment {
                        view: view_ref,
                        depth_ops: Some(wgpu::Operations { load: wgpu_load, store: wgpu_store }),
                        stencil_ops: None,
                    });
                    _resource_refs.push(view_arc as Arc<dyn std::any::Any + Send + Sync>);
                }

                // Ignore stencil_attachment for now (combined with depth in most cases)
                let _ = stencil_attachment;

                let encoder_ptr = &mut encoder as *mut wgpu::CommandEncoder;
                let encoder_ref: &'static mut wgpu::CommandEncoder = unsafe { &mut *encoder_ptr };
                let render_pass = encoder_ref.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("VkDynamicRenderPass"),
                    color_attachments: &wgpu_color_attachments,
                    depth_stencil_attachment: wgpu_depth_stencil,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                active_render_pass = Some(Box::new(render_pass));
            }

            RecordedCommand::EndRendering => {
                debug!("Replay: EndRendering");
                drop(active_render_pass.take());
            }

            RecordedCommand::DrawIndirect {
                buffer,
                offset,
                draw_count,
                stride,
            } => {
                debug!("Replay: DrawIndirect(count={})", draw_count);
                if let Some(ref mut pass) = active_render_pass {
                    let buffer_data = buffer::get_buffer_data(*buffer)
                        .ok_or_else(|| VkError::InvalidHandle("Invalid indirect buffer".to_string()))?;
                    let wgpu_buffer_arc = {
                        let guard = buffer_data.wgpu_buffer.read();
                        guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Indirect buffer not bound".to_string()))?.clone()
                    };
                    let buffer_ref: &'static wgpu::Buffer =
                        unsafe { std::mem::transmute(wgpu_buffer_arc.as_ref()) };
                    _resource_refs.push(wgpu_buffer_arc as Arc<dyn std::any::Any + Send + Sync>);

                    // Apply dynamic state
                    if !current_viewports.is_empty() {
                        let vp = &current_viewports[0];
                        pass.set_viewport(vp.x, vp.y, vp.width, vp.height, vp.min_depth, vp.max_depth);
                    }
                    if !current_scissors.is_empty() {
                        let sc = &current_scissors[0];
                        pass.set_scissor_rect(sc.offset.x as u32, sc.offset.y as u32, sc.extent.width, sc.extent.height);
                    }

                    for i in 0..*draw_count {
                        let draw_offset = offset + (i * stride) as vk::DeviceSize;
                        pass.draw_indirect(buffer_ref, draw_offset);
                    }
                }
            }

            RecordedCommand::DrawIndexedIndirect {
                buffer,
                offset,
                draw_count,
                stride,
            } => {
                debug!("Replay: DrawIndexedIndirect(count={})", draw_count);
                if let Some(ref mut pass) = active_render_pass {
                    let buffer_data = buffer::get_buffer_data(*buffer)
                        .ok_or_else(|| VkError::InvalidHandle("Invalid indirect buffer".to_string()))?;
                    let wgpu_buffer_arc = {
                        let guard = buffer_data.wgpu_buffer.read();
                        guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Indirect buffer not bound".to_string()))?.clone()
                    };
                    let buffer_ref: &'static wgpu::Buffer =
                        unsafe { std::mem::transmute(wgpu_buffer_arc.as_ref()) };
                    _resource_refs.push(wgpu_buffer_arc as Arc<dyn std::any::Any + Send + Sync>);

                    // Apply dynamic state
                    if !current_viewports.is_empty() {
                        let vp = &current_viewports[0];
                        pass.set_viewport(vp.x, vp.y, vp.width, vp.height, vp.min_depth, vp.max_depth);
                    }
                    if !current_scissors.is_empty() {
                        let sc = &current_scissors[0];
                        pass.set_scissor_rect(sc.offset.x as u32, sc.offset.y as u32, sc.extent.width, sc.extent.height);
                    }

                    for i in 0..*draw_count {
                        let draw_offset = offset + (i * stride) as vk::DeviceSize;
                        pass.draw_indexed_indirect(buffer_ref, draw_offset);
                    }
                }
            }

            RecordedCommand::FillBuffer {
                dst_buffer,
                dst_offset,
                size,
                data,
            } => {
                debug!("Replay: FillBuffer(offset={}, size={}, data={:#x})", dst_offset, size, data);
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                if let Some(buf_data) = buffer::get_buffer_data(*dst_buffer) {
                    let wgpu_guard = buf_data.wgpu_buffer.read();
                    if let Some(wgpu_buf) = wgpu_guard.as_ref() {
                        let actual_size = if *size == vk::WHOLE_SIZE {
                            buf_data.size - dst_offset
                        } else {
                            *size
                        };
                        // Build a fill pattern as repeated u32 bytes
                        let data_bytes = data.to_ne_bytes();
                        let byte_count = actual_size as usize;
                        let mut fill_data = Vec::with_capacity(byte_count);
                        for i in 0..byte_count {
                            fill_data.push(data_bytes[i % 4]);
                        }
                        backend.queue.write_buffer(wgpu_buf.as_ref(), *dst_offset, &fill_data);
                    }
                }
            }

            RecordedCommand::UpdateBuffer {
                dst_buffer,
                dst_offset,
                data,
            } => {
                debug!("Replay: UpdateBuffer(offset={}, size={})", dst_offset, data.len());
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                if let Some(buf_data) = buffer::get_buffer_data(*dst_buffer) {
                    let wgpu_guard = buf_data.wgpu_buffer.read();
                    if let Some(wgpu_buf) = wgpu_guard.as_ref() {
                        backend.queue.write_buffer(wgpu_buf.as_ref(), *dst_offset, data);
                    }
                }
            }

            RecordedCommand::CopyBuffer2 { src_buffer, dst_buffer, regions } => {
                debug!("Replay: CopyBuffer2");
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = buffer::get_buffer_data(*src_buffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source buffer".to_string()))?;
                let dst_data = buffer::get_buffer_data(*dst_buffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid destination buffer".to_string()))?;
                let src_guard = src_data.wgpu_buffer.read();
                let dst_guard = dst_data.wgpu_buffer.read();
                let src_wgpu = src_guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Source buffer not bound".to_string()))?;
                let dst_wgpu = dst_guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Dest buffer not bound".to_string()))?;
                for region in regions {
                    encoder.copy_buffer_to_buffer(src_wgpu.as_ref(), region.src_offset, dst_wgpu.as_ref(), region.dst_offset, region.size);
                }
            }

            RecordedCommand::CopyImage2 { src_image, src_image_layout: _, dst_image, dst_image_layout: _, regions } => {
                debug!("Replay: CopyImage2");
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = image::get_image_data(*src_image)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source image".to_string()))?;
                let dst_data = image::get_image_data(*dst_image)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid dest image".to_string()))?;
                let src_guard = src_data.wgpu_texture.read();
                let dst_guard = dst_data.wgpu_texture.read();
                let src_wgpu = src_guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Source image not bound".to_string()))?;
                let dst_wgpu = dst_guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Dest image not bound".to_string()))?;
                for region in regions {
                    encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture {
                            texture: src_wgpu.as_ref(),
                            mip_level: region.src_subresource.mip_level,
                            origin: wgpu::Origin3d { x: region.src_offset.x as u32, y: region.src_offset.y as u32, z: region.src_offset.z as u32 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyTexture {
                            texture: dst_wgpu.as_ref(),
                            mip_level: region.dst_subresource.mip_level,
                            origin: wgpu::Origin3d { x: region.dst_offset.x as u32, y: region.dst_offset.y as u32, z: region.dst_offset.z as u32 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d { width: region.extent.width, height: region.extent.height, depth_or_array_layers: region.extent.depth },
                    );
                }
            }

            RecordedCommand::CopyBufferToImage2 { src_buffer, dst_image, dst_image_layout: _, regions } => {
                debug!("Replay: CopyBufferToImage2");
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = buffer::get_buffer_data(*src_buffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source buffer".to_string()))?;
                let dst_data = image::get_image_data(*dst_image)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid dest image".to_string()))?;
                let dst_guard = dst_data.wgpu_texture.read();
                let dst_wgpu = dst_guard.as_ref()
                    .ok_or_else(|| VkError::InvalidHandle("Dest image not bound".to_string()))?;

                for region in regions {
                    let bytes_per_pixel = crate::format::format_size(dst_data.format)
                        .ok_or_else(|| VkError::FormatNotSupported)?;

                    // buffer_row_length==0 means tightly packed (use image width).
                    let row_px = if region.buffer_row_length == 0 {
                        region.image_extent.width
                    } else {
                        region.buffer_row_length
                    };
                    let actual_bpr = row_px * bytes_per_pixel;
                    // wgpu requires bytes_per_row to be a multiple of 256.
                    let aligned_bpr = (actual_bpr + 255) & !255;

                    let img_height = if region.buffer_image_height == 0 {
                        region.image_extent.height
                    } else {
                        region.buffer_image_height
                    };
                    let depth = region.image_extent.depth.max(1);

                    let dst_tex = wgpu::ImageCopyTexture {
                        texture: dst_wgpu.as_ref(),
                        mip_level: region.image_subresource.mip_level,
                        origin: wgpu::Origin3d {
                            x: region.image_offset.x as u32,
                            y: region.image_offset.y as u32,
                            z: region.image_offset.z as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    };
                    let copy_size = wgpu::Extent3d {
                        width: region.image_extent.width,
                        height: region.image_extent.height,
                        depth_or_array_layers: depth,
                    };

                    if aligned_bpr == actual_bpr {
                        // Already 256-aligned; use the source wgpu buffer directly.
                        let src_guard = src_data.wgpu_buffer.read();
                        let src_wgpu = src_guard.as_ref()
                            .ok_or_else(|| VkError::InvalidHandle("Source buffer not bound".to_string()))?;
                        encoder.copy_buffer_to_texture(
                            wgpu::ImageCopyBuffer {
                                buffer: src_wgpu.as_ref(),
                                layout: wgpu::ImageDataLayout {
                                    offset: region.buffer_offset,
                                    bytes_per_row: Some(aligned_bpr),
                                    rows_per_image: Some(img_height),
                                },
                            },
                            dst_tex,
                            copy_size,
                        );
                    } else {
                        // Row pitch not 256-aligned: re-pack from CPU-side memory.
                        let total_rows = (img_height * depth) as usize;
                        let aligned_total = aligned_bpr as usize * total_rows;
                        let copy_bpr = region.image_extent.width * bytes_per_pixel;
                        let mut packed = vec![0u8; aligned_total];

                        let mem_guard = src_data.memory.read();
                        if let Some(mem_h) = mem_guard.as_ref() {
                            if let Some(mem_data) = crate::memory::get_memory_data(*mem_h) {
                                let cpu = mem_data.data.read();
                                let base = *src_data.memory_offset.read() as usize
                                    + region.buffer_offset as usize;
                                for row in 0..total_rows {
                                    let src_off = base + row * actual_bpr as usize;
                                    let dst_off = row * aligned_bpr as usize;
                                    let len = copy_bpr as usize;
                                    if src_off + len <= cpu.len() {
                                        packed[dst_off..dst_off + len]
                                            .copy_from_slice(&cpu[src_off..src_off + len]);
                                    }
                                }
                            }
                        }

                        let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("CopyBufferToImage2_aligned"),
                            size: aligned_total as u64,
                            usage: wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        });
                        backend.queue.write_buffer(&staging, 0, &packed);
                        let staging_arc = Arc::new(staging);
                        // SAFETY: staging_arc is stored in _resource_refs, keeping
                        // it alive until after queue.submit().
                        let staging_ref: &'static wgpu::Buffer =
                            unsafe { std::mem::transmute(staging_arc.as_ref()) };
                        _resource_refs
                            .push(staging_arc as Arc<dyn std::any::Any + Send + Sync>);

                        encoder.copy_buffer_to_texture(
                            wgpu::ImageCopyBuffer {
                                buffer: staging_ref,
                                layout: wgpu::ImageDataLayout {
                                    offset: 0,
                                    bytes_per_row: Some(aligned_bpr),
                                    rows_per_image: Some(img_height),
                                },
                            },
                            dst_tex,
                            copy_size,
                        );
                        debug!(
                            "CopyBufferToImage2: re-packed {}→{} bytes/row for {}x{} texture",
                            actual_bpr, aligned_bpr,
                            region.image_extent.width, region.image_extent.height
                        );
                    }
                }
            }

            RecordedCommand::CopyImageToBuffer2 { src_image, src_image_layout: _, dst_buffer, regions } => {
                debug!("Replay: CopyImageToBuffer2");
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = image::get_image_data(*src_image)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source image".to_string()))?;
                let dst_data = buffer::get_buffer_data(*dst_buffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid dest buffer".to_string()))?;
                let src_guard = src_data.wgpu_texture.read();
                let dst_guard = dst_data.wgpu_buffer.read();
                let src_wgpu = src_guard.as_ref()
                    .ok_or_else(|| VkError::InvalidHandle("Source image not bound".to_string()))?;
                let dst_wgpu = dst_guard.as_ref()
                    .ok_or_else(|| VkError::InvalidHandle("Dest buffer not bound".to_string()))?;

                for region in regions {
                    let bytes_per_pixel = crate::format::format_size(src_data.format)
                        .ok_or_else(|| VkError::FormatNotSupported)?;

                    // buffer_row_length==0 means tightly packed (use image width).
                    let row_px = if region.buffer_row_length == 0 {
                        region.image_extent.width
                    } else {
                        region.buffer_row_length
                    };
                    let actual_bpr = row_px * bytes_per_pixel;
                    // wgpu requires bytes_per_row to be a multiple of 256.
                    let aligned_bpr = (actual_bpr + 255) & !255;

                    let img_height = if region.buffer_image_height == 0 {
                        region.image_extent.height
                    } else {
                        region.buffer_image_height
                    };

                    if aligned_bpr != actual_bpr {
                        debug!(
                            "CopyImageToBuffer2: actual bpr {} not 256-aligned, using {} (output layout has alignment padding)",
                            actual_bpr, aligned_bpr
                        );
                    }

                    encoder.copy_texture_to_buffer(
                        wgpu::ImageCopyTexture {
                            texture: src_wgpu.as_ref(),
                            mip_level: region.image_subresource.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.image_offset.x as u32,
                                y: region.image_offset.y as u32,
                                z: region.image_offset.z as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyBuffer {
                            buffer: dst_wgpu.as_ref(),
                            layout: wgpu::ImageDataLayout {
                                offset: region.buffer_offset,
                                bytes_per_row: Some(aligned_bpr),
                                rows_per_image: Some(img_height),
                            },
                        },
                        wgpu::Extent3d {
                            width: region.image_extent.width,
                            height: region.image_extent.height,
                            depth_or_array_layers: region.image_extent.depth.max(1),
                        },
                    );
                }
            }

            RecordedCommand::BlitImage2 { .. } => {
                debug!("Replay: BlitImage2 (no-op - use compute shader for real blit)");
            }

            RecordedCommand::ResolveImage { .. } => {
                debug!("Replay: ResolveImage (no-op - WebGPU handles MSAA resolve automatically)");
            }

            // Extended dynamic state - no-op (WebGPU bakes these into pipeline)
            RecordedCommand::SetCullMode { cull_mode } => {
                debug!("Replay: SetCullMode({:?}) (no-op)", cull_mode);
            }
            RecordedCommand::SetFrontFace { front_face } => {
                debug!("Replay: SetFrontFace({:?}) (no-op)", front_face);
            }
            RecordedCommand::SetPrimitiveTopology { primitive_topology } => {
                debug!("Replay: SetPrimitiveTopology({:?}) (no-op)", primitive_topology);
            }
            RecordedCommand::SetDepthTestEnable { depth_test_enable } => {
                debug!("Replay: SetDepthTestEnable({}) (no-op)", depth_test_enable);
            }
            RecordedCommand::SetDepthWriteEnable { depth_write_enable } => {
                debug!("Replay: SetDepthWriteEnable({}) (no-op)", depth_write_enable);
            }
            RecordedCommand::SetDepthCompareOp { depth_compare_op } => {
                debug!("Replay: SetDepthCompareOp({:?}) (no-op)", depth_compare_op);
            }
            RecordedCommand::SetDepthBiasEnable { depth_bias_enable } => {
                debug!("Replay: SetDepthBiasEnable({}) (no-op)", depth_bias_enable);
            }
            RecordedCommand::SetStencilTestEnable { stencil_test_enable } => {
                debug!("Replay: SetStencilTestEnable({}) (no-op)", stencil_test_enable);
            }
            RecordedCommand::SetStencilOp { .. } => {
                debug!("Replay: SetStencilOp (no-op)");
            }
            RecordedCommand::SetDepthBounds { .. } => {
                debug!("Replay: SetDepthBounds (no-op)");
            }
            RecordedCommand::SetLineWidth { .. } => {
                debug!("Replay: SetLineWidth (no-op)");
            }
            RecordedCommand::SetDepthBias { .. } => {
                debug!("Replay: SetDepthBias (no-op)");
            }

            RecordedCommand::BeginRenderPass2 {
                render_pass,
                framebuffer,
                render_area: _,
                clear_values,
            } => {
                debug!("Replay: BeginRenderPass2 (same as BeginRenderPass)");
                // Delegate to same logic as BeginRenderPass by recording equivalent command
                // We need to replay inline. Treat identically to BeginRenderPass.
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let fb_data = framebuffer::get_framebuffer_data(*framebuffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid framebuffer".to_string()))?;
                let rp_data = render_pass::get_render_pass_data(*render_pass)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid render pass".to_string()))?;

                let mut view_arcs: Vec<Arc<wgpu::TextureView>> = Vec::new();
                let mut attachment_info: Vec<(bool, wgpu::Color, f32)> = Vec::new();

                for (i, &image_view_handle) in fb_data.attachments.iter().enumerate() {
                    let view_data = image::get_image_view_data(image_view_handle)
                        .ok_or_else(|| VkError::InvalidHandle("Invalid image view".to_string()))?;
                    let wgpu_view_arc = {
                        let guard = view_data.wgpu_view.read();
                        guard.as_ref().ok_or_else(|| VkError::InvalidHandle("Image view not bound".to_string()))?.clone()
                    };
                    let is_depth_stencil = if i < rp_data.attachments.len() {
                        let fmt = rp_data.attachments[i].format;
                        fmt == vk::Format::D16_UNORM || fmt == vk::Format::D32_SFLOAT
                            || fmt == vk::Format::D24_UNORM_S8_UINT || fmt == vk::Format::D32_SFLOAT_S8_UINT
                    } else { false };
                    let (clear_color, clear_depth) = if is_depth_stencil {
                        let depth = if i < clear_values.len() { unsafe { clear_values[i].depth_stencil.depth } } else { 1.0 };
                        (wgpu::Color::BLACK, depth)
                    } else {
                        let color = if i < clear_values.len() {
                            let cv = unsafe { clear_values[i].color };
                            wgpu::Color { r: unsafe { cv.float32[0] } as f64, g: unsafe { cv.float32[1] } as f64, b: unsafe { cv.float32[2] } as f64, a: unsafe { cv.float32[3] } as f64 }
                        } else { wgpu::Color::BLACK };
                        (color, 1.0)
                    };
                    view_arcs.push(wgpu_view_arc);
                    attachment_info.push((is_depth_stencil, clear_color, clear_depth));
                }

                let mut color_attachments2: Vec<Option<wgpu::RenderPassColorAttachment>> = Vec::new();
                let mut depth_stencil_attachment2: Option<wgpu::RenderPassDepthStencilAttachment> = None;
                for (view_arc, (is_depth, clear_color, clear_depth)) in view_arcs.iter().zip(attachment_info.iter()) {
                    let view_ref: &'static wgpu::TextureView = unsafe { std::mem::transmute(view_arc.as_ref()) };
                    if *is_depth {
                        depth_stencil_attachment2 = Some(wgpu::RenderPassDepthStencilAttachment {
                            view: view_ref,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(*clear_depth), store: wgpu::StoreOp::Store }),
                            stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                        });
                    } else {
                        color_attachments2.push(Some(wgpu::RenderPassColorAttachment {
                            view: view_ref,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(*clear_color), store: wgpu::StoreOp::Store },
                        }));
                    }
                }
                _resource_refs.extend(view_arcs.into_iter().map(|v| v as Arc<dyn std::any::Any + Send + Sync>));

                let encoder_ptr = &mut encoder as *mut wgpu::CommandEncoder;
                let encoder_ref: &'static mut wgpu::CommandEncoder = unsafe { &mut *encoder_ptr };
                let render_pass = encoder_ref.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("VkRenderPass2"),
                    color_attachments: &color_attachments2,
                    depth_stencil_attachment: depth_stencil_attachment2,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                active_render_pass = Some(Box::new(render_pass));
            }

            RecordedCommand::NextSubpass | RecordedCommand::NextSubpass2 => {
                debug!("Replay: NextSubpass (no-op)");
            }

            RecordedCommand::EndRenderPass2 => {
                debug!("Replay: EndRenderPass2");
                drop(active_render_pass.take());
            }

            RecordedCommand::ExecuteCommands { command_buffers } => {
                debug!("Replay: ExecuteCommands({} buffers)", command_buffers.len());
                // Drop active pass before executing secondary commands
                drop(active_render_pass.take());
                drop(active_compute_pass.take());
                // For secondary command buffers, replay their commands inline
                for &secondary_cb in command_buffers {
                    if let Some(secondary_data) = COMMAND_BUFFER_ALLOCATOR.get(secondary_cb.as_raw()) {
                        // We can't directly replay here since replay_commands takes ownership of encoder
                        // The secondary commands will be submitted separately via queue submit
                        debug!("Warning: ExecuteCommands with secondary buffer - secondary buffers should be pre-recorded");
                        let _ = secondary_data;
                    }
                }
            }

            RecordedCommand::DispatchBase {
                base_group_x: _,
                base_group_y: _,
                base_group_z: _,
                group_count_x,
                group_count_y,
                group_count_z,
            } => {
                debug!("Replay: DispatchBase({}x{}x{})", group_count_x, group_count_y, group_count_z);
                // WebGPU doesn't support base group offsets, dispatch with count only
                drop(active_render_pass.take());

                if active_compute_pass.is_none() {
                    let encoder_ptr = &mut encoder as *mut wgpu::CommandEncoder;
                    let encoder_ref: &'static mut wgpu::CommandEncoder = unsafe { &mut *encoder_ptr };
                    let compute_pass = encoder_ref.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("VkComputePassBase"),
                        timestamp_writes: None,
                    });
                    active_compute_pass = Some(Box::new(compute_pass));
                }

                if let Some(ref mut pass) = active_compute_pass {
                    if let Some(ref pipeline_arc) = active_compute_pipeline {
                        let pipeline_ref: &'static wgpu::ComputePipeline =
                            unsafe { std::mem::transmute(pipeline_arc.as_ref()) };
                        _resource_refs.push(pipeline_arc.clone() as Arc<dyn std::any::Any + Send + Sync>);
                        pass.set_pipeline(pipeline_ref);
                    }
                    pass.dispatch_workgroups(*group_count_x, *group_count_y, *group_count_z);
                }
                drop(active_compute_pass.take());
            }
        }
    }

    // Ensure all passes are dropped before finishing
    drop(active_render_pass);
    drop(active_compute_pass);

    // Finish and return the command buffer
    Ok(encoder.finish())
}

#[cfg(target_arch = "wasm32")]
pub fn replay_commands(
    _cmd_data: &VkCommandBufferData,
    _backend: &WebGPUBackend,
) -> Result<CommandBuffer> {
    // WASM implementation would go here
    Err(VkError::FeatureNotSupported(
        "WASM command replay not yet implemented".to_string(),
    ))
}
