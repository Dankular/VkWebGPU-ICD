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

pub fn get_command_buffer_data(
    command_buffer: vk::CommandBuffer,
) -> Option<Arc<VkCommandBufferData>> {
    COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw())
}

/// Replay recorded commands to create a WebGPU command buffer
/// TODO: This is a simplified stub - needs full implementation with proper lifetime management
#[cfg(not(target_arch = "wasm32"))]
pub fn replay_commands(
    cmd_data: &VkCommandBufferData,
    backend: &WebGPUBackend,
) -> Result<CommandBuffer> {
    let encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("VkWebGPU Command Encoder"),
        });

    let commands = cmd_data.commands.read();

    debug!(
        "Replaying {} recorded commands (stub implementation)",
        commands.len()
    );

    // TODO: Implement full command replay with proper WebGPU resource lifetime management
    // The challenge is that WebGPU RenderPass/ComputePass lifetimes conflict with Vulkan's model
    // Need to carefully manage Arc references and ensure WebGPU resources outlive pass encoders

    // Stub implementation - just log commands for now
    for command in commands.iter() {
        match command {
            RecordedCommand::BeginRenderPass { .. } => debug!("Replay: BeginRenderPass"),
            RecordedCommand::EndRenderPass => debug!("Replay: EndRenderPass"),
            RecordedCommand::BindPipeline { .. } => debug!("Replay: BindPipeline"),
            RecordedCommand::BindVertexBuffers { .. } => debug!("Replay: BindVertexBuffers"),
            RecordedCommand::BindIndexBuffer { .. } => debug!("Replay: BindIndexBuffer"),
            RecordedCommand::BindDescriptorSets { .. } => debug!("Replay: BindDescriptorSets"),
            RecordedCommand::Draw { .. } => debug!("Replay: Draw"),
            RecordedCommand::DrawIndexed { .. } => debug!("Replay: DrawIndexed"),
            RecordedCommand::Dispatch { .. } => debug!("Replay: Dispatch"),
            RecordedCommand::CopyBuffer { .. } => debug!("Replay: CopyBuffer"),
            RecordedCommand::CopyBufferToImage { .. } => debug!("Replay: CopyBufferToImage"),
            RecordedCommand::PipelineBarrier { .. } => debug!("Replay: PipelineBarrier"),
        }
    }

    // Return an empty command buffer for now
    Ok(encoder.finish())
}

#[cfg(target_arch = "wasm32")]
pub fn replay_commands(
    _cmd_data: &VkCommandBufferData,
    _backend: &WebGPUBackend,
) -> Result<CommandBuffer> {
    // WASM implementation would go here
    Err(VkError::NotImplemented(
        "WASM command replay not yet implemented".to_string(),
    ))
}
