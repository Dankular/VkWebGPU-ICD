//! Vulkan Command Buffer implementation
//! Maps VkCommandBuffer to WebGPU GPUCommandEncoder

use ash::vk;
use log::debug;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::command_pool;
use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static COMMAND_BUFFER_ALLOCATOR: HandleAllocator<VkCommandBufferData> = HandleAllocator::new();

pub struct VkCommandBufferData {
    pub device: vk::Device,
    pub command_pool: vk::CommandPool,
    pub level: vk::CommandBufferLevel,
    pub state: RwLock<CommandBufferState>,
    #[cfg(not(target_arch = "wasm32"))]
    pub encoder: RwLock<Option<wgpu::CommandEncoder>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub finished_buffers: RwLock<Vec<wgpu::CommandBuffer>>,
}

pub enum CommandBufferState {
    Initial,
    Recording,
    Executable,
    Invalid,
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
            #[cfg(not(target_arch = "wasm32"))]
            encoder: RwLock::new(None),
            #[cfg(not(target_arch = "wasm32"))]
            finished_buffers: RwLock::new(Vec::new()),
        };

        let cmd_handle = COMMAND_BUFFER_ALLOCATOR.allocate(cmd_data);
        *cmd_buffer = vk::CommandBuffer::from_raw(cmd_handle);

        pool_data.allocated_buffers.write().push(*cmd_buffer);
    }

    Ok(())
}

pub unsafe fn begin_command_buffer(
    command_buffer: vk::CommandBuffer,
    p_begin_info: *const vk::CommandBufferBeginInfo,
) -> Result<()> {
    let cmd_data = COMMAND_BUFFER_ALLOCATOR
        .get(command_buffer.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid command buffer".to_string()))?;

    let device_data = device::get_device_data(cmd_data.device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    debug!("Beginning command buffer recording");

    #[cfg(not(target_arch = "wasm32"))]
    {
        let encoder =
            device_data
                .backend
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("VkCommandBuffer"),
                });

        *cmd_data.encoder.write() = Some(encoder);
    }

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
    debug!("Begin render pass");
    // Simplified for now
}

pub unsafe fn cmd_end_render_pass(command_buffer: vk::CommandBuffer) {
    debug!("End render pass");
}

pub unsafe fn cmd_bind_pipeline(
    command_buffer: vk::CommandBuffer,
    _pipeline_bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    debug!("Bind pipeline: {:?}", pipeline);
}

pub unsafe fn cmd_bind_descriptor_sets(
    command_buffer: vk::CommandBuffer,
    _pipeline_bind_point: vk::PipelineBindPoint,
    _layout: vk::PipelineLayout,
    _first_set: u32,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
    _dynamic_offset_count: u32,
    _p_dynamic_offsets: *const u32,
) {
    debug!("Bind {} descriptor sets", descriptor_set_count);
}

pub unsafe fn cmd_draw(
    command_buffer: vk::CommandBuffer,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    debug!(
        "Draw: vertices={}, instances={}, first_vertex={}, first_instance={}",
        vertex_count, instance_count, first_vertex, first_instance
    );
}

pub unsafe fn cmd_draw_indexed(
    command_buffer: vk::CommandBuffer,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
) {
    debug!(
        "Draw indexed: indices={}, instances={}, first_index={}, vertex_offset={}, first_instance={}",
        index_count, instance_count, first_index, vertex_offset, first_instance
    );
}

pub unsafe fn cmd_bind_vertex_buffers(
    command_buffer: vk::CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_buffers: *const vk::Buffer,
    p_offsets: *const vk::DeviceSize,
) {
    debug!(
        "Bind {} vertex buffers starting at binding {}",
        binding_count, first_binding
    );
}

pub unsafe fn cmd_bind_index_buffer(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
) {
    debug!(
        "Bind index buffer: offset={}, type={:?}",
        offset, index_type
    );
}

pub unsafe fn cmd_copy_buffer(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    region_count: u32,
    p_regions: *const vk::BufferCopy,
) {
    debug!("Copy buffer: {} regions", region_count);
}

pub unsafe fn cmd_copy_buffer_to_image(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    _dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::BufferImageCopy,
) {
    debug!("Copy buffer to image: {} regions", region_count);
}

pub unsafe fn cmd_pipeline_barrier(
    command_buffer: vk::CommandBuffer,
    _src_stage_mask: vk::PipelineStageFlags,
    _dst_stage_mask: vk::PipelineStageFlags,
    _dependency_flags: vk::DependencyFlags,
    memory_barrier_count: u32,
    _p_memory_barriers: *const vk::MemoryBarrier,
    buffer_memory_barrier_count: u32,
    _p_buffer_memory_barriers: *const vk::BufferMemoryBarrier,
    image_memory_barrier_count: u32,
    _p_image_memory_barriers: *const vk::ImageMemoryBarrier,
) {
    debug!(
        "Pipeline barrier: mem={}, buf={}, img={}",
        memory_barrier_count, buffer_memory_barrier_count, image_memory_barrier_count
    );
    // WebGPU handles barriers implicitly
}

impl VkCommandBufferData {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn finish(&self) -> Result<Vec<wgpu::CommandBuffer>> {
        let mut encoder = self.encoder.write();
        if let Some(enc) = encoder.take() {
            let buffer = enc.finish();
            Ok(vec![buffer])
        } else {
            Ok(Vec::new())
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn finish(&self) -> Result<Vec<crate::backend::CommandBuffer>> {
        Ok(Vec::new())
    }
}

pub fn get_command_buffer_data(
    command_buffer: vk::CommandBuffer,
) -> Option<Arc<VkCommandBufferData>> {
    COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw())
}
