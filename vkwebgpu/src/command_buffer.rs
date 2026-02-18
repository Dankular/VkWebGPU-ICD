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

pub fn get_command_buffer_data(
    command_buffer: vk::CommandBuffer,
) -> Option<Arc<VkCommandBufferData>> {
    COMMAND_BUFFER_ALLOCATOR.get(command_buffer.as_raw())
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

                // Drop any active passes
                drop(active_render_pass.take());
                drop(active_compute_pass.take());

                let src_data = buffer::get_buffer_data(*src_buffer)
                    .ok_or_else(|| VkError::InvalidHandle("Invalid source buffer".to_string()))?;
                let dst_data = image::get_image_data(*dst_image).ok_or_else(|| {
                    VkError::InvalidHandle("Invalid destination image".to_string())
                })?;

                let src_wgpu_guard = src_data.wgpu_buffer.read();
                let dst_wgpu_guard = dst_data.wgpu_texture.read();

                let src_wgpu = src_wgpu_guard
                    .as_ref()
                    .ok_or_else(|| VkError::InvalidHandle("Source buffer not bound".to_string()))?;
                let dst_wgpu = dst_wgpu_guard.as_ref().ok_or_else(|| {
                    VkError::InvalidHandle("Destination image not bound".to_string())
                })?;

                for region in regions {
                    // Calculate proper bytes_per_row from format
                    let bytes_per_pixel = crate::format::format_size(dst_data.format)
                        .ok_or_else(|| VkError::FormatNotSupported)?;
                    let bytes_per_row = region.image_extent.width * bytes_per_pixel;

                    encoder.copy_buffer_to_texture(
                        wgpu::ImageCopyBuffer {
                            buffer: src_wgpu.as_ref(),
                            layout: wgpu::ImageDataLayout {
                                offset: region.buffer_offset,
                                bytes_per_row: Some(bytes_per_row),
                                rows_per_image: Some(region.image_extent.height),
                            },
                        },
                        wgpu::ImageCopyTexture {
                            texture: dst_wgpu.as_ref(),
                            mip_level: region.image_subresource.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.image_offset.x as u32,
                                y: region.image_offset.y as u32,
                                z: region.image_offset.z as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: region.image_extent.width,
                            height: region.image_extent.height,
                            depth_or_array_layers: region.image_extent.depth,
                        },
                    );
                }
            }

            RecordedCommand::PipelineBarrier { .. } => {
                debug!("Replay: PipelineBarrier (no-op)");
                // WebGPU handles synchronization implicitly
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
