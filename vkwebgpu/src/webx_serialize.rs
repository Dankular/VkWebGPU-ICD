//! WebX frame command serializer
//!
//! Converts a `Vec<RecordedCommand>` into the binary payload sent inside a
//! `WEBX_CMD_QUEUE_SUBMIT` (0x00B0) IPC packet.
//!
//! ## Format
//!
//! The payload of WEBX_CMD_QUEUE_SUBMIT is:
//!
//!   [cmd_count: u32 LE]
//!   For each command:
//!     [opcode: u32 LE][payload_len: u32 LE][payload: payload_len bytes]
//!
//! All integers are little-endian.  Vulkan handles are u64.
//! Fixed-size Vulkan structs (vk::BufferCopy, vk::Viewport, etc.) are serialised
//! as their raw `#[repr(C)]` bytes — safe because ash guarantees this layout.
//!
//! ## Opcodes
//!
//! Inner command opcodes mirror the WEBX_CMD_CMD_* values from protocol/commands.h.
//! Variants that have no dedicated outer-protocol equivalent use new opcodes defined here.

use ash::vk;

use crate::command_buffer::{RecordedCommand, VkCommandBufferData};
use crate::error::Result;
use crate::webx_ipc::WebXIpc;

// ── Outer WEBX command IDs (from protocol/commands.h) ────────────────────────
const WEBX_CMD_QUEUE_SUBMIT: u32 = 0x00B0;

// ── Inner frame-command opcodes (embedded in QUEUE_SUBMIT payload) ────────────
// These match WEBX_CMD_CMD_* in protocol/commands.h.  New opcodes (0x00A5+)
// are also added to commands.h so the host JS side uses the same constants.
const FC_BEGIN_RENDER_PASS:           u32 = 0x0080;
const FC_END_RENDER_PASS:             u32 = 0x0081;
const FC_NEXT_SUBPASS:                u32 = 0x0082;
const FC_BEGIN_RENDERING:             u32 = 0x0083;
const FC_END_RENDERING:               u32 = 0x0084;
const FC_BIND_PIPELINE:               u32 = 0x0085;
const FC_BIND_VERTEX_BUFFERS:         u32 = 0x0086;
const FC_BIND_INDEX_BUFFER:           u32 = 0x0087;
const FC_BIND_DESCRIPTOR_SETS:        u32 = 0x0088;
const FC_PUSH_CONSTANTS:              u32 = 0x0089;
const FC_DRAW:                        u32 = 0x008A;
const FC_DRAW_INDEXED:                u32 = 0x008B;
const FC_DRAW_INDIRECT:               u32 = 0x008C;
const FC_DRAW_INDEXED_INDIRECT:       u32 = 0x008D;
const FC_DISPATCH:                    u32 = 0x008E;
const FC_SET_VIEWPORT:                u32 = 0x0090;
const FC_SET_SCISSOR:                 u32 = 0x0091;
const FC_SET_BLEND_CONSTANTS:         u32 = 0x0092;
const FC_SET_STENCIL_REFERENCE:       u32 = 0x0093;
const FC_SET_DEPTH_BIAS:              u32 = 0x0094;
const FC_SET_LINE_WIDTH:              u32 = 0x0095;
const FC_SET_CULL_MODE:               u32 = 0x0096;
const FC_SET_FRONT_FACE:              u32 = 0x0097;
const FC_SET_PRIMITIVE_TOPOLOGY:      u32 = 0x0098;
const FC_PIPELINE_BARRIER:            u32 = 0x0099; // also PipelineBarrier2
const FC_COPY_BUFFER:                 u32 = 0x009A; // also CopyBuffer2
const FC_COPY_BUFFER_TO_IMAGE:        u32 = 0x009B; // also CopyBufferToImage2
const FC_COPY_IMAGE_TO_BUFFER:        u32 = 0x009C; // also CopyImageToBuffer2
const FC_COPY_IMAGE:                  u32 = 0x009D; // also CopyImage2
const FC_BLIT_IMAGE:                  u32 = 0x009E; // also BlitImage2
const FC_CLEAR_COLOR_IMAGE:           u32 = 0x009F;
const FC_CLEAR_DEPTH_STENCIL_IMAGE:   u32 = 0x00A0;
const FC_CLEAR_ATTACHMENTS:           u32 = 0x00A1;
const FC_FILL_BUFFER:                 u32 = 0x00A2;
const FC_UPDATE_BUFFER:               u32 = 0x00A3;
const FC_EXECUTE_COMMANDS:            u32 = 0x00A4;
// New opcodes (also added to protocol/commands.h)
const FC_RESOLVE_IMAGE:               u32 = 0x00A5;
const FC_SET_DEPTH_TEST_ENABLE:       u32 = 0x00A6;
const FC_SET_DEPTH_WRITE_ENABLE:      u32 = 0x00A7;
const FC_SET_DEPTH_COMPARE_OP:        u32 = 0x00A8;
const FC_SET_DEPTH_BIAS_ENABLE:       u32 = 0x00A9;
const FC_SET_STENCIL_TEST_ENABLE:     u32 = 0x00AA;
const FC_SET_STENCIL_OP:              u32 = 0x00AB;
const FC_SET_DEPTH_BOUNDS:            u32 = 0x00AC;
const FC_DISPATCH_BASE:               u32 = 0x00AD;

// ── Public entry point ────────────────────────────────────────────────────────

/// Serialize the recorded commands from `cmd_data` and send them to the host as a
/// single `WEBX_CMD_QUEUE_SUBMIT` IPC packet.  This is the core "one packet per
/// frame" optimisation — replaces the old per-Vulkan-call IPC model.
pub fn submit_frame(cmd_data: &VkCommandBufferData) -> Result<()> {
    let commands = cmd_data.commands.read();
    let payload = serialize_commands(&commands);
    drop(commands);

    // Send the bulk packet; ignore the (empty) response data.
    WebXIpc::global().send_cmd(WEBX_CMD_QUEUE_SUBMIT, &payload)?;
    Ok(())
}

// ── Serializer core ───────────────────────────────────────────────────────────

fn serialize_commands(commands: &[RecordedCommand]) -> Vec<u8> {
    let mut out = Vec::with_capacity(commands.len() * 32);
    write_u32(&mut out, commands.len() as u32);
    for cmd in commands {
        serialize_one(cmd, &mut out);
    }
    out
}

/// Write one command as [opcode: u32][payload_len: u32][payload...].
fn serialize_one(cmd: &RecordedCommand, out: &mut Vec<u8>) {
    use crate::command_buffer::RenderingAttachment;

    // Helper: write the framed command atomically.
    let mut frame = |opcode: u32, payload: &[u8]| {
        write_u32(out, opcode);
        write_u32(out, payload.len() as u32);
        out.extend_from_slice(payload);
    };

    match cmd {
        RecordedCommand::BeginRenderPass { render_pass, framebuffer, render_area, clear_values } |
        RecordedCommand::BeginRenderPass2 { render_pass, framebuffer, render_area, clear_values } => {
            let opcode = FC_BEGIN_RENDER_PASS;
            let mut p = Vec::new();
            write_u64(&mut p, render_pass.as_raw());
            write_u64(&mut p, framebuffer.as_raw());
            p.extend_from_slice(unsafe { struct_bytes(render_area) }); // 16 bytes
            write_u32(&mut p, clear_values.len() as u32);
            for cv in clear_values {
                p.extend_from_slice(unsafe { struct_bytes(cv) }); // 16 bytes each (union)
            }
            frame(opcode, &p);
        }

        RecordedCommand::EndRenderPass |
        RecordedCommand::EndRenderPass2 => frame(FC_END_RENDER_PASS, &[]),

        RecordedCommand::NextSubpass |
        RecordedCommand::NextSubpass2 => frame(FC_NEXT_SUBPASS, &[]),

        RecordedCommand::BindPipeline { bind_point, pipeline } => {
            let mut p = Vec::new();
            write_u32(&mut p, bind_point.as_raw() as u32);
            write_u64(&mut p, pipeline.as_raw());
            frame(FC_BIND_PIPELINE, &p);
        }

        RecordedCommand::BindVertexBuffers { first_binding, buffers, offsets } => {
            let mut p = Vec::new();
            write_u32(&mut p, *first_binding);
            write_u32(&mut p, buffers.len() as u32);
            for b in buffers { write_u64(&mut p, b.as_raw()); }
            for o in offsets { write_u64(&mut p, *o); }
            frame(FC_BIND_VERTEX_BUFFERS, &p);
        }

        RecordedCommand::BindIndexBuffer { buffer, offset, index_type } => {
            let mut p = Vec::new();
            write_u64(&mut p, buffer.as_raw());
            write_u64(&mut p, *offset);
            write_u32(&mut p, index_type.as_raw() as u32);
            frame(FC_BIND_INDEX_BUFFER, &p);
        }

        RecordedCommand::BindDescriptorSets { bind_point, layout, first_set, descriptor_sets, dynamic_offsets } => {
            let mut p = Vec::new();
            write_u32(&mut p, bind_point.as_raw() as u32);
            write_u64(&mut p, layout.as_raw());
            write_u32(&mut p, *first_set);
            write_u32(&mut p, descriptor_sets.len() as u32);
            for ds in descriptor_sets { write_u64(&mut p, ds.as_raw()); }
            write_u32(&mut p, dynamic_offsets.len() as u32);
            for o in dynamic_offsets { write_u32(&mut p, *o); }
            frame(FC_BIND_DESCRIPTOR_SETS, &p);
        }

        RecordedCommand::PushConstants { layout, stage_flags, offset, size, data } => {
            let mut p = Vec::new();
            write_u64(&mut p, layout.as_raw());
            write_u32(&mut p, stage_flags.as_raw());
            write_u32(&mut p, *offset);
            write_u32(&mut p, *size);
            write_u32(&mut p, data.len() as u32);
            p.extend_from_slice(data);
            frame(FC_PUSH_CONSTANTS, &p);
        }

        RecordedCommand::Draw { vertex_count, instance_count, first_vertex, first_instance } => {
            let mut p = Vec::new();
            write_u32(&mut p, *vertex_count);
            write_u32(&mut p, *instance_count);
            write_u32(&mut p, *first_vertex);
            write_u32(&mut p, *first_instance);
            frame(FC_DRAW, &p);
        }

        RecordedCommand::DrawIndexed { index_count, instance_count, first_index, vertex_offset, first_instance } => {
            let mut p = Vec::new();
            write_u32(&mut p, *index_count);
            write_u32(&mut p, *instance_count);
            write_u32(&mut p, *first_index);
            write_i32(&mut p, *vertex_offset);
            write_u32(&mut p, *first_instance);
            frame(FC_DRAW_INDEXED, &p);
        }

        RecordedCommand::Dispatch { group_count_x, group_count_y, group_count_z } => {
            let mut p = Vec::new();
            write_u32(&mut p, *group_count_x);
            write_u32(&mut p, *group_count_y);
            write_u32(&mut p, *group_count_z);
            frame(FC_DISPATCH, &p);
        }

        RecordedCommand::DispatchBase { base_group_x, base_group_y, base_group_z, group_count_x, group_count_y, group_count_z } => {
            let mut p = Vec::new();
            write_u32(&mut p, *base_group_x);
            write_u32(&mut p, *base_group_y);
            write_u32(&mut p, *base_group_z);
            write_u32(&mut p, *group_count_x);
            write_u32(&mut p, *group_count_y);
            write_u32(&mut p, *group_count_z);
            frame(FC_DISPATCH_BASE, &p);
        }

        RecordedCommand::DrawIndirect { buffer, offset, draw_count, stride } => {
            let mut p = Vec::new();
            write_u64(&mut p, buffer.as_raw());
            write_u64(&mut p, *offset);
            write_u32(&mut p, *draw_count);
            write_u32(&mut p, *stride);
            frame(FC_DRAW_INDIRECT, &p);
        }

        RecordedCommand::DrawIndexedIndirect { buffer, offset, draw_count, stride } => {
            let mut p = Vec::new();
            write_u64(&mut p, buffer.as_raw());
            write_u64(&mut p, *offset);
            write_u32(&mut p, *draw_count);
            write_u32(&mut p, *stride);
            frame(FC_DRAW_INDEXED_INDIRECT, &p);
        }

        RecordedCommand::CopyBuffer { src_buffer, dst_buffer, regions } |
        RecordedCommand::CopyBuffer2 { src_buffer, dst_buffer, regions } => {
            let mut p = Vec::new();
            write_u64(&mut p, src_buffer.as_raw());
            write_u64(&mut p, dst_buffer.as_raw());
            write_u32(&mut p, regions.len() as u32);
            for r in regions { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_COPY_BUFFER, &p);
        }

        RecordedCommand::CopyBufferToImage { src_buffer, dst_image, dst_image_layout, regions } |
        RecordedCommand::CopyBufferToImage2 { src_buffer, dst_image, dst_image_layout, regions } => {
            let mut p = Vec::new();
            write_u64(&mut p, src_buffer.as_raw());
            write_u64(&mut p, dst_image.as_raw());
            write_u32(&mut p, dst_image_layout.as_raw() as u32);
            write_u32(&mut p, regions.len() as u32);
            for r in regions { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_COPY_BUFFER_TO_IMAGE, &p);
        }

        RecordedCommand::CopyImageToBuffer { src_image, src_image_layout, dst_buffer, regions } |
        RecordedCommand::CopyImageToBuffer2 { src_image, src_image_layout, dst_buffer, regions } => {
            let mut p = Vec::new();
            write_u64(&mut p, src_image.as_raw());
            write_u32(&mut p, src_image_layout.as_raw() as u32);
            write_u64(&mut p, dst_buffer.as_raw());
            write_u32(&mut p, regions.len() as u32);
            for r in regions { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_COPY_IMAGE_TO_BUFFER, &p);
        }

        RecordedCommand::CopyImage { src_image, src_image_layout, dst_image, dst_image_layout, regions } |
        RecordedCommand::CopyImage2 { src_image, src_image_layout, dst_image, dst_image_layout, regions } => {
            let mut p = Vec::new();
            write_u64(&mut p, src_image.as_raw());
            write_u32(&mut p, src_image_layout.as_raw() as u32);
            write_u64(&mut p, dst_image.as_raw());
            write_u32(&mut p, dst_image_layout.as_raw() as u32);
            write_u32(&mut p, regions.len() as u32);
            for r in regions { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_COPY_IMAGE, &p);
        }

        RecordedCommand::BlitImage { src_image, src_image_layout, dst_image, dst_image_layout, regions, filter } |
        RecordedCommand::BlitImage2 { src_image, src_image_layout, dst_image, dst_image_layout, regions, filter } => {
            let mut p = Vec::new();
            write_u64(&mut p, src_image.as_raw());
            write_u32(&mut p, src_image_layout.as_raw() as u32);
            write_u64(&mut p, dst_image.as_raw());
            write_u32(&mut p, dst_image_layout.as_raw() as u32);
            write_u32(&mut p, regions.len() as u32);
            for r in regions { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            write_u32(&mut p, filter.as_raw() as u32);
            frame(FC_BLIT_IMAGE, &p);
        }

        RecordedCommand::ResolveImage { src_image, src_image_layout, dst_image, dst_image_layout, regions } => {
            let mut p = Vec::new();
            write_u64(&mut p, src_image.as_raw());
            write_u32(&mut p, src_image_layout.as_raw() as u32);
            write_u64(&mut p, dst_image.as_raw());
            write_u32(&mut p, dst_image_layout.as_raw() as u32);
            write_u32(&mut p, regions.len() as u32);
            for r in regions { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_RESOLVE_IMAGE, &p);
        }

        RecordedCommand::PipelineBarrier { src_stage_mask, dst_stage_mask, dependency_flags } => {
            let mut p = Vec::new();
            write_u32(&mut p, src_stage_mask.as_raw());
            write_u32(&mut p, dst_stage_mask.as_raw());
            write_u32(&mut p, dependency_flags.as_raw());
            frame(FC_PIPELINE_BARRIER, &p);
        }

        RecordedCommand::PipelineBarrier2 => frame(FC_PIPELINE_BARRIER, &[]),

        RecordedCommand::SetViewport { first_viewport, viewports } => {
            let mut p = Vec::new();
            write_u32(&mut p, *first_viewport);
            write_u32(&mut p, viewports.len() as u32);
            for v in viewports { p.extend_from_slice(unsafe { struct_bytes(v) }); }
            frame(FC_SET_VIEWPORT, &p);
        }

        RecordedCommand::SetScissor { first_scissor, scissors } => {
            let mut p = Vec::new();
            write_u32(&mut p, *first_scissor);
            write_u32(&mut p, scissors.len() as u32);
            for s in scissors { p.extend_from_slice(unsafe { struct_bytes(s) }); }
            frame(FC_SET_SCISSOR, &p);
        }

        RecordedCommand::SetBlendConstants { blend_constants } => {
            let mut p = Vec::new();
            for &f in blend_constants { write_f32(&mut p, f); }
            frame(FC_SET_BLEND_CONSTANTS, &p);
        }

        RecordedCommand::SetStencilReference { face_mask, reference } => {
            let mut p = Vec::new();
            write_u32(&mut p, face_mask.as_raw());
            write_u32(&mut p, *reference);
            frame(FC_SET_STENCIL_REFERENCE, &p);
        }

        RecordedCommand::SetDepthBias { depth_bias_constant_factor, depth_bias_clamp, depth_bias_slope_factor } => {
            let mut p = Vec::new();
            write_f32(&mut p, *depth_bias_constant_factor);
            write_f32(&mut p, *depth_bias_clamp);
            write_f32(&mut p, *depth_bias_slope_factor);
            frame(FC_SET_DEPTH_BIAS, &p);
        }

        RecordedCommand::SetLineWidth { line_width } =>
            frame(FC_SET_LINE_WIDTH, &line_width.to_le_bytes()),

        RecordedCommand::SetCullMode { cull_mode } =>
            frame(FC_SET_CULL_MODE, &cull_mode.as_raw().to_le_bytes()),

        RecordedCommand::SetFrontFace { front_face } =>
            frame(FC_SET_FRONT_FACE, &(front_face.as_raw() as u32).to_le_bytes()),

        RecordedCommand::SetPrimitiveTopology { primitive_topology } =>
            frame(FC_SET_PRIMITIVE_TOPOLOGY, &(primitive_topology.as_raw() as u32).to_le_bytes()),

        RecordedCommand::SetDepthTestEnable { depth_test_enable } =>
            frame(FC_SET_DEPTH_TEST_ENABLE, &depth_test_enable.to_le_bytes()),

        RecordedCommand::SetDepthWriteEnable { depth_write_enable } =>
            frame(FC_SET_DEPTH_WRITE_ENABLE, &depth_write_enable.to_le_bytes()),

        RecordedCommand::SetDepthCompareOp { depth_compare_op } =>
            frame(FC_SET_DEPTH_COMPARE_OP, &(depth_compare_op.as_raw() as u32).to_le_bytes()),

        RecordedCommand::SetDepthBiasEnable { depth_bias_enable } =>
            frame(FC_SET_DEPTH_BIAS_ENABLE, &depth_bias_enable.to_le_bytes()),

        RecordedCommand::SetStencilTestEnable { stencil_test_enable } =>
            frame(FC_SET_STENCIL_TEST_ENABLE, &stencil_test_enable.to_le_bytes()),

        RecordedCommand::SetStencilOp { face_mask, fail_op, pass_op, depth_fail_op, compare_op } => {
            let mut p = Vec::new();
            write_u32(&mut p, face_mask.as_raw());
            write_u32(&mut p, fail_op.as_raw() as u32);
            write_u32(&mut p, pass_op.as_raw() as u32);
            write_u32(&mut p, depth_fail_op.as_raw() as u32);
            write_u32(&mut p, compare_op.as_raw() as u32);
            frame(FC_SET_STENCIL_OP, &p);
        }

        RecordedCommand::SetDepthBounds { min_depth_bounds, max_depth_bounds } => {
            let mut p = Vec::new();
            write_f32(&mut p, *min_depth_bounds);
            write_f32(&mut p, *max_depth_bounds);
            frame(FC_SET_DEPTH_BOUNDS, &p);
        }

        RecordedCommand::ClearColorImage { image, image_layout, color, ranges } => {
            let mut p = Vec::new();
            write_u64(&mut p, image.as_raw());
            write_u32(&mut p, image_layout.as_raw() as u32);
            p.extend_from_slice(unsafe { struct_bytes(color) }); // 16 bytes (union)
            write_u32(&mut p, ranges.len() as u32);
            for r in ranges { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_CLEAR_COLOR_IMAGE, &p);
        }

        RecordedCommand::ClearDepthStencilImage { image, image_layout, depth_stencil, ranges } => {
            let mut p = Vec::new();
            write_u64(&mut p, image.as_raw());
            write_u32(&mut p, image_layout.as_raw() as u32);
            p.extend_from_slice(unsafe { struct_bytes(depth_stencil) }); // 8 bytes
            write_u32(&mut p, ranges.len() as u32);
            for r in ranges { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_CLEAR_DEPTH_STENCIL_IMAGE, &p);
        }

        RecordedCommand::ClearAttachments { attachments, rects } => {
            let mut p = Vec::new();
            write_u32(&mut p, attachments.len() as u32);
            for a in attachments { p.extend_from_slice(unsafe { struct_bytes(a) }); }
            write_u32(&mut p, rects.len() as u32);
            for r in rects { p.extend_from_slice(unsafe { struct_bytes(r) }); }
            frame(FC_CLEAR_ATTACHMENTS, &p);
        }

        RecordedCommand::FillBuffer { dst_buffer, dst_offset, size, data } => {
            let mut p = Vec::new();
            write_u64(&mut p, dst_buffer.as_raw());
            write_u64(&mut p, *dst_offset);
            write_u64(&mut p, *size);
            write_u32(&mut p, *data);
            frame(FC_FILL_BUFFER, &p);
        }

        RecordedCommand::UpdateBuffer { dst_buffer, dst_offset, data } => {
            let mut p = Vec::new();
            write_u64(&mut p, dst_buffer.as_raw());
            write_u64(&mut p, *dst_offset);
            write_u32(&mut p, data.len() as u32);
            p.extend_from_slice(data);
            frame(FC_UPDATE_BUFFER, &p);
        }

        RecordedCommand::ExecuteCommands { command_buffers } => {
            let mut p = Vec::new();
            write_u32(&mut p, command_buffers.len() as u32);
            for cb in command_buffers { write_u64(&mut p, cb.as_raw()); }
            frame(FC_EXECUTE_COMMANDS, &p);
        }

        RecordedCommand::BeginRendering { render_area, layer_count, color_attachments, depth_attachment, stencil_attachment } => {
            let mut p = Vec::new();
            p.extend_from_slice(unsafe { struct_bytes(render_area) }); // 16 bytes
            write_u32(&mut p, *layer_count);
            write_u32(&mut p, color_attachments.len() as u32);
            for a in color_attachments { ser_rendering_attachment(&mut p, a); }
            write_option_attachment(&mut p, depth_attachment.as_ref());
            write_option_attachment(&mut p, stencil_attachment.as_ref());
            frame(FC_BEGIN_RENDERING, &p);
        }

        RecordedCommand::EndRendering => frame(FC_END_RENDERING, &[]),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn ser_rendering_attachment(out: &mut Vec<u8>, a: &crate::command_buffer::RenderingAttachment) {
    write_u64(out, a.image_view.as_raw());
    write_u32(out, a.load_op.as_raw() as u32);
    write_u32(out, a.store_op.as_raw() as u32);
    out.extend_from_slice(unsafe { struct_bytes(&a.clear_value) }); // 16 bytes
}

fn write_option_attachment(out: &mut Vec<u8>, att: Option<&crate::command_buffer::RenderingAttachment>) {
    if let Some(a) = att {
        write_u32(out, 1);
        ser_rendering_attachment(out, a);
    } else {
        write_u32(out, 0);
    }
}

/// Reinterpret any `Copy` value as a byte slice.
/// Safe because all vk::* structs used here are `#[repr(C)]` (generated from Vulkan XML).
unsafe fn struct_bytes<T: Copy>(val: &T) -> &[u8] {
    std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>())
}

#[inline] fn write_u32(out: &mut Vec<u8>, v: u32) { out.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn write_i32(out: &mut Vec<u8>, v: i32) { out.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn write_u64(out: &mut Vec<u8>, v: u64) { out.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn write_f32(out: &mut Vec<u8>, v: f32) { out.extend_from_slice(&v.to_bits().to_le_bytes()); }
