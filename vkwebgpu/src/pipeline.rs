//! Vulkan Pipeline implementation
//! Maps VkPipeline to WebGPU GPURenderPipeline/GPUComputePipeline

use ash::vk::{self, Handle};
use log::{debug, warn};
use once_cell::sync::Lazy;
use std::ffi::CStr;
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static PIPELINE_LAYOUT_ALLOCATOR: Lazy<HandleAllocator<VkPipelineLayoutData>> =
    Lazy::new(|| HandleAllocator::new());
pub static PIPELINE_ALLOCATOR: Lazy<HandleAllocator<VkPipelineData>> =
    Lazy::new(|| HandleAllocator::new());
pub static SHADER_MODULE_ALLOCATOR: Lazy<HandleAllocator<VkShaderModuleData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkPipelineLayoutData {
    pub device: vk::Device,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_layout: Arc<wgpu::PipelineLayout>,
}

pub struct VkShaderModuleData {
    pub device: vk::Device,
    pub spirv: Vec<u32>,
}

pub enum VkPipelineData {
    Graphics {
        device: vk::Device,
        layout: vk::PipelineLayout,
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_pipeline: Arc<wgpu::RenderPipeline>,
    },
    Compute {
        device: vk::Device,
        layout: vk::PipelineLayout,
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_pipeline: Arc<wgpu::ComputePipeline>,
    },
}

pub unsafe fn create_shader_module(
    device: vk::Device,
    p_create_info: *const vk::ShaderModuleCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_shader_module: *mut vk::ShaderModule,
) -> Result<()> {
    let create_info = &*p_create_info;

    let spirv = std::slice::from_raw_parts(create_info.p_code, create_info.code_size / 4).to_vec();

    debug!("Creating shader module with {} SPIR-V words", spirv.len());

    let module_data = VkShaderModuleData { device, spirv };

    let module_handle = SHADER_MODULE_ALLOCATOR.allocate(module_data);
    *p_shader_module = Handle::from_raw(module_handle);

    Ok(())
}

pub unsafe fn destroy_shader_module(
    _device: vk::Device,
    shader_module: vk::ShaderModule,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if shader_module == vk::ShaderModule::null() {
        return;
    }

    SHADER_MODULE_ALLOCATOR.remove(shader_module.as_raw());
}

pub unsafe fn create_pipeline_layout(
    device: vk::Device,
    p_create_info: *const vk::PipelineLayoutCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_pipeline_layout: *mut vk::PipelineLayout,
) -> Result<()> {
    let create_info = &*p_create_info;

    let set_layouts = if create_info.set_layout_count > 0 {
        std::slice::from_raw_parts(
            create_info.p_set_layouts,
            create_info.set_layout_count as usize,
        )
        .to_vec()
    } else {
        Vec::new()
    };

    let push_constant_ranges = if create_info.push_constant_range_count > 0 {
        std::slice::from_raw_parts(
            create_info.p_push_constant_ranges,
            create_info.push_constant_range_count as usize,
        )
        .to_vec()
    } else {
        Vec::new()
    };

    debug!(
        "Creating pipeline layout with {} set layouts, {} push constant ranges",
        set_layouts.len(),
        push_constant_ranges.len()
    );

    let device_data = device::get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    #[cfg(not(target_arch = "wasm32"))]
    let wgpu_layout = {
        // Keep Arc references alive for the lifetime of bind_group_layouts
        let layout_datas: Vec<_> = set_layouts
            .iter()
            .filter_map(|&layout| crate::descriptor::get_descriptor_set_layout_data(layout))
            .collect();

        let mut bind_group_layouts: Vec<&wgpu::BindGroupLayout> =
            layout_datas.iter().map(|data| &*data.wgpu_layout).collect();

        // If this pipeline layout has push constants, prepend the push constant bind group layout
        // at set 0. This shifts all user descriptor sets by +1. The shader transformation in
        // shader.rs applies the same shift to WGSL binding group numbers.
        let has_push_constants = !push_constant_ranges.is_empty();
        if has_push_constants {
            bind_group_layouts.insert(0, device_data.push_constant_buffer.bind_group_layout());
        }

        device_data
            .backend
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("VkPipelineLayout"),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            })
    };

    let layout_data = VkPipelineLayoutData {
        device,
        set_layouts,
        push_constant_ranges,
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_layout: Arc::new(wgpu_layout),
    };

    let layout_handle = PIPELINE_LAYOUT_ALLOCATOR.allocate(layout_data);
    *p_pipeline_layout = Handle::from_raw(layout_handle);

    Ok(())
}

pub unsafe fn destroy_pipeline_layout(
    _device: vk::Device,
    pipeline_layout: vk::PipelineLayout,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if pipeline_layout == vk::PipelineLayout::null() {
        return;
    }

    PIPELINE_LAYOUT_ALLOCATOR.remove(pipeline_layout.as_raw());
}

pub unsafe fn create_graphics_pipelines(
    device: vk::Device,
    _pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::GraphicsPipelineCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_pipelines: *mut vk::Pipeline,
) -> Result<()> {
    let create_infos = std::slice::from_raw_parts(p_create_infos, create_info_count as usize);
    let pipelines = std::slice::from_raw_parts_mut(p_pipelines, create_info_count as usize);

    let device_data = device::get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    for (i, create_info) in create_infos.iter().enumerate() {
        debug!("Creating graphics pipeline {}/{}", i + 1, create_info_count);

        #[cfg(not(target_arch = "wasm32"))]
        let wgpu_pipeline = create_wgpu_render_pipeline(device, create_info, &device_data)?;

        let pipeline_data = VkPipelineData::Graphics {
            device,
            layout: create_info.layout,
            #[cfg(not(target_arch = "wasm32"))]
            wgpu_pipeline: Arc::new(wgpu_pipeline),
        };

        let pipeline_handle = PIPELINE_ALLOCATOR.allocate(pipeline_data);
        pipelines[i] = Handle::from_raw(pipeline_handle);
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn create_wgpu_render_pipeline(
    _device: vk::Device,
    create_info: &vk::GraphicsPipelineCreateInfo,
    device_data: &crate::device::VkDeviceData,
) -> Result<wgpu::RenderPipeline> {
    // Get pipeline layout
    let layout_data = PIPELINE_LAYOUT_ALLOCATOR
        .get(create_info.layout.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid pipeline layout".to_string()))?;

    // ── 1. Collect render target formats ─────────────────────────────────────────
    // Priority order:
    //   a) VkPipelineRenderingCreateInfo from the pNext chain (dynamic rendering / Vulkan 1.3)
    //   b) VkRenderPass + subpass index (traditional render pass path)
    //   c) Fallback defaults if neither is available
    let mut color_vk_formats: Vec<vk::Format> = Vec::new();
    let mut depth_vk_format = vk::Format::UNDEFINED;

    // Walk the pNext chain for VkPipelineRenderingCreateInfo
    let mut p_next = create_info.p_next as *const vk::BaseInStructure;
    while !p_next.is_null() {
        let base = &*p_next;
        if base.s_type == vk::StructureType::PIPELINE_RENDERING_CREATE_INFO {
            let pri = &*(p_next as *const vk::PipelineRenderingCreateInfo);
            if pri.color_attachment_count > 0 && !pri.p_color_attachment_formats.is_null() {
                color_vk_formats = std::slice::from_raw_parts(
                    pri.p_color_attachment_formats,
                    pri.color_attachment_count as usize,
                )
                .to_vec();
            }
            depth_vk_format = pri.depth_attachment_format;
            debug!(
                "Pipeline: read {} color format(s) and depth format {:?} from VkPipelineRenderingCreateInfo",
                color_vk_formats.len(),
                depth_vk_format
            );
            break;
        }
        p_next = base.p_next;
    }

    // Fall back to render pass attachment formats when not using dynamic rendering
    if color_vk_formats.is_empty() && create_info.render_pass != vk::RenderPass::null() {
        if let Some(rp_data) = crate::render_pass::get_render_pass_data(create_info.render_pass) {
            let subpass_idx = create_info.subpass as usize;
            if let Some(subpass) = rp_data.subpasses.get(subpass_idx) {
                color_vk_formats = subpass
                    .color_attachment_indices
                    .iter()
                    .filter_map(|&idx| rp_data.attachments.get(idx as usize).map(|a| a.format))
                    .collect();
                if let Some(depth_idx) = subpass.depth_stencil_attachment_index {
                    depth_vk_format = rp_data
                        .attachments
                        .get(depth_idx as usize)
                        .map(|a| a.format)
                        .unwrap_or(vk::Format::UNDEFINED);
                }
                debug!(
                    "Pipeline: read {} color format(s) and depth format {:?} from render pass subpass {}",
                    color_vk_formats.len(),
                    depth_vk_format,
                    subpass_idx
                );
            }
        }
    }

    // ── 2. Process shader stages ──────────────────────────────────────────────────
    let stages =
        std::slice::from_raw_parts(create_info.p_stages, create_info.stage_count as usize);

    let mut vertex_module: Option<wgpu::ShaderModule> = None;
    let mut vertex_entry = String::from("main");
    let mut fragment_module: Option<wgpu::ShaderModule> = None;
    let mut fragment_entry = String::from("main");

    for stage in stages {
        let shader_data = SHADER_MODULE_ALLOCATOR
            .get(stage.module.as_raw())
            .ok_or_else(|| VkError::InvalidHandle("Invalid shader module".to_string()))?;

        let wgsl = device_data
            .shader_cache
            .get_or_translate(&shader_data.spirv)?;

        let module =
            device_data
                .backend
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Shader"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&wgsl)),
                });

        // Use the entry point name specified in the Vulkan stage info (pName).
        // This is the name of the OpEntryPoint in the SPIR-V, which Naga preserves
        // as the function name in the generated WGSL.
        let entry_name = if !stage.p_name.is_null() {
            CStr::from_ptr(stage.p_name)
                .to_str()
                .unwrap_or("main")
                .to_string()
        } else {
            String::from("main")
        };

        if stage.stage.contains(vk::ShaderStageFlags::VERTEX) {
            vertex_entry = entry_name;
            vertex_module = Some(module);
        } else if stage.stage.contains(vk::ShaderStageFlags::FRAGMENT) {
            fragment_entry = entry_name;
            fragment_module = Some(module);
        } else {
            // Geometry and tessellation stages are not supported by WebGPU.
            // Log a warning and skip rather than failing pipeline creation, since DXVK
            // may include these stages but the pipeline can still partially function.
            warn!(
                "Shader stage {:?} is not supported by WebGPU and will be skipped",
                stage.stage
            );
        }
    }

    // ── 3. Vertex input state (no Box::leak) ─────────────────────────────────────
    // We build owned Vec<VertexAttribute> per binding, then borrow them into
    // VertexBufferLayout<'_> within this same scope (all_binding_attrs outlives
    // vertex_buffers because it is declared first in the same scope).
    let mut all_binding_attrs: Vec<Vec<wgpu::VertexAttribute>> = Vec::new();
    let mut owned_binding_descs: Vec<vk::VertexInputBindingDescription> = Vec::new();

    if !create_info.p_vertex_input_state.is_null() {
        let vis = &*create_info.p_vertex_input_state;
        if vis.vertex_binding_description_count > 0
            && !vis.p_vertex_binding_descriptions.is_null()
        {
            let bindings = std::slice::from_raw_parts(
                vis.p_vertex_binding_descriptions,
                vis.vertex_binding_description_count as usize,
            );
            let attr_slice = if vis.vertex_attribute_description_count > 0
                && !vis.p_vertex_attribute_descriptions.is_null()
            {
                std::slice::from_raw_parts(
                    vis.p_vertex_attribute_descriptions,
                    vis.vertex_attribute_description_count as usize,
                )
            } else {
                &[]
            };

            for binding in bindings {
                let attrs: Vec<wgpu::VertexAttribute> = attr_slice
                    .iter()
                    .filter(|attr| attr.binding == binding.binding)
                    .map(|attr| wgpu::VertexAttribute {
                        format: vk_to_wgpu_vertex_format(attr.format),
                        offset: attr.offset as u64,
                        shader_location: attr.location,
                    })
                    .collect();
                all_binding_attrs.push(attrs);
            }
            owned_binding_descs = bindings.to_vec();
        }
    }

    // Borrow from all_binding_attrs; the borrow checker ensures all_binding_attrs
    // lives at least as long as vertex_buffers (it is declared before vertex_buffers).
    let vertex_buffers: Vec<wgpu::VertexBufferLayout> = owned_binding_descs
        .iter()
        .zip(all_binding_attrs.iter())
        .map(|(binding, attrs)| wgpu::VertexBufferLayout {
            array_stride: binding.stride as u64,
            step_mode: if binding.input_rate == vk::VertexInputRate::VERTEX {
                wgpu::VertexStepMode::Vertex
            } else {
                wgpu::VertexStepMode::Instance
            },
            attributes: attrs.as_slice(),
        })
        .collect();

    // ── 4. Primitive state ────────────────────────────────────────────────────────
    let primitive = {
        let topology = if !create_info.p_input_assembly_state.is_null() {
            let ias = &*create_info.p_input_assembly_state;
            vk_to_wgpu_primitive_topology(ias.topology)
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };

        let (front_face, cull_mode, polygon_mode) = if !create_info.p_rasterization_state.is_null()
        {
            let rs = &*create_info.p_rasterization_state;
            let front_face = if rs.front_face == vk::FrontFace::CLOCKWISE {
                wgpu::FrontFace::Cw
            } else {
                wgpu::FrontFace::Ccw
            };
            let cull_mode = match rs.cull_mode {
                vk::CullModeFlags::NONE => None,
                vk::CullModeFlags::FRONT => Some(wgpu::Face::Front),
                vk::CullModeFlags::BACK => Some(wgpu::Face::Back),
                _ => None,
            };
            let polygon_mode = match rs.polygon_mode {
                vk::PolygonMode::FILL => wgpu::PolygonMode::Fill,
                vk::PolygonMode::LINE => wgpu::PolygonMode::Line,
                vk::PolygonMode::POINT => wgpu::PolygonMode::Point,
                _ => wgpu::PolygonMode::Fill,
            };
            (front_face, cull_mode, polygon_mode)
        } else {
            (wgpu::FrontFace::Ccw, None, wgpu::PolygonMode::Fill)
        };

        wgpu::PrimitiveState {
            topology,
            strip_index_format: None,
            front_face,
            cull_mode,
            polygon_mode,
            unclipped_depth: false,
            conservative: false,
        }
    };

    // ── 5. Color targets with real formats ────────────────────────────────────────
    let color_targets: Vec<Option<wgpu::ColorTargetState>> = {
        let (att_count, att_slice): (usize, &[vk::PipelineColorBlendAttachmentState]) =
            if !create_info.p_color_blend_state.is_null() {
                let cbs = &*create_info.p_color_blend_state;
                let count = cbs.attachment_count as usize;
                let slice = if count > 0 && !cbs.p_attachments.is_null() {
                    std::slice::from_raw_parts(cbs.p_attachments, count)
                } else {
                    &[]
                };
                (count, slice)
            } else {
                (0, &[])
            };

        // Number of targets is the max of blend state attachment count and format count.
        // When using dynamic rendering, color_vk_formats is the authoritative count.
        let num_targets = att_count.max(color_vk_formats.len());

        if num_targets == 0 {
            // No color attachments (depth-only pass or unknown configuration).
            Vec::new()
        } else {
            (0..num_targets)
                .map(|i| {
                    // Resolve the wgpu format: prefer the explicit format list, fall back to B8G8R8A8_UNORM.
                    let vk_fmt = color_vk_formats
                        .get(i)
                        .copied()
                        .unwrap_or(vk::Format::B8G8R8A8_UNORM);
                    let wgpu_fmt = crate::format::vk_to_wgpu_format(vk_fmt)
                        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);

                    let att = att_slice.get(i);
                    let blend = att.and_then(|a| {
                        if a.blend_enable == vk::TRUE {
                            Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: vk_to_wgpu_blend_factor(a.src_color_blend_factor),
                                    dst_factor: vk_to_wgpu_blend_factor(a.dst_color_blend_factor),
                                    operation: vk_to_wgpu_blend_operation(a.color_blend_op),
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: vk_to_wgpu_blend_factor(a.src_alpha_blend_factor),
                                    dst_factor: vk_to_wgpu_blend_factor(a.dst_alpha_blend_factor),
                                    operation: vk_to_wgpu_blend_operation(a.alpha_blend_op),
                                },
                            })
                        } else {
                            None
                        }
                    });
                    let write_mask = att
                        .map(|a| vk_to_wgpu_color_write_mask(a.color_write_mask))
                        .unwrap_or(wgpu::ColorWrites::ALL);

                    Some(wgpu::ColorTargetState {
                        format: wgpu_fmt,
                        blend,
                        write_mask,
                    })
                })
                .collect()
        }
    };

    // ── 6. Depth/stencil state with real format ───────────────────────────────────
    let depth_stencil = if !create_info.p_depth_stencil_state.is_null() {
        let dss = &*create_info.p_depth_stencil_state;
        if dss.depth_test_enable == vk::TRUE
            || dss.depth_write_enable == vk::TRUE
            || dss.stencil_test_enable == vk::TRUE
        {
            // Resolve depth format from the information we gathered above.
            let depth_wgpu_format = if depth_vk_format != vk::Format::UNDEFINED {
                crate::format::vk_to_wgpu_format(depth_vk_format)
                    .unwrap_or(wgpu::TextureFormat::Depth24Plus)
            } else {
                // No explicit format provided; default to Depth24Plus which is
                // universally supported and covers D24S8 and plain depth cases.
                wgpu::TextureFormat::Depth24Plus
            };

            Some(wgpu::DepthStencilState {
                format: depth_wgpu_format,
                depth_write_enabled: dss.depth_write_enable == vk::TRUE,
                depth_compare: vk_to_wgpu_compare_function(dss.depth_compare_op),
                stencil: wgpu::StencilState {
                    front: vk_to_wgpu_stencil_face_state(&dss.front),
                    back: vk_to_wgpu_stencil_face_state(&dss.back),
                    read_mask: dss.front.compare_mask,
                    write_mask: dss.front.write_mask,
                },
                bias: wgpu::DepthBiasState::default(),
            })
        } else {
            None
        }
    } else {
        None
    };

    // ── 7. Create the render pipeline ─────────────────────────────────────────────
    let vertex_mod = vertex_module
        .as_ref()
        .ok_or_else(|| VkError::InvalidHandle("Graphics pipeline has no vertex shader".to_string()))?;

    let pipeline =
        device_data
            .backend
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("VkGraphicsPipeline"),
                layout: Some(&layout_data.wgpu_layout),
                vertex: wgpu::VertexState {
                    module: vertex_mod,
                    entry_point: vertex_entry.as_str(),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                primitive,
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                fragment: fragment_module.as_ref().map(|module| wgpu::FragmentState {
                    module,
                    entry_point: fragment_entry.as_str(),
                    targets: &color_targets,
                    compilation_options: Default::default(),
                }),
                multiview: None,
            });

    Ok(pipeline)
}

pub unsafe fn create_compute_pipelines(
    device: vk::Device,
    _pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::ComputePipelineCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_pipelines: *mut vk::Pipeline,
) -> Result<()> {
    let create_infos = std::slice::from_raw_parts(p_create_infos, create_info_count as usize);
    let pipelines = std::slice::from_raw_parts_mut(p_pipelines, create_info_count as usize);

    let device_data = device::get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    for (i, create_info) in create_infos.iter().enumerate() {
        debug!("Creating compute pipeline {}/{}", i + 1, create_info_count);

        #[cfg(not(target_arch = "wasm32"))]
        let wgpu_pipeline = {
            // Get pipeline layout
            let layout_data = PIPELINE_LAYOUT_ALLOCATOR
                .get(create_info.layout.as_raw())
                .ok_or_else(|| VkError::InvalidHandle("Invalid pipeline layout".to_string()))?;

            // Get shader module
            let shader_data = SHADER_MODULE_ALLOCATOR
                .get(create_info.stage.module.as_raw())
                .ok_or_else(|| VkError::InvalidHandle("Invalid shader module".to_string()))?;

            let wgsl = device_data
                .shader_cache
                .get_or_translate(&shader_data.spirv)?;

            let module =
                device_data
                    .backend
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("ComputeShader"),
                        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&wgsl)),
                    });

            // Use the entry point name from the Vulkan stage info.
            let entry_name = if !create_info.stage.p_name.is_null() {
                CStr::from_ptr(create_info.stage.p_name)
                    .to_str()
                    .unwrap_or("main")
                    .to_string()
            } else {
                String::from("main")
            };

            device_data
                .backend
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("VkComputePipeline"),
                    layout: Some(&layout_data.wgpu_layout),
                    module: &module,
                    entry_point: entry_name.as_str(),
                    compilation_options: Default::default(),
                })
        };

        let pipeline_data = VkPipelineData::Compute {
            device,
            layout: create_info.layout,
            #[cfg(not(target_arch = "wasm32"))]
            wgpu_pipeline: Arc::new(wgpu_pipeline),
        };

        let pipeline_handle = PIPELINE_ALLOCATOR.allocate(pipeline_data);
        pipelines[i] = Handle::from_raw(pipeline_handle);
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_vertex_format(format: vk::Format) -> wgpu::VertexFormat {
    match format {
        vk::Format::R32_SFLOAT => wgpu::VertexFormat::Float32,
        vk::Format::R32G32_SFLOAT => wgpu::VertexFormat::Float32x2,
        vk::Format::R32G32B32_SFLOAT => wgpu::VertexFormat::Float32x3,
        vk::Format::R32G32B32A32_SFLOAT => wgpu::VertexFormat::Float32x4,
        vk::Format::R32_SINT => wgpu::VertexFormat::Sint32,
        vk::Format::R32G32_SINT => wgpu::VertexFormat::Sint32x2,
        vk::Format::R32G32B32_SINT => wgpu::VertexFormat::Sint32x3,
        vk::Format::R32G32B32A32_SINT => wgpu::VertexFormat::Sint32x4,
        vk::Format::R32_UINT => wgpu::VertexFormat::Uint32,
        vk::Format::R32G32_UINT => wgpu::VertexFormat::Uint32x2,
        vk::Format::R32G32B32_UINT => wgpu::VertexFormat::Uint32x3,
        vk::Format::R32G32B32A32_UINT => wgpu::VertexFormat::Uint32x4,
        vk::Format::R16G16_SINT => wgpu::VertexFormat::Sint16x2,
        vk::Format::R16G16B16A16_SINT => wgpu::VertexFormat::Sint16x4,
        vk::Format::R16G16_UINT => wgpu::VertexFormat::Uint16x2,
        vk::Format::R16G16B16A16_UINT => wgpu::VertexFormat::Uint16x4,
        vk::Format::R8G8_SINT => wgpu::VertexFormat::Sint8x2,
        vk::Format::R8G8B8A8_SINT => wgpu::VertexFormat::Sint8x4,
        vk::Format::R8G8_UINT => wgpu::VertexFormat::Uint8x2,
        vk::Format::R8G8B8A8_UINT => wgpu::VertexFormat::Uint8x4,
        _ => wgpu::VertexFormat::Float32x3, // Default
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_blend_factor(factor: vk::BlendFactor) -> wgpu::BlendFactor {
    match factor {
        vk::BlendFactor::ZERO => wgpu::BlendFactor::Zero,
        vk::BlendFactor::ONE => wgpu::BlendFactor::One,
        vk::BlendFactor::SRC_COLOR => wgpu::BlendFactor::Src,
        vk::BlendFactor::ONE_MINUS_SRC_COLOR => wgpu::BlendFactor::OneMinusSrc,
        vk::BlendFactor::DST_COLOR => wgpu::BlendFactor::Dst,
        vk::BlendFactor::ONE_MINUS_DST_COLOR => wgpu::BlendFactor::OneMinusDst,
        vk::BlendFactor::SRC_ALPHA => wgpu::BlendFactor::SrcAlpha,
        vk::BlendFactor::ONE_MINUS_SRC_ALPHA => wgpu::BlendFactor::OneMinusSrcAlpha,
        vk::BlendFactor::DST_ALPHA => wgpu::BlendFactor::DstAlpha,
        vk::BlendFactor::ONE_MINUS_DST_ALPHA => wgpu::BlendFactor::OneMinusDstAlpha,
        vk::BlendFactor::CONSTANT_COLOR => wgpu::BlendFactor::Constant,
        vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR => wgpu::BlendFactor::OneMinusConstant,
        _ => wgpu::BlendFactor::One,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_blend_operation(op: vk::BlendOp) -> wgpu::BlendOperation {
    match op {
        vk::BlendOp::ADD => wgpu::BlendOperation::Add,
        vk::BlendOp::SUBTRACT => wgpu::BlendOperation::Subtract,
        vk::BlendOp::REVERSE_SUBTRACT => wgpu::BlendOperation::ReverseSubtract,
        vk::BlendOp::MIN => wgpu::BlendOperation::Min,
        vk::BlendOp::MAX => wgpu::BlendOperation::Max,
        _ => wgpu::BlendOperation::Add,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_color_write_mask(mask: vk::ColorComponentFlags) -> wgpu::ColorWrites {
    let mut writes = wgpu::ColorWrites::empty();
    if mask.contains(vk::ColorComponentFlags::R) {
        writes |= wgpu::ColorWrites::RED;
    }
    if mask.contains(vk::ColorComponentFlags::G) {
        writes |= wgpu::ColorWrites::GREEN;
    }
    if mask.contains(vk::ColorComponentFlags::B) {
        writes |= wgpu::ColorWrites::BLUE;
    }
    if mask.contains(vk::ColorComponentFlags::A) {
        writes |= wgpu::ColorWrites::ALPHA;
    }
    writes
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_compare_function(op: vk::CompareOp) -> wgpu::CompareFunction {
    match op {
        vk::CompareOp::NEVER => wgpu::CompareFunction::Never,
        vk::CompareOp::LESS => wgpu::CompareFunction::Less,
        vk::CompareOp::EQUAL => wgpu::CompareFunction::Equal,
        vk::CompareOp::LESS_OR_EQUAL => wgpu::CompareFunction::LessEqual,
        vk::CompareOp::GREATER => wgpu::CompareFunction::Greater,
        vk::CompareOp::NOT_EQUAL => wgpu::CompareFunction::NotEqual,
        vk::CompareOp::GREATER_OR_EQUAL => wgpu::CompareFunction::GreaterEqual,
        vk::CompareOp::ALWAYS => wgpu::CompareFunction::Always,
        _ => wgpu::CompareFunction::Always,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_stencil_operation(op: vk::StencilOp) -> wgpu::StencilOperation {
    match op {
        vk::StencilOp::KEEP => wgpu::StencilOperation::Keep,
        vk::StencilOp::ZERO => wgpu::StencilOperation::Zero,
        vk::StencilOp::REPLACE => wgpu::StencilOperation::Replace,
        vk::StencilOp::INCREMENT_AND_CLAMP => wgpu::StencilOperation::IncrementClamp,
        vk::StencilOp::DECREMENT_AND_CLAMP => wgpu::StencilOperation::DecrementClamp,
        vk::StencilOp::INVERT => wgpu::StencilOperation::Invert,
        vk::StencilOp::INCREMENT_AND_WRAP => wgpu::StencilOperation::IncrementWrap,
        vk::StencilOp::DECREMENT_AND_WRAP => wgpu::StencilOperation::DecrementWrap,
        _ => wgpu::StencilOperation::Keep,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_stencil_face_state(state: &vk::StencilOpState) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: vk_to_wgpu_compare_function(state.compare_op),
        fail_op: vk_to_wgpu_stencil_operation(state.fail_op),
        depth_fail_op: vk_to_wgpu_stencil_operation(state.depth_fail_op),
        pass_op: vk_to_wgpu_stencil_operation(state.pass_op),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_primitive_topology(topology: vk::PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match topology {
        vk::PrimitiveTopology::POINT_LIST => wgpu::PrimitiveTopology::PointList,
        vk::PrimitiveTopology::LINE_LIST => wgpu::PrimitiveTopology::LineList,
        vk::PrimitiveTopology::LINE_STRIP => wgpu::PrimitiveTopology::LineStrip,
        vk::PrimitiveTopology::TRIANGLE_LIST => wgpu::PrimitiveTopology::TriangleList,
        vk::PrimitiveTopology::TRIANGLE_STRIP => wgpu::PrimitiveTopology::TriangleStrip,
        _ => wgpu::PrimitiveTopology::TriangleList,
    }
}

pub unsafe fn destroy_pipeline(
    _device: vk::Device,
    pipeline: vk::Pipeline,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if pipeline == vk::Pipeline::null() {
        return;
    }

    PIPELINE_ALLOCATOR.remove(pipeline.as_raw());
}

pub fn get_pipeline_data(pipeline: vk::Pipeline) -> Option<Arc<VkPipelineData>> {
    PIPELINE_ALLOCATOR.get(pipeline.as_raw())
}

pub fn get_pipeline_layout_data(layout: vk::PipelineLayout) -> Option<Arc<VkPipelineLayoutData>> {
    PIPELINE_LAYOUT_ALLOCATOR.get(layout.as_raw())
}
