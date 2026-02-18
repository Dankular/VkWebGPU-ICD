//! Vulkan Pipeline implementation
//! Maps VkPipeline to WebGPU GPURenderPipeline/GPUComputePipeline

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
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

        let bind_group_layouts: Vec<&wgpu::BindGroupLayout> =
            layout_datas.iter().map(|data| &*data.wgpu_layout).collect();

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

    // Process shader stages
    let stages = std::slice::from_raw_parts(create_info.p_stages, create_info.stage_count as usize);

    let mut vertex_module = None;
    let mut fragment_module = None;

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

        if stage.stage.contains(vk::ShaderStageFlags::VERTEX) {
            vertex_module = Some(module);
        } else if stage.stage.contains(vk::ShaderStageFlags::FRAGMENT) {
            fragment_module = Some(module);
        }
    }

    // Get vertex input state
    let vertex_buffers = if !create_info.p_vertex_input_state.is_null() {
        let vis = &*create_info.p_vertex_input_state;
        process_vertex_input_state(vis)
    } else {
        Vec::new()
    };

    // Get color targets
    let color_targets = if !create_info.p_color_blend_state.is_null() {
        let cbs = &*create_info.p_color_blend_state;
        process_color_blend_state(cbs)
    } else {
        vec![Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })]
    };

    // Primitive state
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

    // Depth/stencil state
    let depth_stencil = if !create_info.p_depth_stencil_state.is_null() {
        let dss = &*create_info.p_depth_stencil_state;
        if dss.depth_test_enable == vk::TRUE || dss.depth_write_enable == vk::TRUE {
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus, // TODO: Get from render pass
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

    let pipeline =
        device_data
            .backend
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("VkGraphicsPipeline"),
                layout: Some(&layout_data.wgpu_layout),
                vertex: wgpu::VertexState {
                    module: vertex_module.as_ref().unwrap(),
                    entry_point: "main",
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                primitive,
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                fragment: fragment_module.as_ref().map(|module| wgpu::FragmentState {
                    module,
                    entry_point: "main",
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

            device_data
                .backend
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("VkComputePipeline"),
                    layout: Some(&layout_data.wgpu_layout),
                    module: &module,
                    entry_point: "main",
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
fn process_vertex_input_state(
    vis: &vk::PipelineVertexInputStateCreateInfo,
) -> Vec<wgpu::VertexBufferLayout<'static>> {
    if vis.vertex_binding_description_count == 0 {
        return Vec::new();
    }

    let bindings = unsafe {
        std::slice::from_raw_parts(
            vis.p_vertex_binding_descriptions,
            vis.vertex_binding_description_count as usize,
        )
    };

    let attributes = unsafe {
        std::slice::from_raw_parts(
            vis.p_vertex_attribute_descriptions,
            vis.vertex_attribute_description_count as usize,
        )
    };

    bindings
        .iter()
        .map(|binding| {
            let binding_attrs: Vec<wgpu::VertexAttribute> = attributes
                .iter()
                .filter(|attr| attr.binding == binding.binding)
                .map(|attr| wgpu::VertexAttribute {
                    format: vk_to_wgpu_vertex_format(attr.format),
                    offset: attr.offset as u64,
                    shader_location: attr.location,
                })
                .collect();

            wgpu::VertexBufferLayout {
                array_stride: binding.stride as u64,
                step_mode: if binding.input_rate == vk::VertexInputRate::VERTEX {
                    wgpu::VertexStepMode::Vertex
                } else {
                    wgpu::VertexStepMode::Instance
                },
                attributes: Box::leak(binding_attrs.into_boxed_slice()),
            }
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn process_color_blend_state(
    cbs: &vk::PipelineColorBlendStateCreateInfo,
) -> Vec<Option<wgpu::ColorTargetState>> {
    if cbs.attachment_count == 0 {
        return vec![Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
    }

    let attachments =
        unsafe { std::slice::from_raw_parts(cbs.p_attachments, cbs.attachment_count as usize) };

    attachments
        .iter()
        .map(|attachment| {
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8Unorm, // TODO: Get from render pass
                blend: if attachment.blend_enable == vk::TRUE {
                    Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: vk_to_wgpu_blend_factor(attachment.src_color_blend_factor),
                            dst_factor: vk_to_wgpu_blend_factor(attachment.dst_color_blend_factor),
                            operation: vk_to_wgpu_blend_operation(attachment.color_blend_op),
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: vk_to_wgpu_blend_factor(attachment.src_alpha_blend_factor),
                            dst_factor: vk_to_wgpu_blend_factor(attachment.dst_alpha_blend_factor),
                            operation: vk_to_wgpu_blend_operation(attachment.alpha_blend_op),
                        },
                    })
                } else {
                    None
                },
                write_mask: vk_to_wgpu_color_write_mask(attachment.color_write_mask),
            })
        })
        .collect()
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
