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
    let primitive = if !create_info.p_input_assembly_state.is_null() {
        let ias = &*create_info.p_input_assembly_state;
        wgpu::PrimitiveState {
            topology: vk_to_wgpu_primitive_topology(ias.topology),
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        }
    } else {
        wgpu::PrimitiveState::default()
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
                depth_stencil: None,
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

#[cfg(not(target_arch = "wasm32"))]
fn process_vertex_input_state(
    _vis: &vk::PipelineVertexInputStateCreateInfo,
) -> Vec<wgpu::VertexBufferLayout<'static>> {
    // Simplified: return empty for now
    Vec::new()
}

#[cfg(not(target_arch = "wasm32"))]
fn process_color_blend_state(
    _cbs: &vk::PipelineColorBlendStateCreateInfo,
) -> Vec<Option<wgpu::ColorTargetState>> {
    vec![Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Bgra8Unorm,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })]
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
