//! WebGPU backend abstraction
//!
//! Provides a unified interface for both native (wgpu) and WASM (web-sys) WebGPU backends

use crate::error::{Result, VkError};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
pub type Instance = wgpu::Instance;
#[cfg(not(target_arch = "wasm32"))]
pub type Adapter = wgpu::Adapter;
#[cfg(not(target_arch = "wasm32"))]
pub type Device = wgpu::Device;
#[cfg(not(target_arch = "wasm32"))]
pub type Queue = wgpu::Queue;
#[cfg(not(target_arch = "wasm32"))]
pub type Buffer = wgpu::Buffer;
#[cfg(not(target_arch = "wasm32"))]
pub type Texture = wgpu::Texture;
#[cfg(not(target_arch = "wasm32"))]
pub type TextureView = wgpu::TextureView;
#[cfg(not(target_arch = "wasm32"))]
pub type Sampler = wgpu::Sampler;
#[cfg(not(target_arch = "wasm32"))]
pub type BindGroup = wgpu::BindGroup;
#[cfg(not(target_arch = "wasm32"))]
pub type BindGroupLayout = wgpu::BindGroupLayout;
#[cfg(not(target_arch = "wasm32"))]
pub type PipelineLayout = wgpu::PipelineLayout;
#[cfg(not(target_arch = "wasm32"))]
pub type RenderPipeline = wgpu::RenderPipeline;
#[cfg(not(target_arch = "wasm32"))]
pub type ComputePipeline = wgpu::ComputePipeline;
#[cfg(not(target_arch = "wasm32"))]
pub type ShaderModule = wgpu::ShaderModule;
#[cfg(not(target_arch = "wasm32"))]
pub type CommandEncoder = wgpu::CommandEncoder;
#[cfg(not(target_arch = "wasm32"))]
pub type CommandBuffer = wgpu::CommandBuffer;
#[cfg(not(target_arch = "wasm32"))]
pub type RenderPass<'a> = wgpu::RenderPass<'a>;
#[cfg(not(target_arch = "wasm32"))]
pub type ComputePass<'a> = wgpu::ComputePass<'a>;

#[cfg(target_arch = "wasm32")]
pub type Instance = web_sys::Gpu;
#[cfg(target_arch = "wasm32")]
pub type Adapter = web_sys::GpuAdapter;
#[cfg(target_arch = "wasm32")]
pub type Device = web_sys::GpuDevice;
#[cfg(target_arch = "wasm32")]
pub type Queue = web_sys::GpuQueue;
#[cfg(target_arch = "wasm32")]
pub type Buffer = web_sys::GpuBuffer;
#[cfg(target_arch = "wasm32")]
pub type Texture = web_sys::GpuTexture;
#[cfg(target_arch = "wasm32")]
pub type TextureView = web_sys::GpuTextureView;
#[cfg(target_arch = "wasm32")]
pub type Sampler = web_sys::GpuSampler;
#[cfg(target_arch = "wasm32")]
pub type BindGroup = web_sys::GpuBindGroup;
#[cfg(target_arch = "wasm32")]
pub type BindGroupLayout = web_sys::GpuBindGroupLayout;
#[cfg(target_arch = "wasm32")]
pub type PipelineLayout = web_sys::GpuPipelineLayout;
#[cfg(target_arch = "wasm32")]
pub type RenderPipeline = web_sys::GpuRenderPipeline;
#[cfg(target_arch = "wasm32")]
pub type ComputePipeline = web_sys::GpuComputePipeline;
#[cfg(target_arch = "wasm32")]
pub type ShaderModule = web_sys::GpuShaderModule;
#[cfg(target_arch = "wasm32")]
pub type CommandEncoder = web_sys::GpuCommandEncoder;
#[cfg(target_arch = "wasm32")]
pub type CommandBuffer = web_sys::GpuCommandBuffer;
#[cfg(target_arch = "wasm32")]
pub type RenderPass = web_sys::GpuRenderPassEncoder;
#[cfg(target_arch = "wasm32")]
pub type ComputePass = web_sys::GpuComputePassEncoder;

/// Backend-agnostic WebGPU context
pub struct WebGPUBackend {
    #[cfg(not(target_arch = "wasm32"))]
    pub instance: Instance,
    pub adapter: Arc<Adapter>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl WebGPUBackend {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self> {
        use log::{info, warn};
        
        // CRITICAL: Exclude Vulkan backend to prevent infinite recursion!
        // Our ICD translates Vulkan->WebGPU, so wgpu must use native backends (DX12, Metal, etc.)
        // Note: GL backend can hang when called from within GL context, so only use DX12/Metal
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12 | wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            // Do NOT force fallback: DX12/Metal backends already prevent Vulkan recursion.
            // WARP is ~100x slower than a real GPU and would make rendering useless.
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(VkError::AdapterNotAvailable)?;

        // Log adapter features and limits
        let adapter_features = adapter.features();
        let adapter_limits = adapter.limits();
        info!("WebGPU Adapter: {}", adapter.get_info().name);
        info!("Available features: {:?}", adapter_features);
        info!("Adapter limits - max_push_constant_size: {}", adapter_limits.max_push_constant_size);

        // Request only features that are available
        let mut required_features = wgpu::Features::empty();
        
        if adapter_features.contains(wgpu::Features::PUSH_CONSTANTS) {
            required_features |= wgpu::Features::PUSH_CONSTANTS;
            info!("PUSH_CONSTANTS supported");
        } else {
            warn!("PUSH_CONSTANTS not supported, emulation will be used");
        }
        
        if adapter_features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        
        if adapter_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
            required_features |= wgpu::Features::MULTI_DRAW_INDIRECT;
        }
        
        if adapter_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            required_features |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
        }

        // Calculate max push constant size based on adapter limits
        let max_push_constant_size = if adapter_features.contains(wgpu::Features::PUSH_CONSTANTS) {
            adapter_limits.max_push_constant_size.min(128)
        } else {
            0
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("VkWebGPU Device"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(), // Use minimal limits for compatibility
            },
            None,
        ))
        .map_err(|e| {
            info!("Device creation error: {}", e);
            VkError::DeviceCreationFailed(e.to_string())
        })?;

        info!("WebGPU device created successfully");

        Ok(Self {
            instance,
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new() -> Result<Self> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        let window = web_sys::window().ok_or(VkError::InitializationFailed(
            "No window object".to_string(),
        ))?;

        let navigator = window.navigator();
        let gpu = navigator
            .gpu()
            .ok_or(VkError::InitializationFailed(
                "WebGPU not supported".to_string(),
            ))?;

        let adapter_promise = gpu.request_adapter();
        let adapter_value = JsFuture::from(adapter_promise)
            .await
            .map_err(|e| VkError::AdapterNotAvailable)?;
        
        let adapter: web_sys::GpuAdapter = adapter_value
            .dyn_into()
            .map_err(|_| VkError::AdapterNotAvailable)?;

        let device_promise = adapter.request_device();
        let device_value = JsFuture::from(device_promise)
            .await
            .map_err(|e| VkError::DeviceCreationFailed(format!("{:?}", e)))?;

        let device: web_sys::GpuDevice = device_value
            .dyn_into()
            .map_err(|_| VkError::DeviceCreationFailed("Invalid device".to_string()))?;

        let queue = device.queue();

        Ok(Self {
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}
