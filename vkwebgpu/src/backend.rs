//! WebGPU backend abstraction
//!
//! Provides a unified interface for both native (wgpu) and WASM (web-sys) WebGPU backends.
//!
//! Under the `webx` feature the backend is a zero-sized stub — no GPU objects are created
//! on the guest side; all real GPU work is delegated to the browser host via IPC.

use crate::error::{Result, VkError};
use std::sync::Arc;

// ── WebX feature: zero-sized stub backend ────────────────────────────────────
// All Vulkan GPU calls are forwarded to the browser host via x86 I/O port IPC.
// No wgpu types are used at runtime under this feature.

#[cfg(feature = "webx")]
pub struct WebGPUBackend {}

#[cfg(feature = "webx")]
impl WebGPUBackend {
    pub fn new() -> Result<Self> {
        log::info!("[WebX] Using WebX IPC backend (no local wgpu device)");
        Ok(Self {})
    }
}

// CommandBuffer placeholder so command_buffer.rs can still reference the type
// in its (conditionally compiled) replay_commands signature.
#[cfg(feature = "webx")]
pub type CommandBuffer = ();

// ── Non-webx: wgpu type aliases ───────────────────────────────────────────────
// These are only active when NOT building for the webx IPC target.

#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type Instance = wgpu::Instance;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type Adapter = wgpu::Adapter;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type Device = wgpu::Device;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type Queue = wgpu::Queue;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type Buffer = wgpu::Buffer;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type Texture = wgpu::Texture;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type TextureView = wgpu::TextureView;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type Sampler = wgpu::Sampler;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type BindGroup = wgpu::BindGroup;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type BindGroupLayout = wgpu::BindGroupLayout;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type PipelineLayout = wgpu::PipelineLayout;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type RenderPipeline = wgpu::RenderPipeline;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type ComputePipeline = wgpu::ComputePipeline;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type ShaderModule = wgpu::ShaderModule;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type CommandEncoder = wgpu::CommandEncoder;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type CommandBuffer = wgpu::CommandBuffer;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type RenderPass<'a> = wgpu::RenderPass<'a>;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
pub type ComputePass<'a> = wgpu::ComputePass<'a>;

#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type Instance = web_sys::Gpu;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type Adapter = web_sys::GpuAdapter;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type Device = web_sys::GpuDevice;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type Queue = web_sys::GpuQueue;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type Buffer = web_sys::GpuBuffer;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type Texture = web_sys::GpuTexture;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type TextureView = web_sys::GpuTextureView;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type Sampler = web_sys::GpuSampler;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type BindGroup = web_sys::GpuBindGroup;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type BindGroupLayout = web_sys::GpuBindGroupLayout;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type PipelineLayout = web_sys::GpuPipelineLayout;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type RenderPipeline = web_sys::GpuRenderPipeline;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type ComputePipeline = web_sys::GpuComputePipeline;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type ShaderModule = web_sys::GpuShaderModule;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type CommandEncoder = web_sys::GpuCommandEncoder;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type CommandBuffer = web_sys::GpuCommandBuffer;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type RenderPass = web_sys::GpuRenderPassEncoder;
#[cfg(all(target_arch = "wasm32", not(feature = "webx")))]
pub type ComputePass = web_sys::GpuComputePassEncoder;

// ── Non-webx: real WebGPU backend struct ──────────────────────────────────────

/// Backend-agnostic WebGPU context (only exists when NOT using the webx IPC feature)
#[cfg(not(feature = "webx"))]
pub struct WebGPUBackend {
    #[cfg(not(target_arch = "wasm32"))]
    pub instance: Instance,
    pub adapter: Arc<Adapter>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

#[cfg(not(feature = "webx"))]
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
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(VkError::AdapterNotAvailable)?;

        let adapter_features = adapter.features();
        let adapter_limits = adapter.limits();
        info!("WebGPU Adapter: {}", adapter.get_info().name);
        info!("Available features: {:?}", adapter_features);
        info!(
            "Adapter limits - max_push_constant_size: {}",
            adapter_limits.max_push_constant_size
        );

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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("VkWebGPU Device"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
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
        let gpu = navigator.gpu().ok_or(VkError::InitializationFailed(
            "WebGPU not supported".to_string(),
        ))?;

        let adapter_promise = gpu.request_adapter();
        let adapter_value = JsFuture::from(adapter_promise)
            .await
            .map_err(|_| VkError::AdapterNotAvailable)?;

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
