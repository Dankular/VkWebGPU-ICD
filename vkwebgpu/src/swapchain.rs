//! Vulkan Swapchain implementation (KHR extension)
//!
//! Swapchain images are backed by real wgpu Textures so that vkCreateImageView,
//! framebuffer creation, and render passes all work correctly.
//!
//! Presentation:
//!   On Win32 targets a real wgpu::Surface is created from the HWND stored in
//!   VkSurfaceKHR.  At vkQueuePresentKHR time each offscreen swapchain texture
//!   is blitted into the surface's current frame texture and presented.
//!
//!   If surface creation fails (headless, CheerpX/browser, etc.) frames are
//!   rendered into offscreen GPU textures and the present is a silent no-op.

use ash::vk::{self, Handle};
use log::{debug, warn};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::format;
use crate::handle::HandleAllocator;
use crate::image;
use crate::surface;

pub static SWAPCHAIN_ALLOCATOR: Lazy<HandleAllocator<VkSwapchainData>> =
    Lazy::new(|| HandleAllocator::new());

// ---------------------------------------------------------------------------
// Win32 window-handle wrapper for wgpu::Surface creation
// ---------------------------------------------------------------------------

/// A zero-overhead wrapper around a Win32 HWND + HINSTANCE that implements the
/// raw-window-handle traits required by wgpu::Instance::create_surface().
///
/// # Safety
/// The caller (the Vulkan app) is responsible for keeping the Win32 window alive
/// for the duration of the swapchain.  We only store integer copies of the handles.
#[cfg(not(target_arch = "wasm32"))]
struct Win32SurfaceWindow {
    hwnd: std::num::NonZeroIsize,
    hinstance: std::num::NonZeroIsize,
}

#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for Win32SurfaceWindow {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for Win32SurfaceWindow {}

#[cfg(not(target_arch = "wasm32"))]
impl wgpu::rwh::HasWindowHandle for Win32SurfaceWindow {
    fn window_handle(
        &self,
    ) -> std::result::Result<wgpu::rwh::WindowHandle<'_>, wgpu::rwh::HandleError> {
        let mut handle = wgpu::rwh::Win32WindowHandle::new(self.hwnd);
        handle.hinstance = Some(self.hinstance);
        // SAFETY: The Win32 window is guaranteed to outlive this surface by the
        // Vulkan application (it created the VkSurface from its own HWND).
        Ok(unsafe {
            wgpu::rwh::WindowHandle::borrow_raw(wgpu::rwh::RawWindowHandle::Win32(handle))
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl wgpu::rwh::HasDisplayHandle for Win32SurfaceWindow {
    fn display_handle(
        &self,
    ) -> std::result::Result<wgpu::rwh::DisplayHandle<'_>, wgpu::rwh::HandleError> {
        Ok(unsafe {
            wgpu::rwh::DisplayHandle::borrow_raw(wgpu::rwh::RawDisplayHandle::Windows(
                wgpu::rwh::WindowsDisplayHandle::new(),
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Swapchain data
// ---------------------------------------------------------------------------

pub struct VkSwapchainData {
    pub device: vk::Device,
    pub surface: vk::SurfaceKHR,
    pub image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,

    /// Real VkImage handles backed by wgpu Textures (registered in IMAGE_ALLOCATOR).
    pub images: Vec<vk::Image>,

    /// Tracks which image index was most recently acquired.
    pub current_image_index: AtomicU32,

    /// Real wgpu::Surface for Win32 window presentation (None in headless/browser mode).
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_surface: Option<wgpu::Surface<'static>>,

    /// Format actually configured on the wgpu surface (may differ from the Vulkan
    /// swapchain format if the surface didn't support it directly).
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_surface_format: wgpu::TextureFormat,

    /// Present blit pipeline: used when the surface format differs from the
    /// offscreen texture format and a copy_texture_to_texture is not valid.
    #[cfg(not(target_arch = "wasm32"))]
    pub blit_pipeline: Option<Arc<wgpu::RenderPipeline>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub blit_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub blit_sampler: Option<Arc<wgpu::Sampler>>,
}

// wgpu::Surface<'static> and wgpu::RenderPipeline are Send + Sync.
unsafe impl Send for VkSwapchainData {}
unsafe impl Sync for VkSwapchainData {}

/// Helper function to get swapchain data from handle
unsafe fn get_swapchain_data(swapchain: vk::SwapchainKHR) -> Result<Arc<VkSwapchainData>> {
    SWAPCHAIN_ALLOCATOR
        .get(swapchain.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid swapchain handle".to_string()))
}

// ---------------------------------------------------------------------------
// WGSL present-blit shader
// ---------------------------------------------------------------------------
// A fullscreen triangle that samples from the offscreen texture (binding 0)
// and outputs to the render attachment (the wgpu surface texture).
// This handles format-converting copies (e.g. sRGB ↔ UNORM) correctly.

#[cfg(not(target_arch = "wasm32"))]
const PRESENT_BLIT_WGSL: &str = r#"
struct VertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle.  Three vertices cover the entire [-1,1]^2 NDC viewport.
// UV mapping: u = (ndc_x + 1) / 2, v = (1 - ndc_y) / 2  (Y-flipped for tex).
@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> VertOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VertOut;
    o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
    o.uv  = uv[vi];
    return o;
}

@group(0) @binding(0) var src_tex:  texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

@fragment
fn fs_blit(in: VertOut) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_samp, in.uv);
}
"#;

// ---------------------------------------------------------------------------
// create_swapchain_khr
// ---------------------------------------------------------------------------

pub unsafe fn create_swapchain_khr(
    device: vk::Device,
    p_create_info: *const vk::SwapchainCreateInfoKHR,
    _p_allocator: *const vk::AllocationCallbacks,
    p_swapchain: *mut vk::SwapchainKHR,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating swapchain: {}x{}, format={:?}, min_image_count={}",
        create_info.image_extent.width,
        create_info.image_extent.height,
        create_info.image_format,
        create_info.min_image_count
    );

    // Clamp image count to a sensible range.
    let image_count = create_info.min_image_count.clamp(2, 8).max(3);

    let device_data = device::get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    // Map the Vulkan format to a wgpu TextureFormat for the offscreen render targets.
    let wgpu_format = format::vk_to_wgpu_format(create_info.image_format)
        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);

    // Build wgpu texture usage flags from the application request.
    let wgpu_usage = {
        let mut u = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING;
        if create_info
            .image_usage
            .contains(vk::ImageUsageFlags::STORAGE)
        {
            u |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        u
    };

    // Create one real wgpu Texture per swapchain image and register each as a VkImage.
    let mut images: Vec<vk::Image> = Vec::with_capacity(image_count as usize);
    for idx in 0..image_count {
        let texture = device_data.backend.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("SwapchainImage[{}]", idx)),
            size: wgpu::Extent3d {
                width: create_info.image_extent.width,
                height: create_info.image_extent.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu_usage,
            view_formats: &[],
        });

        let vk_image = image::create_swapchain_image(
            device,
            create_info.image_format,
            create_info.image_extent,
            texture,
        );
        images.push(vk_image);
        debug!("Created swapchain image[{}]: {:?}", idx, vk_image);
    }

    // --- Real wgpu::Surface for Win32 presentation ---
    #[cfg(not(target_arch = "wasm32"))]
    let (wgpu_surface, wgpu_surface_format, blit_pipeline, blit_bind_group_layout, blit_sampler) =
        create_wgpu_surface(create_info, &device_data.backend, wgpu_format);

    let swapchain_data = VkSwapchainData {
        device,
        surface: create_info.surface,
        image_count,
        image_format: create_info.image_format,
        image_color_space: create_info.image_color_space,
        image_extent: create_info.image_extent,
        image_array_layers: create_info.image_array_layers,
        image_usage: create_info.image_usage,
        pre_transform: create_info.pre_transform,
        composite_alpha: create_info.composite_alpha,
        present_mode: create_info.present_mode,
        images,
        current_image_index: AtomicU32::new(0),
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_surface,
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_surface_format,
        #[cfg(not(target_arch = "wasm32"))]
        blit_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        blit_bind_group_layout,
        #[cfg(not(target_arch = "wasm32"))]
        blit_sampler,
    };

    let swapchain_handle = SWAPCHAIN_ALLOCATOR.allocate(swapchain_data);
    *p_swapchain = Handle::from_raw(swapchain_handle);

    debug!("Created swapchain with {} wgpu-backed images", image_count);
    Ok(())
}

/// Attempt to create a real wgpu::Surface from the Win32 HWND stored in the
/// VkSurfaceKHR, configure it, and (if the surface format differs from the
/// offscreen texture format) build a present-blit render pipeline.
///
/// Returns a tuple: (surface, surface_format, blit_pipeline, blit_layout, blit_sampler).
/// All values are None on failure or in headless/browser mode.
#[cfg(not(target_arch = "wasm32"))]
fn create_wgpu_surface(
    create_info: &vk::SwapchainCreateInfoKHR,
    backend: &crate::backend::WebGPUBackend,
    offscreen_format: wgpu::TextureFormat,
) -> (
    Option<wgpu::Surface<'static>>,
    wgpu::TextureFormat,
    Option<Arc<wgpu::RenderPipeline>>,
    Option<Arc<wgpu::BindGroupLayout>>,
    Option<Arc<wgpu::Sampler>>,
) {
    use std::num::NonZeroIsize;

    // Retrieve the HWND from the VkSurfaceKHR.
    let surf_data = match surface::get_surface_data(create_info.surface) {
        Some(d) => d,
        None => return (None, offscreen_format, None, None, None),
    };

    let hwnd = match NonZeroIsize::new(surf_data.hwnd as isize) {
        Some(h) => h,
        None => return (None, offscreen_format, None, None, None),
    };
    let hinstance = NonZeroIsize::new(surf_data.hinstance as isize)
        .unwrap_or_else(|| NonZeroIsize::new(1).unwrap());

    let window = Win32SurfaceWindow { hwnd, hinstance };

    // Create the wgpu::Surface from the HWND.
    // SAFETY: Win32SurfaceWindow holds valid Win32 handles for the lifetime of
    // the swapchain (the app keeps its window alive throughout).
    let wgpu_surf = match backend.instance.create_surface(window) {
        Ok(s) => s,
        Err(e) => {
            warn!("create_surface from HWND failed: {e}");
            return (None, offscreen_format, None, None, None);
        }
    };

    // Pick the best surface format.
    let caps = wgpu_surf.get_capabilities(&backend.adapter);
    debug!("Surface capabilities: formats={:?}", caps.formats);

    let surface_format = if caps.formats.contains(&offscreen_format) {
        offscreen_format
    } else {
        // Prefer the sRGB / non-sRGB sibling of the requested format.
        let sibling = sibling_format(offscreen_format);
        if caps.formats.contains(&sibling) {
            sibling
        } else {
            // Fall back to the first available surface format.
            caps.formats
                .first()
                .copied()
                .unwrap_or(wgpu::TextureFormat::Bgra8Unorm)
        }
    };

    // Configure the surface.
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
        format: surface_format,
        width: create_info.image_extent.width,
        height: create_info.image_extent.height,
        present_mode: vk_present_mode_to_wgpu(create_info.present_mode),
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    wgpu_surf.configure(&backend.device, &config);

    debug!(
        "Configured wgpu surface: format={:?}, present_mode={:?}",
        surface_format, config.present_mode
    );

    // If the surface format matches the offscreen format we can use
    // copy_texture_to_texture at present time (fast path, no shader needed).
    // If they differ we build a blit render pipeline for the slow path.
    let (blit_pipeline, blit_layout, blit_sampler) = if surface_format == offscreen_format {
        (None, None, None)
    } else {
        debug!(
            "Surface format {:?} != offscreen format {:?}: building present-blit pipeline",
            surface_format, offscreen_format
        );
        build_blit_pipeline(&backend.device, offscreen_format, surface_format)
    };

    (
        Some(wgpu_surf),
        surface_format,
        blit_pipeline,
        blit_layout,
        blit_sampler,
    )
}

/// Return the sRGB ↔ linear sibling of a texture format (if one exists).
#[cfg(not(target_arch = "wasm32"))]
fn sibling_format(fmt: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match fmt {
        wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8UnormSrgb,
        other => other,
    }
}

/// Map a Vulkan present mode to the closest wgpu equivalent.
#[cfg(not(target_arch = "wasm32"))]
fn vk_present_mode_to_wgpu(mode: vk::PresentModeKHR) -> wgpu::PresentMode {
    match mode {
        vk::PresentModeKHR::IMMEDIATE => wgpu::PresentMode::Immediate,
        vk::PresentModeKHR::MAILBOX => wgpu::PresentMode::Mailbox,
        vk::PresentModeKHR::FIFO_RELAXED => wgpu::PresentMode::FifoRelaxed,
        _ => wgpu::PresentMode::Fifo,
    }
}

/// Build the render pipeline used for format-converting blits at present time.
///
/// The pipeline renders a fullscreen triangle that samples from `src_format`
/// and writes to `dst_format` (the surface).
#[cfg(not(target_arch = "wasm32"))]
fn build_blit_pipeline(
    device: &wgpu::Device,
    _src_format: wgpu::TextureFormat,
    dst_format: wgpu::TextureFormat,
) -> (
    Option<Arc<wgpu::RenderPipeline>>,
    Option<Arc<wgpu::BindGroupLayout>>,
    Option<Arc<wgpu::Sampler>>,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("present_blit"),
        source: wgpu::ShaderSource::Wgsl(PRESENT_BLIT_WGSL.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("present_blit_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("present_blit_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("present_blit_rp"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_blit",
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_blit",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: dst_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("present_blit_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    (
        Some(Arc::new(pipeline)),
        Some(Arc::new(bind_group_layout)),
        Some(Arc::new(sampler)),
    )
}

// ---------------------------------------------------------------------------
// destroy_swapchain_khr
// ---------------------------------------------------------------------------

pub unsafe fn destroy_swapchain_khr(
    swapchain: vk::SwapchainKHR,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if swapchain == vk::SwapchainKHR::null() {
        return;
    }

    // Unregister all backing VkImages before dropping the swapchain.
    if let Some(data) = SWAPCHAIN_ALLOCATOR.get(swapchain.as_raw()) {
        for &img in &data.images {
            image::destroy_swapchain_image(img);
        }
    }

    debug!("Destroying swapchain");
    SWAPCHAIN_ALLOCATOR.remove(swapchain.as_raw());
}

// ---------------------------------------------------------------------------
// get_swapchain_images_khr
// ---------------------------------------------------------------------------

pub unsafe fn get_swapchain_images_khr(
    swapchain: vk::SwapchainKHR,
    p_swapchain_image_count: *mut u32,
    p_swapchain_images: *mut vk::Image,
) -> Result<()> {
    let swapchain_data = get_swapchain_data(swapchain)?;

    if p_swapchain_images.is_null() {
        *p_swapchain_image_count = swapchain_data.image_count;
        debug!(
            "Querying swapchain image count: {}",
            swapchain_data.image_count
        );
    } else {
        let count = (*p_swapchain_image_count).min(swapchain_data.image_count);
        let images = std::slice::from_raw_parts_mut(p_swapchain_images, count as usize);
        images.copy_from_slice(&swapchain_data.images[..count as usize]);
        *p_swapchain_image_count = count;
        debug!("Retrieved {} swapchain images", count);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// acquire_next_image_khr
// ---------------------------------------------------------------------------

pub unsafe fn acquire_next_image_khr(
    swapchain: vk::SwapchainKHR,
    _timeout: u64,
    _semaphore: vk::Semaphore,
    _fence: vk::Fence,
    p_image_index: *mut u32,
) -> Result<()> {
    let swapchain_data = get_swapchain_data(swapchain)?;

    // Cycle through the available images in round-robin order.
    let current = swapchain_data.current_image_index.load(Ordering::Relaxed);
    let next = (current + 1) % swapchain_data.image_count;
    swapchain_data
        .current_image_index
        .store(next, Ordering::Relaxed);

    *p_image_index = next;

    debug!(
        "Acquired swapchain image index: {} ({}/{})",
        next,
        next + 1,
        swapchain_data.image_count
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// queue_present_khr
// ---------------------------------------------------------------------------

pub unsafe fn queue_present_khr(
    _queue: vk::Queue,
    p_present_info: *const vk::PresentInfoKHR,
) -> Result<()> {
    let present_info = &*p_present_info;

    if present_info.swapchain_count == 0 {
        debug!("Queue present called with 0 swapchains");
        return Ok(());
    }

    let swapchains = std::slice::from_raw_parts(
        present_info.p_swapchains,
        present_info.swapchain_count as usize,
    );
    let image_indices = std::slice::from_raw_parts(
        present_info.p_image_indices,
        present_info.swapchain_count as usize,
    );

    for (i, (&swapchain, &image_index)) in swapchains.iter().zip(image_indices.iter()).enumerate()
    {
        let swapchain_data = get_swapchain_data(swapchain)?;

        debug!(
            "Presenting swapchain {} image[{}] ({}x{} {:?})",
            i,
            image_index,
            swapchain_data.image_extent.width,
            swapchain_data.image_extent.height,
            swapchain_data.image_format
        );

        if image_index >= swapchain_data.image_count {
            return Err(VkError::InvalidHandle(format!(
                "Invalid image index: {} (max: {})",
                image_index, swapchain_data.image_count
            )));
        }

        // --- Real Win32 presentation ---
        #[cfg(not(target_arch = "wasm32"))]
        present_to_wgpu_surface(&swapchain_data, image_index as usize)?;

        if !present_info.p_results.is_null() {
            let results = std::slice::from_raw_parts_mut(
                present_info.p_results as *mut vk::Result,
                present_info.swapchain_count as usize,
            );
            results[i] = vk::Result::SUCCESS;
        }
    }

    debug!(
        "Queue present completed for {} swapchain(s)",
        present_info.swapchain_count
    );
    Ok(())
}

/// Blit the offscreen swapchain texture at `image_index` into the wgpu surface
/// texture and call `present()`.  Does nothing if there is no real surface.
#[cfg(not(target_arch = "wasm32"))]
fn present_to_wgpu_surface(
    swapchain_data: &VkSwapchainData,
    image_index: usize,
) -> Result<()> {
    let wgpu_surf = match &swapchain_data.wgpu_surface {
        Some(s) => s,
        None => return Ok(()), // headless / no surface
    };

    // Get the current surface texture.
    let surface_tex = match wgpu_surf.get_current_texture() {
        Ok(t) => t,
        Err(e) => {
            warn!("get_current_texture() failed: {:?}", e);
            return Ok(());
        }
    };

    // Get the offscreen VkImage texture for this swapchain image slot.
    let vk_img = swapchain_data.images[image_index];
    let img_data = match image::get_image_data(vk_img) {
        Some(d) => d,
        None => {
            warn!("present_to_wgpu_surface: offscreen image data missing");
            surface_tex.present();
            return Ok(());
        }
    };
    let img_guard = img_data.wgpu_texture.read();
    let offscreen = match img_guard.as_ref() {
        Some(t) => t,
        None => {
            warn!("present_to_wgpu_surface: offscreen texture not bound");
            surface_tex.present();
            return Ok(());
        }
    };

    // Get the backend (device + queue) from the VkDevice stored in swapchain.
    let device_data = match device::get_device_data(swapchain_data.device) {
        Some(d) => d,
        None => {
            warn!("present_to_wgpu_surface: invalid device");
            surface_tex.present();
            return Ok(());
        }
    };
    let backend = &device_data.backend;

    let mut encoder =
        backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PresentBlit"),
            });

    let extent = wgpu::Extent3d {
        width: swapchain_data.image_extent.width,
        height: swapchain_data.image_extent.height,
        depth_or_array_layers: 1,
    };

    if swapchain_data.blit_pipeline.is_none() {
        // Fast path: formats match — use copy_texture_to_texture.
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: offscreen.as_ref(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &surface_tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            extent,
        );
        debug!("Present: copy_texture_to_texture (fast path)");
    } else {
        // Slow path: render-pass blit (handles format conversion).
        let pipeline = swapchain_data.blit_pipeline.as_ref().unwrap();
        let layout = swapchain_data.blit_bind_group_layout.as_ref().unwrap();
        let sampler = swapchain_data.blit_sampler.as_ref().unwrap();

        let offscreen_view = offscreen.create_view(&wgpu::TextureViewDescriptor::default());
        let surface_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present_blit_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&offscreen_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present_blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        debug!("Present: render-pass blit (slow path, format conversion)");
    }

    backend.queue.submit([encoder.finish()]);
    surface_tex.present();

    Ok(())
}
