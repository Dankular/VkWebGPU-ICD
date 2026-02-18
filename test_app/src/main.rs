// VkWebGPU-ICD end-to-end triangle test
// vkCreateInstance → Win32 surface → device → swapchain → render pass →
// graphics pipeline → command buffers → vkCmdDraw → vkQueuePresentKHR

use std::ffi::CString;

use ash::vk;
use winit::{
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

// SPIR-V blobs compiled at build time from WGSL by build.rs
static VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vert.spv"));
static FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv"));

fn bytes_to_spv_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V not 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── App state ────────────────────────────────────────────────────────────────

struct App {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    _physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    _queue_family_index: u32,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    _swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    _swapchain_extent: vk::Extent2D,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight_fence: vk::Fence,

    frame_count: u64,
}

impl App {
    unsafe fn new(window: &winit::window::Window) -> Self {
        // ── Entry & instance ──────────────────────────────────────────────
        let entry = ash::Entry::load().expect("Failed to load vulkan-1.dll");

        let app_name = CString::new("VkTriangleTest").unwrap();
        let engine_name = CString::new("VkWebGPU-ICD test").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_1);

        let ext_names = [
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];

        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(&ext_names),
                None,
            )
            .expect("vkCreateInstance failed");
        log::info!("Instance created");

        // ── Win32 surface ─────────────────────────────────────────────────
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
        let (hwnd, hinstance) = match window.window_handle().unwrap().as_raw() {
            RawWindowHandle::Win32(h) => {
                // ash 0.38: HWND = HINSTANCE = isize (matches NonZero<isize>::get())
                let hwnd: vk::HWND = h.hwnd.get();
                let hinstance: vk::HINSTANCE = h.hinstance.map(|i| i.get()).unwrap_or(0);
                (hwnd, hinstance)
            }
            _ => panic!("Expected Win32 window handle"),
        };

        let win32_surface = ash::khr::win32_surface::Instance::new(&entry, &instance);
        let surface = win32_surface
            .create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR::default()
                    .hwnd(hwnd)
                    .hinstance(hinstance),
                None,
            )
            .expect("vkCreateWin32SurfaceKHR failed");
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        log::info!("Surface created");

        // ── Physical device ───────────────────────────────────────────────
        let phys = instance
            .enumerate_physical_devices()
            .expect("vkEnumeratePhysicalDevices failed");
        assert!(!phys.is_empty(), "No physical devices");

        let (physical_device, qfi) = phys
            .iter()
            .find_map(|&pd| {
                let qfams = instance.get_physical_device_queue_family_properties(pd);
                qfams.iter().enumerate().find_map(|(i, qf)| {
                    let gfx = qf.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                    let pres = surface_loader
                        .get_physical_device_surface_support(pd, i as u32, surface)
                        .unwrap_or(false);
                    if gfx && pres { Some((pd, i as u32)) } else { None }
                })
            })
            .expect("No physical device with graphics+present");

        {
            use std::ffi::CStr;
            let props = instance.get_physical_device_properties(physical_device);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            log::info!("Device: {} (queue family {})", name, qfi);
        }

        // ── Logical device ────────────────────────────────────────────────
        let queue_prio = [1.0_f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&queue_prio);
        let dev_exts = [ash::khr::swapchain::NAME.as_ptr()];

        let device = instance
            .create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(std::slice::from_ref(&queue_ci))
                    .enabled_extension_names(&dev_exts),
                None,
            )
            .expect("vkCreateDevice failed");
        let graphics_queue = device.get_device_queue(qfi, 0);
        log::info!("Logical device created");

        // ── Swapchain ─────────────────────────────────────────────────────
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let caps = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .expect("surface capabilities");
        let fmts = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("surface formats");
        let pmodes = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .expect("present modes");

        let fmt = fmts
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| fmts.iter().find(|f| f.format == vk::Format::B8G8R8A8_UNORM))
            .unwrap_or(&fmts[0]);

        let pmode = pmodes
            .iter()
            .find(|&&m| m == vk::PresentModeKHR::MAILBOX)
            .copied()
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let inner = window.inner_size();
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: inner.width.clamp(
                    caps.min_image_extent.width,
                    caps.max_image_extent.width,
                ),
                height: inner.height.clamp(
                    caps.min_image_extent.height,
                    caps.max_image_extent.height,
                ),
            }
        };

        let img_count = (caps.min_image_count + 1)
            .min(if caps.max_image_count > 0 { caps.max_image_count } else { u32::MAX });

        let swapchain = swapchain_loader
            .create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(surface)
                    .min_image_count(img_count)
                    .image_format(fmt.format)
                    .image_color_space(fmt.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(pmode)
                    .clipped(true),
                None,
            )
            .expect("vkCreateSwapchainKHR failed");

        let swapchain_images = swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("vkGetSwapchainImagesKHR failed");

        log::info!(
            "Swapchain: {}x{}, {} images, {:?}",
            extent.width, extent.height, swapchain_images.len(), fmt.format
        );

        // ── Image views ───────────────────────────────────────────────────
        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|&img| {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(img)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(fmt.format)
                            .components(vk::ComponentMapping {
                                r: vk::ComponentSwizzle::IDENTITY,
                                g: vk::ComponentSwizzle::IDENTITY,
                                b: vk::ComponentSwizzle::IDENTITY,
                                a: vk::ComponentSwizzle::IDENTITY,
                            })
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        None,
                    )
                    .expect("vkCreateImageView failed")
            })
            .collect();

        // ── Render pass ───────────────────────────────────────────────────
        let color_att = vk::AttachmentDescription::default()
            .format(fmt.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref));
        let dep = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        };

        let render_pass = device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(&color_att))
                    .subpasses(std::slice::from_ref(&subpass))
                    .dependencies(std::slice::from_ref(&dep)),
                None,
            )
            .expect("vkCreateRenderPass failed");

        // ── Shaders ───────────────────────────────────────────────────────
        let vert_words = bytes_to_spv_words(VERT_SPV);
        let frag_words = bytes_to_spv_words(FRAG_SPV);
        let vert_mod = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&vert_words),
                None,
            )
            .expect("vert shader module");
        let frag_mod = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_words),
                None,
            )
            .expect("frag shader module");

        // ── Pipeline ──────────────────────────────────────────────────────
        let pipeline_layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .expect("pipeline layout");

        let entry_vert = CString::new("vs_main").unwrap();
        let entry_frag = CString::new("fs_main").unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_mod)
                .name(&entry_vert),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_mod)
                .name(&entry_frag),
        ];

        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let blend_att = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };

        let pipeline = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                    .input_assembly_state(
                        &vk::PipelineInputAssemblyStateCreateInfo::default()
                            .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                    )
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::default()
                            .viewport_count(1)
                            .scissor_count(1),
                    )
                    .rasterization_state(
                        &vk::PipelineRasterizationStateCreateInfo::default()
                            .polygon_mode(vk::PolygonMode::FILL)
                            .line_width(1.0)
                            .cull_mode(vk::CullModeFlags::NONE)
                            .front_face(vk::FrontFace::CLOCKWISE),
                    )
                    .multisample_state(
                        &vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                    )
                    .color_blend_state(
                        &vk::PipelineColorBlendStateCreateInfo::default()
                            .attachments(std::slice::from_ref(&blend_att)),
                    )
                    .dynamic_state(
                        &vk::PipelineDynamicStateCreateInfo::default()
                            .dynamic_states(&dyn_states),
                    )
                    .layout(pipeline_layout)
                    .render_pass(render_pass)
                    .subpass(0)],
                None,
            )
            .expect("vkCreateGraphicsPipelines")[0];

        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
        log::info!("Graphics pipeline created");

        // ── Framebuffers ──────────────────────────────────────────────────
        let framebuffers: Vec<_> = swapchain_image_views
            .iter()
            .map(|&view| {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(std::slice::from_ref(&view))
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .expect("vkCreateFramebuffer failed")
            })
            .collect();

        // ── Command pool & buffers ────────────────────────────────────────
        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(qfi),
                None,
            )
            .expect("command pool");

        let command_buffers = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(swapchain_images.len() as u32),
            )
            .expect("command buffers");

        for (i, &cb) in command_buffers.iter().enumerate() {
            record_triangle(&device, cb, render_pass, framebuffers[i], pipeline, extent);
        }

        // ── Sync ──────────────────────────────────────────────────────────
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let image_available = device.create_semaphore(&sem_ci, None).expect("semaphore");
        let render_finished = device.create_semaphore(&sem_ci, None).expect("semaphore");
        let in_flight_fence = device
            .create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .expect("fence");

        log::info!("Ready — orange triangle on dark blue. Close window or press Esc to exit.");

        Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            _physical_device: physical_device,
            device,
            graphics_queue,
            _queue_family_index: qfi,
            swapchain_loader,
            swapchain,
            _swapchain_images: swapchain_images,
            swapchain_image_views,
            _swapchain_extent: extent,
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight_fence,
            frame_count: 0,
        }
    }

    unsafe fn draw_frame(&mut self) {
        self.device
            .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
            .unwrap();
        self.device.reset_fences(&[self.in_flight_fence]).unwrap();

        let (img_idx, _) = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
            .expect("acquire_next_image");

        let cb = self.command_buffers[img_idx as usize];
        let wait_sems = [self.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_sems = [self.render_finished];

        self.device
            .queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&wait_sems)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(std::slice::from_ref(&cb))
                    .signal_semaphores(&signal_sems)],
                self.in_flight_fence,
            )
            .expect("vkQueueSubmit");

        self.swapchain_loader
            .queue_present(
                self.graphics_queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal_sems)
                    .swapchains(std::slice::from_ref(&self.swapchain))
                    .image_indices(std::slice::from_ref(&img_idx)),
            )
            .expect("vkQueuePresentKHR");

        self.frame_count += 1;
        if self.frame_count % 120 == 0 {
            log::info!("Frame {}", self.frame_count);
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_semaphore(self.render_finished, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_command_pool(self.command_pool, None);
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
        log::info!("Cleaned up after {} frames", self.frame_count);
    }
}

unsafe fn record_triangle(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline: vk::Pipeline,
    extent: vk::Extent2D,
) {
    device
        .begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
        )
        .unwrap();

    device.cmd_begin_render_pass(
        cb,
        &vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&[vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.2, 1.0], // dark blue
                },
            }]),
        vk::SubpassContents::INLINE,
    );

    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);

    device.cmd_set_viewport(
        cb,
        0,
        &[vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
    );
    device.cmd_set_scissor(
        cb,
        0,
        &[vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        }],
    );

    device.cmd_draw(cb, 3, 1, 0, 0); // 3 verts from shader, 1 instance

    device.cmd_end_render_pass(cb);
    device.end_command_buffer(cb).unwrap();
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("VkWebGPU-ICD Triangle Test");

    // winit 0.29: EventLoop::new() returns Result<EventLoop<()>, EventLoopError>
    let event_loop = EventLoop::new().expect("EventLoop::new failed");

    let window = WindowBuilder::new()
        .with_title("VkWebGPU-ICD Triangle Test")
        .with_inner_size(winit::dpi::LogicalSize::new(800u32, 600u32))
        .build(&event_loop)
        .expect("Window creation failed");

    let mut app = unsafe { App::new(&window) };

    // winit 0.29: closure is FnMut(Event<T>, &EventLoopWindowTarget<T>)
    // Control flow set via elwt.set_control_flow(); exit via elwt.exit()
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { event: key_event, .. },
                    ..
                } => {
                    if key_event.state == ElementState::Pressed {
                        if key_event.logical_key == Key::Named(NamedKey::Escape) {
                            elwt.exit();
                        }
                    }
                }
                Event::AboutToWait => {
                    // winit 0.29 equivalent of MainEventsCleared → drive render loop
                    unsafe { app.draw_frame() };
                }
                _ => {}
            }
        })
        .expect("event loop error");
}
