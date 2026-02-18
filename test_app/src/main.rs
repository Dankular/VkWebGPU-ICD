use ash::vk;
use std::ffi::CStr;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

fn main() {
    println!("VkWebGPU-ICD Test Application");
    println!("==============================\n");

    // Create window
    let event_loop = EventLoop::new().unwrap();
    let _window = WindowBuilder::new()
        .with_title("VkWebGPU Test")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    // Create Vulkan instance
    println!("1. Creating Vulkan instance...");
    let entry = unsafe { ash::Entry::load().unwrap() };

    let app_info = vk::ApplicationInfo::default()
        .application_name(CStr::from_bytes_with_nul(b"VkWebGPU Test\0").unwrap())
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul(b"No Engine\0").unwrap())
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_3);

    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create Vulkan instance")
    };
    println!("   ✓ Instance created");

    // Enumerate physical devices
    println!("\n2. Enumerating physical devices...");
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
    };
    println!("   Found {} device(s)", physical_devices.len());

    if physical_devices.is_empty() {
        println!("   ✗ ERROR: No Vulkan devices found!");
        println!("   Make sure VK_DRIVER_FILES is set correctly");
        return;
    }

    let physical_device = physical_devices[0];

    // Get device properties
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };

    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };
    println!("   Device: {}", device_name);
    println!(
        "   API Version: {}.{}.{}",
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version),
        vk::api_version_patch(properties.api_version)
    );

    // Enumerate device extensions
    println!("\n3. Querying device extensions...");
    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to enumerate extensions")
    };
    println!("   Found {} extensions:", extensions.len());
    for ext in extensions.iter().take(5) {
        let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
        println!("     - {}", name.to_string_lossy());
    }
    if extensions.len() > 5 {
        println!("     ... and {} more", extensions.len() - 5);
    }

    // Create logical device
    println!("\n4. Creating logical device...");
    let queue_family_index = 0; // Assuming first queue family

    let queue_priorities = [1.0];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info));

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical device")
    };
    println!("   ✓ Logical device created");

    // Success!
    println!("\n✅ SUCCESS! VkWebGPU-ICD is working!");
    println!("\nAll basic Vulkan operations completed successfully.");
    println!("The ICD can:");
    println!("  ✓ Create instance");
    println!("  ✓ Enumerate devices");
    println!("  ✓ Query extensions");
    println!("  ✓ Create logical device");

    // Cleanup
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    println!("\nPress Ctrl+C to exit or close window...");

    // Simple event loop
    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            _ => {}
        })
        .unwrap();
}
