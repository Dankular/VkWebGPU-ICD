# VkWebGPU-ICD Test Application

Simple Vulkan test app to verify the ICD works without needing the Vulkan SDK.

## Build and Run

```cmd
cd test_app
run_test.bat
```

Or manually:

```cmd
cd test_app
set VK_DRIVER_FILES=..\vkwebgpu_icd.json
cargo run --release
```

## What It Tests

1. **Vulkan instance creation** - Tests vkCreateInstance
2. **Physical device enumeration** - Tests vkEnumeratePhysicalDevices  
3. **Device properties** - Tests vkGetPhysicalDeviceProperties
4. **Extension queries** - Tests vkEnumerateDeviceExtensionProperties
5. **Logical device creation** - Tests vkCreateDevice

If all tests pass, the ICD is working correctly!

## Expected Output

```
VkWebGPU-ICD Test Application
==============================

1. Creating Vulkan instance...
   ✓ Instance created

2. Enumerating physical devices...
   Found 1 device(s)
   Device: WebGPU Adapter
   API Version: 1.3.0

3. Querying device extensions...
   Found X extensions:
     - VK_KHR_swapchain
     - ...

4. Creating logical device...
   ✓ Logical device created

✅ SUCCESS! VkWebGPU-ICD is working!

All basic Vulkan operations completed successfully.
The ICD can:
  ✓ Create instance
  ✓ Enumerate devices
  ✓ Query extensions
  ✓ Create logical device
```

## Requirements

- Rust toolchain (cargo)
- VkWebGPU-ICD built in parent directory
- No Vulkan SDK required!

## Notes

This is a minimal test application that verifies the ICD can handle basic Vulkan operations.
It does not perform actual rendering, but validates the core ICD functionality.
