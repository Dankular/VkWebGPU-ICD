# Testing Sauerbraten with VkWebGPU-ICD

## Issue: Sauerbraten Uses OpenGL, Not Vulkan

Sauerbraten is an OpenGL-based game, so it **won't use our Vulkan ICD** directly.

**Options to test with Sauerbraten:**

### Option 1: Use DXVK (OpenGL → Vulkan Translation)
DXVK also supports OpenGL → Vulkan translation via its GL implementation.

**Setup:**
1. Download DXVK latest release
2. Copy DXVK DLLs to Sauerbraten bin64 folder:
   - `dxvk_d3d9.dll`
   - `dxvk_d3d11.dll`
   - `dxvk_dxgi.dll`
   - `dxvk_opengl32.dll` ← For OpenGL games

3. Set environment to use our ICD:
   ```cmd
   cd "C:\Program Files (x86)\Sauerbraten\bin64"
   set VK_DRIVER_FILES=Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json
   set RUST_LOG=debug
   sauerbraten.exe
   ```

### Option 2: Test with Vulkan Applications First

**Better approach - test with native Vulkan apps:**

1. **vulkaninfo.exe** (from Vulkan SDK)
   - Queries ICD capabilities
   - Shows if ICD loads correctly

2. **vkcube.exe** (from Vulkan SDK)
   - Simple spinning cube
   - Tests basic rendering
   - Good first test!

3. **DXVK Test Apps**
   - Test DXVK → Vulkan → VkWebGPU → WebGPU chain
   - Closer to actual game scenario

4. **Then Sauerbraten + DXVK**
   - OpenGL → DXVK → Vulkan → VkWebGPU → WebGPU

## Recommended Testing Order

### 1. Quick ICD Test (5 minutes)
```cmd
cd Z:\source\Repos\VkWebGPU-ICD
test_icd.bat
vulkaninfo > vulkaninfo_output.txt
```

**Expected:** ICD loads, shows "VkWebGPU" device, lists extensions

### 2. Simple Rendering Test (10 minutes)
Download Vulkan SDK, then:
```cmd
test_icd.bat
vkcube
```

**Expected:** Spinning cube rendered via WebGPU backend

### 3. DXVK Test (if vkcube works)
Get DXVK, create test app, verify Vulkan translation

### 4. Sauerbraten + DXVK (final test)
Combine DXVK OpenGL wrapper with our Vulkan ICD

## Alternative: Simple Vulkan Test App

We could create a minimal Vulkan test app that:
- Creates instance
- Enumerates devices
- Creates swapchain
- Renders a triangle
- Uses push constants

This would directly test our ICD without needing Vulkan SDK.

## Current Best Option

**Run vulkaninfo first:**
```cmd
cd Z:\source\Repos\VkWebGPU-ICD
test_icd.bat

REM If vulkaninfo not installed:
REM Download from: https://vulkan.lunarg.com/sdk/home
REM Or use vkcube from Vulkan SDK
```

This will show if the ICD loads and what capabilities it reports.

Would you like to:
1. Download Vulkan SDK and test with vulkaninfo/vkcube?
2. Create a simple Vulkan test application?
3. Set up DXVK for Sauerbraten testing?
