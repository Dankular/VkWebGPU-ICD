# Sauerbraten Setup Guide - OpenGL → Vulkan → WebGPU

## Overview

This guide will help you run Sauerbraten using the VkWebGPU ICD via Mesa Zink (OpenGL → Vulkan translation).

## Prerequisites

- ✅ Windows system
- ✅ Sauerbraten installed (already present at `C:\Program Files (x86)\Sauerbraten`)
- ✅ VkWebGPU ICD built and working (already done!)
- ⏳ Mesa3D with Zink driver (to be downloaded)

## Step-by-Step Setup

### Step 1: Download Mesa3D with Zink

1. Visit: https://github.com/pal1000/mesa-dist-win/releases
2. Download the latest release (example: `mesa3d-24.x.x-release-msvc.7z`)
3. Extract the archive to a temporary location

### Step 2: Install Zink DLLs

Copy the following files from the extracted Mesa archive to `C:\Program Files (x86)\Sauerbraten\bin64\`:

**Required files (from the `x64` folder in Mesa archive):**
- `opengl32.dll` (This replaces Windows OpenGL with Mesa Zink)
- `libgallium_wgl.dll`
- `libglapi.dll`

**Possibly required (copy if present):**
- `zlib1.dll`
- `libssp-0.dll`
- `dxil.dll`
- `dxcompiler.dll`

**Note**: The exact DLL list may vary depending on Mesa version. If Sauerbraten doesn't launch, copy all `.dll` files from the `x64` folder.

### Step 3: Backup Original DLLs (Optional but Recommended)

Before copying Mesa DLLs:
```powershell
cd "C:\Program Files (x86)\Sauerbraten\bin64"
mkdir backup_dlls
# If opengl32.dll exists (unlikely):
copy opengl32.dll backup_dlls\
```

### Step 4: Launch Sauerbraten with WebGPU

Run the launch script:
```powershell
cd Z:\source\Repos\VkWebGPU-ICD
.\launch_sauerbraten_webgpu.ps1
```

Or manually:
```powershell
$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
$env:RUST_LOG = "info"
cd "C:\Program Files (x86)\Sauerbraten\bin64"
.\sauerbraten.exe
```

## Expected Behavior

### Success
- Sauerbraten window opens
- Main menu is visible and responsive
- Game renders correctly
- You can start a game and play

### Partial Success
- Game launches but has rendering artifacts
- Performance is lower than normal
- Some effects don't work

### Failure Modes

#### Error: "opengl32.dll not found"
**Solution**: Copy Mesa Zink DLLs as described in Step 2

#### Error: "The procedure entry point...could not be located"
**Solution**: You're missing dependent DLLs. Copy all DLLs from Mesa's x64 folder

#### Sauerbraten crashes immediately
**Possible causes**:
1. Zink requires Vulkan 1.3 (our ICD is 1.2)
2. Missing Vulkan extensions
3. Incompatible Mesa version

**Debug steps**:
1. Run with `$env:RUST_LOG = "debug"` for verbose output
2. Check console for Vulkan errors
3. Try different Mesa version

#### Black screen / No rendering
**Possible causes**:
1. Rendering commands not implemented in VkWebGPU
2. Format conversion issues
3. Swapchain problems

**Debug steps**:
1. Check logs for unimplemented functions
2. Test with simpler OpenGL app first
3. Enable debug logging in Mesa: `$env:MESA_DEBUG = "1"`

## Debugging Tools

### Enable Maximum Logging
```powershell
$env:RUST_LOG = "debug"
$env:MESA_DEBUG = "1"
$env:VK_LOADER_DEBUG = "all"
$env:GALLIUM_HUD = "fps"  # Show FPS overlay
```

### Test with Simple OpenGL App
Before Sauerbraten, test Zink works:
```powershell
# If you have glxgears or similar OpenGL test app:
$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
glxgears  # Or any simple OpenGL program
```

## Troubleshooting

### Issue: Vulkan Version Mismatch
**Symptom**: Zink says "Vulkan 1.3 required"
**Solution**: Upgrade VkWebGPU ICD to report version 1.3
- Edit `vkwebgpu/src/icd.rs`
- Change `VK_API_VERSION_1_2` to `VK_API_VERSION_1_3`
- Rebuild ICD

### Issue: Missing Vulkan Extensions
**Symptom**: Errors about missing extensions
**Solution**: Implement the required extensions in VkWebGPU
- Check which extensions are needed
- Add stub implementations if needed
- Full implementation if used by Zink

### Issue: Performance Too Low
**Symptom**: Game runs but < 15 FPS
**Why**: Multiple translation layers add overhead
**Solutions**:
- Use hardware acceleration (not Basic Render Driver)
- Optimize VkWebGPU command replay
- Profile and find bottlenecks

## Alternative Testing

If Sauerbraten doesn't work immediately:

### Test 1: Simple Vulkan App
```powershell
# Already working!
cd Z:\source\Repos\VkWebGPU-ICD\test_app
.\run_test_clean.ps1
```

### Test 2: vulkaninfo
```powershell
# Already tested!
.\test_vulkaninfo.ps1
```

### Test 3: Different OpenGL Game
Try a simpler OpenGL game first:
- SuperTuxKart (if available)
- Any OpenGL 3.3 game
- GLUT demo programs

## Success Metrics

- [ ] Mesa Zink DLLs installed
- [ ] Sauerbraten launches without crashing  
- [ ] Main menu renders correctly
- [ ] Can start a game
- [ ] Game is playable (>20 FPS)
- [ ] No critical rendering bugs

## Next Steps After Success

1. Document what worked
2. Test different game modes
3. Profile performance
4. Optimize bottlenecks
5. Test other OpenGL games
6. Eventually: Enter the Gungeon via DXVK!

## Resources

- Mesa3D Windows: https://github.com/pal1000/mesa-dist-win
- Zink Documentation: https://docs.mesa3d.org/drivers/zink.html
- VkWebGPU Source: Z:\source\Repos\VkWebGPU-ICD
- Our Discord/Issues: (add link if you have one)
