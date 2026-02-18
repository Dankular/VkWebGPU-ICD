# Sauerbraten Testing Plan

## Current Status

- ✅ VkWebGPU ICD is working and loads successfully
- ✅ vulkaninfo can enumerate our ICD
- ✅ Sauerbraten is installed at `C:\Program Files (x86)\Sauerbraten`
- ❌ Sauerbraten uses OpenGL, not Vulkan

## The Challenge

Sauerbraten → OpenGL → **MISSING LAYER** → Vulkan → VkWebGPU → WebGPU → DX12/Metal/GL

We need an OpenGL → Vulkan translation layer!

## Options for OpenGL → Vulkan Translation

### Option 1: Zink (Mesa's OpenGL-on-Vulkan) ⭐ RECOMMENDED
- **What it is**: Official Mesa driver that implements OpenGL via Vulkan
- **Status**: Available in Mesa 20.1+
- **Windows builds**: Available from https://github.com/pal1000/mesa-dist-win
- **How to use**:
  1. Download latest Mesa3D Windows build
  2. Extract `opengl32.dll` from the archive
  3. Place in Sauerbraten's bin64 folder
  4. Set environment to use our Vulkan ICD
  5. Run Sauerbraten

### Option 2: Build Our Own Simple OpenGL Wrapper
- Create minimal OpenGL → Vulkan wrapper for Sauerbraten's specific needs
- **Pros**: Full control, optimized for our use case
- **Cons**: Huge amount of work (OpenGL API is massive)
- **Status**: Not realistic for quick testing

### Option 3: Wait for Browser WebGL Support
- Run Sauerbraten compiled to WebAssembly
- Use WebGL → WebGPU in browser
- **Status**: Different approach entirely

## Recommended Next Steps

### Step 1: Download Mesa3D with Zink (5 minutes)
```powershell
# Download from https://github.com/pal1000/mesa-dist-win/releases
# Look for latest release with Zink support
# Extract opengl32.dll, libgallium_wgl.dll, etc.
```

### Step 2: Set Up Test Environment (5 minutes)
```powershell
cd "C:\Program Files (x86)\Sauerbraten\bin64"

# Backup original OpenGL DLL (if exists)
# Copy Mesa Zink DLLs to Sauerbraten folder

# Set environment
$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "info"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"  # Force Zink driver

# Run Sauerbraten
.\sauerbraten.exe
```

### Step 3: Debug and Test (varies)
Expected chain:
```
Sauerbraten (OpenGL calls)
    ↓
Mesa Zink (OpenGL → Vulkan)
    ↓
VkWebGPU ICD (Vulkan → WebGPU)
    ↓
wgpu (WebGPU → DX12)
    ↓
DirectX 12
    ↓
GPU
```

## Alternative: Test with Simple OpenGL App First

Before Sauerbraten, test with a minimal OpenGL app:
- glxgears (if available)
- Simple OpenGL triangle demo
- Verify the Zink → VkWebGPU chain works

## Current Blocker

Need to download and set up Mesa3D with Zink support for Windows.

## Questions to Resolve

1. Does latest Mesa3D for Windows include Zink?
2. What DLLs do we need to copy?
3. Any additional environment variables needed?
4. Will Zink work with our Vulkan 1.2 ICD (it might need 1.3)?

## Success Criteria

- [ ] Mesa Zink loads successfully
- [ ] Zink creates Vulkan instance via our ICD
- [ ] Simple OpenGL app renders (test first)
- [ ] Sauerbraten launches without crashing
- [ ] Sauerbraten renders frames (even if glitchy)
- [ ] Acceptable performance (30+ FPS)
