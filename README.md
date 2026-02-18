# VkWebGPU-ICD

A Vulkan Installable Client Driver (ICD) that translates Vulkan API calls to WebGPU, enabling Vulkan applications (including DXVK-translated DirectX games) to run in web browsers.

## Architecture

```
Game → DirectX → DXVK → Vulkan API → VkWebGPU ICD → WebGPU → Browser GPU
```

## Status

**Phase 1: Core Implementation** - Complete architecture, needs API fixes

- ✅ Project structure and build system
- ✅ Backend abstraction (native wgpu + WASM web-sys)
- ✅ Vulkan ICD entry points (vk_icd.h interface)
- ✅ Instance → GPUAdapter mapping
- ✅ Physical device enumeration
- ✅ Device → GPUDevice creation
- ✅ Queue management
- ✅ Memory allocation and mapping
- ✅ Buffer creation and binding
- ✅ Image/texture creation and views
- ✅ Sampler creation
- ✅ Descriptor sets → Bind groups
- ✅ Pipeline layout and creation
- ✅ Shader module (SPIR-V → WGSL via Naga)
- ✅ Render passes and framebuffers
- ✅ Command pools and buffers
- ✅ Command recording (draw, copy, barriers)
- ✅ Synchronization (fences, semaphores)
- ✅ Format conversion (Vulkan ↔ WebGPU)
- ⏳ Build fixes needed for Ash 0.38 API

## Components

### Core Modules

- **backend.rs** - WebGPU backend abstraction (wgpu/web-sys)
- **error.rs** - Error types and Vulkan result mapping
- **handle.rs** - Thread-safe handle allocation
- **format.rs** - Format conversion tables
- **shader.rs** - SPIR-V → WGSL translation with caching

### Vulkan Implementation

- **instance.rs** - VkInstance → GPUAdapter
- **device.rs** - VkDevice → GPUDevice
- **queue.rs** - VkQueue → GPUQueue
- **memory.rs** - Memory allocation (emulated)
- **buffer.rs** - VkBuffer → GPUBuffer
- **image.rs** - VkImage/VkImageView → GPUTexture/GPUTextureView
- **sampler.rs** - VkSampler → GPUSampler
- **descriptor.rs** - VkDescriptorSet → GPUBindGroup
- **pipeline.rs** - VkPipeline → GPURenderPipeline/GPUComputePipeline
- **render_pass.rs** - VkRenderPass tracking
- **framebuffer.rs** - VkFramebuffer tracking
- **command_pool.rs** - Command buffer pool management
- **command_buffer.rs** - VkCommandBuffer → GPUCommandEncoder
- **sync.rs** - Fences and semaphores
- **swapchain.rs** - Swapchain support (KHR extension)
- **icd.rs** - ICD entry points and function dispatch

## Build Issues to Fix

1. **Ash handle conversion**: Ash 0.38 uses newtype patterns, not `from_raw`/`as_raw`
   - Solution: Use `vk::Handle` trait or direct construction

2. **Naga API**: `parse_u32_slice` moved to different location
   - Solution: Use `naga::front::spv::Parser::parse`

3. **Const initialization**: `HandleAllocator::new()` not const
   - Solution: Use `OnceCell` or lazy_static pattern

4. **Lifetime issues**: Backend types need proper lifetime annotations
   - Solution: Review render pass encoder lifetimes

## Next Steps

### Immediate (Build Fixes)

1. Update handle conversion to use `vk::Handle` trait
2. Fix Naga parser API usage  
3. Convert static allocators to use `OnceCell<HandleAllocator<T>>`
4. Fix render pass encoder lifetime issues

### Phase 2 (DXVK Compatibility)

1. Map DXVK-specific Vulkan usage patterns
2. Implement push constants via uniform buffer emulation
3. Test with simple DXVK-translated DX9 game
4. Optimize shader cache performance

### Phase 3 (Real Game Support)

1. Complete texture format coverage
2. Implement compute pipeline path
3. Add dynamic state support
4. Buffer/texture streaming optimizations
5. Test with Enter the Gungeon (target game)

### Phase 4 (Integration)

1. Package as `.so`/`.dll` for CheerpX
2. Configure as Vulkan ICD via `VK_DRIVER_FILES`
3. Integration testing with Proton runtime
4. Performance profiling and optimization

## Technical Highlights

### Shader Translation

Uses Naga to translate SPIR-V (from DXVK) to WGSL:
- Hash-based shader cache
- Validation pipeline
- Coordinate space adjustments

### Memory Model

Vulkan's explicit allocation → WebGPU's implicit model:
- Track Vulkan allocations
- Create WebGPU resources on bind
- Host-visible memory emulated via staging

### Synchronization

WebGPU's implicit sync → Simplified from Vulkan:
- Pipeline barriers → No-ops (WebGPU auto-barriers)
- Fences → Tracked state
- Semaphores → Sequential submission

### Format Support

Comprehensive format tables:
- Standard formats (R/RG/RGBA 8/16/32-bit)
- Depth/stencil formats
- BC1-7 compression
- ETC2/EAC compression
- ASTC compression

## Dependencies

- `ash` 0.38 - Vulkan bindings
- `naga` 0.20 - Shader translation
- `wgpu` 0.20 - Native WebGPU backend
- `web-sys` - WASM WebGPU bindings
- `parking_lot` - Fast synchronization
- `rustc-hash` - Fast hashing

## License

MIT OR Apache-2.0
