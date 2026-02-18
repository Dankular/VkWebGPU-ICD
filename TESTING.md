# VkWebGPU-ICD Testing Guide

## Overview

This guide explains how to test the VkWebGPU-ICD (Vulkan to WebGPU translation layer) with various Vulkan applications.

## Prerequisites

- Rust toolchain (stable)
- Vulkan SDK (for test applications like `vulkaninfo` and `vkcube`)
- WebGPU-compatible graphics driver

## Setup

### Windows

1. **Build the ICD:**
   ```cmd
   cargo build --release
   ```

2. **Set environment variable to load the ICD:**
   ```cmd
   set VK_DRIVER_FILES=%CD%\vkwebgpu_icd.json
   ```

3. **Optional - Enable Vulkan validation layers:**
   ```cmd
   set VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
   set VK_LOADER_DEBUG=all
   ```

4. **Optional - Enable debug logging:**
   ```cmd
   set RUST_LOG=debug
   ```

### Linux

1. **Build the ICD:**
   ```bash
   cargo build --release
   ```

2. **Set environment variable to load the ICD:**
   ```bash
   export VK_DRIVER_FILES=$PWD/vkwebgpu_icd_linux.json
   ```

3. **Optional - Enable Vulkan validation layers:**
   ```bash
   export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
   export VK_LOADER_DEBUG=all
   ```

4. **Optional - Enable debug logging:**
   ```bash
   export RUST_LOG=debug
   ```

### macOS

1. **Build the ICD:**
   ```bash
   cargo build --release
   ```

2. **Set environment variable to load the ICD:**
   ```bash
   export VK_DRIVER_FILES=$PWD/vkwebgpu_icd_macos.json
   ```
   
   Note: You may need to create `vkwebgpu_icd_macos.json` with the appropriate `.dylib` path.

## Quick Start with Test Script

### Windows

Run the provided test script:
```cmd
test_icd.bat
```

This will build the ICD and set up the environment automatically.

### Linux/macOS

Run the provided test script:
```bash
./test_icd.sh
```

Make sure the script is executable:
```bash
chmod +x test_icd.sh
```

## Testing with Vulkan Applications

### 1. vulkaninfo

The simplest test - displays information about Vulkan devices:

```bash
vulkaninfo
```

You should see VkWebGPU-ICD listed as an available device.

### 2. vkcube

A basic rendering test that displays a spinning cube:

```bash
vkcube
```

If successful, you should see a spinning cube rendered through the WebGPU backend.

### 3. Vulkan SDK Samples

Download and run Vulkan SDK samples with the environment configured to use VkWebGPU-ICD.

### 4. DXVK Applications (Windows)

1. Get DXVK test applications or games
2. Ensure `VK_DRIVER_FILES` is set to use VkWebGPU-ICD
3. Run the DXVK application

Note: This is advanced testing and may require additional DXVK setup.

## Debugging

### Enable Verbose Logging

**Windows:**
```cmd
set RUST_LOG=trace
set VK_LOADER_DEBUG=all
```

**Linux/macOS:**
```bash
export RUST_LOG=trace
export VK_LOADER_DEBUG=all
```

### Check Which ICD is Loaded

Run `vulkaninfo` with loader debugging:

**Windows:**
```cmd
set VK_LOADER_DEBUG=all
vulkaninfo
```

**Linux/macOS:**
```bash
export VK_LOADER_DEBUG=all
vulkaninfo
```

Look for lines mentioning `vkwebgpu` in the output.

### Common Issues

#### ICD Not Found

**Problem:** Vulkan loader doesn't find the ICD.

**Solution:**
- Verify `VK_DRIVER_FILES` points to the correct manifest JSON file
- Use absolute path if relative path doesn't work
- Check that the DLL/SO file exists in `target/release/`

#### Build Errors

**Problem:** Compilation fails.

**Solution:**
- Ensure Rust toolchain is up to date: `rustup update`
- Check that all dependencies are available
- Review build output for specific errors

#### Runtime Crashes

**Problem:** Application crashes when using VkWebGPU-ICD.

**Solution:**
- Enable debug logging: `RUST_LOG=debug`
- Check if WebGPU backend is available on your system
- Try running with Vulkan validation layers to get more information
- Review application logs for error messages

#### Missing WebGPU Support

**Problem:** WebGPU backend initialization fails.

**Solution:**
- Ensure graphics drivers are up to date
- Verify WebGPU support: run a simple WebGPU test (browser-based)
- Check if hardware supports required WebGPU features

## Development Testing

### Running Unit Tests

```bash
cargo test
```

### Running Integration Tests

```bash
cargo test --test '*'
```

### Building with Debug Symbols

For debugging with tools like GDB or Visual Studio:

```bash
cargo build
```

The debug build will be in `target/debug/`.

## Performance Testing

### Benchmarking

Run criterion benchmarks (if available):

```bash
cargo bench
```

### Profiling

Use your platform's profiling tools:

- **Windows:** Visual Studio Profiler, PIX
- **Linux:** `perf`, Valgrind
- **macOS:** Instruments

## CI/CD Testing

The project should include automated tests in the CI pipeline. Check `.github/workflows/` for CI configuration.

## Reporting Issues

When reporting issues, please include:

1. Operating system and version
2. Graphics hardware and driver version
3. Rust version (`rustc --version`)
4. Full command used to run the test
5. Complete error output with `RUST_LOG=debug` and `VK_LOADER_DEBUG=all`
6. Steps to reproduce

## Additional Resources

- [Vulkan Loader Documentation](https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [VkWebGPU-ICD README](README.md)
