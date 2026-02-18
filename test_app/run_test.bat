@echo off
cd /d "%~dp0"
echo Setting up VkWebGPU-ICD test environment...
set VK_DRIVER_FILES=%~dp0..\vkwebgpu_icd.json
set RUST_LOG=debug

echo.
echo VK_DRIVER_FILES=%VK_DRIVER_FILES%
echo.
echo Running test application...
cargo run --release
