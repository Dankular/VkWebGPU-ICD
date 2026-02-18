@echo off
cd /d "%~dp0"
echo Setting up VkWebGPU-ICD test environment...
set VK_DRIVER_FILES=Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json
set RUST_LOG=debug

echo.
echo VK_DRIVER_FILES=%VK_DRIVER_FILES%
echo.
echo Running test application...
target\release\vk_test_app.exe
pause
