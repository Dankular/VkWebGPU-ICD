@echo off
echo ============================================
echo VkWebGPU-ICD Testing Setup
echo ============================================
echo.

echo Building VkWebGPU-ICD...
cargo build --release
if %errorlevel% neq 0 (
    echo.
    echo Build failed! Please fix the errors and try again.
    exit /b %errorlevel%
)

echo.
echo Build successful!
echo.
echo Setting up environment...
set VK_DRIVER_FILES=%~dp0vkwebgpu_icd.json
set RUST_LOG=debug

echo.
echo ============================================
echo VkWebGPU-ICD ready for testing!
echo ============================================
echo VK_DRIVER_FILES=%VK_DRIVER_FILES%
echo RUST_LOG=%RUST_LOG%
echo.
echo You can now run Vulkan applications:
echo   vulkaninfo       - Display device information
echo   vkcube           - Test basic rendering
echo.
echo For verbose debugging, also set:
echo   set VK_LOADER_DEBUG=all
echo   set RUST_LOG=trace
echo.
echo Press Ctrl+C to exit this environment.
echo ============================================
echo.

cmd /k
