@echo off
echo Building VkWebGPU Test Application...
cd test_app
cargo build --release
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Test app built successfully!
echo.
echo To run:
echo   cd test_app
echo   set VK_DRIVER_FILES=..\vkwebgpu_icd.json
echo   cargo run --release
