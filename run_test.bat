@echo off
set VK_DRIVER_FILES=\\efret-hpv-02\Development Share\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json
set RUST_LOG=debug
set MESA_LOADER_DRIVER_OVERRIDE=zink
set MESA_DEBUG=1
set VK_LOADER_DEBUG=all

echo Testing Sauerbraten with Zink + VkWebGPU...
echo.

cd /d "C:\Program Files (x86)\Sauerbraten\bin64"
sauerbraten.exe 2>&1

echo.
echo Test complete.
