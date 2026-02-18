$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "info"
$env:RUST_BACKTRACE = "1"
$env:VK_LOADER_DEBUG = "all"

Write-Host "VK_DRIVER_FILES=$env:VK_DRIVER_FILES"
Write-Host "Running test application..."
Write-Host ""

& ".\target\release\vk_test_app.exe"
