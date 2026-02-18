$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "info"

Write-Host "Running VkWebGPU ICD Test..."
Write-Host "============================`n"

& ".\target\release\vk_test_app.exe"

Write-Host "`n`nTest completed!"
