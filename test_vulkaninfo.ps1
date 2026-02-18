$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "info"

Write-Host "Testing VkWebGPU ICD with vulkaninfo..."
Write-Host "========================================`n"

vulkaninfo --summary

Write-Host "`n`nTest completed!"
