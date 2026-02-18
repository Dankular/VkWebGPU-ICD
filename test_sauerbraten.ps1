$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "debug"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
$env:MESA_DEBUG = "1"
$env:VK_LOADER_DEBUG = "warn,error"

Write-Host "Testing Sauerbraten with Zink + VkWebGPU..." -ForegroundColor Cyan
Write-Host ""

cd "C:\Program Files (x86)\Sauerbraten\bin64"
.\sauerbraten.exe

Write-Host ""
Write-Host "Test complete." -ForegroundColor Yellow
