$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "debug"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
$env:GALLIUM_DRIVER = "zink"

Write-Host "Launching Sauerbraten with full debug logging..." -ForegroundColor Cyan
Write-Host "Check the console for errors. Game window should appear shortly." -ForegroundColor Yellow
Write-Host ""

cd "C:\Program Files (x86)\Sauerbraten\bin64"

# Run directly so we can see output
.\sauerbraten.exe

Write-Host ""
Write-Host "Sauerbraten exited." -ForegroundColor Yellow
