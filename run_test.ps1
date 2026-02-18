$env:VK_DRIVER_FILES = "\\efret-hpv-02\Development Share\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "debug"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
$env:MESA_DEBUG = "1"
$env:VK_LOADER_DEBUG = "all"

$logPath = "\\efret-hpv-02\Development Share\source\Repos\VkWebGPU-ICD\sauerbraten_test.log"

Write-Host "Testing Sauerbraten with Zink + VkWebGPU..." -ForegroundColor Cyan

cd "C:\Program Files (x86)\Sauerbraten\bin64"
& .\sauerbraten.exe 2>&1 | Tee-Object -FilePath $logPath

Write-Host ""
Write-Host "Test complete. Log saved to: $logPath" -ForegroundColor Yellow
