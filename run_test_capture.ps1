$env:VK_DRIVER_FILES = "\\efret-hpv-02\Development Share\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "debug"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
$env:MESA_DEBUG = "1"
$env:VK_LOADER_DEBUG = "all"

$logPath = "C:\temp\sauerbraten_test.log"
$errorPath = "C:\temp\sauerbraten_error.log"

Write-Host "Testing Sauerbraten with Zink + VkWebGPU..." -ForegroundColor Cyan
Write-Host "Logs will be saved to C:\temp\" -ForegroundColor Yellow

Set-Location "C:\Program Files (x86)\Sauerbraten\bin64"
$startParams = @{
    FilePath = ".\sauerbraten.exe"
    RedirectStandardOutput = $logPath
    RedirectStandardError = $errorPath
    UseNewEnvironment = $false
}

$proc = Start-Process @startParams -PassThru
Write-Host "Process started with PID: $($proc.Id)" -ForegroundColor Green
Write-Host "Waiting 10 seconds for initialization..." -ForegroundColor Yellow

Start-Sleep -Seconds 10

if (Test-Path $errorPath) {
    Write-Host "`n=== Error Log ===" -ForegroundColor Red
    Get-Content $errorPath | Select-Object -First 100
}

if (Test-Path $logPath) {
    Write-Host "`n=== Output Log ===" -ForegroundColor Green
    Get-Content $logPath | Select-Object -First 100
}

Write-Host "`nTest running. Check C:\temp\ for logs." -ForegroundColor Yellow
