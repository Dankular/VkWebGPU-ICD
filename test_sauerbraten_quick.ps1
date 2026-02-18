$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "info"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
$env:MESA_DEBUG = "1"

Write-Host "Starting Sauerbraten..." -ForegroundColor Cyan

# Redirect output to file
cd "C:\Program Files (x86)\Sauerbraten\bin64"
Start-Process -FilePath ".\sauerbraten.exe" -RedirectStandardError "Z:\sauerbraten_error.log" -RedirectStandardOutput "Z:\sauerbraten_output.log"

Write-Host "Sauerbraten started. Check Z:\sauerbraten_error.log for output." -ForegroundColor Green
Write-Host "Press Ctrl+C when done testing." -ForegroundColor Yellow

# Wait a bit then show logs
Start-Sleep -Seconds 5
if (Test-Path "Z:\sauerbraten_error.log") {
    Write-Host "`nError log:" -ForegroundColor Red
    Get-Content "Z:\sauerbraten_error.log" | Select-Object -First 50
}
if (Test-Path "Z:\sauerbraten_output.log") {
    Write-Host "`nOutput log:" -ForegroundColor Green
    Get-Content "Z:\sauerbraten_output.log" | Select-Object -First 50
}
