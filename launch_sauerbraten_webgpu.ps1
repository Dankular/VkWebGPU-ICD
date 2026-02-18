# Sauerbraten via OpenGL -> Vulkan -> WebGPU Translation
# ======================================================

Write-Host "Sauerbraten WebGPU Launch Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Set environment to use our Vulkan ICD
$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "info"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"
$env:GALLIUM_DRIVER = "zink"

Write-Host "Translation Chain:" -ForegroundColor Green
Write-Host "  Sauerbraten (OpenGL)" -ForegroundColor White
Write-Host "      |" -ForegroundColor DarkGray
Write-Host "  Mesa Zink (OpenGL -> Vulkan)" -ForegroundColor White
Write-Host "      |" -ForegroundColor DarkGray
Write-Host "  VkWebGPU ICD (Vulkan -> WebGPU)" -ForegroundColor White
Write-Host "      |" -ForegroundColor DarkGray
Write-Host "  wgpu (WebGPU -> DX12)" -ForegroundColor White
Write-Host "      |" -ForegroundColor DarkGray
Write-Host "  DirectX 12" -ForegroundColor White
Write-Host ""

Write-Host "Environment:" -ForegroundColor Yellow
Write-Host "  VK_DRIVER_FILES = $env:VK_DRIVER_FILES" -ForegroundColor White
Write-Host "  MESA_LOADER_DRIVER_OVERRIDE = $env:MESA_LOADER_DRIVER_OVERRIDE" -ForegroundColor White
Write-Host ""

# Check if Zink DLLs are present
$sauerbratenPath = "C:\Program Files (x86)\Sauerbraten\bin64"
$zinkDll = Join-Path $sauerbratenPath "opengl32.dll"

if (-not (Test-Path $zinkDll)) {
    Write-Host "WARNING: Mesa Zink opengl32.dll not found!" -ForegroundColor Red
    Write-Host "You need to install Mesa3D with Zink first." -ForegroundColor Red
    Write-Host "" 
    Write-Host "Steps to install Mesa Zink:" -ForegroundColor Yellow
    Write-Host "1. Download Mesa3D from: https://github.com/pal1000/mesa-dist-win/releases" -ForegroundColor White
    Write-Host "2. Extract the archive" -ForegroundColor White
    Write-Host "3. Copy these files to: $sauerbratenPath" -ForegroundColor White
    Write-Host "   - x64/opengl32.dll" -ForegroundColor Cyan
    Write-Host "   - x64/libgallium_wgl.dll" -ForegroundColor Cyan
    Write-Host "   - x64/libglapi.dll" -ForegroundColor Cyan
    Write-Host "   - Any other x64 DLLs Mesa needs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "Mesa Zink DLL found" -ForegroundColor Green
Write-Host "VkWebGPU ICD configured" -ForegroundColor Green
Write-Host ""
Write-Host "Launching Sauerbraten..." -ForegroundColor Green
Write-Host ""

# Run from bin64 directory
cd $sauerbratenPath
.\sauerbraten.exe

Write-Host ""
Write-Host "Sauerbraten exited." -ForegroundColor Yellow
