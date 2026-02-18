$env:VK_DRIVER_FILES = "Z:\source\Repos\VkWebGPU-ICD\vkwebgpu_icd.json"
$env:RUST_LOG = "info"
$env:MESA_LOADER_DRIVER_OVERRIDE = "zink"

cd "C:\Program Files (x86)\Sauerbraten\bin64"
.\sauerbraten.exe
