#!/bin/bash

echo "============================================"
echo "VkWebGPU-ICD Testing Setup"
echo "============================================"
echo ""

echo "Building VkWebGPU-ICD..."
cargo build --release
if [ $? -ne 0 ]; then
    echo ""
    echo "Build failed! Please fix the errors and try again."
    exit 1
fi

echo ""
echo "Build successful!"
echo ""
echo "Setting up environment..."
export VK_DRIVER_FILES="$PWD/vkwebgpu_icd_linux.json"
export RUST_LOG=debug

echo ""
echo "============================================"
echo "VkWebGPU-ICD ready for testing!"
echo "============================================"
echo "VK_DRIVER_FILES=$VK_DRIVER_FILES"
echo "RUST_LOG=$RUST_LOG"
echo ""
echo "You can now run Vulkan applications:"
echo "  vulkaninfo       - Display device information"
echo "  vkcube           - Test basic rendering"
echo ""
echo "For verbose debugging, also set:"
echo "  export VK_LOADER_DEBUG=all"
echo "  export RUST_LOG=trace"
echo ""
echo "Press Ctrl+D to exit this environment."
echo "============================================"
echo ""

bash
