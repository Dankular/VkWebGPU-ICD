//! Vulkan ICD (Installable Client Driver) entry points
//!
//! This module implements the core Vulkan ICD interface that the Vulkan loader uses.

use ash::vk;
use log::{debug, info};
use std::ffi::CStr;
use std::os::raw::c_char;

/// ICD interface version
pub const VK_ICD_INTERFACE_VERSION: u32 = 5;

/// Get the instance proc addr - core ICD function
#[no_mangle]
pub unsafe extern "system" fn vk_icdGetInstanceProcAddr(
    _instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }

    let name = CStr::from_ptr(p_name);
    let name_str = name.to_bytes();

    let result = match name_str {
        b"vkCreateInstance" => std::mem::transmute(vkCreateInstance as vk::PFN_vkCreateInstance),
        b"vkDestroyInstance" => std::mem::transmute(vkDestroyInstance as vk::PFN_vkDestroyInstance),
        b"vkEnumerateInstanceVersion" => {
            std::mem::transmute(vkEnumerateInstanceVersion as vk::PFN_vkEnumerateInstanceVersion)
        }
        b"vkEnumerateInstanceExtensionProperties" => std::mem::transmute(
            vkEnumerateInstanceExtensionProperties
                as vk::PFN_vkEnumerateInstanceExtensionProperties,
        ),
        b"vkEnumeratePhysicalDevices" => {
            std::mem::transmute(vkEnumeratePhysicalDevices as vk::PFN_vkEnumeratePhysicalDevices)
        }
        b"vkGetPhysicalDeviceProperties" => std::mem::transmute(
            vkGetPhysicalDeviceProperties as vk::PFN_vkGetPhysicalDeviceProperties,
        ),
        b"vkGetPhysicalDeviceFeatures" => {
            std::mem::transmute(vkGetPhysicalDeviceFeatures as vk::PFN_vkGetPhysicalDeviceFeatures)
        }
        b"vkGetPhysicalDeviceQueueFamilyProperties" => std::mem::transmute(
            vkGetPhysicalDeviceQueueFamilyProperties
                as vk::PFN_vkGetPhysicalDeviceQueueFamilyProperties,
        ),
        b"vkGetPhysicalDeviceMemoryProperties" => std::mem::transmute(
            vkGetPhysicalDeviceMemoryProperties as vk::PFN_vkGetPhysicalDeviceMemoryProperties,
        ),
        b"vkEnumerateDeviceExtensionProperties" => std::mem::transmute(
            vkEnumerateDeviceExtensionProperties as vk::PFN_vkEnumerateDeviceExtensionProperties,
        ),
        b"vkGetPhysicalDeviceFormatProperties" => std::mem::transmute(
            vkGetPhysicalDeviceFormatProperties as vk::PFN_vkGetPhysicalDeviceFormatProperties,
        ),
        b"vkGetPhysicalDeviceImageFormatProperties" => std::mem::transmute(
            vkGetPhysicalDeviceImageFormatProperties
                as vk::PFN_vkGetPhysicalDeviceImageFormatProperties,
        ),
        b"vkCreateDevice" => std::mem::transmute(vkCreateDevice as vk::PFN_vkCreateDevice),
        b"vkGetDeviceProcAddr" => {
            std::mem::transmute(vkGetDeviceProcAddr as vk::PFN_vkGetDeviceProcAddr)
        }
        b"vkGetPhysicalDeviceSparseImageFormatProperties" => std::mem::transmute(
            vkGetPhysicalDeviceSparseImageFormatProperties
                as vk::PFN_vkGetPhysicalDeviceSparseImageFormatProperties,
        ),
        
        // Surface functions (KHR)
        b"vkCreateWin32SurfaceKHR" => {
            std::mem::transmute(vkCreateWin32SurfaceKHR as vk::PFN_vkCreateWin32SurfaceKHR)
        }
        b"vkDestroySurfaceKHR" => {
            std::mem::transmute(vkDestroySurfaceKHR as vk::PFN_vkDestroySurfaceKHR)
        }
        b"vkGetPhysicalDeviceSurfaceSupportKHR" => {
            std::mem::transmute(vkGetPhysicalDeviceSurfaceSupportKHR as vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR)
        }
        b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR" => {
            std::mem::transmute(vkGetPhysicalDeviceSurfaceCapabilitiesKHR as vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
        }
        b"vkGetPhysicalDeviceSurfaceFormatsKHR" => {
            std::mem::transmute(vkGetPhysicalDeviceSurfaceFormatsKHR as vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)
        }
        b"vkGetPhysicalDeviceSurfacePresentModesKHR" => {
            std::mem::transmute(vkGetPhysicalDeviceSurfacePresentModesKHR as vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)
        }

        // Physical device properties2 (core 1.1 + KHR aliases)
        b"vkGetPhysicalDeviceProperties2" | b"vkGetPhysicalDeviceProperties2KHR" => {
            std::mem::transmute(vkGetPhysicalDeviceProperties2 as vk::PFN_vkGetPhysicalDeviceProperties2)
        }
        b"vkGetPhysicalDeviceFeatures2" | b"vkGetPhysicalDeviceFeatures2KHR" => {
            std::mem::transmute(vkGetPhysicalDeviceFeatures2 as vk::PFN_vkGetPhysicalDeviceFeatures2)
        }
        b"vkGetPhysicalDeviceMemoryProperties2" | b"vkGetPhysicalDeviceMemoryProperties2KHR" => {
            std::mem::transmute(vkGetPhysicalDeviceMemoryProperties2 as vk::PFN_vkGetPhysicalDeviceMemoryProperties2)
        }
        b"vkGetPhysicalDeviceQueueFamilyProperties2" | b"vkGetPhysicalDeviceQueueFamilyProperties2KHR" => {
            std::mem::transmute(vkGetPhysicalDeviceQueueFamilyProperties2 as vk::PFN_vkGetPhysicalDeviceQueueFamilyProperties2)
        }
        b"vkGetPhysicalDeviceFormatProperties2" | b"vkGetPhysicalDeviceFormatProperties2KHR" => {
            std::mem::transmute(vkGetPhysicalDeviceFormatProperties2 as vk::PFN_vkGetPhysicalDeviceFormatProperties2)
        }
        b"vkGetPhysicalDeviceImageFormatProperties2" | b"vkGetPhysicalDeviceImageFormatProperties2KHR" => {
            std::mem::transmute(vkGetPhysicalDeviceImageFormatProperties2 as vk::PFN_vkGetPhysicalDeviceImageFormatProperties2)
        }
        b"vkGetPhysicalDeviceSparseImageFormatProperties2" | b"vkGetPhysicalDeviceSparseImageFormatProperties2KHR" => {
            std::mem::transmute(vkGetPhysicalDeviceSparseImageFormatProperties2 as vk::PFN_vkGetPhysicalDeviceSparseImageFormatProperties2)
        }
        b"vkGetPhysicalDeviceExternalBufferProperties" | b"vkGetPhysicalDeviceExternalBufferPropertiesKHR" => {
            std::mem::transmute(vkGetPhysicalDeviceExternalBufferProperties as vk::PFN_vkGetPhysicalDeviceExternalBufferProperties)
        }
        b"vkGetPhysicalDeviceExternalFenceProperties" | b"vkGetPhysicalDeviceExternalFencePropertiesKHR" => {
            std::mem::transmute(vkGetPhysicalDeviceExternalFenceProperties as vk::PFN_vkGetPhysicalDeviceExternalFenceProperties)
        }
        b"vkGetPhysicalDeviceExternalSemaphoreProperties" | b"vkGetPhysicalDeviceExternalSemaphorePropertiesKHR" => {
            std::mem::transmute(vkGetPhysicalDeviceExternalSemaphoreProperties as vk::PFN_vkGetPhysicalDeviceExternalSemaphoreProperties)
        }

        _ => None,
    };

    // Log if we don't have this function
    if result.is_none() {
        info!("MISSING FUNCTION: {:?}", name);
    }

    result
}

#[no_mangle]
pub unsafe extern "system" fn vk_icdNegotiateLoaderICDInterfaceVersion(
    p_supported_version: *mut u32,
) -> vk::Result {
    if p_supported_version.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let requested_version = *p_supported_version;
    info!(
        "ICD version negotiation: loader requests v{}",
        requested_version
    );

    if requested_version > VK_ICD_INTERFACE_VERSION {
        *p_supported_version = VK_ICD_INTERFACE_VERSION;
    }

    vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vk_icdGetPhysicalDeviceProcAddr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    vk_icdGetInstanceProcAddr(instance, p_name)
}

// Instance functions

#[no_mangle]
pub unsafe extern "system" fn vkEnumerateInstanceVersion(p_api_version: *mut u32) -> vk::Result {
    if p_api_version.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // Report Vulkan 1.3 support (required by Zink)
    *p_api_version = vk::API_VERSION_1_3;
    vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateInstance(
    p_create_info: *const vk::InstanceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    crate::init();
    info!("vkCreateInstance called");

    match crate::instance::create_instance(p_create_info, p_allocator, p_instance) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyInstance(
    instance: vk::Instance,
    p_allocator: *const vk::AllocationCallbacks,
) {
    info!("vkDestroyInstance called");
    crate::instance::destroy_instance(instance, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumeratePhysicalDevices(
    instance: vk::Instance,
    p_physical_device_count: *mut u32,
    p_physical_devices: *mut vk::PhysicalDevice,
) -> vk::Result {
    match crate::instance::enumerate_physical_devices(
        instance,
        p_physical_device_count,
        p_physical_devices,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceProperties(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties,
) {
    crate::instance::get_physical_device_properties(physical_device, p_properties);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFeatures(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures,
) {
    crate::instance::get_physical_device_features(physical_device, p_features);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceQueueFamilyProperties(
    physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties,
) {
    crate::instance::get_physical_device_queue_family_properties(
        physical_device,
        p_queue_family_property_count,
        p_queue_family_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceMemoryProperties(
    physical_device: vk::PhysicalDevice,
    p_memory_properties: *mut vk::PhysicalDeviceMemoryProperties,
) {
    crate::memory::get_physical_device_memory_properties(physical_device, p_memory_properties);
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumerateInstanceExtensionProperties(
    p_layer_name: *const c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    match crate::instance::enumerate_instance_extension_properties(
        p_layer_name,
        p_property_count,
        p_properties,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumerateDeviceExtensionProperties(
    physical_device: vk::PhysicalDevice,
    p_layer_name: *const c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    match crate::device::enumerate_device_extension_properties(
        physical_device,
        p_layer_name,
        p_property_count,
        p_properties,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFormatProperties(
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    p_format_properties: *mut vk::FormatProperties,
) {
    crate::instance::get_physical_device_format_properties(
        physical_device,
        format,
        p_format_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSparseImageFormatProperties(
    _physical_device: vk::PhysicalDevice,
    _format: vk::Format,
    _type: vk::ImageType,
    _samples: vk::SampleCountFlags,
    _usage: vk::ImageUsageFlags,
    _tiling: vk::ImageTiling,
    p_property_count: *mut u32,
    _p_properties: *mut vk::SparseImageFormatProperties,
) {
    // WebGPU doesn't support sparse images, so return 0 properties
    if !p_property_count.is_null() {
        *p_property_count = 0;
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceImageFormatProperties(
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    image_type: vk::ImageType,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    flags: vk::ImageCreateFlags,
    p_image_format_properties: *mut vk::ImageFormatProperties,
) -> vk::Result {
    match crate::instance::get_physical_device_image_format_properties(
        physical_device,
        format,
        image_type,
        tiling,
        usage,
        flags,
        p_image_format_properties,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateDevice(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_device: *mut vk::Device,
) -> vk::Result {
    info!("vkCreateDevice called");

    match crate::device::create_device(physical_device, p_create_info, p_allocator, p_device) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceProcAddr(
    _device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }

    let name = CStr::from_ptr(p_name);
    debug!("vkGetDeviceProcAddr: {:?}", name);

    match name.to_bytes() {
        b"vkDestroyDevice" => std::mem::transmute(vkDestroyDevice as vk::PFN_vkDestroyDevice),
        b"vkGetDeviceQueue" => std::mem::transmute(vkGetDeviceQueue as vk::PFN_vkGetDeviceQueue),
        b"vkDeviceWaitIdle" => std::mem::transmute(vkDeviceWaitIdle as vk::PFN_vkDeviceWaitIdle),

        // Memory
        b"vkAllocateMemory" => std::mem::transmute(vkAllocateMemory as vk::PFN_vkAllocateMemory),
        b"vkFreeMemory" => std::mem::transmute(vkFreeMemory as vk::PFN_vkFreeMemory),
        b"vkMapMemory" => std::mem::transmute(vkMapMemory as vk::PFN_vkMapMemory),
        b"vkUnmapMemory" => std::mem::transmute(vkUnmapMemory as vk::PFN_vkUnmapMemory),
        b"vkFlushMappedMemoryRanges" => {
            std::mem::transmute(vkFlushMappedMemoryRanges as vk::PFN_vkFlushMappedMemoryRanges)
        }

        // Buffers
        b"vkCreateBuffer" => std::mem::transmute(vkCreateBuffer as vk::PFN_vkCreateBuffer),
        b"vkDestroyBuffer" => std::mem::transmute(vkDestroyBuffer as vk::PFN_vkDestroyBuffer),
        b"vkBindBufferMemory" => {
            std::mem::transmute(vkBindBufferMemory as vk::PFN_vkBindBufferMemory)
        }
        b"vkGetBufferMemoryRequirements" => std::mem::transmute(
            vkGetBufferMemoryRequirements as vk::PFN_vkGetBufferMemoryRequirements,
        ),

        // Images
        b"vkCreateImage" => std::mem::transmute(vkCreateImage as vk::PFN_vkCreateImage),
        b"vkDestroyImage" => std::mem::transmute(vkDestroyImage as vk::PFN_vkDestroyImage),
        b"vkBindImageMemory" => std::mem::transmute(vkBindImageMemory as vk::PFN_vkBindImageMemory),
        b"vkGetImageMemoryRequirements" => std::mem::transmute(
            vkGetImageMemoryRequirements as vk::PFN_vkGetImageMemoryRequirements,
        ),
        b"vkCreateImageView" => std::mem::transmute(vkCreateImageView as vk::PFN_vkCreateImageView),
        b"vkDestroyImageView" => {
            std::mem::transmute(vkDestroyImageView as vk::PFN_vkDestroyImageView)
        }

        // Samplers
        b"vkCreateSampler" => std::mem::transmute(vkCreateSampler as vk::PFN_vkCreateSampler),
        b"vkDestroySampler" => std::mem::transmute(vkDestroySampler as vk::PFN_vkDestroySampler),

        // Descriptors
        b"vkCreateDescriptorSetLayout" => {
            std::mem::transmute(vkCreateDescriptorSetLayout as vk::PFN_vkCreateDescriptorSetLayout)
        }
        b"vkDestroyDescriptorSetLayout" => std::mem::transmute(
            vkDestroyDescriptorSetLayout as vk::PFN_vkDestroyDescriptorSetLayout,
        ),
        b"vkCreateDescriptorPool" => {
            std::mem::transmute(vkCreateDescriptorPool as vk::PFN_vkCreateDescriptorPool)
        }
        b"vkDestroyDescriptorPool" => {
            std::mem::transmute(vkDestroyDescriptorPool as vk::PFN_vkDestroyDescriptorPool)
        }
        b"vkAllocateDescriptorSets" => {
            std::mem::transmute(vkAllocateDescriptorSets as vk::PFN_vkAllocateDescriptorSets)
        }
        b"vkUpdateDescriptorSets" => {
            std::mem::transmute(vkUpdateDescriptorSets as vk::PFN_vkUpdateDescriptorSets)
        }

        // Pipelines
        b"vkCreateShaderModule" => {
            std::mem::transmute(vkCreateShaderModule as vk::PFN_vkCreateShaderModule)
        }
        b"vkDestroyShaderModule" => {
            std::mem::transmute(vkDestroyShaderModule as vk::PFN_vkDestroyShaderModule)
        }
        b"vkCreatePipelineLayout" => {
            std::mem::transmute(vkCreatePipelineLayout as vk::PFN_vkCreatePipelineLayout)
        }
        b"vkDestroyPipelineLayout" => {
            std::mem::transmute(vkDestroyPipelineLayout as vk::PFN_vkDestroyPipelineLayout)
        }
        b"vkCreateGraphicsPipelines" => {
            std::mem::transmute(vkCreateGraphicsPipelines as vk::PFN_vkCreateGraphicsPipelines)
        }
        b"vkCreateComputePipelines" => {
            std::mem::transmute(vkCreateComputePipelines as vk::PFN_vkCreateComputePipelines)
        }
        b"vkDestroyPipeline" => std::mem::transmute(vkDestroyPipeline as vk::PFN_vkDestroyPipeline),

        // Render passes
        b"vkCreateRenderPass" => {
            std::mem::transmute(vkCreateRenderPass as vk::PFN_vkCreateRenderPass)
        }
        b"vkDestroyRenderPass" => {
            std::mem::transmute(vkDestroyRenderPass as vk::PFN_vkDestroyRenderPass)
        }
        b"vkCreateFramebuffer" => {
            std::mem::transmute(vkCreateFramebuffer as vk::PFN_vkCreateFramebuffer)
        }
        b"vkDestroyFramebuffer" => {
            std::mem::transmute(vkDestroyFramebuffer as vk::PFN_vkDestroyFramebuffer)
        }

        // Command buffers
        b"vkCreateCommandPool" => {
            std::mem::transmute(vkCreateCommandPool as vk::PFN_vkCreateCommandPool)
        }
        b"vkDestroyCommandPool" => {
            std::mem::transmute(vkDestroyCommandPool as vk::PFN_vkDestroyCommandPool)
        }
        b"vkAllocateCommandBuffers" => {
            std::mem::transmute(vkAllocateCommandBuffers as vk::PFN_vkAllocateCommandBuffers)
        }
        b"vkBeginCommandBuffer" => {
            std::mem::transmute(vkBeginCommandBuffer as vk::PFN_vkBeginCommandBuffer)
        }
        b"vkEndCommandBuffer" => {
            std::mem::transmute(vkEndCommandBuffer as vk::PFN_vkEndCommandBuffer)
        }
        b"vkCmdBeginRenderPass" => {
            std::mem::transmute(vkCmdBeginRenderPass as vk::PFN_vkCmdBeginRenderPass)
        }
        b"vkCmdEndRenderPass" => {
            std::mem::transmute(vkCmdEndRenderPass as vk::PFN_vkCmdEndRenderPass)
        }
        b"vkCmdBindPipeline" => std::mem::transmute(vkCmdBindPipeline as vk::PFN_vkCmdBindPipeline),
        b"vkCmdBindDescriptorSets" => {
            std::mem::transmute(vkCmdBindDescriptorSets as vk::PFN_vkCmdBindDescriptorSets)
        }
        b"vkCmdBindVertexBuffers" => {
            std::mem::transmute(vkCmdBindVertexBuffers as vk::PFN_vkCmdBindVertexBuffers)
        }
        b"vkCmdBindIndexBuffer" => {
            std::mem::transmute(vkCmdBindIndexBuffer as vk::PFN_vkCmdBindIndexBuffer)
        }
        b"vkCmdDraw" => std::mem::transmute(vkCmdDraw as vk::PFN_vkCmdDraw),
        b"vkCmdDrawIndexed" => std::mem::transmute(vkCmdDrawIndexed as vk::PFN_vkCmdDrawIndexed),
        b"vkCmdCopyBuffer" => std::mem::transmute(vkCmdCopyBuffer as vk::PFN_vkCmdCopyBuffer),
        b"vkCmdCopyBufferToImage" => {
            std::mem::transmute(vkCmdCopyBufferToImage as vk::PFN_vkCmdCopyBufferToImage)
        }
        b"vkCmdPipelineBarrier" => {
            std::mem::transmute(vkCmdPipelineBarrier as vk::PFN_vkCmdPipelineBarrier)
        }
        b"vkCmdPushConstants" => {
            std::mem::transmute(vkCmdPushConstants as vk::PFN_vkCmdPushConstants)
        }
        b"vkCmdDispatch" => std::mem::transmute(vkCmdDispatch as vk::PFN_vkCmdDispatch),

        // Dynamic state commands
        b"vkCmdSetViewport" => std::mem::transmute(vkCmdSetViewport as vk::PFN_vkCmdSetViewport),
        b"vkCmdSetScissor" => std::mem::transmute(vkCmdSetScissor as vk::PFN_vkCmdSetScissor),
        b"vkCmdSetBlendConstants" => {
            std::mem::transmute(vkCmdSetBlendConstants as vk::PFN_vkCmdSetBlendConstants)
        }
        b"vkCmdSetStencilReference" => {
            std::mem::transmute(vkCmdSetStencilReference as vk::PFN_vkCmdSetStencilReference)
        }

        // Clear commands
        b"vkCmdClearColorImage" => {
            std::mem::transmute(vkCmdClearColorImage as vk::PFN_vkCmdClearColorImage)
        }
        b"vkCmdClearDepthStencilImage" => {
            std::mem::transmute(vkCmdClearDepthStencilImage as vk::PFN_vkCmdClearDepthStencilImage)
        }
        b"vkCmdClearAttachments" => {
            std::mem::transmute(vkCmdClearAttachments as vk::PFN_vkCmdClearAttachments)
        }

        // Copy commands
        b"vkCmdCopyImage" => std::mem::transmute(vkCmdCopyImage as vk::PFN_vkCmdCopyImage),
        b"vkCmdCopyImageToBuffer" => {
            std::mem::transmute(vkCmdCopyImageToBuffer as vk::PFN_vkCmdCopyImageToBuffer)
        }
        b"vkCmdBlitImage" => std::mem::transmute(vkCmdBlitImage as vk::PFN_vkCmdBlitImage),

        // Queue operations
        b"vkQueueSubmit" => std::mem::transmute(vkQueueSubmit as vk::PFN_vkQueueSubmit),
        b"vkQueueWaitIdle" => std::mem::transmute(vkQueueWaitIdle as vk::PFN_vkQueueWaitIdle),

        // Synchronization
        b"vkCreateFence" => std::mem::transmute(vkCreateFence as vk::PFN_vkCreateFence),
        b"vkDestroyFence" => std::mem::transmute(vkDestroyFence as vk::PFN_vkDestroyFence),
        b"vkResetFences" => std::mem::transmute(vkResetFences as vk::PFN_vkResetFences),
        b"vkWaitForFences" => std::mem::transmute(vkWaitForFences as vk::PFN_vkWaitForFences),
        b"vkCreateSemaphore" => std::mem::transmute(vkCreateSemaphore as vk::PFN_vkCreateSemaphore),
        b"vkDestroySemaphore" => {
            std::mem::transmute(vkDestroySemaphore as vk::PFN_vkDestroySemaphore)
        }

        // Swapchain (KHR extension)
        b"vkCreateSwapchainKHR" => {
            std::mem::transmute(vkCreateSwapchainKHR as vk::PFN_vkCreateSwapchainKHR)
        }
        b"vkDestroySwapchainKHR" => {
            std::mem::transmute(vkDestroySwapchainKHR as vk::PFN_vkDestroySwapchainKHR)
        }
        b"vkGetSwapchainImagesKHR" => {
            std::mem::transmute(vkGetSwapchainImagesKHR as vk::PFN_vkGetSwapchainImagesKHR)
        }
        b"vkAcquireNextImageKHR" => {
            std::mem::transmute(vkAcquireNextImageKHR as vk::PFN_vkAcquireNextImageKHR)
        }
        b"vkQueuePresentKHR" => std::mem::transmute(vkQueuePresentKHR as vk::PFN_vkQueuePresentKHR),

        // Command buffer management
        b"vkFreeCommandBuffers" => {
            std::mem::transmute(vkFreeCommandBuffers as vk::PFN_vkFreeCommandBuffers)
        }
        b"vkResetCommandBuffer" => {
            std::mem::transmute(vkResetCommandBuffer as vk::PFN_vkResetCommandBuffer)
        }
        b"vkResetCommandPool" => {
            std::mem::transmute(vkResetCommandPool as vk::PFN_vkResetCommandPool)
        }

        // Descriptor management
        b"vkFreeDescriptorSets" => {
            std::mem::transmute(vkFreeDescriptorSets as vk::PFN_vkFreeDescriptorSets)
        }
        b"vkResetDescriptorPool" => {
            std::mem::transmute(vkResetDescriptorPool as vk::PFN_vkResetDescriptorPool)
        }

        // Image extras
        b"vkGetImageSubresourceLayout" => {
            std::mem::transmute(vkGetImageSubresourceLayout as vk::PFN_vkGetImageSubresourceLayout)
        }
        b"vkGetImageMemoryRequirements2" | b"vkGetImageMemoryRequirements2KHR" => {
            std::mem::transmute(vkGetImageMemoryRequirements2 as vk::PFN_vkGetImageMemoryRequirements2)
        }
        b"vkBindImageMemory2" | b"vkBindImageMemory2KHR" => {
            std::mem::transmute(vkBindImageMemory2 as vk::PFN_vkBindImageMemory2)
        }
        b"vkGetDeviceImageMemoryRequirements" | b"vkGetDeviceImageMemoryRequirementsKHR" => {
            std::mem::transmute(vkGetDeviceImageMemoryRequirements as vk::PFN_vkGetDeviceImageMemoryRequirements)
        }

        // Buffer extras
        b"vkGetBufferMemoryRequirements2" | b"vkGetBufferMemoryRequirements2KHR" => {
            std::mem::transmute(vkGetBufferMemoryRequirements2 as vk::PFN_vkGetBufferMemoryRequirements2)
        }
        b"vkBindBufferMemory2" | b"vkBindBufferMemory2KHR" => {
            std::mem::transmute(vkBindBufferMemory2 as vk::PFN_vkBindBufferMemory2)
        }
        b"vkGetBufferDeviceAddress" | b"vkGetBufferDeviceAddressKHR" | b"vkGetBufferDeviceAddressEXT" => {
            std::mem::transmute(vkGetBufferDeviceAddress as vk::PFN_vkGetBufferDeviceAddress)
        }
        b"vkGetDeviceBufferMemoryRequirements" | b"vkGetDeviceBufferMemoryRequirementsKHR" => {
            std::mem::transmute(vkGetDeviceBufferMemoryRequirements as vk::PFN_vkGetDeviceBufferMemoryRequirements)
        }

        // Render pass 2
        b"vkCreateRenderPass2" | b"vkCreateRenderPass2KHR" => {
            std::mem::transmute(vkCreateRenderPass2 as vk::PFN_vkCreateRenderPass2)
        }

        // Dynamic rendering (Vulkan 1.3 / VK_KHR_dynamic_rendering)
        b"vkCmdBeginRendering" | b"vkCmdBeginRenderingKHR" => {
            std::mem::transmute(vkCmdBeginRendering as vk::PFN_vkCmdBeginRendering)
        }
        b"vkCmdEndRendering" | b"vkCmdEndRenderingKHR" => {
            std::mem::transmute(vkCmdEndRendering as vk::PFN_vkCmdEndRendering)
        }

        // Indirect draw
        b"vkCmdDrawIndirect" => {
            std::mem::transmute(vkCmdDrawIndirect as vk::PFN_vkCmdDrawIndirect)
        }
        b"vkCmdDrawIndexedIndirect" => {
            std::mem::transmute(vkCmdDrawIndexedIndirect as vk::PFN_vkCmdDrawIndexedIndirect)
        }

        // Buffer fill / update
        b"vkCmdFillBuffer" => std::mem::transmute(vkCmdFillBuffer as vk::PFN_vkCmdFillBuffer),
        b"vkCmdUpdateBuffer" => std::mem::transmute(vkCmdUpdateBuffer as vk::PFN_vkCmdUpdateBuffer),

        // Synchronization2 (VK_KHR_synchronization2 / core 1.3)
        b"vkCmdPipelineBarrier2" | b"vkCmdPipelineBarrier2KHR" => {
            std::mem::transmute(vkCmdPipelineBarrier2 as vk::PFN_vkCmdPipelineBarrier2)
        }
        b"vkQueueSubmit2" | b"vkQueueSubmit2KHR" => {
            std::mem::transmute(vkQueueSubmit2 as vk::PFN_vkQueueSubmit2)
        }

        // Copy commands 2 (VK_KHR_copy_commands2 / core 1.3)
        b"vkCmdCopyBuffer2" | b"vkCmdCopyBuffer2KHR" => {
            std::mem::transmute(vkCmdCopyBuffer2 as vk::PFN_vkCmdCopyBuffer2)
        }
        b"vkCmdCopyImage2" | b"vkCmdCopyImage2KHR" => {
            std::mem::transmute(vkCmdCopyImage2 as vk::PFN_vkCmdCopyImage2)
        }
        b"vkCmdCopyBufferToImage2" | b"vkCmdCopyBufferToImage2KHR" => {
            std::mem::transmute(vkCmdCopyBufferToImage2 as vk::PFN_vkCmdCopyBufferToImage2)
        }
        b"vkCmdCopyImageToBuffer2" | b"vkCmdCopyImageToBuffer2KHR" => {
            std::mem::transmute(vkCmdCopyImageToBuffer2 as vk::PFN_vkCmdCopyImageToBuffer2)
        }
        b"vkCmdBlitImage2" | b"vkCmdBlitImage2KHR" => {
            std::mem::transmute(vkCmdBlitImage2 as vk::PFN_vkCmdBlitImage2)
        }

        // Resolve image
        b"vkCmdResolveImage" => {
            std::mem::transmute(vkCmdResolveImage as vk::PFN_vkCmdResolveImage)
        }

        // Extended dynamic state (VK_EXT_extended_dynamic_state / core 1.3)
        b"vkCmdSetCullMode" | b"vkCmdSetCullModeEXT" => {
            std::mem::transmute(vkCmdSetCullMode as vk::PFN_vkCmdSetCullMode)
        }
        b"vkCmdSetFrontFace" | b"vkCmdSetFrontFaceEXT" => {
            std::mem::transmute(vkCmdSetFrontFace as vk::PFN_vkCmdSetFrontFace)
        }
        b"vkCmdSetPrimitiveTopology" | b"vkCmdSetPrimitiveTopologyEXT" => {
            std::mem::transmute(vkCmdSetPrimitiveTopology as vk::PFN_vkCmdSetPrimitiveTopology)
        }
        b"vkCmdSetDepthTestEnable" | b"vkCmdSetDepthTestEnableEXT" => {
            std::mem::transmute(vkCmdSetDepthTestEnable as vk::PFN_vkCmdSetDepthTestEnable)
        }
        b"vkCmdSetDepthWriteEnable" | b"vkCmdSetDepthWriteEnableEXT" => {
            std::mem::transmute(vkCmdSetDepthWriteEnable as vk::PFN_vkCmdSetDepthWriteEnable)
        }
        b"vkCmdSetDepthCompareOp" | b"vkCmdSetDepthCompareOpEXT" => {
            std::mem::transmute(vkCmdSetDepthCompareOp as vk::PFN_vkCmdSetDepthCompareOp)
        }
        b"vkCmdSetDepthBiasEnable" | b"vkCmdSetDepthBiasEnableEXT" => {
            std::mem::transmute(vkCmdSetDepthBiasEnable as vk::PFN_vkCmdSetDepthBiasEnable)
        }
        b"vkCmdSetStencilTestEnable" | b"vkCmdSetStencilTestEnableEXT" => {
            std::mem::transmute(vkCmdSetStencilTestEnable as vk::PFN_vkCmdSetStencilTestEnable)
        }
        b"vkCmdSetStencilOp" | b"vkCmdSetStencilOpEXT" => {
            std::mem::transmute(vkCmdSetStencilOp as vk::PFN_vkCmdSetStencilOp)
        }
        b"vkCmdSetDepthBounds" => {
            std::mem::transmute(vkCmdSetDepthBounds as vk::PFN_vkCmdSetDepthBounds)
        }
        b"vkCmdSetLineWidth" => {
            std::mem::transmute(vkCmdSetLineWidth as vk::PFN_vkCmdSetLineWidth)
        }
        b"vkCmdSetDepthBias" => {
            std::mem::transmute(vkCmdSetDepthBias as vk::PFN_vkCmdSetDepthBias)
        }

        // Render pass 2 commands (VK_KHR_create_renderpass2 / core 1.2)
        b"vkCmdBeginRenderPass2" | b"vkCmdBeginRenderPass2KHR" => {
            std::mem::transmute(vkCmdBeginRenderPass2 as vk::PFN_vkCmdBeginRenderPass2)
        }
        b"vkCmdNextSubpass" => {
            std::mem::transmute(vkCmdNextSubpass as vk::PFN_vkCmdNextSubpass)
        }
        b"vkCmdNextSubpass2" | b"vkCmdNextSubpass2KHR" => {
            std::mem::transmute(vkCmdNextSubpass2 as vk::PFN_vkCmdNextSubpass2)
        }
        b"vkCmdEndRenderPass2" | b"vkCmdEndRenderPass2KHR" => {
            std::mem::transmute(vkCmdEndRenderPass2 as vk::PFN_vkCmdEndRenderPass2)
        }

        // Secondary command buffers
        b"vkCmdExecuteCommands" => {
            std::mem::transmute(vkCmdExecuteCommands as vk::PFN_vkCmdExecuteCommands)
        }

        // Dispatch base
        b"vkCmdDispatchBase" | b"vkCmdDispatchBaseKHR" => {
            std::mem::transmute(vkCmdDispatchBase as vk::PFN_vkCmdDispatchBase)
        }

        // Queue
        b"vkGetDeviceQueue2" => {
            std::mem::transmute(vkGetDeviceQueue2 as vk::PFN_vkGetDeviceQueue2)
        }

        // Fence / semaphore extras
        b"vkGetFenceStatus" => {
            std::mem::transmute(vkGetFenceStatus as vk::PFN_vkGetFenceStatus)
        }
        b"vkGetSemaphoreCounterValue" | b"vkGetSemaphoreCounterValueKHR" => {
            std::mem::transmute(vkGetSemaphoreCounterValue as vk::PFN_vkGetSemaphoreCounterValue)
        }
        b"vkSignalSemaphore" | b"vkSignalSemaphoreKHR" => {
            std::mem::transmute(vkSignalSemaphore as vk::PFN_vkSignalSemaphore)
        }
        b"vkWaitSemaphores" | b"vkWaitSemaphoresKHR" => {
            std::mem::transmute(vkWaitSemaphores as vk::PFN_vkWaitSemaphores)
        }

        // Query pools
        b"vkCreateQueryPool" => {
            std::mem::transmute(vkCreateQueryPool as vk::PFN_vkCreateQueryPool)
        }
        b"vkDestroyQueryPool" => {
            std::mem::transmute(vkDestroyQueryPool as vk::PFN_vkDestroyQueryPool)
        }
        b"vkGetQueryPoolResults" => {
            std::mem::transmute(vkGetQueryPoolResults as vk::PFN_vkGetQueryPoolResults)
        }
        b"vkResetQueryPool" | b"vkResetQueryPoolEXT" => {
            std::mem::transmute(vkResetQueryPool as vk::PFN_vkResetQueryPool)
        }
        b"vkCmdBeginQuery" => {
            std::mem::transmute(vkCmdBeginQuery as vk::PFN_vkCmdBeginQuery)
        }
        b"vkCmdEndQuery" => {
            std::mem::transmute(vkCmdEndQuery as vk::PFN_vkCmdEndQuery)
        }
        b"vkCmdResetQueryPool" => {
            std::mem::transmute(vkCmdResetQueryPool as vk::PFN_vkCmdResetQueryPool)
        }
        b"vkCmdWriteTimestamp" => {
            std::mem::transmute(vkCmdWriteTimestamp as vk::PFN_vkCmdWriteTimestamp)
        }
        b"vkCmdWriteTimestamp2" | b"vkCmdWriteTimestamp2KHR" => {
            std::mem::transmute(vkCmdWriteTimestamp2 as vk::PFN_vkCmdWriteTimestamp2)
        }
        b"vkCmdCopyQueryPoolResults" => {
            std::mem::transmute(vkCmdCopyQueryPoolResults as vk::PFN_vkCmdCopyQueryPoolResults)
        }

        // Private data (VK_EXT_private_data / core 1.3)
        b"vkCreatePrivateDataSlot" | b"vkCreatePrivateDataSlotEXT" => {
            std::mem::transmute(vkCreatePrivateDataSlot as vk::PFN_vkCreatePrivateDataSlot)
        }
        b"vkDestroyPrivateDataSlot" | b"vkDestroyPrivateDataSlotEXT" => {
            std::mem::transmute(vkDestroyPrivateDataSlot as vk::PFN_vkDestroyPrivateDataSlot)
        }
        b"vkSetPrivateData" | b"vkSetPrivateDataEXT" => {
            std::mem::transmute(vkSetPrivateData as vk::PFN_vkSetPrivateData)
        }
        b"vkGetPrivateData" | b"vkGetPrivateDataEXT" => {
            std::mem::transmute(vkGetPrivateData as vk::PFN_vkGetPrivateData)
        }

        // Descriptor set layout support
        b"vkGetDescriptorSetLayoutSupport" | b"vkGetDescriptorSetLayoutSupportKHR" => {
            std::mem::transmute(vkGetDescriptorSetLayoutSupport as vk::PFN_vkGetDescriptorSetLayoutSupport)
        }

        _ => None,
    }
}

// Device functions

#[no_mangle]
pub unsafe extern "system" fn vkDestroyDevice(
    device: vk::Device,
    p_allocator: *const vk::AllocationCallbacks,
) {
    info!("vkDestroyDevice called");
    crate::device::destroy_device(device, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceQueue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    crate::queue::get_device_queue(device, queue_family_index, queue_index, p_queue);
}

#[no_mangle]
pub unsafe extern "system" fn vkDeviceWaitIdle(device: vk::Device) -> vk::Result {
    match crate::device::device_wait_idle(device) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// Memory functions

#[no_mangle]
pub unsafe extern "system" fn vkAllocateMemory(
    device: vk::Device,
    p_allocate_info: *const vk::MemoryAllocateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_memory: *mut vk::DeviceMemory,
) -> vk::Result {
    match crate::memory::allocate_memory(device, p_allocate_info, p_allocator, p_memory) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkFreeMemory(
    device: vk::Device,
    memory: vk::DeviceMemory,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::memory::free_memory(device, memory, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkMapMemory(
    device: vk::Device,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    flags: vk::MemoryMapFlags,
    pp_data: *mut *mut std::ffi::c_void,
) -> vk::Result {
    match crate::memory::map_memory(device, memory, offset, size, flags, pp_data) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkUnmapMemory(device: vk::Device, memory: vk::DeviceMemory) {
    crate::memory::unmap_memory(device, memory);
}

#[no_mangle]
pub unsafe extern "system" fn vkFlushMappedMemoryRanges(
    device: vk::Device,
    memory_range_count: u32,
    p_memory_ranges: *const vk::MappedMemoryRange,
) -> vk::Result {
    match crate::memory::flush_mapped_memory_ranges(device, memory_range_count, p_memory_ranges) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// Buffer functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateBuffer(
    device: vk::Device,
    p_create_info: *const vk::BufferCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_buffer: *mut vk::Buffer,
) -> vk::Result {
    match crate::buffer::create_buffer(device, p_create_info, p_allocator, p_buffer) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyBuffer(
    device: vk::Device,
    buffer: vk::Buffer,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::buffer::destroy_buffer(device, buffer, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkBindBufferMemory(
    device: vk::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_offset: vk::DeviceSize,
) -> vk::Result {
    match crate::buffer::bind_buffer_memory(device, buffer, memory, memory_offset) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferMemoryRequirements(
    device: vk::Device,
    buffer: vk::Buffer,
    p_memory_requirements: *mut vk::MemoryRequirements,
) {
    crate::buffer::get_buffer_memory_requirements(device, buffer, p_memory_requirements);
}

// Image functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateImage(
    device: vk::Device,
    p_create_info: *const vk::ImageCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_image: *mut vk::Image,
) -> vk::Result {
    match crate::image::create_image(device, p_create_info, p_allocator, p_image) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyImage(
    device: vk::Device,
    image: vk::Image,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::image::destroy_image(device, image, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkBindImageMemory(
    device: vk::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    memory_offset: vk::DeviceSize,
) -> vk::Result {
    match crate::image::bind_image_memory(device, image, memory, memory_offset) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetImageMemoryRequirements(
    device: vk::Device,
    image: vk::Image,
    p_memory_requirements: *mut vk::MemoryRequirements,
) {
    crate::image::get_image_memory_requirements(device, image, p_memory_requirements);
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateImageView(
    device: vk::Device,
    p_create_info: *const vk::ImageViewCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_view: *mut vk::ImageView,
) -> vk::Result {
    match crate::image::create_image_view(device, p_create_info, p_allocator, p_view) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyImageView(
    device: vk::Device,
    image_view: vk::ImageView,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::image::destroy_image_view(device, image_view, p_allocator);
}

// Sampler functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateSampler(
    device: vk::Device,
    p_create_info: *const vk::SamplerCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_sampler: *mut vk::Sampler,
) -> vk::Result {
    match crate::sampler::create_sampler(device, p_create_info, p_allocator, p_sampler) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySampler(
    device: vk::Device,
    sampler: vk::Sampler,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::sampler::destroy_sampler(device, sampler, p_allocator);
}

// Descriptor functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateDescriptorSetLayout(
    device: vk::Device,
    p_create_info: *const vk::DescriptorSetLayoutCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_set_layout: *mut vk::DescriptorSetLayout,
) -> vk::Result {
    match crate::descriptor::create_descriptor_set_layout(
        device,
        p_create_info,
        p_allocator,
        p_set_layout,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyDescriptorSetLayout(
    device: vk::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::descriptor::destroy_descriptor_set_layout(device, descriptor_set_layout, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateDescriptorPool(
    device: vk::Device,
    p_create_info: *const vk::DescriptorPoolCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_descriptor_pool: *mut vk::DescriptorPool,
) -> vk::Result {
    match crate::descriptor::create_descriptor_pool(
        device,
        p_create_info,
        p_allocator,
        p_descriptor_pool,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyDescriptorPool(
    device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::descriptor::destroy_descriptor_pool(device, descriptor_pool, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkAllocateDescriptorSets(
    device: vk::Device,
    p_allocate_info: *const vk::DescriptorSetAllocateInfo,
    p_descriptor_sets: *mut vk::DescriptorSet,
) -> vk::Result {
    match crate::descriptor::allocate_descriptor_sets(device, p_allocate_info, p_descriptor_sets) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkUpdateDescriptorSets(
    device: vk::Device,
    descriptor_write_count: u32,
    p_descriptor_writes: *const vk::WriteDescriptorSet,
    descriptor_copy_count: u32,
    p_descriptor_copies: *const vk::CopyDescriptorSet,
) {
    crate::descriptor::update_descriptor_sets(
        device,
        descriptor_write_count,
        p_descriptor_writes,
        descriptor_copy_count,
        p_descriptor_copies,
    );
}

// Pipeline functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateShaderModule(
    device: vk::Device,
    p_create_info: *const vk::ShaderModuleCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_shader_module: *mut vk::ShaderModule,
) -> vk::Result {
    match crate::pipeline::create_shader_module(device, p_create_info, p_allocator, p_shader_module)
    {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyShaderModule(
    device: vk::Device,
    shader_module: vk::ShaderModule,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::pipeline::destroy_shader_module(device, shader_module, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkCreatePipelineLayout(
    device: vk::Device,
    p_create_info: *const vk::PipelineLayoutCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_pipeline_layout: *mut vk::PipelineLayout,
) -> vk::Result {
    match crate::pipeline::create_pipeline_layout(
        device,
        p_create_info,
        p_allocator,
        p_pipeline_layout,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyPipelineLayout(
    device: vk::Device,
    pipeline_layout: vk::PipelineLayout,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::pipeline::destroy_pipeline_layout(device, pipeline_layout, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateGraphicsPipelines(
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::GraphicsPipelineCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    match crate::pipeline::create_graphics_pipelines(
        device,
        pipeline_cache,
        create_info_count,
        p_create_infos,
        p_allocator,
        p_pipelines,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateComputePipelines(
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::ComputePipelineCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    match crate::pipeline::create_compute_pipelines(
        device,
        pipeline_cache,
        create_info_count,
        p_create_infos,
        p_allocator,
        p_pipelines,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyPipeline(
    device: vk::Device,
    pipeline: vk::Pipeline,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::pipeline::destroy_pipeline(device, pipeline, p_allocator);
}

// Render pass functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateRenderPass(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_render_pass: *mut vk::RenderPass,
) -> vk::Result {
    match crate::render_pass::create_render_pass(device, p_create_info, p_allocator, p_render_pass)
    {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyRenderPass(
    device: vk::Device,
    render_pass: vk::RenderPass,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::render_pass::destroy_render_pass(device, render_pass, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateFramebuffer(
    device: vk::Device,
    p_create_info: *const vk::FramebufferCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_framebuffer: *mut vk::Framebuffer,
) -> vk::Result {
    match crate::framebuffer::create_framebuffer(device, p_create_info, p_allocator, p_framebuffer)
    {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyFramebuffer(
    device: vk::Device,
    framebuffer: vk::Framebuffer,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::framebuffer::destroy_framebuffer(device, framebuffer, p_allocator);
}

// Command pool and buffer functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateCommandPool(
    device: vk::Device,
    p_create_info: *const vk::CommandPoolCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_command_pool: *mut vk::CommandPool,
) -> vk::Result {
    match crate::command_pool::create_command_pool(
        device,
        p_create_info,
        p_allocator,
        p_command_pool,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyCommandPool(
    device: vk::Device,
    command_pool: vk::CommandPool,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::command_pool::destroy_command_pool(device, command_pool, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkAllocateCommandBuffers(
    device: vk::Device,
    p_allocate_info: *const vk::CommandBufferAllocateInfo,
    p_command_buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    match crate::command_buffer::allocate_command_buffers(
        device,
        p_allocate_info,
        p_command_buffers,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkBeginCommandBuffer(
    command_buffer: vk::CommandBuffer,
    p_begin_info: *const vk::CommandBufferBeginInfo,
) -> vk::Result {
    match crate::command_buffer::begin_command_buffer(command_buffer, p_begin_info) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkEndCommandBuffer(command_buffer: vk::CommandBuffer) -> vk::Result {
    match crate::command_buffer::end_command_buffer(command_buffer) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginRenderPass(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo,
    contents: vk::SubpassContents,
) {
    crate::command_buffer::cmd_begin_render_pass(command_buffer, p_render_pass_begin, contents);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndRenderPass(command_buffer: vk::CommandBuffer) {
    crate::command_buffer::cmd_end_render_pass(command_buffer);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindPipeline(
    command_buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    crate::command_buffer::cmd_bind_pipeline(command_buffer, pipeline_bind_point, pipeline);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindDescriptorSets(
    command_buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    first_set: u32,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
    dynamic_offset_count: u32,
    p_dynamic_offsets: *const u32,
) {
    crate::command_buffer::cmd_bind_descriptor_sets(
        command_buffer,
        pipeline_bind_point,
        layout,
        first_set,
        descriptor_set_count,
        p_descriptor_sets,
        dynamic_offset_count,
        p_dynamic_offsets,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindVertexBuffers(
    command_buffer: vk::CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_buffers: *const vk::Buffer,
    p_offsets: *const vk::DeviceSize,
) {
    crate::command_buffer::cmd_bind_vertex_buffers(
        command_buffer,
        first_binding,
        binding_count,
        p_buffers,
        p_offsets,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindIndexBuffer(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
) {
    crate::command_buffer::cmd_bind_index_buffer(command_buffer, buffer, offset, index_type);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDraw(
    command_buffer: vk::CommandBuffer,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    crate::command_buffer::cmd_draw(
        command_buffer,
        vertex_count,
        instance_count,
        first_vertex,
        first_instance,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndexed(
    command_buffer: vk::CommandBuffer,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
) {
    crate::command_buffer::cmd_draw_indexed(
        command_buffer,
        index_count,
        instance_count,
        first_index,
        vertex_offset,
        first_instance,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyBuffer(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    region_count: u32,
    p_regions: *const vk::BufferCopy,
) {
    crate::command_buffer::cmd_copy_buffer(
        command_buffer,
        src_buffer,
        dst_buffer,
        region_count,
        p_regions,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyBufferToImage(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::BufferImageCopy,
) {
    crate::command_buffer::cmd_copy_buffer_to_image(
        command_buffer,
        src_buffer,
        dst_image,
        dst_image_layout,
        region_count,
        p_regions,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdPipelineBarrier(
    command_buffer: vk::CommandBuffer,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barrier_count: u32,
    p_memory_barriers: *const vk::MemoryBarrier,
    buffer_memory_barrier_count: u32,
    p_buffer_memory_barriers: *const vk::BufferMemoryBarrier,
    image_memory_barrier_count: u32,
    p_image_memory_barriers: *const vk::ImageMemoryBarrier,
) {
    crate::command_buffer::cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        dependency_flags,
        memory_barrier_count,
        p_memory_barriers,
        buffer_memory_barrier_count,
        p_buffer_memory_barriers,
        image_memory_barrier_count,
        p_image_memory_barriers,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdPushConstants(
    command_buffer: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    stage_flags: vk::ShaderStageFlags,
    offset: u32,
    size: u32,
    p_values: *const std::ffi::c_void,
) {
    crate::command_buffer::cmd_push_constants(
        command_buffer,
        layout,
        stage_flags,
        offset,
        size,
        p_values,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDispatch(
    command_buffer: vk::CommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) {
    crate::command_buffer::cmd_dispatch(
        command_buffer,
        group_count_x,
        group_count_y,
        group_count_z,
    );
}

// Queue functions

#[no_mangle]
pub unsafe extern "system" fn vkQueueSubmit(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo,
    fence: vk::Fence,
) -> vk::Result {
    match crate::queue::queue_submit(queue, submit_count, p_submits, fence) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkQueueWaitIdle(queue: vk::Queue) -> vk::Result {
    match crate::queue::queue_wait_idle(queue) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// Synchronization functions

#[no_mangle]
pub unsafe extern "system" fn vkCreateFence(
    device: vk::Device,
    p_create_info: *const vk::FenceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_fence: *mut vk::Fence,
) -> vk::Result {
    match crate::sync::create_fence(device, p_create_info, p_allocator, p_fence) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyFence(
    device: vk::Device,
    fence: vk::Fence,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::sync::destroy_fence(device, fence, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkResetFences(
    device: vk::Device,
    fence_count: u32,
    p_fences: *const vk::Fence,
) -> vk::Result {
    match crate::sync::reset_fences(device, fence_count, p_fences) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkWaitForFences(
    device: vk::Device,
    fence_count: u32,
    p_fences: *const vk::Fence,
    wait_all: vk::Bool32,
    timeout: u64,
) -> vk::Result {
    match crate::sync::wait_for_fences(device, fence_count, p_fences, wait_all, timeout) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateSemaphore(
    device: vk::Device,
    p_create_info: *const vk::SemaphoreCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_semaphore: *mut vk::Semaphore,
) -> vk::Result {
    match crate::sync::create_semaphore(device, p_create_info, p_allocator, p_semaphore) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySemaphore(
    device: vk::Device,
    semaphore: vk::Semaphore,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::sync::destroy_semaphore(device, semaphore, p_allocator);
}

// Swapchain functions (KHR extension)

#[no_mangle]
pub unsafe extern "system" fn vkCreateSwapchainKHR(
    device: vk::Device,
    p_create_info: *const vk::SwapchainCreateInfoKHR,
    p_allocator: *const vk::AllocationCallbacks,
    p_swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    match crate::swapchain::create_swapchain_khr(device, p_create_info, p_allocator, p_swapchain) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySwapchainKHR(
    _device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::swapchain::destroy_swapchain_khr(swapchain, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetSwapchainImagesKHR(
    _device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_swapchain_image_count: *mut u32,
    p_swapchain_images: *mut vk::Image,
) -> vk::Result {
    match crate::swapchain::get_swapchain_images_khr(
        swapchain,
        p_swapchain_image_count,
        p_swapchain_images,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkAcquireNextImageKHR(
    _device: vk::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    p_image_index: *mut u32,
) -> vk::Result {
    match crate::swapchain::acquire_next_image_khr(
        swapchain,
        timeout,
        semaphore,
        fence,
        p_image_index,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkQueuePresentKHR(
    queue: vk::Queue,
    p_present_info: *const vk::PresentInfoKHR,
) -> vk::Result {
    match crate::swapchain::queue_present_khr(queue, p_present_info) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// Dynamic state commands

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetViewport(
    command_buffer: vk::CommandBuffer,
    first_viewport: u32,
    viewport_count: u32,
    p_viewports: *const vk::Viewport,
) {
    crate::command_buffer::cmd_set_viewport(
        command_buffer,
        first_viewport,
        viewport_count,
        p_viewports,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetScissor(
    command_buffer: vk::CommandBuffer,
    first_scissor: u32,
    scissor_count: u32,
    p_scissors: *const vk::Rect2D,
) {
    crate::command_buffer::cmd_set_scissor(
        command_buffer,
        first_scissor,
        scissor_count,
        p_scissors,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetBlendConstants(
    command_buffer: vk::CommandBuffer,
    blend_constants: *const [f32; 4],
) {
    crate::command_buffer::cmd_set_blend_constants(command_buffer, blend_constants);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetStencilReference(
    command_buffer: vk::CommandBuffer,
    face_mask: vk::StencilFaceFlags,
    reference: u32,
) {
    crate::command_buffer::cmd_set_stencil_reference(command_buffer, face_mask, reference);
}

// Clear commands

#[no_mangle]
pub unsafe extern "system" fn vkCmdClearColorImage(
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_layout: vk::ImageLayout,
    p_color: *const vk::ClearColorValue,
    range_count: u32,
    p_ranges: *const vk::ImageSubresourceRange,
) {
    crate::command_buffer::cmd_clear_color_image(
        command_buffer,
        image,
        image_layout,
        p_color,
        range_count,
        p_ranges,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdClearDepthStencilImage(
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_layout: vk::ImageLayout,
    p_depth_stencil: *const vk::ClearDepthStencilValue,
    range_count: u32,
    p_ranges: *const vk::ImageSubresourceRange,
) {
    crate::command_buffer::cmd_clear_depth_stencil_image(
        command_buffer,
        image,
        image_layout,
        p_depth_stencil,
        range_count,
        p_ranges,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdClearAttachments(
    command_buffer: vk::CommandBuffer,
    attachment_count: u32,
    p_attachments: *const vk::ClearAttachment,
    rect_count: u32,
    p_rects: *const vk::ClearRect,
) {
    crate::command_buffer::cmd_clear_attachments(
        command_buffer,
        attachment_count,
        p_attachments,
        rect_count,
        p_rects,
    );
}

// Copy commands

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyImage(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::ImageCopy,
) {
    crate::command_buffer::cmd_copy_image(
        command_buffer,
        src_image,
        src_image_layout,
        dst_image,
        dst_image_layout,
        region_count,
        p_regions,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyImageToBuffer(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_buffer: vk::Buffer,
    region_count: u32,
    p_regions: *const vk::BufferImageCopy,
) {
    crate::command_buffer::cmd_copy_image_to_buffer(
        command_buffer,
        src_image,
        src_image_layout,
        dst_buffer,
        region_count,
        p_regions,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBlitImage(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::ImageBlit,
    filter: vk::Filter,
) {
    crate::command_buffer::cmd_blit_image(
        command_buffer,
        src_image,
        src_image_layout,
        dst_image,
        dst_image_layout,
        region_count,
        p_regions,
        filter,
    );
}

// ============================================================================
// Physical device 2 functions (Vulkan 1.1 / KHR aliases)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceProperties2(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties2,
) {
    crate::instance::get_physical_device_properties2(physical_device, p_properties);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFeatures2(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures2,
) {
    crate::instance::get_physical_device_features2(physical_device, p_features);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceMemoryProperties2(
    physical_device: vk::PhysicalDevice,
    p_memory_properties: *mut vk::PhysicalDeviceMemoryProperties2,
) {
    crate::instance::get_physical_device_memory_properties2(physical_device, p_memory_properties);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceQueueFamilyProperties2(
    physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties2,
) {
    crate::instance::get_physical_device_queue_family_properties2(
        physical_device,
        p_queue_family_property_count,
        p_queue_family_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFormatProperties2(
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    p_format_properties: *mut vk::FormatProperties2,
) {
    crate::instance::get_physical_device_format_properties2(
        physical_device,
        format,
        p_format_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceImageFormatProperties2(
    physical_device: vk::PhysicalDevice,
    p_image_format_info: *const vk::PhysicalDeviceImageFormatInfo2,
    p_image_format_properties: *mut vk::ImageFormatProperties2,
) -> vk::Result {
    match crate::instance::get_physical_device_image_format_properties2(
        physical_device,
        p_image_format_info,
        p_image_format_properties,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSparseImageFormatProperties2(
    physical_device: vk::PhysicalDevice,
    p_format_info: *const vk::PhysicalDeviceSparseImageFormatInfo2,
    p_property_count: *mut u32,
    p_properties: *mut vk::SparseImageFormatProperties2,
) {
    crate::instance::get_physical_device_sparse_image_format_properties2(
        physical_device,
        p_format_info,
        p_property_count,
        p_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceExternalBufferProperties(
    physical_device: vk::PhysicalDevice,
    p_external_buffer_info: *const vk::PhysicalDeviceExternalBufferInfo,
    p_external_buffer_properties: *mut vk::ExternalBufferProperties,
) {
    crate::instance::get_physical_device_external_buffer_properties(
        physical_device,
        p_external_buffer_info,
        p_external_buffer_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceExternalFenceProperties(
    physical_device: vk::PhysicalDevice,
    p_external_fence_info: *const vk::PhysicalDeviceExternalFenceInfo,
    p_external_fence_properties: *mut vk::ExternalFenceProperties,
) {
    crate::instance::get_physical_device_external_fence_properties(
        physical_device,
        p_external_fence_info,
        p_external_fence_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceExternalSemaphoreProperties(
    physical_device: vk::PhysicalDevice,
    p_external_semaphore_info: *const vk::PhysicalDeviceExternalSemaphoreInfo,
    p_external_semaphore_properties: *mut vk::ExternalSemaphoreProperties,
) {
    crate::instance::get_physical_device_external_semaphore_properties(
        physical_device,
        p_external_semaphore_info,
        p_external_semaphore_properties,
    );
}

// Surface functions (KHR)

#[no_mangle]
pub unsafe extern "system" fn vkCreateWin32SurfaceKHR(
    instance: vk::Instance,
    p_create_info: *const vk::Win32SurfaceCreateInfoKHR,
    p_allocator: *const vk::AllocationCallbacks,
    p_surface: *mut vk::SurfaceKHR,
) -> vk::Result {
    match crate::surface::create_win32_surface_khr(instance, p_create_info, p_allocator, p_surface)
    {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySurfaceKHR(
    instance: vk::Instance,
    surface: vk::SurfaceKHR,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::surface::destroy_surface_khr(instance, surface, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceSupportKHR(
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    surface: vk::SurfaceKHR,
    p_supported: *mut vk::Bool32,
) -> vk::Result {
    match crate::surface::get_physical_device_surface_support_khr(
        physical_device,
        queue_family_index,
        surface,
        p_supported,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    p_surface_capabilities: *mut vk::SurfaceCapabilitiesKHR,
) -> vk::Result {
    match crate::surface::get_physical_device_surface_capabilities_khr(
        physical_device,
        surface,
        p_surface_capabilities,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceFormatsKHR(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    p_surface_format_count: *mut u32,
    p_surface_formats: *mut vk::SurfaceFormatKHR,
) -> vk::Result {
    match crate::surface::get_physical_device_surface_formats_khr(
        physical_device,
        surface,
        p_surface_format_count,
        p_surface_formats,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfacePresentModesKHR(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    p_present_mode_count: *mut u32,
    p_present_modes: *mut vk::PresentModeKHR,
) -> vk::Result {
    match crate::surface::get_physical_device_surface_present_modes_khr(
        physical_device,
        surface,
        p_present_mode_count,
        p_present_modes,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// ============================================================================
// Command buffer management (free / reset)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkFreeCommandBuffers(
    device: vk::Device,
    command_pool: vk::CommandPool,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    crate::command_buffer::free_command_buffers(
        device,
        command_pool,
        command_buffer_count,
        p_command_buffers,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkResetCommandBuffer(
    command_buffer: vk::CommandBuffer,
    flags: vk::CommandBufferResetFlags,
) -> vk::Result {
    match crate::command_buffer::reset_command_buffer(command_buffer, flags) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkResetCommandPool(
    device: vk::Device,
    command_pool: vk::CommandPool,
    flags: vk::CommandPoolResetFlags,
) -> vk::Result {
    match crate::command_buffer::reset_command_pool(device, command_pool, flags) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// ============================================================================
// Descriptor management (free / reset)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkFreeDescriptorSets(
    device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
) -> vk::Result {
    match crate::descriptor::free_descriptor_sets(
        device,
        descriptor_pool,
        descriptor_set_count,
        p_descriptor_sets,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkResetDescriptorPool(
    device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    flags: vk::DescriptorPoolResetFlags,
) -> vk::Result {
    match crate::descriptor::reset_descriptor_pool(device, descriptor_pool, flags) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// ============================================================================
// Image extras
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkGetImageSubresourceLayout(
    device: vk::Device,
    image: vk::Image,
    p_subresource: *const vk::ImageSubresource,
    p_layout: *mut vk::SubresourceLayout,
) {
    crate::image::get_image_subresource_layout(device, image, p_subresource, p_layout);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetImageMemoryRequirements2(
    device: vk::Device,
    p_info: *const vk::ImageMemoryRequirementsInfo2,
    p_memory_requirements: *mut vk::MemoryRequirements2,
) {
    crate::image::get_image_memory_requirements2(device, p_info, p_memory_requirements);
}

#[no_mangle]
pub unsafe extern "system" fn vkBindImageMemory2(
    device: vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const vk::BindImageMemoryInfo,
) -> vk::Result {
    match crate::image::bind_image_memory2(device, bind_info_count, p_bind_infos) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceImageMemoryRequirements(
    device: vk::Device,
    p_info: *const vk::DeviceImageMemoryRequirements,
    p_memory_requirements: *mut vk::MemoryRequirements2,
) {
    crate::image::get_device_image_memory_requirements(device, p_info, p_memory_requirements);
}

// ============================================================================
// Buffer extras
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferMemoryRequirements2(
    device: vk::Device,
    p_info: *const vk::BufferMemoryRequirementsInfo2,
    p_memory_requirements: *mut vk::MemoryRequirements2,
) {
    crate::buffer::get_buffer_memory_requirements2(device, p_info, p_memory_requirements);
}

#[no_mangle]
pub unsafe extern "system" fn vkBindBufferMemory2(
    device: vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const vk::BindBufferMemoryInfo,
) -> vk::Result {
    match crate::buffer::bind_buffer_memory2(device, bind_info_count, p_bind_infos) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferDeviceAddress(
    device: vk::Device,
    p_info: *const vk::BufferDeviceAddressInfo,
) -> vk::DeviceAddress {
    crate::buffer::get_buffer_device_address(device, p_info)
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceBufferMemoryRequirements(
    device: vk::Device,
    p_info: *const vk::DeviceBufferMemoryRequirements,
    p_memory_requirements: *mut vk::MemoryRequirements2,
) {
    crate::buffer::get_device_buffer_memory_requirements(device, p_info, p_memory_requirements);
}

// ============================================================================
// Render pass 2
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCreateRenderPass2(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo2,
    p_allocator: *const vk::AllocationCallbacks,
    p_render_pass: *mut vk::RenderPass,
) -> vk::Result {
    match crate::render_pass::create_render_pass2(device, p_create_info, p_allocator, p_render_pass)
    {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// ============================================================================
// Dynamic rendering (VK_KHR_dynamic_rendering / core 1.3)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginRendering(
    command_buffer: vk::CommandBuffer,
    p_rendering_info: *const vk::RenderingInfo,
) {
    crate::command_buffer::cmd_begin_rendering(command_buffer, p_rendering_info);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndRendering(command_buffer: vk::CommandBuffer) {
    crate::command_buffer::cmd_end_rendering(command_buffer);
}

// ============================================================================
// Indirect draw
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndirect(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    crate::command_buffer::cmd_draw_indirect(command_buffer, buffer, offset, draw_count, stride);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndexedIndirect(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    crate::command_buffer::cmd_draw_indexed_indirect(
        command_buffer,
        buffer,
        offset,
        draw_count,
        stride,
    );
}

// ============================================================================
// Buffer fill / update
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdFillBuffer(
    command_buffer: vk::CommandBuffer,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    data: u32,
) {
    crate::command_buffer::cmd_fill_buffer(command_buffer, dst_buffer, dst_offset, size, data);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdUpdateBuffer(
    command_buffer: vk::CommandBuffer,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    data_size: vk::DeviceSize,
    p_data: *const std::ffi::c_void,
) {
    crate::command_buffer::cmd_update_buffer(
        command_buffer,
        dst_buffer,
        dst_offset,
        data_size,
        p_data,
    );
}

// ============================================================================
// Synchronization 2 (VK_KHR_synchronization2 / core 1.3)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdPipelineBarrier2(
    command_buffer: vk::CommandBuffer,
    p_dependency_info: *const vk::DependencyInfo,
) {
    crate::command_buffer::cmd_pipeline_barrier2(command_buffer, p_dependency_info);
}

#[no_mangle]
pub unsafe extern "system" fn vkQueueSubmit2(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo2,
    fence: vk::Fence,
) -> vk::Result {
    match crate::queue::queue_submit2(queue, submit_count, p_submits, fence) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// ============================================================================
// Copy commands 2 (VK_KHR_copy_commands2 / core 1.3)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyBuffer2(
    command_buffer: vk::CommandBuffer,
    p_copy_buffer_info: *const vk::CopyBufferInfo2,
) {
    crate::command_buffer::cmd_copy_buffer2(command_buffer, p_copy_buffer_info);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyImage2(
    command_buffer: vk::CommandBuffer,
    p_copy_image_info: *const vk::CopyImageInfo2,
) {
    crate::command_buffer::cmd_copy_image2(command_buffer, p_copy_image_info);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyBufferToImage2(
    command_buffer: vk::CommandBuffer,
    p_copy_buffer_to_image_info: *const vk::CopyBufferToImageInfo2,
) {
    crate::command_buffer::cmd_copy_buffer_to_image2(
        command_buffer,
        p_copy_buffer_to_image_info,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyImageToBuffer2(
    command_buffer: vk::CommandBuffer,
    p_copy_image_to_buffer_info: *const vk::CopyImageToBufferInfo2,
) {
    crate::command_buffer::cmd_copy_image_to_buffer2(
        command_buffer,
        p_copy_image_to_buffer_info,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBlitImage2(
    command_buffer: vk::CommandBuffer,
    p_blit_image_info: *const vk::BlitImageInfo2,
) {
    crate::command_buffer::cmd_blit_image2(command_buffer, p_blit_image_info);
}

// ============================================================================
// Resolve image
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdResolveImage(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::ImageResolve,
) {
    crate::command_buffer::cmd_resolve_image(
        command_buffer,
        src_image,
        src_image_layout,
        dst_image,
        dst_image_layout,
        region_count,
        p_regions,
    );
}

// ============================================================================
// Extended dynamic state (VK_EXT_extended_dynamic_state / core 1.3)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetCullMode(
    command_buffer: vk::CommandBuffer,
    cull_mode: vk::CullModeFlags,
) {
    crate::command_buffer::cmd_set_cull_mode(command_buffer, cull_mode);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetFrontFace(
    command_buffer: vk::CommandBuffer,
    front_face: vk::FrontFace,
) {
    crate::command_buffer::cmd_set_front_face(command_buffer, front_face);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetPrimitiveTopology(
    command_buffer: vk::CommandBuffer,
    primitive_topology: vk::PrimitiveTopology,
) {
    crate::command_buffer::cmd_set_primitive_topology(command_buffer, primitive_topology);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthTestEnable(
    command_buffer: vk::CommandBuffer,
    depth_test_enable: vk::Bool32,
) {
    crate::command_buffer::cmd_set_depth_test_enable(command_buffer, depth_test_enable);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthWriteEnable(
    command_buffer: vk::CommandBuffer,
    depth_write_enable: vk::Bool32,
) {
    crate::command_buffer::cmd_set_depth_write_enable(command_buffer, depth_write_enable);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthCompareOp(
    command_buffer: vk::CommandBuffer,
    depth_compare_op: vk::CompareOp,
) {
    crate::command_buffer::cmd_set_depth_compare_op(command_buffer, depth_compare_op);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthBiasEnable(
    command_buffer: vk::CommandBuffer,
    depth_bias_enable: vk::Bool32,
) {
    crate::command_buffer::cmd_set_depth_bias_enable(command_buffer, depth_bias_enable);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetStencilTestEnable(
    command_buffer: vk::CommandBuffer,
    stencil_test_enable: vk::Bool32,
) {
    crate::command_buffer::cmd_set_stencil_test_enable(command_buffer, stencil_test_enable);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetStencilOp(
    command_buffer: vk::CommandBuffer,
    face_mask: vk::StencilFaceFlags,
    fail_op: vk::StencilOp,
    pass_op: vk::StencilOp,
    depth_fail_op: vk::StencilOp,
    compare_op: vk::CompareOp,
) {
    crate::command_buffer::cmd_set_stencil_op(
        command_buffer,
        face_mask,
        fail_op,
        pass_op,
        depth_fail_op,
        compare_op,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthBounds(
    command_buffer: vk::CommandBuffer,
    min_depth_bounds: f32,
    max_depth_bounds: f32,
) {
    crate::command_buffer::cmd_set_depth_bounds(
        command_buffer,
        min_depth_bounds,
        max_depth_bounds,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetLineWidth(
    command_buffer: vk::CommandBuffer,
    line_width: f32,
) {
    crate::command_buffer::cmd_set_line_width(command_buffer, line_width);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthBias(
    command_buffer: vk::CommandBuffer,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
) {
    crate::command_buffer::cmd_set_depth_bias(
        command_buffer,
        depth_bias_constant_factor,
        depth_bias_clamp,
        depth_bias_slope_factor,
    );
}

// ============================================================================
// Render pass 2 commands (VK_KHR_create_renderpass2 / core 1.2)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginRenderPass2(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo,
    p_subpass_begin_info: *const vk::SubpassBeginInfo,
) {
    crate::command_buffer::cmd_begin_render_pass2(
        command_buffer,
        p_render_pass_begin,
        p_subpass_begin_info,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdNextSubpass(
    command_buffer: vk::CommandBuffer,
    contents: vk::SubpassContents,
) {
    crate::command_buffer::cmd_next_subpass(command_buffer, contents);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdNextSubpass2(
    command_buffer: vk::CommandBuffer,
    p_subpass_begin_info: *const vk::SubpassBeginInfo,
    p_subpass_end_info: *const vk::SubpassEndInfo,
) {
    crate::command_buffer::cmd_next_subpass2(
        command_buffer,
        p_subpass_begin_info,
        p_subpass_end_info,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndRenderPass2(
    command_buffer: vk::CommandBuffer,
    p_subpass_end_info: *const vk::SubpassEndInfo,
) {
    crate::command_buffer::cmd_end_render_pass2(command_buffer, p_subpass_end_info);
}

// ============================================================================
// Secondary command buffers
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdExecuteCommands(
    command_buffer: vk::CommandBuffer,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    crate::command_buffer::cmd_execute_commands(
        command_buffer,
        command_buffer_count,
        p_command_buffers,
    );
}

// ============================================================================
// Dispatch base
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCmdDispatchBase(
    command_buffer: vk::CommandBuffer,
    base_group_x: u32,
    base_group_y: u32,
    base_group_z: u32,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) {
    crate::command_buffer::cmd_dispatch_base(
        command_buffer,
        base_group_x,
        base_group_y,
        base_group_z,
        group_count_x,
        group_count_y,
        group_count_z,
    );
}

// ============================================================================
// Queue extras
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceQueue2(
    device: vk::Device,
    p_queue_info: *const vk::DeviceQueueInfo2,
    p_queue: *mut vk::Queue,
) {
    crate::queue::get_device_queue2(device, p_queue_info, p_queue);
}

// ============================================================================
// Fence / semaphore extras
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkGetFenceStatus(
    device: vk::Device,
    fence: vk::Fence,
) -> vk::Result {
    match crate::sync::get_fence_status(device, fence) {
        Ok(r) => r,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetSemaphoreCounterValue(
    device: vk::Device,
    semaphore: vk::Semaphore,
    p_value: *mut u64,
) -> vk::Result {
    match crate::sync::get_semaphore_counter_value(device, semaphore, p_value) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkSignalSemaphore(
    device: vk::Device,
    p_signal_info: *const vk::SemaphoreSignalInfo,
) -> vk::Result {
    match crate::sync::signal_semaphore(device, p_signal_info) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkWaitSemaphores(
    device: vk::Device,
    p_wait_info: *const vk::SemaphoreWaitInfo,
    timeout: u64,
) -> vk::Result {
    match crate::sync::wait_semaphores(device, p_wait_info, timeout) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

// ============================================================================
// Query pools
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCreateQueryPool(
    device: vk::Device,
    p_create_info: *const vk::QueryPoolCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_query_pool: *mut vk::QueryPool,
) -> vk::Result {
    match crate::query::create_query_pool(device, p_create_info, p_allocator, p_query_pool) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyQueryPool(
    device: vk::Device,
    query_pool: vk::QueryPool,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::query::destroy_query_pool(device, query_pool, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetQueryPoolResults(
    device: vk::Device,
    query_pool: vk::QueryPool,
    first_query: u32,
    query_count: u32,
    data_size: usize,
    p_data: *mut std::ffi::c_void,
    stride: vk::DeviceSize,
    flags: vk::QueryResultFlags,
) -> vk::Result {
    crate::query::get_query_pool_results(
        device,
        query_pool,
        first_query,
        query_count,
        data_size,
        p_data,
        stride,
        flags,
    )
}

#[no_mangle]
pub unsafe extern "system" fn vkResetQueryPool(
    device: vk::Device,
    query_pool: vk::QueryPool,
    first_query: u32,
    query_count: u32,
) {
    crate::query::reset_query_pool(device, query_pool, first_query, query_count);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginQuery(
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    query: u32,
    flags: vk::QueryControlFlags,
) {
    // Record for completeness; no-op in replay since WebGPU doesn't expose occlusion queries.
    let _ = (command_buffer, query_pool, query, flags);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndQuery(
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    query: u32,
) {
    let _ = (command_buffer, query_pool, query);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdResetQueryPool(
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    first_query: u32,
    query_count: u32,
) {
    let _ = (command_buffer, query_pool, first_query, query_count);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdWriteTimestamp(
    command_buffer: vk::CommandBuffer,
    pipeline_stage: vk::PipelineStageFlags,
    query_pool: vk::QueryPool,
    query: u32,
) {
    let _ = (command_buffer, pipeline_stage, query_pool, query);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdWriteTimestamp2(
    command_buffer: vk::CommandBuffer,
    stage: vk::PipelineStageFlags2,
    query_pool: vk::QueryPool,
    query: u32,
) {
    let _ = (command_buffer, stage, query_pool, query);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyQueryPoolResults(
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    first_query: u32,
    query_count: u32,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    stride: vk::DeviceSize,
    flags: vk::QueryResultFlags,
) {
    let _ = (
        command_buffer,
        query_pool,
        first_query,
        query_count,
        dst_buffer,
        dst_offset,
        stride,
        flags,
    );
}

// ============================================================================
// Private data (VK_EXT_private_data / core 1.3)
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkCreatePrivateDataSlot(
    device: vk::Device,
    p_create_info: *const vk::PrivateDataSlotCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_private_data_slot: *mut vk::PrivateDataSlot,
) -> vk::Result {
    match crate::device::create_private_data_slot(
        device,
        p_create_info,
        p_allocator,
        p_private_data_slot,
    ) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyPrivateDataSlot(
    device: vk::Device,
    private_data_slot: vk::PrivateDataSlot,
    p_allocator: *const vk::AllocationCallbacks,
) {
    crate::device::destroy_private_data_slot(device, private_data_slot, p_allocator);
}

#[no_mangle]
pub unsafe extern "system" fn vkSetPrivateData(
    device: vk::Device,
    object_type: vk::ObjectType,
    object_handle: u64,
    private_data_slot: vk::PrivateDataSlot,
    data: u64,
) -> vk::Result {
    match crate::device::set_private_data(device, object_type, object_handle, private_data_slot, data) {
        Ok(_) => vk::Result::SUCCESS,
        Err(e) => e.to_vk_result(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPrivateData(
    device: vk::Device,
    object_type: vk::ObjectType,
    object_handle: u64,
    private_data_slot: vk::PrivateDataSlot,
    p_data: *mut u64,
) {
    crate::device::get_private_data(device, object_type, object_handle, private_data_slot, p_data);
}

// ============================================================================
// Descriptor set layout support
// ============================================================================

#[no_mangle]
pub unsafe extern "system" fn vkGetDescriptorSetLayoutSupport(
    device: vk::Device,
    p_create_info: *const vk::DescriptorSetLayoutCreateInfo,
    p_support: *mut vk::DescriptorSetLayoutSupport,
) {
    crate::device::get_descriptor_set_layout_support(device, p_create_info, p_support);
}
