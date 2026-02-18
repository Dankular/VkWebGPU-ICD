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
    debug!("vk_icdGetInstanceProcAddr: {:?}", name);

    match name.to_bytes() {
        b"vkCreateInstance" => std::mem::transmute(vkCreateInstance as vk::PFN_vkCreateInstance),
        b"vkDestroyInstance" => std::mem::transmute(vkDestroyInstance as vk::PFN_vkDestroyInstance),
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
        b"vkCreateDevice" => std::mem::transmute(vkCreateDevice as vk::PFN_vkCreateDevice),
        b"vkGetDeviceProcAddr" => {
            std::mem::transmute(vkGetDeviceProcAddr as vk::PFN_vkGetDeviceProcAddr)
        }
        _ => None,
    }
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
pub unsafe extern "system" fn vkUnmapMemory(device: vk::Device,
    memory: vk::DeviceMemory) {
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
