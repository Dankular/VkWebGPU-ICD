//! Vulkan Queue implementation
//! Maps VkQueue to WebGPU GPUQueue

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::command_buffer;
use crate::device;
use crate::memory;
use crate::error::{Result, VkError};
use crate::handle::{self, HandleAllocator};

pub static QUEUE_ALLOCATOR: Lazy<HandleAllocator<VkQueueData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkQueueData {
    pub device: vk::Device,
    pub family_index: u32,
    pub queue_index: u32,
}

pub unsafe fn get_device_queue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    debug!(
        "Getting queue family {} index {}",
        queue_family_index, queue_index
    );

    let queue_data = VkQueueData {
        device,
        family_index: queue_family_index,
        queue_index,
    };

    let queue_index = QUEUE_ALLOCATOR.allocate(queue_data);
    let queue_ptr = handle::alloc_dispatchable(queue_index);
    *p_queue = Handle::from_raw(queue_ptr);

    // Store in device's queue list
    if let Some(device_data) = device::get_device_data(device) {
        let mut queues = device_data.queues.write();
        queues.push(Handle::from_raw(queue_ptr));
    }
}

pub unsafe fn queue_submit(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo,
    fence: vk::Fence,
) -> Result<()> {
    let queue_data = QUEUE_ALLOCATOR
        .get_dispatchable(queue.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid queue".to_string()))?;

    let device_data = device::get_device_data(queue_data.device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    if submit_count == 0 {
        return Ok(());
    }

    // Flush HOST_COHERENT mapped memory before command replay.
    // Applications that allocate HOST_COHERENT memory do not call
    // vkFlushMappedMemoryRanges — they rely on automatic coherence.
    // We must push their CPU writes to the wgpu Buffers now, before
    // any recorded copy / draw commands reference those buffers.
    memory::flush_all_mapped_memory(queue_data.device);

    let submits = std::slice::from_raw_parts(p_submits, submit_count as usize);

    for submit in submits {
        // Process wait semaphores (WebGPU handles sync implicitly, so we just note them)
        if submit.wait_semaphore_count > 0 {
            debug!("Wait semaphores: {}", submit.wait_semaphore_count);
        }

        // Submit command buffers
        if submit.command_buffer_count > 0 {
            let cmd_buffers = std::slice::from_raw_parts(
                submit.p_command_buffers,
                submit.command_buffer_count as usize,
            );

            for &cmd_buffer in cmd_buffers {
                if let Some(cmd_data) = command_buffer::get_command_buffer_data(cmd_buffer) {
                    // Replay commands to create WebGPU command buffer
                    let webgpu_cmd_buffer =
                        command_buffer::replay_commands(&cmd_data, &device_data.backend)?;

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        device_data
                            .backend
                            .queue
                            .submit(std::iter::once(webgpu_cmd_buffer));
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        use wasm_bindgen::JsCast;
                        let js_array = js_sys::Array::new();
                        js_array.push(&webgpu_cmd_buffer);
                        device_data.backend.queue.submit(&js_array);
                    }
                }
            }
        }

        // Process signal semaphores
        if submit.signal_semaphore_count > 0 {
            debug!("Signal semaphores: {}", submit.signal_semaphore_count);
        }
    }

    // Handle fence signaling
    if fence != vk::Fence::null() {
        crate::sync::signal_fence(fence)?;
    }

    Ok(())
}

/// vkQueueSubmit2 (Vulkan 1.3 / VK_KHR_synchronization2)
pub unsafe fn queue_submit2(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo2,
    fence: vk::Fence,
) -> Result<()> {
    let queue_data = QUEUE_ALLOCATOR
        .get_dispatchable(queue.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid queue".to_string()))?;

    let device_data = device::get_device_data(queue_data.device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    if submit_count == 0 {
        // Fence-only submit: just signal the fence
        if fence != vk::Fence::null() {
            crate::sync::signal_fence(fence)?;
        }
        return Ok(());
    }

    // Flush HOST_COHERENT mapped memory before command replay (same reason as queue_submit).
    memory::flush_all_mapped_memory(queue_data.device);

    let submits = std::slice::from_raw_parts(p_submits, submit_count as usize);

    for submit in submits {
        if submit.wait_semaphore_info_count > 0 {
            debug!(
                "QueueSubmit2: wait semaphores: {}",
                submit.wait_semaphore_info_count
            );
        }

        // Submit command buffers
        if submit.command_buffer_info_count > 0 {
            let cmd_buffer_infos = std::slice::from_raw_parts(
                submit.p_command_buffer_infos,
                submit.command_buffer_info_count as usize,
            );

            for info in cmd_buffer_infos {
                let cmd_buffer = info.command_buffer;
                if let Some(cmd_data) = command_buffer::get_command_buffer_data(cmd_buffer) {
                    let webgpu_cmd_buffer =
                        command_buffer::replay_commands(&cmd_data, &device_data.backend)?;

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        device_data
                            .backend
                            .queue
                            .submit(std::iter::once(webgpu_cmd_buffer));
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        use wasm_bindgen::JsCast;
                        let js_array = js_sys::Array::new();
                        js_array.push(&webgpu_cmd_buffer);
                        device_data.backend.queue.submit(&js_array);
                    }
                }
            }
        }

        // Update timeline semaphore values for signal operations
        if submit.signal_semaphore_info_count > 0 {
            let signal_infos = std::slice::from_raw_parts(
                submit.p_signal_semaphore_infos,
                submit.signal_semaphore_info_count as usize,
            );
            for sig in signal_infos {
                if let Some(sem_data) =
                    crate::sync::SEMAPHORE_ALLOCATOR.get(sig.semaphore.as_raw())
                {
                    if sem_data.semaphore_type == vk::SemaphoreType::TIMELINE {
                        use std::sync::atomic::Ordering;
                        sem_data.timeline_value.store(sig.value, Ordering::SeqCst);
                        debug!("QueueSubmit2: signaled timeline semaphore value={}", sig.value);
                    }
                }
            }
        }
    }

    if fence != vk::Fence::null() {
        crate::sync::signal_fence(fence)?;
    }

    Ok(())
}

/// vkGetDeviceQueue2 (Vulkan 1.1)
pub unsafe fn get_device_queue2(
    device: vk::Device,
    p_queue_info: *const vk::DeviceQueueInfo2,
    p_queue: *mut vk::Queue,
) {
    if p_queue_info.is_null() {
        return;
    }
    let info = &*p_queue_info;
    get_device_queue(device, info.queue_family_index, info.queue_index, p_queue);
}

pub unsafe fn queue_wait_idle(queue: vk::Queue) -> Result<()> {
    let queue_data = QUEUE_ALLOCATOR
        .get_dispatchable(queue.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid queue".to_string()))?;

    device::device_wait_idle(queue_data.device)
}

pub fn get_queue_data(queue: vk::Queue) -> Option<Arc<VkQueueData>> {
    QUEUE_ALLOCATOR.get_dispatchable(queue.as_raw())
}
