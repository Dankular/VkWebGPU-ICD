//! Vulkan Queue implementation
//! Maps VkQueue to WebGPU GPUQueue

use ash::vk;
use log::debug;
use std::sync::Arc;

use crate::command_buffer;
use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static QUEUE_ALLOCATOR: HandleAllocator<VkQueueData> = HandleAllocator::new();

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

    let queue_handle = QUEUE_ALLOCATOR.allocate(queue_data);
    *p_queue = vk::Queue::from_raw(queue_handle);

    // Store in device's queue list
    if let Some(device_data) = device::get_device_data(device) {
        let mut queues = device_data.queues.write();
        queues.push(vk::Queue::from_raw(queue_handle));
    }
}

pub unsafe fn queue_submit(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo,
    fence: vk::Fence,
) -> Result<()> {
    let queue_data = QUEUE_ALLOCATOR
        .get(queue.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid queue".to_string()))?;

    let device_data = device::get_device_data(queue_data.device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    if submit_count == 0 {
        return Ok(());
    }

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
                    let finished_buffers = cmd_data.finish()?;

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        device_data.backend.queue.submit(finished_buffers);
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        for buffer in finished_buffers {
                            use wasm_bindgen::JsCast;
                            let js_array = js_sys::Array::new();
                            js_array.push(&buffer);
                            device_data.backend.queue.submit(&js_array);
                        }
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

pub unsafe fn queue_wait_idle(queue: vk::Queue) -> Result<()> {
    let queue_data = QUEUE_ALLOCATOR
        .get(queue.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid queue".to_string()))?;

    device::device_wait_idle(queue_data.device)
}

pub fn get_queue_data(queue: vk::Queue) -> Option<Arc<VkQueueData>> {
    QUEUE_ALLOCATOR.get(queue.as_raw())
}
