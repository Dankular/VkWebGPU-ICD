//! Vulkan Synchronization primitives

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::error::Result;
use crate::handle::HandleAllocator;

pub static FENCE_ALLOCATOR: Lazy<HandleAllocator<VkFenceData>> =
    Lazy::new(|| HandleAllocator::new());
pub static SEMAPHORE_ALLOCATOR: Lazy<HandleAllocator<VkSemaphoreData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkFenceData {
    pub device: vk::Device,
    pub signaled: RwLock<bool>,
}

pub struct VkSemaphoreData {
    pub device: vk::Device,
}

pub unsafe fn create_fence(
    device: vk::Device,
    p_create_info: *const vk::FenceCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_fence: *mut vk::Fence,
) -> Result<()> {
    let create_info = &*p_create_info;

    let signaled = create_info.flags.contains(vk::FenceCreateFlags::SIGNALED);

    debug!("Creating fence (signaled={})", signaled);

    let fence_data = VkFenceData {
        device,
        signaled: RwLock::new(signaled),
    };

    let fence_handle = FENCE_ALLOCATOR.allocate(fence_data);
    *p_fence = Handle::from_raw(fence_handle);

    Ok(())
}

pub unsafe fn destroy_fence(
    _device: vk::Device,
    fence: vk::Fence,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if fence == vk::Fence::null() {
        return;
    }

    FENCE_ALLOCATOR.remove(fence.as_raw());
}

pub unsafe fn reset_fences(
    device: vk::Device,
    fence_count: u32,
    p_fences: *const vk::Fence,
) -> Result<()> {
    let fences = std::slice::from_raw_parts(p_fences, fence_count as usize);

    for &fence in fences {
        if let Some(fence_data) = FENCE_ALLOCATOR.get(fence.as_raw()) {
            *fence_data.signaled.write() = false;
        }
    }

    debug!("Reset {} fences", fence_count);
    Ok(())
}

pub unsafe fn wait_for_fences(
    device: vk::Device,
    fence_count: u32,
    p_fences: *const vk::Fence,
    wait_all: vk::Bool32,
    timeout: u64,
) -> Result<()> {
    let fences = std::slice::from_raw_parts(p_fences, fence_count as usize);

    debug!("Waiting for {} fences (wait_all={})", fence_count, wait_all);

    // Simplified: assume all fences are signaled
    Ok(())
}

pub fn signal_fence(fence: vk::Fence) -> Result<()> {
    if let Some(fence_data) = FENCE_ALLOCATOR.get(fence.as_raw()) {
        *fence_data.signaled.write() = true;
    }
    Ok(())
}

pub unsafe fn create_semaphore(
    device: vk::Device,
    _p_create_info: *const vk::SemaphoreCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_semaphore: *mut vk::Semaphore,
) -> Result<()> {
    debug!("Creating semaphore");

    let semaphore_data = VkSemaphoreData { device };

    let semaphore_handle = SEMAPHORE_ALLOCATOR.allocate(semaphore_data);
    *p_semaphore = Handle::from_raw(semaphore_handle);

    Ok(())
}

pub unsafe fn destroy_semaphore(
    _device: vk::Device,
    semaphore: vk::Semaphore,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if semaphore == vk::Semaphore::null() {
        return;
    }

    SEMAPHORE_ALLOCATOR.remove(semaphore.as_raw());
}
