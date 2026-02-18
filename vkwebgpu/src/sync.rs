//! Vulkan Synchronization primitives

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Result, VkError};
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
    /// Semaphore type: BINARY or TIMELINE
    pub semaphore_type: vk::SemaphoreType,
    /// Current timeline counter value (only meaningful for TIMELINE semaphores)
    pub timeline_value: AtomicU64,
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
    _device: vk::Device,
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
    _device: vk::Device,
    fence_count: u32,
    p_fences: *const vk::Fence,
    wait_all: vk::Bool32,
    _timeout: u64,
) -> Result<()> {
    let _fences = std::slice::from_raw_parts(p_fences, fence_count as usize);

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
    p_create_info: *const vk::SemaphoreCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_semaphore: *mut vk::Semaphore,
) -> Result<()> {
    // Check pNext for VkSemaphoreTypeCreateInfo (timeline semaphores)
    let mut semaphore_type = vk::SemaphoreType::BINARY;
    let mut initial_value: u64 = 0;

    if !p_create_info.is_null() {
        let mut p_next = (*p_create_info).p_next as *const vk::BaseInStructure;
        while !p_next.is_null() {
            if (*p_next).s_type == vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO {
                let type_info = &*(p_next as *const vk::SemaphoreTypeCreateInfo);
                semaphore_type = type_info.semaphore_type;
                initial_value = type_info.initial_value;
                break;
            }
            p_next = (*p_next).p_next as *const vk::BaseInStructure;
        }
    }

    debug!(
        "Creating {:?} semaphore (initial_value={})",
        semaphore_type, initial_value
    );

    let semaphore_data = VkSemaphoreData {
        device,
        semaphore_type,
        timeline_value: AtomicU64::new(initial_value),
    };

    let semaphore_handle = SEMAPHORE_ALLOCATOR.allocate(semaphore_data);
    *p_semaphore = Handle::from_raw(semaphore_handle);

    Ok(())
}

pub unsafe fn get_fence_status(
    _device: vk::Device,
    fence: vk::Fence,
) -> Result<vk::Result> {
    if fence == vk::Fence::null() {
        return Ok(vk::Result::SUCCESS);
    }
    if let Some(fence_data) = FENCE_ALLOCATOR.get(fence.as_raw()) {
        if *fence_data.signaled.read() {
            Ok(vk::Result::SUCCESS)
        } else {
            Ok(vk::Result::NOT_READY)
        }
    } else {
        Err(VkError::InvalidHandle("Invalid fence".to_string()))
    }
}

pub unsafe fn get_semaphore_counter_value(
    _device: vk::Device,
    semaphore: vk::Semaphore,
    p_value: *mut u64,
) -> Result<()> {
    let sem_data = SEMAPHORE_ALLOCATOR
        .get(semaphore.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid semaphore".to_string()))?;
    *p_value = sem_data.timeline_value.load(Ordering::SeqCst);
    debug!("GetSemaphoreCounterValue: {}", *p_value);
    Ok(())
}

pub unsafe fn signal_semaphore(
    _device: vk::Device,
    p_signal_info: *const vk::SemaphoreSignalInfo,
) -> Result<()> {
    if p_signal_info.is_null() {
        return Ok(());
    }
    let info = &*p_signal_info;
    if let Some(sem_data) = SEMAPHORE_ALLOCATOR.get(info.semaphore.as_raw()) {
        sem_data.timeline_value.store(info.value, Ordering::SeqCst);
        debug!("SignalSemaphore: set timeline={}", info.value);
    }
    Ok(())
}

pub unsafe fn wait_semaphores(
    _device: vk::Device,
    p_wait_info: *const vk::SemaphoreWaitInfo,
    _timeout: u64,
) -> Result<()> {
    if p_wait_info.is_null() {
        return Ok(());
    }
    let info = &*p_wait_info;
    debug!(
        "WaitSemaphores: {} semaphores (WebGPU manages ordering)",
        info.semaphore_count
    );
    // WebGPU handles GPU-side ordering implicitly. For host-side waits on timeline
    // semaphores we optimistically return success, which is correct in our CPU-side
    // simulation model.
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
