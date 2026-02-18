//! Vulkan Query Pool implementation
//!
//! WebGPU has limited query support (timestamps via QuerySet).
//! Occlusion and pipeline statistics queries are not supported.
//! We implement stubs that return zero results to prevent crashes.

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::error::Result;
use crate::handle::HandleAllocator;

pub static QUERY_POOL_ALLOCATOR: Lazy<HandleAllocator<VkQueryPoolData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkQueryPoolData {
    pub device: vk::Device,
    pub query_type: vk::QueryType,
    pub query_count: u32,
    /// Stored results — all 0 unless a real backend writes them.
    pub results: RwLock<Vec<u64>>,
}

pub unsafe fn create_query_pool(
    device: vk::Device,
    p_create_info: *const vk::QueryPoolCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_query_pool: *mut vk::QueryPool,
) -> Result<()> {
    let create_info = &*p_create_info;
    let count = create_info.query_count as usize;

    debug!(
        "Creating query pool: type={:?}, count={}",
        create_info.query_type, count
    );

    let data = VkQueryPoolData {
        device,
        query_type: create_info.query_type,
        query_count: create_info.query_count,
        results: RwLock::new(vec![0u64; count]),
    };

    let handle = QUERY_POOL_ALLOCATOR.allocate(data);
    *p_query_pool = Handle::from_raw(handle);

    Ok(())
}

pub unsafe fn destroy_query_pool(
    _device: vk::Device,
    query_pool: vk::QueryPool,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if query_pool == vk::QueryPool::null() {
        return;
    }
    QUERY_POOL_ALLOCATOR.remove(query_pool.as_raw());
}

/// vkResetQueryPool / vkResetQueryPoolEXT — zero out the requested range.
pub unsafe fn reset_query_pool(
    _device: vk::Device,
    query_pool: vk::QueryPool,
    first_query: u32,
    query_count: u32,
) {
    if let Some(data) = QUERY_POOL_ALLOCATOR.get(query_pool.as_raw()) {
        let mut results = data.results.write();
        let end = (first_query + query_count) as usize;
        let end = end.min(results.len());
        for r in &mut results[first_query as usize..end] {
            *r = 0;
        }
    }
}

/// vkGetQueryPoolResults — copy zeroed results into the caller's buffer.
pub unsafe fn get_query_pool_results(
    _device: vk::Device,
    query_pool: vk::QueryPool,
    first_query: u32,
    query_count: u32,
    data_size: usize,
    p_data: *mut std::ffi::c_void,
    stride: vk::DeviceSize,
    flags: vk::QueryResultFlags,
) -> vk::Result {
    let pool = match QUERY_POOL_ALLOCATOR.get(query_pool.as_raw()) {
        Some(p) => p,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let results = pool.results.read();
    let use_64bit = flags.contains(vk::QueryResultFlags::TYPE_64);
    let with_availability = flags.contains(vk::QueryResultFlags::WITH_AVAILABILITY);

    // stride == 0 means tightly packed
    let slot_size: usize = if use_64bit { 8 } else { 4 };
    let slots_per_query: usize = if with_availability { 2 } else { 1 };
    let effective_stride = if stride == 0 {
        (slot_size * slots_per_query) as u64
    } else {
        stride
    };

    let needed = (query_count as usize) * effective_stride as usize;
    if data_size < needed {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let dst = p_data as *mut u8;
    for i in 0..query_count as usize {
        let idx = first_query as usize + i;
        let val = if idx < results.len() { results[idx] } else { 0 };
        let base = i * effective_stride as usize;

        if use_64bit {
            let ptr = dst.add(base) as *mut u64;
            *ptr = val;
            if with_availability {
                *(ptr.add(1)) = 1; // available
            }
        } else {
            let ptr = dst.add(base) as *mut u32;
            *ptr = val as u32;
            if with_availability {
                *(ptr.add(1)) = 1u32;
            }
        }
    }

    vk::Result::SUCCESS
}

pub fn get_query_pool_data(query_pool: vk::QueryPool) -> Option<Arc<VkQueryPoolData>> {
    QUERY_POOL_ALLOCATOR.get(query_pool.as_raw())
}
