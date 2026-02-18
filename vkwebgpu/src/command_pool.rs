//! Vulkan Command Pool implementation

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::error::Result;
use crate::handle::HandleAllocator;

pub static COMMAND_POOL_ALLOCATOR: Lazy<HandleAllocator<VkCommandPoolData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkCommandPoolData {
    pub device: vk::Device,
    pub queue_family_index: u32,
    pub flags: vk::CommandPoolCreateFlags,
    pub allocated_buffers: RwLock<Vec<vk::CommandBuffer>>,
}

pub unsafe fn create_command_pool(
    device: vk::Device,
    p_create_info: *const vk::CommandPoolCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_command_pool: *mut vk::CommandPool,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating command pool for queue family {}",
        create_info.queue_family_index
    );

    let pool_data = VkCommandPoolData {
        device,
        queue_family_index: create_info.queue_family_index,
        flags: create_info.flags,
        allocated_buffers: RwLock::new(Vec::new()),
    };

    let pool_handle = COMMAND_POOL_ALLOCATOR.allocate(pool_data);
    *p_command_pool = Handle::from_raw(pool_handle);

    Ok(())
}

pub unsafe fn destroy_command_pool(
    _device: vk::Device,
    command_pool: vk::CommandPool,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if command_pool == vk::CommandPool::null() {
        return;
    }

    COMMAND_POOL_ALLOCATOR.remove(command_pool.as_raw());
}

pub fn get_command_pool_data(command_pool: vk::CommandPool) -> Option<Arc<VkCommandPoolData>> {
    COMMAND_POOL_ALLOCATOR.get(command_pool.as_raw())
}
