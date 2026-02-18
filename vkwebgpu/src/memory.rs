//! Vulkan Memory implementation
//!
//! WebGPU doesn't expose explicit memory allocation like Vulkan.
//! We track allocations but map them to WebGPU resources lazily.

use ash::vk;
use log::debug;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static MEMORY_ALLOCATOR: HandleAllocator<VkDeviceMemoryData> = HandleAllocator::new();

pub struct VkDeviceMemoryData {
    pub device: vk::Device,
    pub size: vk::DeviceSize,
    pub memory_type_index: u32,
    pub mapped_ptr: RwLock<Option<*mut u8>>,
    pub data: RwLock<Vec<u8>>,
}

unsafe impl Send for VkDeviceMemoryData {}
unsafe impl Sync for VkDeviceMemoryData {}

pub unsafe fn allocate_memory(
    device: vk::Device,
    p_allocate_info: *const vk::MemoryAllocateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_memory: *mut vk::DeviceMemory,
) -> Result<()> {
    let allocate_info = &*p_allocate_info;

    debug!(
        "Allocating {} bytes of memory type {}",
        allocate_info.allocation_size, allocate_info.memory_type_index
    );

    let memory_data = VkDeviceMemoryData {
        device,
        size: allocate_info.allocation_size,
        memory_type_index: allocate_info.memory_type_index,
        mapped_ptr: RwLock::new(None),
        data: RwLock::new(vec![0u8; allocate_info.allocation_size as usize]),
    };

    let memory_handle = MEMORY_ALLOCATOR.allocate(memory_data);
    *p_memory = vk::DeviceMemory::from_raw(memory_handle);

    Ok(())
}

pub unsafe fn free_memory(memory: vk::DeviceMemory, _p_allocator: *const vk::AllocationCallbacks) {
    if memory == vk::DeviceMemory::null() {
        return;
    }

    MEMORY_ALLOCATOR.remove(memory.as_raw());
    debug!("Freed device memory");
}

pub unsafe fn map_memory(
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    _flags: vk::MemoryMapFlags,
    pp_data: *mut *mut std::ffi::c_void,
) -> Result<()> {
    let memory_data = MEMORY_ALLOCATOR
        .get(memory.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid device memory".to_string()))?;

    let mut data = memory_data.data.write();
    let slice = if size == vk::WHOLE_SIZE {
        &mut data[offset as usize..]
    } else {
        &mut data[offset as usize..(offset + size) as usize]
    };

    let ptr = slice.as_mut_ptr() as *mut std::ffi::c_void;
    *pp_data = ptr;

    let mut mapped_ptr = memory_data.mapped_ptr.write();
    *mapped_ptr = Some(ptr as *mut u8);

    debug!("Mapped memory at offset {} size {}", offset, size);
    Ok(())
}

pub unsafe fn unmap_memory(memory: vk::DeviceMemory) {
    if let Some(memory_data) = MEMORY_ALLOCATOR.get(memory.as_raw()) {
        let mut mapped_ptr = memory_data.mapped_ptr.write();
        *mapped_ptr = None;
        debug!("Unmapped memory");
    }
}

pub unsafe fn flush_mapped_memory_ranges(
    _device: vk::Device,
    memory_range_count: u32,
    p_memory_ranges: *const vk::MappedMemoryRange,
) -> Result<()> {
    if memory_range_count == 0 {
        return Ok(());
    }

    let ranges = std::slice::from_raw_parts(p_memory_ranges, memory_range_count as usize);
    for range in ranges {
        debug!(
            "Flushing memory range: offset={}, size={}",
            range.offset, range.size
        );
    }

    Ok(())
}

pub unsafe fn invalidate_mapped_memory_ranges(
    _device: vk::Device,
    memory_range_count: u32,
    p_memory_ranges: *const vk::MappedMemoryRange,
) -> Result<()> {
    if memory_range_count == 0 {
        return Ok(());
    }

    let ranges = std::slice::from_raw_parts(p_memory_ranges, memory_range_count as usize);
    for range in ranges {
        debug!(
            "Invalidating memory range: offset={}, size={}",
            range.offset, range.size
        );
    }

    Ok(())
}

pub unsafe fn get_physical_device_memory_properties(
    _physical_device: vk::PhysicalDevice,
    p_memory_properties: *mut vk::PhysicalDeviceMemoryProperties,
) {
    let props = &mut *p_memory_properties;

    // Define memory types matching WebGPU's capabilities
    // Type 0: Device local (for GPU resources)
    props.memory_types[0] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        heap_index: 0,
    };

    // Type 1: Host visible + coherent (for staging/uniform buffers)
    props.memory_types[1] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
        heap_index: 1,
    };

    // Type 2: Host visible + cached (for readback)
    props.memory_types[2] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_CACHED,
        heap_index: 1,
    };

    props.memory_type_count = 3;

    // Define memory heaps
    // Heap 0: Device local memory (256 MB default)
    props.memory_heaps[0] = vk::MemoryHeap {
        size: 256 * 1024 * 1024,
        flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
    };

    // Heap 1: Host memory (assume 1 GB)
    props.memory_heaps[1] = vk::MemoryHeap {
        size: 1024 * 1024 * 1024,
        flags: vk::MemoryHeapFlags::empty(),
    };

    props.memory_heap_count = 2;
}

pub fn get_memory_data(memory: vk::DeviceMemory) -> Option<Arc<VkDeviceMemoryData>> {
    MEMORY_ALLOCATOR.get(memory.as_raw())
}
