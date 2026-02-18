//! Vulkan Memory implementation
//!
//! WebGPU doesn't expose explicit memory allocation like Vulkan.
//! We use a CPU-side Vec<u8> as the "host-visible" backing store for each
//! VkDeviceMemory.  When host memory is unmapped (vkUnmapMemory) or explicitly
//! flushed (vkFlushMappedMemoryRanges), we write the modified bytes back to all
//! wgpu Buffers that are bound to that memory range.

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static MEMORY_ALLOCATOR: Lazy<HandleAllocator<VkDeviceMemoryData>> =
    Lazy::new(|| HandleAllocator::new());

/// All currently-mapped VkDeviceMemory handles.
/// Used by `flush_all_mapped_memory` to flush HOST_COHERENT memory at vkQueueSubmit
/// time — apps relying on coherent semantics never call vkFlushMappedMemoryRanges.
pub static MAPPED_MEMORIES: Lazy<RwLock<Vec<vk::DeviceMemory>>> =
    Lazy::new(|| RwLock::new(Vec::new()));

/// Describes a wgpu Buffer that is bound to a VkDeviceMemory allocation.
pub struct BoundBufferInfo {
    /// The wgpu buffer to flush CPU data into.
    pub wgpu_buffer: Arc<wgpu::Buffer>,
    /// Byte offset within the VkDeviceMemory where this buffer starts.
    pub memory_offset: vk::DeviceSize,
    /// Byte size of the buffer (number of bytes to copy on flush).
    pub buffer_size: vk::DeviceSize,
}

pub struct VkDeviceMemoryData {
    pub device: vk::Device,
    pub size: vk::DeviceSize,
    pub memory_type_index: u32,
    /// Raw pointer returned to the application by vkMapMemory (inside `data`).
    pub mapped_ptr: RwLock<Option<*mut u8>>,
    /// CPU-side backing store.  Applications write here through the mapped pointer.
    pub data: RwLock<Vec<u8>>,
    /// All wgpu Buffers bound to this memory, tracked so we can flush on unmap.
    pub bound_buffers: RwLock<Vec<BoundBufferInfo>>,
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
        bound_buffers: RwLock::new(Vec::new()),
    };

    let memory_handle = MEMORY_ALLOCATOR.allocate(memory_data);
    *p_memory = Handle::from_raw(memory_handle);

    Ok(())
}

pub unsafe fn free_memory(
    _device: vk::Device,
    memory: vk::DeviceMemory,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if memory == vk::DeviceMemory::null() {
        return;
    }

    MEMORY_ALLOCATOR.remove(memory.as_raw());
    debug!("Freed device memory");
}

pub unsafe fn map_memory(
    _device: vk::Device,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    _flags: vk::MemoryMapFlags,
    pp_data: *mut *mut std::ffi::c_void,
) -> Result<()> {
    let memory_data = MEMORY_ALLOCATOR
        .get(memory.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid device memory".to_string()))?;

    // We take a write lock momentarily to get the pointer, then release it.
    // The caller writes to the raw pointer directly; the lock is NOT held during the map.
    let ptr = {
        let mut data = memory_data.data.write();
        let slice = if size == vk::WHOLE_SIZE {
            &mut data[offset as usize..]
        } else {
            &mut data[offset as usize..(offset + size) as usize]
        };
        slice.as_mut_ptr() as *mut std::ffi::c_void
    };

    *pp_data = ptr;

    // Store map state (offset so we can flush the right range on unmap).
    *memory_data.mapped_ptr.write() = Some(ptr as *mut u8);

    // Track as mapped so queue_submit can flush HOST_COHERENT memory automatically.
    MAPPED_MEMORIES.write().push(memory);

    debug!("Mapped memory at offset {} size {}", offset, size);
    Ok(())
}

pub unsafe fn unmap_memory(device: vk::Device, memory: vk::DeviceMemory) {
    let memory_data = match MEMORY_ALLOCATOR.get(memory.as_raw()) {
        Some(d) => d,
        None => return,
    };

    *memory_data.mapped_ptr.write() = None;

    // Deregister before flushing so queue_submit doesn't double-flush.
    MAPPED_MEMORIES.write().retain(|&m| m != memory);

    // Flush all CPU-side writes to every wgpu Buffer bound to this memory.
    flush_bound_buffers(device, &memory_data);

    debug!("Unmapped memory");
}

pub unsafe fn flush_mapped_memory_ranges(
    device: vk::Device,
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

        if let Some(memory_data) = MEMORY_ALLOCATOR.get(range.memory.as_raw()) {
            flush_bound_buffers(device, &memory_data);
        }
    }

    Ok(())
}

/// Write CPU-side `data` back to all wgpu Buffers bound to `memory_data`.
/// Call this after the application has finished writing through a mapped pointer.
#[cfg(not(target_arch = "wasm32"))]
fn flush_bound_buffers(device: vk::Device, memory_data: &VkDeviceMemoryData) {
    let device_data = match device::get_device_data(device) {
        Some(d) => d,
        None => return,
    };

    let cpu_data = memory_data.data.read();
    let bound = memory_data.bound_buffers.read();

    for info in bound.iter() {
        let start = info.memory_offset as usize;
        let end = start + info.buffer_size as usize;

        if end > cpu_data.len() {
            debug!(
                "flush_bound_buffers: buffer range [{}, {}) out of bounds (mem size {})",
                start,
                end,
                cpu_data.len()
            );
            continue;
        }

        let slice = &cpu_data[start..end];
        device_data
            .backend
            .queue
            .write_buffer(&info.wgpu_buffer, 0, slice);

        debug!(
            "Flushed {} bytes from memory offset {} to wgpu buffer",
            info.buffer_size, info.memory_offset
        );
    }
}

/// No-op fallback for wasm32 where wgpu buffer writes work differently.
#[cfg(target_arch = "wasm32")]
fn flush_bound_buffers(_device: vk::Device, _memory_data: &VkDeviceMemoryData) {}

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

/// Register a wgpu Buffer bound to this memory so it can be flushed on unmap.
/// Called by buffer::bind_buffer_memory after the wgpu Buffer is created.
/// Flush all currently-mapped memory allocations belonging to `device` to their
/// wgpu Buffers.
///
/// Must be called at `vkQueueSubmit` time before command replay.  Applications
/// that allocate HOST_COHERENT memory rely on the implementation providing
/// automatic coherence — they never call `vkFlushMappedMemoryRanges`.  Without
/// this call their staged data would never reach the wgpu Buffer and the GPU
/// would read stale (all-zero) bytes.
pub fn flush_all_mapped_memory(device: vk::Device) {
    // Snapshot the list while holding the read lock, then release before doing
    // any wgpu work (write_buffer internally acquires no extra memory locks).
    let mapped: Vec<vk::DeviceMemory> = MAPPED_MEMORIES.read().clone();
    for memory in mapped {
        if let Some(memory_data) = MEMORY_ALLOCATOR.get(memory.as_raw()) {
            if memory_data.device == device {
                flush_bound_buffers(device, &memory_data);
            }
        }
    }
}

pub fn register_bound_buffer(
    memory: vk::DeviceMemory,
    wgpu_buffer: Arc<wgpu::Buffer>,
    memory_offset: vk::DeviceSize,
    buffer_size: vk::DeviceSize,
) {
    if let Some(memory_data) = MEMORY_ALLOCATOR.get(memory.as_raw()) {
        memory_data.bound_buffers.write().push(BoundBufferInfo {
            wgpu_buffer,
            memory_offset,
            buffer_size,
        });
    }
}

/// Unregister a wgpu Buffer from memory tracking (called on buffer destruction).
pub fn unregister_bound_buffer(memory: vk::DeviceMemory, wgpu_buffer: &Arc<wgpu::Buffer>) {
    if let Some(memory_data) = MEMORY_ALLOCATOR.get(memory.as_raw()) {
        let target = Arc::as_ptr(wgpu_buffer) as usize;
        memory_data.bound_buffers.write().retain(|info| {
            Arc::as_ptr(&info.wgpu_buffer) as usize != target
        });
    }
}
