//! Vulkan Buffer implementation
//! Maps VkBuffer to WebGPU GPUBuffer

use ash::vk;
use log::debug;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;
use crate::memory;

pub static BUFFER_ALLOCATOR: HandleAllocator<VkBufferData> = HandleAllocator::new();

pub struct VkBufferData {
    pub device: vk::Device,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub memory: RwLock<Option<vk::DeviceMemory>>,
    pub memory_offset: RwLock<vk::DeviceSize>,
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_buffer: RwLock<Option<Arc<wgpu::Buffer>>>,
}

pub unsafe fn create_buffer(
    device: vk::Device,
    p_create_info: *const vk::BufferCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_buffer: *mut vk::Buffer,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating buffer: size={}, usage={:?}",
        create_info.size, create_info.usage
    );

    let buffer_data = VkBufferData {
        device,
        size: create_info.size,
        usage: create_info.usage,
        sharing_mode: create_info.sharing_mode,
        memory: RwLock::new(None),
        memory_offset: RwLock::new(0),
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_buffer: RwLock::new(None),
    };

    let buffer_handle = BUFFER_ALLOCATOR.allocate(buffer_data);
    *p_buffer = vk::Buffer::from_raw(buffer_handle);

    Ok(())
}

pub unsafe fn destroy_buffer(buffer: vk::Buffer, _p_allocator: *const vk::AllocationCallbacks) {
    if buffer == vk::Buffer::null() {
        return;
    }

    BUFFER_ALLOCATOR.remove(buffer.as_raw());
    debug!("Destroyed buffer");
}

pub unsafe fn bind_buffer_memory(
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_offset: vk::DeviceSize,
) -> Result<()> {
    let buffer_data = BUFFER_ALLOCATOR
        .get(buffer.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid buffer".to_string()))?;

    let device_data = device::get_device_data(buffer_data.device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    // Store memory binding
    *buffer_data.memory.write() = Some(memory);
    *buffer_data.memory_offset.write() = memory_offset;

    // Create actual WebGPU buffer
    #[cfg(not(target_arch = "wasm32"))]
    {
        let usage = vk_to_wgpu_buffer_usage(buffer_data.usage);

        let wgpu_buffer = device_data
            .backend
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("VkBuffer"),
                size: buffer_data.size,
                usage,
                mapped_at_creation: false,
            });

        // If we have memory data, copy it to the buffer
        if let Some(memory_data) = memory::get_memory_data(memory) {
            let memory_data_vec = memory_data.data.read();
            let data_slice = &memory_data_vec
                [memory_offset as usize..(memory_offset + buffer_data.size) as usize];

            device_data
                .backend
                .queue
                .write_buffer(&wgpu_buffer, 0, data_slice);
        }

        *buffer_data.wgpu_buffer.write() = Some(Arc::new(wgpu_buffer));
    }

    debug!("Bound buffer to memory at offset {}", memory_offset);

    Ok(())
}

pub unsafe fn get_buffer_memory_requirements(
    buffer: vk::Buffer,
    p_memory_requirements: *mut vk::MemoryRequirements,
) {
    let buffer_data = match BUFFER_ALLOCATOR.get(buffer.as_raw()) {
        Some(data) => data,
        None => return,
    };

    let requirements = &mut *p_memory_requirements;

    requirements.size = buffer_data.size;
    requirements.alignment = 256; // WebGPU typically requires 256-byte alignment

    // All memory types are suitable for buffers
    requirements.memory_type_bits = 0b111; // Types 0, 1, 2
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_buffer_usage(vk_usage: vk::BufferUsageFlags) -> wgpu::BufferUsages {
    let mut usage = wgpu::BufferUsages::empty();

    if vk_usage.contains(vk::BufferUsageFlags::TRANSFER_SRC) {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    if vk_usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    if vk_usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
        usage |= wgpu::BufferUsages::UNIFORM;
    }
    if vk_usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
        usage |= wgpu::BufferUsages::STORAGE;
    }
    if vk_usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
        usage |= wgpu::BufferUsages::INDEX;
    }
    if vk_usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
        usage |= wgpu::BufferUsages::VERTEX;
    }
    if vk_usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
        usage |= wgpu::BufferUsages::INDIRECT;
    }

    // Ensure at least one usage is set
    if usage.is_empty() {
        usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    }

    usage
}

pub fn get_buffer_data(buffer: vk::Buffer) -> Option<Arc<VkBufferData>> {
    BUFFER_ALLOCATOR.get(buffer.as_raw())
}
