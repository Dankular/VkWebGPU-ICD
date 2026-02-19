//! Vulkan Buffer implementation
//! Maps VkBuffer to WebGPU GPUBuffer

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;
use crate::memory;

pub static BUFFER_ALLOCATOR: Lazy<HandleAllocator<VkBufferData>> =
    Lazy::new(|| HandleAllocator::new());

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
    *p_buffer = Handle::from_raw(buffer_handle);

    #[cfg(feature = "webx")]
    {
        // payload: handle(u64) + size(u64) + vkUsage(u32)
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&buffer_handle.to_le_bytes());
        payload.extend_from_slice(&create_info.size.to_le_bytes());
        payload.extend_from_slice(&create_info.usage.as_raw().to_le_bytes());
        if let Err(e) = crate::webx_ipc::WebXIpc::global().send_cmd(0x0040, &payload) {
            log::warn!("[webx] create_buffer IPC failed: {:?}", e);
        }
    }

    Ok(())
}

pub unsafe fn destroy_buffer(
    _device: vk::Device,
    buffer: vk::Buffer,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if buffer == vk::Buffer::null() {
        return;
    }

    // Unregister from memory flush tracking before removing the buffer.
    if let Some(buf_data) = BUFFER_ALLOCATOR.get(buffer.as_raw()) {
        let mem_guard = buf_data.memory.read();
        let wgpu_guard = buf_data.wgpu_buffer.read();
        if let (Some(mem), Some(wgpu_buf)) = (mem_guard.as_ref(), wgpu_guard.as_ref()) {
            memory::unregister_bound_buffer(*mem, wgpu_buf);
        }
    }

    BUFFER_ALLOCATOR.remove(buffer.as_raw());
    debug!("Destroyed buffer");
}

pub unsafe fn bind_buffer_memory(
    _device: vk::Device,
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
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "webx")))]
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

        let wgpu_buffer_arc = Arc::new(wgpu_buffer);

        // Register in memory's flush list so future map/unmap cycles keep the
        // wgpu buffer in sync with CPU writes.
        memory::register_bound_buffer(
            memory,
            Arc::clone(&wgpu_buffer_arc),
            memory_offset,
            buffer_data.size,
        );

        *buffer_data.wgpu_buffer.write() = Some(wgpu_buffer_arc);
    }

    debug!("Bound buffer to memory at offset {}", memory_offset);

    Ok(())
}

pub unsafe fn get_buffer_memory_requirements(
    _device: vk::Device,
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

    // Always include copy usages.
    // COPY_DST is required by queue.write_buffer() — our path for uploading CPU
    // writes (mapped memory) to the GPU.  Without it, any flush on a staging
    // buffer (TRANSFER_SRC only) or a uniform buffer silently drops the data.
    // COPY_SRC is required when this buffer is the source of CmdCopyBuffer /
    // CmdCopyBufferToImage.  Both are near-zero cost (just a usage flag).
    usage |= wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;

    usage
}

pub fn get_buffer_data(buffer: vk::Buffer) -> Option<Arc<VkBufferData>> {
    BUFFER_ALLOCATOR.get(buffer.as_raw())
}

pub unsafe fn get_buffer_memory_requirements2(
    device: vk::Device,
    p_info: *const vk::BufferMemoryRequirementsInfo2,
    p_memory_requirements: *mut vk::MemoryRequirements2,
) {
    let info = &*p_info;
    get_buffer_memory_requirements(
        device,
        info.buffer,
        &mut (*p_memory_requirements).memory_requirements,
    );
}

pub unsafe fn bind_buffer_memory2(
    device: vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const vk::BindBufferMemoryInfo,
) -> Result<()> {
    if bind_info_count == 0 || p_bind_infos.is_null() {
        return Ok(());
    }
    let infos = std::slice::from_raw_parts(p_bind_infos, bind_info_count as usize);
    for info in infos {
        bind_buffer_memory(device, info.buffer, info.memory, info.memory_offset)?;
    }
    Ok(())
}

pub unsafe fn get_buffer_device_address(
    _device: vk::Device,
    p_info: *const vk::BufferDeviceAddressInfo,
) -> vk::DeviceAddress {
    if p_info.is_null() {
        return 0;
    }
    let info = &*p_info;
    // Return the handle raw value as a fake device address.
    // This is a stub - real buffer device address requires GPU VA space management.
    // The value is non-zero and unique per buffer, which is enough to not crash.
    info.buffer.as_raw()
}

pub unsafe fn get_device_buffer_memory_requirements(
    _device: vk::Device,
    p_info: *const vk::DeviceBufferMemoryRequirements,
    p_memory_requirements: *mut vk::MemoryRequirements2,
) {
    if p_info.is_null() || (*p_info).p_create_info.is_null() {
        return;
    }
    let create_info = &*(*p_info).p_create_info;

    let reqs = &mut (*p_memory_requirements).memory_requirements;
    reqs.size = create_info.size;
    reqs.alignment = 256;
    reqs.memory_type_bits = 0b111;
}
