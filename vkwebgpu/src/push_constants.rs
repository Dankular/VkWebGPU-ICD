//! Push Constants Emulation for WebGPU
//!
//! Since WebGPU doesn't have native push constants, we emulate them using a ring buffer
//! of uniform data. This is critical for DXVK compatibility, which uses push constants
//! extensively for per-sprite transforms, material parameters, etc.
//!
//! Implementation:
//! - 64KB ring buffer (typical push constant usage is 128-256 bytes per draw)
//! - Write data on vkCmdPushConstants
//! - Bind as uniform buffer at set 0, binding 0 by convention
//! - Automatic wrap-around when buffer fills
//! - Per-frame reset to avoid fragmentation

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use wgpu;

/// Ring buffer for push constant emulation
///
/// The ring buffer allows efficient per-draw push constant updates without
/// creating new buffers. It automatically wraps around when full.
pub struct PushConstantRingBuffer {
    #[cfg(not(target_arch = "wasm32"))]
    buffer: Arc<wgpu::Buffer>,
    #[cfg(not(target_arch = "wasm32"))]
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    offset: AtomicU32,
    capacity: u32,
    alignment: u32,
}

impl PushConstantRingBuffer {
    /// Default capacity: 64KB (supports ~256 draws with 256-byte push constants)
    pub const DEFAULT_CAPACITY: u32 = 65536;

    /// Minimum uniform buffer offset alignment (WebGPU spec requires 256 bytes)
    pub const MIN_UNIFORM_BUFFER_ALIGNMENT: u32 = 256;

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let alignment = Self::MIN_UNIFORM_BUFFER_ALIGNMENT;

        // Ensure capacity is aligned
        let aligned_capacity = ((capacity + alignment - 1) / alignment) * alignment;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PushConstantRingBuffer"),
            size: aligned_capacity as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout for push constants
        // This will be used at set 0, binding 0 in pipelines that use push constants
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PushConstantBindGroupLayout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        Self {
            buffer: Arc::new(buffer),
            bind_group_layout: Arc::new(bind_group_layout),
            offset: AtomicU32::new(0),
            capacity: aligned_capacity,
            alignment,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new(_device: &crate::backend::Device, capacity: u32) -> Self {
        let alignment = Self::MIN_UNIFORM_BUFFER_ALIGNMENT;
        let aligned_capacity = ((capacity + alignment - 1) / alignment) * alignment;

        Self {
            offset: AtomicU32::new(0),
            capacity: aligned_capacity,
            alignment,
        }
    }

    /// Push data to the ring buffer and return the offset
    ///
    /// Returns the aligned offset where the data was written.
    /// This offset should be used when binding the buffer as a uniform buffer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn push(&self, queue: &wgpu::Queue, data: &[u8]) -> u32 {
        if data.is_empty() {
            return 0;
        }

        // Calculate aligned size
        let size = data.len() as u32;
        let aligned_size = ((size + self.alignment - 1) / self.alignment) * self.alignment;

        // Get current offset and increment atomically
        let current_offset = self.offset.fetch_add(aligned_size, Ordering::SeqCst);

        // Check if we need to wrap around
        let actual_offset = if current_offset + aligned_size > self.capacity {
            // Wrap around to the beginning
            self.offset.store(aligned_size, Ordering::SeqCst);
            0
        } else {
            current_offset
        };

        // Write data to the buffer
        queue.write_buffer(&self.buffer, actual_offset as u64, data);

        actual_offset
    }

    #[cfg(target_arch = "wasm32")]
    pub fn push(&self, _queue: &crate::backend::Queue, data: &[u8]) -> u32 {
        if data.is_empty() {
            return 0;
        }

        let size = data.len() as u32;
        let aligned_size = ((size + self.alignment - 1) / self.alignment) * self.alignment;

        let current_offset = self.offset.fetch_add(aligned_size, Ordering::SeqCst);

        let actual_offset = if current_offset + aligned_size > self.capacity {
            self.offset.store(aligned_size, Ordering::SeqCst);
            0
        } else {
            current_offset
        };

        // TODO: Implement WASM buffer write
        actual_offset
    }

    /// Reset the ring buffer (typically called once per frame)
    pub fn reset(&self) {
        self.offset.store(0, Ordering::SeqCst);
    }

    /// Get the underlying WebGPU buffer
    #[cfg(not(target_arch = "wasm32"))]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the bind group layout for push constants
    #[cfg(not(target_arch = "wasm32"))]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group for a specific offset in the ring buffer
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PushConstantBindGroup"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        })
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get current offset
    pub fn current_offset(&self) -> u32 {
        self.offset.load(Ordering::SeqCst)
    }

    /// Get alignment requirement
    pub fn alignment(&self) -> u32 {
        self.alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let capacity = 1000;
        let _alignment = PushConstantRingBuffer::MIN_UNIFORM_BUFFER_ALIGNMENT;
        let aligned = ((capacity + _alignment - 1) / _alignment) * _alignment;
        assert_eq!(aligned, 1024); // 1000 rounded up to next multiple of 256
    }

    #[test]
    fn test_offset_wrap() {
        // This test would require a WebGPU device, so we just test the logic
        let alignment = PushConstantRingBuffer::MIN_UNIFORM_BUFFER_ALIGNMENT;
        let capacity = 1024u32;

        let offset = AtomicU32::new(900);
        let data_size = 256u32;

        let current = offset.fetch_add(data_size, Ordering::SeqCst);
        let should_wrap = current + data_size > capacity;

        assert!(should_wrap);
    }
}
