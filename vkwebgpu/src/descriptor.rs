//! Vulkan Descriptor Set implementation
//! Maps VkDescriptorSet to WebGPU GPUBindGroup

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static DESCRIPTOR_SET_LAYOUT_ALLOCATOR: Lazy<HandleAllocator<VkDescriptorSetLayoutData>> =
    Lazy::new(|| HandleAllocator::new());
pub static DESCRIPTOR_POOL_ALLOCATOR: Lazy<HandleAllocator<VkDescriptorPoolData>> =
    Lazy::new(|| HandleAllocator::new());
pub static DESCRIPTOR_SET_ALLOCATOR: Lazy<HandleAllocator<VkDescriptorSetData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkDescriptorSetLayoutData {
    pub device: vk::Device,
    pub bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_layout: Arc<wgpu::BindGroupLayout>,
}

pub struct VkDescriptorPoolData {
    pub device: vk::Device,
    pub max_sets: u32,
    pub pool_sizes: Vec<vk::DescriptorPoolSize>,
    pub allocated_sets: RwLock<Vec<vk::DescriptorSet>>,
}

pub struct VkDescriptorSetData {
    pub device: vk::Device,
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub bindings: RwLock<Vec<DescriptorBinding>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_bind_group: RwLock<Option<Arc<wgpu::BindGroup>>>,
}

#[derive(Clone)]
pub enum DescriptorBinding {
    Buffer {
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    },
    ImageView {
        image_view: vk::ImageView,
        layout: vk::ImageLayout,
    },
    Sampler {
        sampler: vk::Sampler,
    },
    CombinedImageSampler {
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    },
}

pub unsafe fn create_descriptor_set_layout(
    device: vk::Device,
    p_create_info: *const vk::DescriptorSetLayoutCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_set_layout: *mut vk::DescriptorSetLayout,
) -> Result<()> {
    let create_info = &*p_create_info;

    let bindings: Vec<vk::DescriptorSetLayoutBinding<'static>> = if create_info.binding_count > 0 {
        std::slice::from_raw_parts(create_info.p_bindings, create_info.binding_count as usize)
            .iter()
            .map(|b| {
                // Create a new binding with 'static lifetime
                // This is safe because we're not using p_immutable_samplers
                vk::DescriptorSetLayoutBinding {
                    binding: b.binding,
                    descriptor_type: b.descriptor_type,
                    descriptor_count: b.descriptor_count,
                    stage_flags: b.stage_flags,
                    p_immutable_samplers: std::ptr::null(), // We don't support immutable samplers yet
                    _marker: std::marker::PhantomData,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    debug!(
        "Creating descriptor set layout with {} bindings",
        bindings.len()
    );

    let device_data = device::get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    #[cfg(not(target_arch = "wasm32"))]
    let wgpu_layout = {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .map(|binding| vk_to_wgpu_bind_group_layout_entry(binding))
            .collect();

        device_data
            .backend
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("VkDescriptorSetLayout"),
                entries: &entries,
            })
    };

    let layout_data = VkDescriptorSetLayoutData {
        device,
        bindings,
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_layout: Arc::new(wgpu_layout),
    };

    let layout_handle = DESCRIPTOR_SET_LAYOUT_ALLOCATOR.allocate(layout_data);
    *p_set_layout = Handle::from_raw(layout_handle);

    Ok(())
}

pub unsafe fn destroy_descriptor_set_layout(
    _device: vk::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if descriptor_set_layout == vk::DescriptorSetLayout::null() {
        return;
    }

    DESCRIPTOR_SET_LAYOUT_ALLOCATOR.remove(descriptor_set_layout.as_raw());
    debug!("Destroyed descriptor set layout");
}

pub unsafe fn create_descriptor_pool(
    device: vk::Device,
    p_create_info: *const vk::DescriptorPoolCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_descriptor_pool: *mut vk::DescriptorPool,
) -> Result<()> {
    let create_info = &*p_create_info;

    let pool_sizes = if create_info.pool_size_count > 0 {
        std::slice::from_raw_parts(
            create_info.p_pool_sizes,
            create_info.pool_size_count as usize,
        )
        .to_vec()
    } else {
        Vec::new()
    };

    debug!(
        "Creating descriptor pool: max_sets={}, pool_sizes={}",
        create_info.max_sets,
        pool_sizes.len()
    );

    let pool_data = VkDescriptorPoolData {
        device,
        max_sets: create_info.max_sets,
        pool_sizes,
        allocated_sets: RwLock::new(Vec::new()),
    };

    let pool_handle = DESCRIPTOR_POOL_ALLOCATOR.allocate(pool_data);
    *p_descriptor_pool = Handle::from_raw(pool_handle);

    Ok(())
}

pub unsafe fn destroy_descriptor_pool(
    _device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    _p_allocator: *const vk::AllocationCallbacks,
) {
    if descriptor_pool == vk::DescriptorPool::null() {
        return;
    }

    // Free all allocated sets
    if let Some(pool_data) = DESCRIPTOR_POOL_ALLOCATOR.get(descriptor_pool.as_raw()) {
        let sets = pool_data.allocated_sets.read().clone();
        for set in sets {
            DESCRIPTOR_SET_ALLOCATOR.remove(set.as_raw());
        }
    }

    DESCRIPTOR_POOL_ALLOCATOR.remove(descriptor_pool.as_raw());
    debug!("Destroyed descriptor pool");
}

pub unsafe fn allocate_descriptor_sets(
    device: vk::Device,
    p_allocate_info: *const vk::DescriptorSetAllocateInfo,
    p_descriptor_sets: *mut vk::DescriptorSet,
) -> Result<()> {
    let allocate_info = &*p_allocate_info;

    let layouts = std::slice::from_raw_parts(
        allocate_info.p_set_layouts,
        allocate_info.descriptor_set_count as usize,
    );

    let pool_data = DESCRIPTOR_POOL_ALLOCATOR
        .get(allocate_info.descriptor_pool.as_raw())
        .ok_or_else(|| VkError::InvalidHandle("Invalid descriptor pool".to_string()))?;

    debug!("Allocating {} descriptor sets", layouts.len());

    let dest_sets = std::slice::from_raw_parts_mut(
        p_descriptor_sets,
        allocate_info.descriptor_set_count as usize,
    );

    for (i, &layout) in layouts.iter().enumerate() {
        let set_data = VkDescriptorSetData {
            device,
            layout,
            pool: allocate_info.descriptor_pool,
            bindings: RwLock::new(Vec::new()),
            #[cfg(not(target_arch = "wasm32"))]
            wgpu_bind_group: RwLock::new(None),
        };

        let set_handle = DESCRIPTOR_SET_ALLOCATOR.allocate(set_data);
        let descriptor_set = Handle::from_raw(set_handle);
        dest_sets[i] = descriptor_set;

        pool_data.allocated_sets.write().push(descriptor_set);
    }

    Ok(())
}

pub unsafe fn update_descriptor_sets(
    _device: vk::Device,
    descriptor_write_count: u32,
    p_descriptor_writes: *const vk::WriteDescriptorSet,
    descriptor_copy_count: u32,
    p_descriptor_copies: *const vk::CopyDescriptorSet,
) {
    // Process writes
    if descriptor_write_count > 0 {
        let writes =
            std::slice::from_raw_parts(p_descriptor_writes, descriptor_write_count as usize);

        for write in writes {
            if let Some(set_data) = DESCRIPTOR_SET_ALLOCATOR.get(write.dst_set.as_raw()) {
                let mut bindings = set_data.bindings.write();

                // Ensure bindings vector is large enough
                let required_size = (write.dst_binding + write.descriptor_count) as usize;
                if bindings.len() < required_size {
                    bindings.resize(
                        required_size,
                        DescriptorBinding::Buffer {
                            buffer: vk::Buffer::null(),
                            offset: 0,
                            range: 0,
                        },
                    );
                }

                match write.descriptor_type {
                    vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => {
                        if !write.p_buffer_info.is_null() {
                            let buffer_infos = std::slice::from_raw_parts(
                                write.p_buffer_info,
                                write.descriptor_count as usize,
                            );

                            for (i, info) in buffer_infos.iter().enumerate() {
                                bindings[(write.dst_binding + i as u32) as usize] =
                                    DescriptorBinding::Buffer {
                                        buffer: info.buffer,
                                        offset: info.offset,
                                        range: info.range,
                                    };
                            }
                        }
                    }
                    vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::STORAGE_IMAGE => {
                        if !write.p_image_info.is_null() {
                            let image_infos = std::slice::from_raw_parts(
                                write.p_image_info,
                                write.descriptor_count as usize,
                            );

                            for (i, info) in image_infos.iter().enumerate() {
                                bindings[(write.dst_binding + i as u32) as usize] =
                                    DescriptorBinding::ImageView {
                                        image_view: info.image_view,
                                        layout: info.image_layout,
                                    };
                            }
                        }
                    }
                    vk::DescriptorType::SAMPLER => {
                        if !write.p_image_info.is_null() {
                            let image_infos = std::slice::from_raw_parts(
                                write.p_image_info,
                                write.descriptor_count as usize,
                            );

                            for (i, info) in image_infos.iter().enumerate() {
                                bindings[(write.dst_binding + i as u32) as usize] =
                                    DescriptorBinding::Sampler {
                                        sampler: info.sampler,
                                    };
                            }
                        }
                    }
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                        if !write.p_image_info.is_null() {
                            let image_infos = std::slice::from_raw_parts(
                                write.p_image_info,
                                write.descriptor_count as usize,
                            );

                            for (i, info) in image_infos.iter().enumerate() {
                                bindings[(write.dst_binding + i as u32) as usize] =
                                    DescriptorBinding::CombinedImageSampler {
                                        image_view: info.image_view,
                                        sampler: info.sampler,
                                        layout: info.image_layout,
                                    };
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Process copies
    if descriptor_copy_count > 0 {
        let copies =
            std::slice::from_raw_parts(p_descriptor_copies, descriptor_copy_count as usize);

        for copy in copies {
            let src_data = DESCRIPTOR_SET_ALLOCATOR.get(copy.src_set.as_raw());
            let dst_data = DESCRIPTOR_SET_ALLOCATOR.get(copy.dst_set.as_raw());

            if let (Some(src), Some(dst)) = (src_data, dst_data) {
                let src_bindings = src.bindings.read();
                let mut dst_bindings = dst.bindings.write();

                let required_size = (copy.dst_binding + copy.descriptor_count) as usize;
                if dst_bindings.len() < required_size {
                    dst_bindings.resize(
                        required_size,
                        DescriptorBinding::Buffer {
                            buffer: vk::Buffer::null(),
                            offset: 0,
                            range: 0,
                        },
                    );
                }

                for i in 0..copy.descriptor_count as usize {
                    let src_idx = (copy.src_binding as usize) + i;
                    let dst_idx = (copy.dst_binding as usize) + i;
                    if src_idx < src_bindings.len() {
                        dst_bindings[dst_idx] = src_bindings[src_idx].clone();
                    }
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_bind_group_layout_entry(
    binding: &vk::DescriptorSetLayoutBinding,
) -> wgpu::BindGroupLayoutEntry {
    let visibility = vk_to_wgpu_shader_stages(binding.stage_flags);
    let ty = vk_to_wgpu_binding_type(binding.descriptor_type, binding.descriptor_count);

    wgpu::BindGroupLayoutEntry {
        binding: binding.binding,
        visibility,
        ty,
        count: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_shader_stages(stages: vk::ShaderStageFlags) -> wgpu::ShaderStages {
    let mut result = wgpu::ShaderStages::empty();

    if stages.contains(vk::ShaderStageFlags::VERTEX) {
        result |= wgpu::ShaderStages::VERTEX;
    }
    if stages.contains(vk::ShaderStageFlags::FRAGMENT) {
        result |= wgpu::ShaderStages::FRAGMENT;
    }
    if stages.contains(vk::ShaderStageFlags::COMPUTE) {
        result |= wgpu::ShaderStages::COMPUTE;
    }

    result
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_binding_type(desc_type: vk::DescriptorType, _count: u32) -> wgpu::BindingType {
    match desc_type {
        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: desc_type == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                min_binding_size: None,
            }
        }
        vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: desc_type == vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                min_binding_size: None,
            }
        }
        vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        }
        vk::DescriptorType::STORAGE_IMAGE => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        vk::DescriptorType::SAMPLER => {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        }
        _ => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}

pub fn get_descriptor_set_layout_data(
    layout: vk::DescriptorSetLayout,
) -> Option<Arc<VkDescriptorSetLayoutData>> {
    DESCRIPTOR_SET_LAYOUT_ALLOCATOR.get(layout.as_raw())
}

pub fn get_descriptor_set_data(set: vk::DescriptorSet) -> Option<Arc<VkDescriptorSetData>> {
    DESCRIPTOR_SET_ALLOCATOR.get(set.as_raw())
}
