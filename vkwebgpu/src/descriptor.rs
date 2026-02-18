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
use crate::{buffer, image, sampler};

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
                vk::DescriptorSetLayoutBinding {
                    binding: b.binding,
                    descriptor_type: b.descriptor_type,
                    descriptor_count: b.descriptor_count,
                    stage_flags: b.stage_flags,
                    p_immutable_samplers: std::ptr::null(),
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
        // For COMBINED_IMAGE_SAMPLER, we emit both a Texture entry and a Sampler entry.
        // The sampler is placed at a synthetic binding = original_binding | 0x8000_0000
        // so it doesn't collide with real bindings (which are always < 2^16 in practice).
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        for binding in &bindings {
            entries.push(vk_to_wgpu_bind_group_layout_entry(binding));
            if binding.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                // Add the paired sampler entry at the synthetic slot.
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: combined_sampler_binding(binding.binding),
                    visibility: vk_to_wgpu_shader_stages(binding.stage_flags),
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                });
            }
        }

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

/// Synthetic binding slot for the sampler half of a COMBINED_IMAGE_SAMPLER.
/// Must not collide with real Vulkan descriptor bindings (which top out well under 2^16).
#[inline]
fn combined_sampler_binding(texture_binding: u32) -> u32 {
    texture_binding | 0x8000_0000
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

    if let Some(pool_data) = DESCRIPTOR_POOL_ALLOCATOR.get(descriptor_pool.as_raw()) {
        let sets = pool_data.allocated_sets.read().clone();
        for set in sets {
            DESCRIPTOR_SET_ALLOCATOR.remove(set.as_raw());
        }
    }

    DESCRIPTOR_POOL_ALLOCATOR.remove(descriptor_pool.as_raw());
    debug!("Destroyed descriptor pool");
}

pub unsafe fn free_descriptor_sets(
    _device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
) -> Result<()> {
    if descriptor_set_count == 0 || p_descriptor_sets.is_null() {
        return Ok(());
    }
    let sets = std::slice::from_raw_parts(p_descriptor_sets, descriptor_set_count as usize);

    if let Some(pool_data) = DESCRIPTOR_POOL_ALLOCATOR.get(descriptor_pool.as_raw()) {
        let mut allocated = pool_data.allocated_sets.write();
        for &set in sets {
            if set != vk::DescriptorSet::null() {
                DESCRIPTOR_SET_ALLOCATOR.remove(set.as_raw());
                allocated.retain(|&s| s != set);
            }
        }
    }

    debug!("Freed {} descriptor sets", descriptor_set_count);
    Ok(())
}

pub unsafe fn reset_descriptor_pool(
    _device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    _flags: vk::DescriptorPoolResetFlags,
) -> Result<()> {
    if let Some(pool_data) = DESCRIPTOR_POOL_ALLOCATOR.get(descriptor_pool.as_raw()) {
        let sets = pool_data.allocated_sets.read().clone();
        for set in sets {
            DESCRIPTOR_SET_ALLOCATOR.remove(set.as_raw());
        }
        pool_data.allocated_sets.write().clear();
    }
    debug!("Reset descriptor pool");
    Ok(())
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
                // Update CPU-side bindings in a scoped lock so it's dropped before
                // we rebuild the wgpu BindGroup (which also needs to read bindings).
                {
                    let mut bindings = set_data.bindings.write();

                    let required_size =
                        (write.dst_binding + write.descriptor_count) as usize;
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
                        vk::DescriptorType::UNIFORM_BUFFER
                        | vk::DescriptorType::STORAGE_BUFFER
                        | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
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
                        vk::DescriptorType::SAMPLED_IMAGE
                        | vk::DescriptorType::STORAGE_IMAGE
                        | vk::DescriptorType::INPUT_ATTACHMENT => {
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
                } // write lock released

                // Rebuild the wgpu BindGroup now that bindings are updated.
                // Returns None silently if any resource is not yet ready.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    match rebuild_bind_group(set_data.device, &set_data) {
                        Some(bg) => {
                            *set_data.wgpu_bind_group.write() = Some(bg);
                            debug!("Rebuilt BindGroup for descriptor set after write");
                        }
                        None => {
                            debug!(
                                "Descriptor set not yet fully populated; BindGroup deferred"
                            );
                        }
                    }
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
                {
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
                } // both locks released

                // Rebuild the destination set's BindGroup.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(bg) = rebuild_bind_group(dst.device, &dst) {
                        *dst.wgpu_bind_group.write() = Some(bg);
                        debug!("Rebuilt BindGroup for copied descriptor set");
                    }
                }
            }
        }
    }
}

/// Rebuild the wgpu BindGroup for a descriptor set from its current CPU-side bindings.
///
/// Returns `None` silently if any required resource is not yet available (partial update,
/// null handle, or wgpu object not yet created). The caller should retry after all writes.
#[cfg(not(target_arch = "wasm32"))]
fn rebuild_bind_group(
    device: vk::Device,
    set_data: &VkDescriptorSetData,
) -> Option<Arc<wgpu::BindGroup>> {
    let layout_data = DESCRIPTOR_SET_LAYOUT_ALLOCATOR.get(set_data.layout.as_raw())?;
    let device_data = device::get_device_data(device)?;
    let bindings = set_data.bindings.read();

    // Collected resources: hold Arcs so the wgpu objects remain alive for the
    // duration of create_bind_group. Entries are built in a second pass that
    // borrows from these Arcs, which avoids Vec-reallocation lifetime issues.
    enum Collected {
        Buffer(Arc<wgpu::Buffer>, u64, Option<wgpu::BufferSize>),
        View(Arc<wgpu::TextureView>),
        Sampler(Arc<wgpu::Sampler>),
    }
    let mut collected: Vec<(u32, Collected)> =
        Vec::with_capacity(layout_data.bindings.len() * 2);

    for layout_binding in &layout_data.bindings {
        let binding_idx = layout_binding.binding as usize;
        if binding_idx >= bindings.len() {
            // This slot has not been written yet — defer BindGroup creation.
            return None;
        }

        match &bindings[binding_idx] {
            DescriptorBinding::Buffer { buffer, offset, range } => {
                if *buffer == vk::Buffer::null() {
                    // Placeholder padding slot; not a real write yet.
                    return None;
                }
                let buf_data = buffer::get_buffer_data(*buffer)?;
                let wgpu_buf = {
                    let g = buf_data.wgpu_buffer.read();
                    Arc::clone(g.as_ref()?)
                };
                // vk::WHOLE_SIZE == u64::MAX; map to None (remaining buffer).
                let size = if *range == vk::WHOLE_SIZE || *range == 0 {
                    None
                } else {
                    wgpu::BufferSize::new(*range)
                };
                collected.push((
                    layout_binding.binding,
                    Collected::Buffer(wgpu_buf, *offset, size),
                ));
            }

            DescriptorBinding::ImageView { image_view, .. } => {
                if *image_view == vk::ImageView::null() {
                    return None;
                }
                let view_data = image::get_image_view_data(*image_view)?;
                let wgpu_view = {
                    let g = view_data.wgpu_view.read();
                    Arc::clone(g.as_ref()?)
                };
                collected.push((layout_binding.binding, Collected::View(wgpu_view)));
            }

            DescriptorBinding::Sampler { sampler } => {
                if *sampler == vk::Sampler::null() {
                    return None;
                }
                let samp_data = sampler::get_sampler_data(*sampler)?;
                collected.push((
                    layout_binding.binding,
                    Collected::Sampler(Arc::clone(&samp_data.wgpu_sampler)),
                ));
            }

            DescriptorBinding::CombinedImageSampler {
                image_view,
                sampler,
                ..
            } => {
                if *image_view == vk::ImageView::null() {
                    return None;
                }
                let view_data = image::get_image_view_data(*image_view)?;
                let wgpu_view = {
                    let g = view_data.wgpu_view.read();
                    Arc::clone(g.as_ref()?)
                };
                // Texture at the original binding.
                collected.push((layout_binding.binding, Collected::View(wgpu_view)));

                // Sampler at the synthetic slot that matches the layout we created.
                if *sampler != vk::Sampler::null() {
                    if let Some(samp_data) = sampler::get_sampler_data(*sampler) {
                        collected.push((
                            combined_sampler_binding(layout_binding.binding),
                            Collected::Sampler(Arc::clone(&samp_data.wgpu_sampler)),
                        ));
                    }
                }
            }
        }
    }

    // Second pass: build BindGroupEntries borrowing from the collected Arcs.
    // The lifetimes work because `collected` outlives `entries` and both outlive
    // the `create_bind_group` call.
    let entries: Vec<wgpu::BindGroupEntry> = collected
        .iter()
        .map(|(binding, res)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: match res {
                Collected::Buffer(buf, offset, size) => {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: *offset,
                        size: *size,
                    })
                }
                Collected::View(view) => wgpu::BindingResource::TextureView(view),
                Collected::Sampler(samp) => wgpu::BindingResource::Sampler(samp),
            },
        })
        .collect();

    let bind_group =
        device_data
            .backend
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VkDescriptorSet"),
                layout: &layout_data.wgpu_layout,
                entries: &entries,
            });

    Some(Arc::new(bind_group))
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
        vk::DescriptorType::SAMPLED_IMAGE
        | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        | vk::DescriptorType::INPUT_ATTACHMENT => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
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
