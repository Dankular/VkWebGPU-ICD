//! Vulkan Sampler implementation
//! Maps VkSampler to WebGPU GPUSampler

use ash::vk::{self, Handle};
use log::debug;
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::device;
use crate::error::{Result, VkError};
use crate::handle::HandleAllocator;

pub static SAMPLER_ALLOCATOR: Lazy<HandleAllocator<VkSamplerData>> =
    Lazy::new(|| HandleAllocator::new());

pub struct VkSamplerData {
    pub device: vk::Device,
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: bool,
    pub max_anisotropy: f32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: vk::BorderColor,
    #[cfg(not(target_arch = "wasm32"))]
    pub wgpu_sampler: Arc<wgpu::Sampler>,
}

pub unsafe fn create_sampler(
    device: vk::Device,
    p_create_info: *const vk::SamplerCreateInfo,
    _p_allocator: *const vk::AllocationCallbacks,
    p_sampler: *mut vk::Sampler,
) -> Result<()> {
    let create_info = &*p_create_info;

    debug!(
        "Creating sampler: mag={:?}, min={:?}, mipmap={:?}",
        create_info.mag_filter, create_info.min_filter, create_info.mipmap_mode
    );

    let device_data = device::get_device_data(device)
        .ok_or_else(|| VkError::InvalidHandle("Invalid device".to_string()))?;

    #[cfg(not(target_arch = "wasm32"))]
    let wgpu_sampler = {
        let address_mode_u = vk_to_wgpu_address_mode(create_info.address_mode_u);
        let address_mode_v = vk_to_wgpu_address_mode(create_info.address_mode_v);
        let address_mode_w = vk_to_wgpu_address_mode(create_info.address_mode_w);

        let mag_filter = vk_to_wgpu_filter_mode(create_info.mag_filter);
        let min_filter = vk_to_wgpu_filter_mode(create_info.min_filter);
        let mipmap_filter = vk_to_wgpu_mipmap_filter(create_info.mipmap_mode);

        let compare = if create_info.compare_enable == vk::TRUE {
            Some(vk_to_wgpu_compare_function(create_info.compare_op))
        } else {
            None
        };

        let max_anisotropy = if create_info.anisotropy_enable == vk::TRUE {
            create_info.max_anisotropy.clamp(1.0, 16.0) as u16
        } else {
            1
        };

        device_data
            .backend
            .device
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some("VkSampler"),
                address_mode_u,
                address_mode_v,
                address_mode_w,
                mag_filter,
                min_filter,
                mipmap_filter,
                lod_min_clamp: create_info.min_lod,
                lod_max_clamp: create_info.max_lod,
                compare,
                anisotropy_clamp: max_anisotropy,
                border_color: None,
            })
    };

    #[cfg(target_arch = "wasm32")]
    let wgpu_sampler = {
        // WASM implementation would go here
        unimplemented!("WASM sampler creation not yet implemented")
    };

    let sampler_data = VkSamplerData {
        device,
        mag_filter: create_info.mag_filter,
        min_filter: create_info.min_filter,
        mipmap_mode: create_info.mipmap_mode,
        address_mode_u: create_info.address_mode_u,
        address_mode_v: create_info.address_mode_v,
        address_mode_w: create_info.address_mode_w,
        mip_lod_bias: create_info.mip_lod_bias,
        anisotropy_enable: create_info.anisotropy_enable == vk::TRUE,
        max_anisotropy: create_info.max_anisotropy,
        compare_enable: create_info.compare_enable == vk::TRUE,
        compare_op: create_info.compare_op,
        min_lod: create_info.min_lod,
        max_lod: create_info.max_lod,
        border_color: create_info.border_color,
        #[cfg(not(target_arch = "wasm32"))]
        wgpu_sampler: Arc::new(wgpu_sampler),
    };

    let sampler_handle = SAMPLER_ALLOCATOR.allocate(sampler_data);
    *p_sampler = Handle::from_raw(sampler_handle);

    Ok(())
}

pub unsafe fn destroy_sampler(_device: vk::Device, sampler: vk::Sampler, _p_allocator: *const vk::AllocationCallbacks) {
    if sampler == vk::Sampler::null() {
        return;
    }

    SAMPLER_ALLOCATOR.remove(sampler.as_raw());
    debug!("Destroyed sampler");
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_address_mode(mode: vk::SamplerAddressMode) -> wgpu::AddressMode {
    match mode {
        vk::SamplerAddressMode::REPEAT => wgpu::AddressMode::Repeat,
        vk::SamplerAddressMode::MIRRORED_REPEAT => wgpu::AddressMode::MirrorRepeat,
        vk::SamplerAddressMode::CLAMP_TO_EDGE => wgpu::AddressMode::ClampToEdge,
        vk::SamplerAddressMode::CLAMP_TO_BORDER => wgpu::AddressMode::ClampToBorder,
        _ => wgpu::AddressMode::ClampToEdge,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_filter_mode(filter: vk::Filter) -> wgpu::FilterMode {
    match filter {
        vk::Filter::NEAREST => wgpu::FilterMode::Nearest,
        vk::Filter::LINEAR => wgpu::FilterMode::Linear,
        _ => wgpu::FilterMode::Linear,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_mipmap_filter(mode: vk::SamplerMipmapMode) -> wgpu::FilterMode {
    match mode {
        vk::SamplerMipmapMode::NEAREST => wgpu::FilterMode::Nearest,
        vk::SamplerMipmapMode::LINEAR => wgpu::FilterMode::Linear,
        _ => wgpu::FilterMode::Linear,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vk_to_wgpu_compare_function(op: vk::CompareOp) -> wgpu::CompareFunction {
    match op {
        vk::CompareOp::NEVER => wgpu::CompareFunction::Never,
        vk::CompareOp::LESS => wgpu::CompareFunction::Less,
        vk::CompareOp::EQUAL => wgpu::CompareFunction::Equal,
        vk::CompareOp::LESS_OR_EQUAL => wgpu::CompareFunction::LessEqual,
        vk::CompareOp::GREATER => wgpu::CompareFunction::Greater,
        vk::CompareOp::NOT_EQUAL => wgpu::CompareFunction::NotEqual,
        vk::CompareOp::GREATER_OR_EQUAL => wgpu::CompareFunction::GreaterEqual,
        vk::CompareOp::ALWAYS => wgpu::CompareFunction::Always,
        _ => wgpu::CompareFunction::Always,
    }
}

pub fn get_sampler_data(sampler: vk::Sampler) -> Option<Arc<VkSamplerData>> {
    SAMPLER_ALLOCATOR.get(sampler.as_raw())
}
