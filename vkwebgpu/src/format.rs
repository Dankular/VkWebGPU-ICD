//! Vulkan to WebGPU format conversion

use ash::vk;

#[cfg(not(target_arch = "wasm32"))]
use wgpu::TextureFormat;

/// Convert Vulkan format to WebGPU format
#[cfg(not(target_arch = "wasm32"))]
pub fn vk_to_wgpu_format(format: vk::Format) -> Option<TextureFormat> {
    match format {
        // 8-bit formats
        vk::Format::R8_UNORM => Some(TextureFormat::R8Unorm),
        vk::Format::R8_SNORM => Some(TextureFormat::R8Snorm),
        vk::Format::R8_UINT => Some(TextureFormat::R8Uint),
        vk::Format::R8_SINT => Some(TextureFormat::R8Sint),

        // 16-bit formats
        vk::Format::R16_UINT => Some(TextureFormat::R16Uint),
        vk::Format::R16_SINT => Some(TextureFormat::R16Sint),
        vk::Format::R16_SFLOAT => Some(TextureFormat::R16Float),
        vk::Format::R16_UNORM => Some(TextureFormat::R16Unorm),
        vk::Format::R16_SNORM => Some(TextureFormat::R16Snorm),

        // 32-bit formats
        vk::Format::R32_UINT => Some(TextureFormat::R32Uint),
        vk::Format::R32_SINT => Some(TextureFormat::R32Sint),
        vk::Format::R32_SFLOAT => Some(TextureFormat::R32Float),

        // RG formats
        vk::Format::R8G8_UNORM => Some(TextureFormat::Rg8Unorm),
        vk::Format::R8G8_SNORM => Some(TextureFormat::Rg8Snorm),
        vk::Format::R8G8_UINT => Some(TextureFormat::Rg8Uint),
        vk::Format::R8G8_SINT => Some(TextureFormat::Rg8Sint),

        vk::Format::R16G16_UINT => Some(TextureFormat::Rg16Uint),
        vk::Format::R16G16_SINT => Some(TextureFormat::Rg16Sint),
        vk::Format::R16G16_SFLOAT => Some(TextureFormat::Rg16Float),
        vk::Format::R16G16_UNORM => Some(TextureFormat::Rg16Unorm),
        vk::Format::R16G16_SNORM => Some(TextureFormat::Rg16Snorm),

        vk::Format::R32G32_UINT => Some(TextureFormat::Rg32Uint),
        vk::Format::R32G32_SINT => Some(TextureFormat::Rg32Sint),
        vk::Format::R32G32_SFLOAT => Some(TextureFormat::Rg32Float),

        // RGBA formats
        vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        vk::Format::R8G8B8A8_SNORM => Some(TextureFormat::Rgba8Snorm),
        vk::Format::R8G8B8A8_UINT => Some(TextureFormat::Rgba8Uint),
        vk::Format::R8G8B8A8_SINT => Some(TextureFormat::Rgba8Sint),
        vk::Format::R8G8B8A8_SRGB => Some(TextureFormat::Rgba8UnormSrgb),

        vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
        vk::Format::B8G8R8A8_SRGB => Some(TextureFormat::Bgra8UnormSrgb),

        vk::Format::R16G16B16A16_UINT => Some(TextureFormat::Rgba16Uint),
        vk::Format::R16G16B16A16_SINT => Some(TextureFormat::Rgba16Sint),
        vk::Format::R16G16B16A16_SFLOAT => Some(TextureFormat::Rgba16Float),
        vk::Format::R16G16B16A16_UNORM => Some(TextureFormat::Rgba16Unorm),
        vk::Format::R16G16B16A16_SNORM => Some(TextureFormat::Rgba16Snorm),

        vk::Format::R32G32B32A32_UINT => Some(TextureFormat::Rgba32Uint),
        vk::Format::R32G32B32A32_SINT => Some(TextureFormat::Rgba32Sint),
        vk::Format::R32G32B32A32_SFLOAT => Some(TextureFormat::Rgba32Float),

        // Packed formats
        vk::Format::A2B10G10R10_UNORM_PACK32 => Some(TextureFormat::Rgb10a2Unorm),
        vk::Format::B10G11R11_UFLOAT_PACK32 => Some(TextureFormat::Rg11b10Float),
        // A8B8G8R8 packed variants — component order swapped vs R8G8B8A8 but
        // wgpu Rgba8* layouts match how the GPU sees ABGR-packed bytes on LE hardware.
        vk::Format::A8B8G8R8_UNORM_PACK32 => Some(TextureFormat::Rgba8Unorm),
        vk::Format::A8B8G8R8_SRGB_PACK32 => Some(TextureFormat::Rgba8UnormSrgb),
        vk::Format::A8B8G8R8_SNORM_PACK32 => Some(TextureFormat::Rgba8Snorm),

        // Depth/Stencil formats
        vk::Format::D16_UNORM => Some(TextureFormat::Depth16Unorm),
        vk::Format::X8_D24_UNORM_PACK32 => Some(TextureFormat::Depth24Plus),
        vk::Format::D32_SFLOAT => Some(TextureFormat::Depth32Float),
        vk::Format::D24_UNORM_S8_UINT => Some(TextureFormat::Depth24PlusStencil8),
        vk::Format::D32_SFLOAT_S8_UINT => Some(TextureFormat::Depth32FloatStencil8),
        vk::Format::S8_UINT => Some(TextureFormat::Stencil8),

        // BC compressed formats
        vk::Format::BC1_RGB_UNORM_BLOCK => Some(TextureFormat::Bc1RgbaUnorm),
        vk::Format::BC1_RGB_SRGB_BLOCK => Some(TextureFormat::Bc1RgbaUnormSrgb),
        vk::Format::BC1_RGBA_UNORM_BLOCK => Some(TextureFormat::Bc1RgbaUnorm),
        vk::Format::BC1_RGBA_SRGB_BLOCK => Some(TextureFormat::Bc1RgbaUnormSrgb),

        vk::Format::BC2_UNORM_BLOCK => Some(TextureFormat::Bc2RgbaUnorm),
        vk::Format::BC2_SRGB_BLOCK => Some(TextureFormat::Bc2RgbaUnormSrgb),

        vk::Format::BC3_UNORM_BLOCK => Some(TextureFormat::Bc3RgbaUnorm),
        vk::Format::BC3_SRGB_BLOCK => Some(TextureFormat::Bc3RgbaUnormSrgb),

        vk::Format::BC4_UNORM_BLOCK => Some(TextureFormat::Bc4RUnorm),
        vk::Format::BC4_SNORM_BLOCK => Some(TextureFormat::Bc4RSnorm),

        vk::Format::BC5_UNORM_BLOCK => Some(TextureFormat::Bc5RgUnorm),
        vk::Format::BC5_SNORM_BLOCK => Some(TextureFormat::Bc5RgSnorm),

        vk::Format::BC6H_UFLOAT_BLOCK => Some(TextureFormat::Bc6hRgbUfloat),
        vk::Format::BC6H_SFLOAT_BLOCK => Some(TextureFormat::Bc6hRgbFloat),

        vk::Format::BC7_UNORM_BLOCK => Some(TextureFormat::Bc7RgbaUnorm),
        vk::Format::BC7_SRGB_BLOCK => Some(TextureFormat::Bc7RgbaUnormSrgb),

        // ETC2 compressed formats
        vk::Format::ETC2_R8G8B8_UNORM_BLOCK => Some(TextureFormat::Etc2Rgb8Unorm),
        vk::Format::ETC2_R8G8B8_SRGB_BLOCK => Some(TextureFormat::Etc2Rgb8UnormSrgb),
        vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK => Some(TextureFormat::Etc2Rgb8A1Unorm),
        vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK => Some(TextureFormat::Etc2Rgb8A1UnormSrgb),
        vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK => Some(TextureFormat::Etc2Rgba8Unorm),
        vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK => Some(TextureFormat::Etc2Rgba8UnormSrgb),

        vk::Format::EAC_R11_UNORM_BLOCK => Some(TextureFormat::EacR11Unorm),
        vk::Format::EAC_R11_SNORM_BLOCK => Some(TextureFormat::EacR11Snorm),
        vk::Format::EAC_R11G11_UNORM_BLOCK => Some(TextureFormat::EacRg11Unorm),
        vk::Format::EAC_R11G11_SNORM_BLOCK => Some(TextureFormat::EacRg11Snorm),

        // ASTC compressed formats
        vk::Format::ASTC_4X4_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_4X4_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_5X4_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x4,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_5X4_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x4,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_5X5_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_5X5_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_6X5_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x5,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_6X5_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x5,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_6X6_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_6X6_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_8X5_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x5,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_8X5_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x5,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_8X6_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x6,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_8X6_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x6,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_8X8_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_8X8_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_10X5_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x5,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_10X5_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x5,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_10X6_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x6,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_10X6_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x6,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_10X8_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x8,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_10X8_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x8,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_10X10_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x10,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_10X10_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x10,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_12X10_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B12x10,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_12X10_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B12x10,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        vk::Format::ASTC_12X12_UNORM_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B12x12,
            channel: wgpu::AstcChannel::Unorm,
        }),
        vk::Format::ASTC_12X12_SRGB_BLOCK => Some(TextureFormat::Astc {
            block: wgpu::AstcBlock::B12x12,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),

        _ => None,
    }
}

/// Get the byte size of a format
pub fn format_size(format: vk::Format) -> Option<u32> {
    match format {
        vk::Format::R8_UNORM | vk::Format::R8_SNORM | vk::Format::R8_UINT | vk::Format::R8_SINT => {
            Some(1)
        }
        vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT => Some(2),
        vk::Format::R8G8B8_UNORM
        | vk::Format::R8G8B8_SNORM
        | vk::Format::R8G8B8_UINT
        | vk::Format::R8G8B8_SINT => Some(3),
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT => Some(4),
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB => {
            Some(4)
        }

        vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT
        | vk::Format::R16_UNORM
        | vk::Format::R16_SNORM => Some(2),
        vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM => Some(4),
        vk::Format::R16G16B16_UINT
        | vk::Format::R16G16B16_SINT
        | vk::Format::R16G16B16_SFLOAT
        | vk::Format::R16G16B16_UNORM
        | vk::Format::R16G16B16_SNORM => Some(6),
        vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM => Some(8),

        vk::Format::R32_UINT | vk::Format::R32_SINT | vk::Format::R32_SFLOAT => Some(4),
        vk::Format::R32G32_UINT | vk::Format::R32G32_SINT | vk::Format::R32G32_SFLOAT => Some(8),
        vk::Format::R32G32B32_UINT | vk::Format::R32G32B32_SINT | vk::Format::R32G32B32_SFLOAT => {
            Some(12)
        }
        vk::Format::R32G32B32A32_UINT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R32G32B32A32_SFLOAT => Some(16),

        vk::Format::D16_UNORM => Some(2),
        vk::Format::D32_SFLOAT => Some(4),
        vk::Format::D24_UNORM_S8_UINT => Some(4),
        vk::Format::D32_SFLOAT_S8_UINT => Some(8),

        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn vk_to_wgpu_vertex_format(format: vk::Format) -> Option<wgpu::VertexFormat> {
    match format {
        vk::Format::R8_UINT => Some(wgpu::VertexFormat::Uint8x2),
        vk::Format::R8_SINT => Some(wgpu::VertexFormat::Sint8x2),
        vk::Format::R8_UNORM => Some(wgpu::VertexFormat::Unorm8x2),
        vk::Format::R8_SNORM => Some(wgpu::VertexFormat::Snorm8x2),

        vk::Format::R8G8_UINT => Some(wgpu::VertexFormat::Uint8x2),
        vk::Format::R8G8_SINT => Some(wgpu::VertexFormat::Sint8x2),
        vk::Format::R8G8_UNORM => Some(wgpu::VertexFormat::Unorm8x2),
        vk::Format::R8G8_SNORM => Some(wgpu::VertexFormat::Snorm8x2),

        vk::Format::R8G8B8A8_UINT => Some(wgpu::VertexFormat::Uint8x4),
        vk::Format::R8G8B8A8_SINT => Some(wgpu::VertexFormat::Sint8x4),
        vk::Format::R8G8B8A8_UNORM => Some(wgpu::VertexFormat::Unorm8x4),
        vk::Format::R8G8B8A8_SNORM => Some(wgpu::VertexFormat::Snorm8x4),

        vk::Format::R16G16_UINT => Some(wgpu::VertexFormat::Uint16x2),
        vk::Format::R16G16_SINT => Some(wgpu::VertexFormat::Sint16x2),
        vk::Format::R16G16_UNORM => Some(wgpu::VertexFormat::Unorm16x2),
        vk::Format::R16G16_SNORM => Some(wgpu::VertexFormat::Snorm16x2),
        vk::Format::R16G16_SFLOAT => Some(wgpu::VertexFormat::Float16x2),

        vk::Format::R16G16B16A16_UINT => Some(wgpu::VertexFormat::Uint16x4),
        vk::Format::R16G16B16A16_SINT => Some(wgpu::VertexFormat::Sint16x4),
        vk::Format::R16G16B16A16_UNORM => Some(wgpu::VertexFormat::Unorm16x4),
        vk::Format::R16G16B16A16_SNORM => Some(wgpu::VertexFormat::Snorm16x4),
        vk::Format::R16G16B16A16_SFLOAT => Some(wgpu::VertexFormat::Float16x4),

        vk::Format::R32_UINT => Some(wgpu::VertexFormat::Uint32),
        vk::Format::R32_SINT => Some(wgpu::VertexFormat::Sint32),
        vk::Format::R32_SFLOAT => Some(wgpu::VertexFormat::Float32),

        vk::Format::R32G32_UINT => Some(wgpu::VertexFormat::Uint32x2),
        vk::Format::R32G32_SINT => Some(wgpu::VertexFormat::Sint32x2),
        vk::Format::R32G32_SFLOAT => Some(wgpu::VertexFormat::Float32x2),

        vk::Format::R32G32B32_UINT => Some(wgpu::VertexFormat::Uint32x3),
        vk::Format::R32G32B32_SINT => Some(wgpu::VertexFormat::Sint32x3),
        vk::Format::R32G32B32_SFLOAT => Some(wgpu::VertexFormat::Float32x3),

        vk::Format::R32G32B32A32_UINT => Some(wgpu::VertexFormat::Uint32x4),
        vk::Format::R32G32B32A32_SINT => Some(wgpu::VertexFormat::Sint32x4),
        vk::Format::R32G32B32A32_SFLOAT => Some(wgpu::VertexFormat::Float32x4),

        _ => None,
    }
}
