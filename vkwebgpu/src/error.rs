//! Error types for VkWebGPU

use thiserror::Error;

#[derive(Error, Debug)]
pub enum VkError {
    #[error("Vulkan initialization failed: {0}")]
    InitializationFailed(String),

    #[error("WebGPU adapter not available")]
    AdapterNotAvailable,

    #[error("WebGPU device creation failed: {0}")]
    DeviceCreationFailed(String),

    #[error("Shader translation failed: {0}")]
    ShaderTranslationFailed(String),

    #[error("Invalid handle: {0}")]
    InvalidHandle(String),

    #[error("Out of host memory")]
    OutOfHostMemory,

    #[error("Out of device memory")]
    OutOfDeviceMemory,

    #[error("Feature not supported: {0}")]
    FeatureNotSupported(String),

    #[error("Invalid format")]
    InvalidFormat,

    #[error("Command buffer recording error: {0}")]
    CommandBufferError(String),

    #[error("WebGPU error: {0}")]
    WebGPUError(String),

    #[error("Extension not present: {0}")]
    ExtensionNotPresent(String),

    #[error("Layer not present: {0}")]
    LayerNotPresent(String),

    #[error("Incompatible driver")]
    IncompatibleDriver,

    #[error("Too many objects")]
    TooManyObjects,

    #[error("Format not supported")]
    FormatNotSupported,

    #[error("Fragmented pool")]
    FragmentedPool,

    #[error("Surface lost")]
    SurfaceLost,

    #[error("Native window in use")]
    NativeWindowInUse,

    #[error("Out of date")]
    OutOfDate,
}

pub type Result<T> = std::result::Result<T, VkError>;

impl VkError {
    /// Convert VkError to Vulkan result code
    pub fn to_vk_result(&self) -> ash::vk::Result {
        match self {
            VkError::OutOfHostMemory => ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            VkError::OutOfDeviceMemory => ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            VkError::InvalidHandle(_) => ash::vk::Result::ERROR_DEVICE_LOST,
            VkError::InitializationFailed(_) | VkError::DeviceCreationFailed(_) => {
                ash::vk::Result::ERROR_INITIALIZATION_FAILED
            }
            VkError::FeatureNotSupported(_) => ash::vk::Result::ERROR_FEATURE_NOT_PRESENT,
            VkError::InvalidFormat | VkError::FormatNotSupported => {
                ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED
            }
            VkError::ExtensionNotPresent(_) => ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            VkError::LayerNotPresent(_) => ash::vk::Result::ERROR_LAYER_NOT_PRESENT,
            VkError::IncompatibleDriver => ash::vk::Result::ERROR_INCOMPATIBLE_DRIVER,
            VkError::TooManyObjects => ash::vk::Result::ERROR_TOO_MANY_OBJECTS,
            VkError::FragmentedPool => ash::vk::Result::ERROR_FRAGMENTED_POOL,
            VkError::SurfaceLost => ash::vk::Result::ERROR_SURFACE_LOST_KHR,
            VkError::NativeWindowInUse => ash::vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR,
            VkError::OutOfDate => ash::vk::Result::ERROR_OUT_OF_DATE_KHR,
            _ => ash::vk::Result::ERROR_UNKNOWN,
        }
    }
}
