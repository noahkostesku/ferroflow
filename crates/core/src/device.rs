use thiserror::Error;

/// Error parsing a device string.
#[derive(Debug, Error)]
pub enum DeviceError {
    /// The string did not match any known device specifier.
    #[error("unknown device {0:?}; expected \"cpu\", \"cuda\", or \"cuda:<N>\"")]
    Unknown(String),
}

/// Compute device for tensor operations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Device {
    /// Host CPU (default).
    #[default]
    Cpu,
    /// NVIDIA GPU identified by device index (0 = first GPU).
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl Device {
    /// Parse a device string into a [`Device`].
    ///
    /// Accepted values:
    /// - `"cpu"` → [`Device::Cpu`]
    /// - `"cuda"` or `"cuda:0"` → [`Device::Cuda(0)`]  (requires the `cuda` feature)
    /// - `"cuda:N"` → [`Device::Cuda(N)`]  (requires the `cuda` feature)
    ///
    /// # Errors
    /// Returns [`DeviceError::Unknown`] for unrecognised strings.
    pub fn from_str(s: &str) -> Result<Self, DeviceError> {
        match s {
            "cpu" => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Ok(Device::Cuda(0)),
            #[cfg(feature = "cuda")]
            s if s.starts_with("cuda:") => {
                let idx = s["cuda:".len()..]
                    .parse::<usize>()
                    .map_err(|_| DeviceError::Unknown(s.to_string()))?;
                Ok(Device::Cuda(idx))
            }
            other => Err(DeviceError::Unknown(other.to_string())),
        }
    }
}

/// Returns a cached [`Arc<CudaDevice>`] for the given device index.
///
/// The first call for a given `idx` opens the CUDA context; subsequent calls
/// return the same handle from a process-wide cache.
///
/// # Errors
/// Returns [`DeviceError`] if the CUDA driver fails to open the device.
#[cfg(feature = "cuda")]
pub(crate) fn get_cuda_device(
    idx: usize,
) -> Result<std::sync::Arc<cudarc::driver::CudaDevice>, DeviceError> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<usize, std::sync::Arc<cudarc::driver::CudaDevice>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache
        .lock()
        .map_err(|_| DeviceError::Unknown("CUDA device cache mutex poisoned".into()))?;
    if let Some(dev) = map.get(&idx) {
        return Ok(std::sync::Arc::clone(dev));
    }
    let dev = cudarc::driver::CudaDevice::new(idx)
        .map_err(|e| DeviceError::Unknown(format!("CUDA device {idx}: {e}")))?;
    map.insert(idx, std::sync::Arc::clone(&dev));
    Ok(dev)
}
