use thiserror::Error;

/// Error parsing a device string.
#[derive(Debug, Error)]
pub enum DeviceError {
    /// The string did not match any known device specifier.
    #[error("unknown device {0:?}; expected \"cpu\", \"cuda\", \"cuda:<N>\", or \"auto\"")]
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
    #[allow(clippy::should_implement_trait)]
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

/// Per-op device placement policy.
///
/// Determines where each op executes.  Pass to
/// [`execute_op_auto`](crate::ops::execute_op_auto) or to the work-stealing
/// scheduler via `with_policy`.
#[derive(Debug, Clone)]
pub enum DevicePolicy {
    /// All ops execute on the host CPU.
    AllCpu,
    /// All ops execute on CUDA device 0; falls back to CPU if no GPU is present.
    #[cfg(feature = "cuda")]
    AllGpu,
    /// Route each op to the best device based on op kind and output size.
    ///
    /// [`OpKind::Matmul`](crate::op::OpKind::Matmul) ops whose output has at
    /// least `gpu_matmul_threshold` elements, and all
    /// [`OpKind::Conv2d`](crate::op::OpKind::Conv2d) ops, go to the GPU.
    /// All other ops stay on the CPU.  When no GPU is available the policy
    /// transparently falls back to CPU for every op.
    Auto { gpu_matmul_threshold: usize },
}

impl DevicePolicy {
    /// Parse a policy string into a [`DevicePolicy`].
    ///
    /// Accepted values:
    /// - `"cpu"` → [`DevicePolicy::AllCpu`]
    /// - `"cuda"` / `"cuda:N"` → [`DevicePolicy::AllGpu`]  (requires `cuda` feature)
    /// - `"auto"` → [`DevicePolicy::Auto`] with a 512×512 default threshold
    ///
    /// # Errors
    /// Returns [`DeviceError::Unknown`] for unrecognised strings.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, DeviceError> {
        match s {
            "cpu" => Ok(DevicePolicy::AllCpu),
            #[cfg(feature = "cuda")]
            "cuda" | "cuda:0" => Ok(DevicePolicy::AllGpu),
            #[cfg(feature = "cuda")]
            s if s.starts_with("cuda:") => {
                s["cuda:".len()..]
                    .parse::<usize>()
                    .map_err(|_| DeviceError::Unknown(s.to_string()))?;
                Ok(DevicePolicy::AllGpu)
            }
            "auto" => Ok(DevicePolicy::Auto {
                gpu_matmul_threshold: 512 * 512,
            }),
            other => Err(DeviceError::Unknown(other.to_string())),
        }
    }

    /// Override the matmul size threshold for [`DevicePolicy::Auto`].
    ///
    /// Matmul ops whose output element count is ≥ `threshold` are sent to the
    /// GPU; smaller ones stay on the CPU to avoid transfer overhead.  No-op for
    /// non-`Auto` variants.
    pub fn with_matmul_threshold(self, threshold: usize) -> Self {
        match self {
            DevicePolicy::Auto { .. } => DevicePolicy::Auto {
                gpu_matmul_threshold: threshold,
            },
            other => other,
        }
    }

    /// Resolves the compute device for `op` under this policy.
    ///
    /// `gpu_available` should be the cached result of [`gpu_available()`]; the
    /// caller is responsible for not calling the CUDA driver in a hot loop.
    /// When `gpu_available` is `false` every variant returns [`Device::Cpu`].
    pub fn device_for_op(&self, op: &crate::op::Op, gpu_available: bool) -> Device {
        match self {
            DevicePolicy::AllCpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            DevicePolicy::AllGpu => {
                if gpu_available {
                    Device::Cuda(0)
                } else {
                    Device::Cpu
                }
            }
            DevicePolicy::Auto { gpu_matmul_threshold } => {
                #[cfg(feature = "cuda")]
                if gpu_available {
                    use crate::op::OpKind;
                    return match &op.kind {
                        OpKind::Matmul { .. } => {
                            let size = op.output_shape.iter().product::<usize>();
                            if size >= *gpu_matmul_threshold {
                                Device::Cuda(0)
                            } else {
                                Device::Cpu
                            }
                        }
                        OpKind::Conv2d { .. } => Device::Cuda(0),
                        _ => Device::Cpu,
                    };
                }
                // Suppress unused-variable warnings in non-cuda builds.
                let _ = (op, gpu_available, gpu_matmul_threshold);
                Device::Cpu
            }
        }
    }
}

/// Returns `true` if CUDA device 0 is accessible on this machine.
///
/// The result is computed once on the first call and cached for the process
/// lifetime, so subsequent calls are a single atomic load.
pub fn gpu_available() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::CudaDevice::new(0).is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    })
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
