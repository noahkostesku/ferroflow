#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::cublas::{sys::cublasOperation_t::CUBLAS_OP_N, CudaBlas, Gemm, GemmConfig};

use ndarray::{Array, ArrayD, IxDyn};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Errors from [`Tensor`] construction and device transfers.
#[derive(Debug, Error)]
pub enum TensorError {
    /// The flat data length does not match the product of `shape`.
    #[error("data length {got} does not match shape {shape:?} (expected {expected})")]
    ShapeMismatch {
        shape: Vec<usize>,
        expected: usize,
        got: usize,
    },
    /// A CPU-only operation was called on a GPU tensor.
    #[error("expected a CPU tensor, but found a GPU tensor")]
    NotCpuTensor,
    /// A CUDA driver or cuBLAS operation failed.
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(String),
}

/// Backing storage for a [`Tensor`]: either host RAM or a CUDA device buffer.
pub enum TensorStorage {
    /// Host memory, backed by an [`ndarray::ArrayD`].
    Cpu(ArrayD<f32>),
    /// Device memory on a CUDA GPU.
    ///
    /// Wrapped in [`Arc`] so cloning a GPU tensor is O(1) (refcount increment),
    /// not a device-to-host-to-device roundtrip.
    #[cfg(feature = "cuda")]
    Cuda(Arc<cudarc::driver::CudaSlice<f32>>),
}

/// A dense f32 tensor with optional GPU acceleration.
///
/// GPU tensors use [`Arc`]-backed storage so they can stay on the device
/// between consecutive GPU ops without a round-trip copy.  Use
/// [`Tensor::to_device_cached`] to obtain a tensor on a specific device
/// without transferring when it is already there.
pub struct Tensor {
    pub(crate) storage: TensorStorage,
    pub(crate) shape: Vec<usize>,
}

impl Tensor {
    /// Creates a CPU tensor from a shape and flat data buffer.
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `data.len() != shape.iter().product()`.
    pub fn from_shape_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();
        let got = data.len();
        if got != expected {
            return Err(TensorError::ShapeMismatch {
                shape: shape.to_vec(),
                expected,
                got,
            });
        }
        let arr = Array::from_shape_vec(IxDyn(shape), data).map_err(|_| {
            TensorError::ShapeMismatch {
                shape: shape.to_vec(),
                expected,
                got,
            }
        })?;
        Ok(Self {
            storage: TensorStorage::Cpu(arr),
            shape: shape.to_vec(),
        })
    }

    /// Creates a zero-filled CPU tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            storage: TensorStorage::Cpu(ArrayD::zeros(IxDyn(shape))),
            shape: shape.to_vec(),
        }
    }

    /// Creates a CPU tensor filled with `value`.
    pub fn full(shape: &[usize], value: f32) -> Self {
        Self {
            storage: TensorStorage::Cpu(ArrayD::from_elem(IxDyn(shape), value)),
            shape: shape.to_vec(),
        }
    }

    /// Shape of the tensor (dimension sizes).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns a reference to the underlying CPU [`ndarray::ArrayD`].
    ///
    /// # Errors
    /// Returns [`TensorError::NotCpuTensor`] if the tensor resides on a GPU.
    pub fn cpu_array(&self) -> Result<&ArrayD<f32>, TensorError> {
        match &self.storage {
            TensorStorage::Cpu(a) => Ok(a),
            #[cfg(feature = "cuda")]
            TensorStorage::Cuda(_) => Err(TensorError::NotCpuTensor),
        }
    }

    /// Returns `true` if the tensor is stored in CUDA device memory.
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        if matches!(&self.storage, TensorStorage::Cuda(_)) {
            return true;
        }
        false
    }

    /// Transfers the tensor to the specified device, copying data if necessary.
    ///
    /// Transfer directions and costs:
    /// - CPU → CPU: clone (ndarray deep copy)
    /// - CPU → CUDA: host-to-device copy via `htod_copy`
    /// - CUDA → CPU: synchronous device-to-host copy via `dtoh_sync_copy`
    /// - CUDA → CUDA: O(1) Arc refcount increment (no data movement)
    ///
    /// Prefer [`to_device_cached`](Self::to_device_cached) when the tensor may
    /// already be on the target device.
    ///
    /// # Errors
    /// Returns [`TensorError::Cuda`] on CUDA driver failures.
    pub fn to_device(&self, device: &crate::device::Device) -> Result<Self, TensorError> {
        use crate::device::Device;
        match (device, &self.storage) {
            (Device::Cpu, TensorStorage::Cpu(_)) => Ok(self.clone()),
            #[cfg(feature = "cuda")]
            (Device::Cuda(idx), TensorStorage::Cpu(arr)) => {
                let dev = crate::device::get_cuda_device(*idx)
                    .map_err(|e| TensorError::Cuda(e.to_string()))?;
                let flat: Vec<f32> = arr.iter().copied().collect();
                let slice = dev
                    .htod_copy(flat)
                    .map_err(|e| TensorError::Cuda(e.to_string()))?;
                Ok(Self {
                    storage: TensorStorage::Cuda(Arc::new(slice)),
                    shape: self.shape.clone(),
                })
            }
            #[cfg(feature = "cuda")]
            (Device::Cpu, TensorStorage::Cuda(arc)) => {
                let dev = arc.device();
                let flat = dev
                    .dtoh_sync_copy(&**arc)
                    .map_err(|e| TensorError::Cuda(e.to_string()))?;
                Tensor::from_shape_vec(&self.shape, flat)
            }
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), TensorStorage::Cuda(_)) => Ok(self.clone()),
        }
    }

    /// Returns `true` if the tensor is already resident on `device`.
    ///
    /// When this returns `true`, [`to_device_cached`](Self::to_device_cached)
    /// skips the transfer entirely and returns an O(1) clone.
    pub fn is_on_device(&self, device: &crate::device::Device) -> bool {
        use crate::device::Device;
        match device {
            Device::Cpu => matches!(&self.storage, TensorStorage::Cpu(_)),
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => matches!(&self.storage, TensorStorage::Cuda(_)),
        }
    }

    /// Returns a tensor on `device`, transferring only if the tensor is not
    /// already resident there.
    ///
    /// - Already on device: O(1) clone (Arc refcount for GPU, ndarray view+clone for CPU).
    /// - Not on device: performs a host↔device transfer via [`to_device`](Self::to_device).
    ///
    /// Use this in hot paths (scheduler input fetch, op dispatch) to eliminate
    /// redundant CPU↔GPU round-trips on consecutive GPU ops.
    ///
    /// # Errors
    /// Returns [`TensorError::Cuda`] on CUDA driver failures during transfer.
    pub fn to_device_cached(&self, device: &crate::device::Device) -> Result<Self, TensorError> {
        if self.is_on_device(device) {
            Ok(self.clone())
        } else {
            self.to_device(device)
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        match &self.storage {
            TensorStorage::Cpu(a) => Self {
                storage: TensorStorage::Cpu(a.clone()),
                shape: self.shape.clone(),
            },
            // GPU clone: O(1) Arc refcount increment — no data movement.
            #[cfg(feature = "cuda")]
            TensorStorage::Cuda(arc) => Self {
                storage: TensorStorage::Cuda(Arc::clone(arc)),
                shape: self.shape.clone(),
            },
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.storage {
            TensorStorage::Cpu(a) => write!(f, "Tensor::Cpu(shape={:?})", a.shape()),
            #[cfg(feature = "cuda")]
            TensorStorage::Cuda(_) => write!(f, "Tensor::Cuda(shape={:?})", self.shape),
        }
    }
}

// ── GPU worker context ────────────────────────────────────────────────────────

/// Per-worker CUDA execution context: one persistent cuBLAS handle and stream.
///
/// Create one per Tokio worker with [`GpuWorkerContext::new`]. Never share
/// between workers — each worker must own its own context so submissions to
/// `stream` are independent and can be synced per-batch without blocking peers.
#[cfg(feature = "cuda")]
pub struct GpuWorkerContext {
    pub(crate) device: Arc<cudarc::driver::CudaDevice>,
    pub(crate) cublas: CudaBlas,
    pub(crate) stream: cudarc::driver::CudaStream,
}

#[cfg(feature = "cuda")]
impl GpuWorkerContext {
    /// Creates a new context for `device_idx` (0 = first GPU).
    ///
    /// Opens a new CUDA stream and a dedicated cuBLAS handle bound to that
    /// stream. All subsequent `matmul_cuda_async` calls using this context
    /// queue kernels on the same stream; call [`sync_stream`] to wait for them.
    ///
    /// # Errors
    /// Returns [`TensorError::Cuda`] if the CUDA driver or cuBLAS fails to
    /// initialise.
    pub fn new(device_idx: usize) -> Result<Self, TensorError> {
        let device = crate::device::get_cuda_device(device_idx)
            .map_err(|e| TensorError::Cuda(e.to_string()))?;
        let stream = device
            .fork_default_stream()
            .map_err(|e| TensorError::Cuda(e.to_string()))?;
        let cublas = CudaBlas::new(Arc::clone(&device))
            .map_err(|e| TensorError::Cuda(e.to_string()))?;
        // Bind the cuBLAS handle to ctx.stream so subsequent gemm calls are
        // queued on that stream and can be awaited with a single sync.
        unsafe {
            cublas
                .set_stream(Some(&stream))
                .map_err(|e| TensorError::Cuda(e.to_string()))?;
        }
        Ok(Self { device, cublas, stream })
    }
}

/// Submits a matrix multiply `A·B` to `ctx.stream` without waiting for the GPU
/// to finish.
///
/// Returns a [`Tensor`] backed by the allocated output buffer.  The buffer
/// contents are **not valid** until [`sync_stream`] has been called on the same
/// context.  Call this in a loop for all independent ops, then call
/// [`sync_stream`] once to materialise all results.
///
/// Inputs may already reside on the GPU (O(1) Arc clone) or on the CPU (one
/// H→D transfer per input).
///
/// # Errors
/// Returns [`TensorError::Cuda`] on shape mismatches or CUDA driver failures.
#[cfg(feature = "cuda")]
pub fn matmul_cuda_async(
    a: &Tensor,
    b: &Tensor,
    ctx: &GpuWorkerContext,
) -> Result<Tensor, TensorError> {
    if a.shape.len() != 2 {
        return Err(TensorError::Cuda(format!(
            "matmul_cuda_async: A must be 2-D, got {:?}",
            a.shape
        )));
    }
    if b.shape.len() != 2 {
        return Err(TensorError::Cuda(format!(
            "matmul_cuda_async: B must be 2-D, got {:?}",
            b.shape
        )));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let (kb, n) = (b.shape[0], b.shape[1]);
    if k != kb {
        return Err(TensorError::Cuda(format!(
            "matmul_cuda_async: A({m},{k}) · B({kb},{n}) — inner dims must match"
        )));
    }

    let a_gpu = match &a.storage {
        TensorStorage::Cuda(arc) => Arc::clone(arc),
        TensorStorage::Cpu(arr) => Arc::new(
            ctx.device
                .htod_copy(arr.iter().copied().collect::<Vec<f32>>())
                .map_err(|e| TensorError::Cuda(e.to_string()))?,
        ),
    };
    let b_gpu = match &b.storage {
        TensorStorage::Cuda(arc) => Arc::clone(arc),
        TensorStorage::Cpu(arr) => Arc::new(
            ctx.device
                .htod_copy(arr.iter().copied().collect::<Vec<f32>>())
                .map_err(|e| TensorError::Cuda(e.to_string()))?,
        ),
    };

    // beta=0 so the output buffer's initial values are irrelevant; alloc_zeros
    // is used for safety but the zeroing and the GEMM may overlap on different
    // streams — acceptable because beta=0 means GEMM never reads old C.
    let mut c_gpu = ctx
        .device
        .alloc_zeros::<f32>(m * n)
        .map_err(|e| TensorError::Cuda(e.to_string()))?;

    // cuBLAS is column-major: C = A·B (row-major) ⟺ C^T = B^T·A^T (col-major).
    // Swap operand order: m=N, n=M, k=K; a=B_data (lda=N), b=A_data (ldb=K).
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };
    // Queues the GEMM kernel on ctx.stream — non-blocking on the CPU side.
    unsafe { ctx.cublas.gemm(cfg, &*b_gpu, &*a_gpu, &mut c_gpu) }
        .map_err(|e| TensorError::Cuda(e.to_string()))?;

    Ok(Tensor {
        storage: TensorStorage::Cuda(Arc::new(c_gpu)),
        shape: vec![m, n],
    })
}

/// Blocks the calling thread until all kernels queued on `ctx.stream` have
/// completed.
///
/// Call this after a batch of [`matmul_cuda_async`] submissions to materialise
/// all output tensors in the batch at once.
///
/// # Errors
/// Returns [`TensorError::Cuda`] if the CUDA driver reports a stream error.
#[cfg(feature = "cuda")]
pub fn sync_stream(ctx: &GpuWorkerContext) -> Result<(), TensorError> {
    // CudaStream has no .synchronize() method in cudarc 0.12.
    // The stream-level sync lives in result::stream::synchronize, which takes
    // the raw sys::CUstream handle exposed by the public `.stream` field.
    // We use per-stream sync rather than device.synchronize() so we only block
    // on ctx.stream's work, leaving other workers' streams unaffected.
    unsafe {
        cudarc::driver::result::stream::synchronize(ctx.stream.stream)
            .map_err(|e| TensorError::Cuda(e.to_string()))
    }
}

// ── Serde ─────────────────────────────────────────────────────────────────────

/// Wire format: flat shape + flat f32 data. Only CPU tensors can be serialized.
#[derive(Serialize, Deserialize)]
struct TensorRepr {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Serialize for Tensor {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.storage {
            TensorStorage::Cpu(arr) => TensorRepr {
                shape: arr.shape().to_vec(),
                data: arr.iter().copied().collect(),
            }
            .serialize(s),
            #[cfg(feature = "cuda")]
            TensorStorage::Cuda(_) => Err(serde::ser::Error::custom(
                "cannot serialize a GPU tensor — call to_device(&Device::Cpu) first",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let repr = TensorRepr::deserialize(d)?;
        Tensor::from_shape_vec(&repr.shape, repr.data).map_err(serde::de::Error::custom)
    }
}
