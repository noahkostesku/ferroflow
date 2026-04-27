use ndarray::{Axis, Ix2, Ix4, IxDyn};
use thiserror::Error;

use crate::device::Device;
use crate::op::{Op, OpKind};
use crate::tensor::Tensor;

/// Errors produced when executing a single [`Op`].
#[derive(Debug, Error)]
pub enum OpError {
    /// Wrong number of input tensors supplied.
    #[error("op {op}: expected {expected} input(s), got {got}")]
    InputCount {
        op: &'static str,
        expected: usize,
        got: usize,
    },

    /// Tensor shapes are incompatible for this operation.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// A tensor did not have the required number of dimensions.
    #[error("dimensionality error: {0}")]
    Dimensionality(String),

    /// Op kind is parsed but not yet implemented.
    #[error("op '{0}' is not yet implemented")]
    Unimplemented(String),

    /// A tensor error propagated from the tensor layer.
    #[error("tensor error: {0}")]
    Tensor(#[from] crate::tensor::TensorError),

    /// CUDA driver error (only present when the `cuda` feature is enabled).
    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    CudaDriver(String),

    /// cuBLAS error (only present when the `cuda` feature is enabled).
    #[cfg(feature = "cuda")]
    #[error("cuBLAS error: {0}")]
    CudaBlas(String),
}

/// Executes `op` given its ordered input `tensors` on the specified `device`.
///
/// `tensors` must be ordered to match `op.input_ids`.
///
/// When `device` is [`Device::Cuda`], only `Matmul` is accelerated; all other
/// ops execute on the CPU.  The GPU path performs host-to-device and
/// device-to-host copies around the kernel — per-op transfer overhead is a
/// known limitation documented for v0.1.
///
/// # Errors
/// Returns [`OpError`] on shape or arity mismatches, or CUDA driver failures.
pub fn execute_op(op: &Op, tensors: &[&Tensor], device: &Device) -> Result<Tensor, OpError> {
    match device {
        Device::Cpu => execute_op_cpu(op, tensors),
        #[cfg(feature = "cuda")]
        Device::Cuda(idx) => {
            let dev = crate::device::get_cuda_device(*idx)
                .map_err(|e| OpError::CudaDriver(e.to_string()))?;
            match &op.kind {
                OpKind::Matmul { .. } => {
                    require_inputs("matmul", tensors, 2)?;
                    matmul_cuda(tensors[0], tensors[1], &dev)
                }
                // All other ops run on the CPU in v0.1.
                _ => execute_op_cpu(op, tensors),
            }
        }
    }
}

// ── CPU execution path ────────────────────────────────────────────────────────

fn execute_op_cpu(op: &Op, tensors: &[&Tensor]) -> Result<Tensor, OpError> {
    match &op.kind {
        OpKind::Matmul { .. } => {
            require_inputs("matmul", tensors, 2)?;
            matmul(tensors[0], tensors[1])
        }
        OpKind::Relu { .. } => {
            require_inputs("relu", tensors, 1)?;
            relu(tensors[0])
        }
        OpKind::LayerNorm { .. } => {
            require_inputs("layer_norm", tensors, 1)?;
            layer_norm(tensors[0])
        }
        OpKind::Reduce { axis, .. } => {
            require_inputs("reduce", tensors, 1)?;
            reduce(tensors[0], *axis)
        }
        OpKind::Softmax { .. } => {
            require_inputs("softmax", tensors, 1)?;
            softmax(tensors[0])
        }
        OpKind::BatchNorm { epsilon } => {
            require_inputs("batch_norm", tensors, 5)?;
            batch_norm(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], *epsilon)
        }
        OpKind::Conv2d {
            kernel_size,
            stride,
            padding,
        } => {
            require_inputs("conv2d", tensors, 2)?;
            conv2d(tensors[0], tensors[1], *kernel_size, *stride, *padding)
        }
        OpKind::Add => {
            require_inputs("add", tensors, 2)?;
            add(tensors[0], tensors[1])
        }
        OpKind::MaxPool {
            kernel_size,
            stride,
            padding,
        } => {
            require_inputs("maxpool", tensors, 1)?;
            maxpool(tensors[0], *kernel_size, *stride, *padding)
        }
        OpKind::Reshape { target_shape } => {
            require_inputs("reshape", tensors, 1)?;
            reshape(tensors[0], target_shape)
        }
        OpKind::Slow { duration_ms } => {
            require_inputs("slow", tensors, 1)?;
            std::thread::sleep(std::time::Duration::from_millis(*duration_ms));
            Ok(tensors[0].clone())
        }
        OpKind::Unsupported { name } => Err(OpError::Unimplemented(name.clone())),
    }
}

// ── GPU matmul via cuBLAS ─────────────────────────────────────────────────────

/// Runs matrix multiply on the GPU: `C = A · B`.
///
/// Both inputs must be 2-D CPU tensors `[M, K]` and `[K, N]`.  The function
/// copies them to the GPU, runs SGEMM via cuBLAS, copies the result back, and
/// returns a CPU tensor `[M, N]`.
///
/// cuBLAS is column-major.  For row-major inputs we use the identity
/// `C = A·B  ⟺  C^T = B^T · A^T` and swap operand order in `gemm`.
///
/// # Errors
/// Returns [`OpError::CudaDriver`] or [`OpError::CudaBlas`] on failure.
#[cfg(feature = "cuda")]
fn matmul_cuda(
    a: &Tensor,
    b: &Tensor,
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor, OpError> {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;

    let a_arr = a.cpu_array()?;
    let b_arr = b.cpu_array()?;

    let a2 = a_arr.view().into_dimensionality::<Ix2>().map_err(|_| {
        OpError::Dimensionality(format!(
            "matmul_cuda input A must be 2-D, got shape {:?}",
            a.shape()
        ))
    })?;
    let b2 = b_arr.view().into_dimensionality::<Ix2>().map_err(|_| {
        OpError::Dimensionality(format!(
            "matmul_cuda input B must be 2-D, got shape {:?}",
            b.shape()
        ))
    })?;

    let (m, k) = (a2.nrows(), a2.ncols());
    let (kb, n) = (b2.nrows(), b2.ncols());

    if k != kb {
        return Err(OpError::ShapeMismatch(format!(
            "matmul_cuda: A({m},{k}), B({kb},{n}) — inner dims must match"
        )));
    }

    // Upload inputs (row-major flat order).
    let a_gpu = dev
        .htod_copy(a_arr.iter().copied().collect::<Vec<f32>>())
        .map_err(|e| OpError::CudaDriver(e.to_string()))?;
    let b_gpu = dev
        .htod_copy(b_arr.iter().copied().collect::<Vec<f32>>())
        .map_err(|e| OpError::CudaDriver(e.to_string()))?;
    let mut c_gpu = dev
        .alloc_zeros::<f32>(m * n)
        .map_err(|e| OpError::CudaDriver(e.to_string()))?;

    // cuBLAS SGEMM: C^T = B · A (col-major) ≡ C = A · B (row-major).
    // m=N, n=M, k=K; a=B_data (lda=N), b=A_data (ldb=K), c=C_data (ldc=N).
    let blas = CudaBlas::new(std::sync::Arc::clone(dev))
        .map_err(|e| OpError::CudaBlas(e.to_string()))?;
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
    // Safety: shapes are validated above; GPU buffers are correctly sized.
    unsafe { blas.gemm(cfg, &b_gpu, &a_gpu, &mut c_gpu) }
        .map_err(|e| OpError::CudaBlas(e.to_string()))?;

    // Download result and wrap as CPU tensor.
    let c_flat = dev
        .dtoh_sync_copy(&c_gpu)
        .map_err(|e| OpError::CudaDriver(e.to_string()))?;
    Tensor::from_shape_vec(&[m, n], c_flat).map_err(|e| OpError::ShapeMismatch(e.to_string()))
}

// ── private CPU helpers ───────────────────────────────────────────────────────

fn require_inputs(op: &'static str, tensors: &[&Tensor], expected: usize) -> Result<(), OpError> {
    if tensors.len() != expected {
        return Err(OpError::InputCount {
            op,
            expected,
            got: tensors.len(),
        });
    }
    Ok(())
}

/// Dense matrix multiply: C = A · B.
///
/// Both inputs must be 2-D. A's column count must equal B's row count.
fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    let a_arr = a.cpu_array()?;
    let b_arr = b.cpu_array()?;
    let a2 = a_arr.view().into_dimensionality::<Ix2>().map_err(|_| {
        OpError::Dimensionality(format!(
            "matmul input A must be 2-D, got shape {:?}",
            a.shape()
        ))
    })?;
    let b2 = b_arr.view().into_dimensionality::<Ix2>().map_err(|_| {
        OpError::Dimensionality(format!(
            "matmul input B must be 2-D, got shape {:?}",
            b.shape()
        ))
    })?;
    if a2.ncols() != b2.nrows() {
        return Err(OpError::ShapeMismatch(format!(
            "matmul: A({},{}), B({},{}) — inner dims must match",
            a2.nrows(),
            a2.ncols(),
            b2.nrows(),
            b2.ncols()
        )));
    }
    Ok(Tensor {
        storage: crate::tensor::TensorStorage::Cpu(a2.dot(&b2).into_dyn()),
        shape: vec![a2.nrows(), b2.ncols()],
    })
}

/// Element-wise ReLU: out = max(0, x).
fn relu(x: &Tensor) -> Result<Tensor, OpError> {
    let arr = x.cpu_array()?;
    let out = arr.map(|&v| v.max(0.0));
    let shape = out.shape().to_vec();
    Ok(Tensor {
        storage: crate::tensor::TensorStorage::Cpu(out),
        shape,
    })
}

/// Layer normalisation: normalise all elements to mean=0, std≈1.
///
/// Uses a global mean/variance over all elements (flattened). A small ε=1e-5
/// is added to the variance for numerical stability.
fn layer_norm(x: &Tensor) -> Result<Tensor, OpError> {
    let arr = x.cpu_array()?;
    let n = arr.len() as f32;
    let mean = arr.sum() / n;
    let var = arr.map(|&v| (v - mean).powi(2)).sum() / n;
    let std = (var + 1e-5).sqrt();
    let out = arr.map(|&v| (v - mean) / std);
    let shape = out.shape().to_vec();
    Ok(Tensor {
        storage: crate::tensor::TensorStorage::Cpu(out),
        shape,
    })
}

/// Sum-reduction along `axis`.
///
/// # Errors
/// Returns [`OpError::ShapeMismatch`] if `axis >= x.ndim()`.
fn reduce(x: &Tensor, axis: usize) -> Result<Tensor, OpError> {
    if axis >= x.ndim() {
        return Err(OpError::ShapeMismatch(format!(
            "reduce axis {axis} out of bounds for ndim {}",
            x.ndim()
        )));
    }
    let arr = x.cpu_array()?;
    let out = arr.sum_axis(Axis(axis)).into_dyn();
    let shape = out.shape().to_vec();
    Ok(Tensor {
        storage: crate::tensor::TensorStorage::Cpu(out),
        shape,
    })
}

/// Numerically-stable softmax along the last axis.
///
/// For each row: out = exp(x − max(x)) / Σexp(x − max(x)).
/// Input shape [batch, classes] is preserved.
///
/// # Errors
/// Returns [`OpError::Dimensionality`] if the input is not 2-D.
fn softmax(x: &Tensor) -> Result<Tensor, OpError> {
    let arr = x.cpu_array()?;
    let x2 = arr.view().into_dimensionality::<Ix2>().map_err(|_| {
        OpError::Dimensionality(format!(
            "softmax input must be 2-D [batch, classes], got shape {:?}",
            x.shape()
        ))
    })?;
    let mut out = x2.to_owned();
    for mut row in out.rows_mut() {
        let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    let dyn_out = out.into_dyn();
    let shape = dyn_out.shape().to_vec();
    Ok(Tensor {
        storage: crate::tensor::TensorStorage::Cpu(dyn_out),
        shape,
    })
}

/// Batch normalisation (inference mode): scale * (x − mean) / sqrt(var + ε) + bias.
///
/// All of `scale`, `bias`, `mean`, `var` must be 1-D with length equal to the
/// channel dimension (axis 1 for 4-D, total elements for 1-D/2-D).
///
/// # Errors
/// Returns [`OpError::ShapeMismatch`] if parameter lengths are inconsistent.
fn batch_norm(
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    let c = scale.numel();
    for (name, t) in [("bias", bias), ("mean", mean), ("var", var)] {
        if t.numel() != c {
            return Err(OpError::ShapeMismatch(format!(
                "batch_norm: scale has {c} elements but {name} has {}",
                t.numel()
            )));
        }
    }
    let x_arr = x.cpu_array()?;
    let scale_arr = scale.cpu_array()?;
    let bias_arr = bias.cpu_array()?;
    let mean_arr = mean.cpu_array()?;
    let var_arr = var.cpu_array()?;

    let scale_s = scale_arr
        .as_slice()
        .ok_or_else(|| OpError::ShapeMismatch("scale not contiguous".into()))?;
    let bias_s = bias_arr
        .as_slice()
        .ok_or_else(|| OpError::ShapeMismatch("bias not contiguous".into()))?;
    let mean_s = mean_arr
        .as_slice()
        .ok_or_else(|| OpError::ShapeMismatch("mean not contiguous".into()))?;
    let var_s = var_arr
        .as_slice()
        .ok_or_else(|| OpError::ShapeMismatch("var not contiguous".into()))?;

    let spatial = if x.ndim() == 4 {
        x.shape()[2] * x.shape()[3]
    } else {
        1
    };

    let data: Vec<f32> = x_arr
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let ch = (i / spatial) % c;
            scale_s[ch] * (v - mean_s[ch]) / (var_s[ch] + epsilon).sqrt() + bias_s[ch]
        })
        .collect();
    Tensor::from_shape_vec(x.shape(), data).map_err(|e| OpError::ShapeMismatch(e.to_string()))
}

/// 2-D convolution via im2col + GEMM.
///
/// Input:  `x`      — shape [N, C_in,  H,    W]
/// Kernel: `weight` — shape [C_out, C_in, kH, kW]
/// Output: shape [N, C_out, out_H, out_W]
///   where out_H = (H + 2*padding − kH) / stride + 1
///
/// # Errors
/// Returns [`OpError::Dimensionality`] if inputs are not 4-D.
/// Returns [`OpError::ShapeMismatch`] if channel counts disagree.
fn conv2d(
    x: &Tensor,
    weight: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Tensor, OpError> {
    if x.ndim() != 4 {
        return Err(OpError::Dimensionality(format!(
            "conv2d input must be 4-D [N,C,H,W], got {:?}",
            x.shape()
        )));
    }
    if weight.ndim() != 4 {
        return Err(OpError::Dimensionality(format!(
            "conv2d weight must be 4-D [C_out,C_in,kH,kW], got {:?}",
            weight.shape()
        )));
    }

    let xs = x.shape();
    let ws = weight.shape();
    let (n, c_in, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let (c_out, wc_in, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);

    if wc_in != c_in {
        return Err(OpError::ShapeMismatch(format!(
            "conv2d: input C_in={c_in} but weight C_in={wc_in}"
        )));
    }
    let _ = kernel_size;

    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;

    let col_rows = kh * kw * c_in;
    let col_cols = n * out_h * out_w;
    let mut col = vec![0f32; col_rows * col_cols];

    let x_arr = x.cpu_array()?;
    let x4 = x_arr.view().into_dimensionality::<Ix4>().map_err(|_| {
        OpError::Dimensionality("conv2d: failed to view input as Ix4".into())
    })?;

    for ni in 0..n {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let col_c = ni * out_h * out_w + oh * out_w + ow;
                for ci in 0..c_in {
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let ih = oh * stride + khi;
                            let iw = ow * stride + kwi;
                            let val = if padding > 0 {
                                let ihi = ih as isize - padding as isize;
                                let iwi = iw as isize - padding as isize;
                                if ihi < 0
                                    || ihi >= h as isize
                                    || iwi < 0
                                    || iwi >= w as isize
                                {
                                    0.0
                                } else {
                                    x4[[ni, ci, ihi as usize, iwi as usize]]
                                }
                            } else {
                                x4[[ni, ci, ih, iw]]
                            };
                            let row_r = ci * kh * kw + khi * kw + kwi;
                            col[row_r * col_cols + col_c] = val;
                        }
                    }
                }
            }
        }
    }

    let w_arr = weight.cpu_array()?;
    let w4 = w_arr.view().into_dimensionality::<Ix4>().map_err(|_| {
        OpError::Dimensionality("conv2d: failed to view weight as Ix4".into())
    })?;
    let wmat_data: Vec<f32> = w4.iter().copied().collect();

    let mut out_data = vec![0f32; c_out * col_cols];
    for oi in 0..c_out {
        for cj in 0..col_cols {
            let mut s = 0f32;
            for k in 0..col_rows {
                s += wmat_data[oi * col_rows + k] * col[k * col_cols + cj];
            }
            out_data[oi * col_cols + cj] = s;
        }
    }

    let mut final_data = vec![0f32; n * c_out * out_h * out_w];
    for ni in 0..n {
        for oi in 0..c_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let src_col = ni * out_h * out_w + oh * out_w + ow;
                    let dst =
                        ni * c_out * out_h * out_w + oi * out_h * out_w + oh * out_w + ow;
                    final_data[dst] = out_data[oi * col_cols + src_col];
                }
            }
        }
    }

    Tensor::from_shape_vec(&[n, c_out, out_h, out_w], final_data)
        .map_err(|e| OpError::ShapeMismatch(e.to_string()))
}

/// Element-wise addition of two tensors with NumPy-style broadcasting.
///
/// If shapes are identical the sum is computed directly. Otherwise the smaller
/// tensor is broadcast to the larger shape per standard NumPy rules.
///
/// # Errors
/// Returns [`OpError::ShapeMismatch`] if the shapes are not broadcast-compatible.
fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    let a_arr = a.cpu_array()?;
    let b_arr = b.cpu_array()?;
    if a.shape() == b.shape() {
        let out = a_arr + b_arr;
        let shape = out.shape().to_vec();
        return Ok(Tensor {
            storage: crate::tensor::TensorStorage::Cpu(out),
            shape,
        });
    }
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let a_bc = a_arr.broadcast(IxDyn(&out_shape)).ok_or_else(|| {
        OpError::ShapeMismatch(format!(
            "add: cannot broadcast {:?} to {:?}",
            a.shape(),
            out_shape
        ))
    })?;
    let b_bc = b_arr.broadcast(IxDyn(&out_shape)).ok_or_else(|| {
        OpError::ShapeMismatch(format!(
            "add: cannot broadcast {:?} to {:?}",
            b.shape(),
            out_shape
        ))
    })?;
    let out = &a_bc + &b_bc;
    let shape = out.shape().to_vec();
    Ok(Tensor {
        storage: crate::tensor::TensorStorage::Cpu(out),
        shape,
    })
}

/// Compute the output shape when broadcasting `a` and `b` together.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, OpError> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    let a_off = rank - a.len();
    let b_off = rank - b.len();
    for i in 0..rank {
        let ai = if i >= a_off { a[i - a_off] } else { 1 };
        let bi = if i >= b_off { b[i - b_off] } else { 1 };
        out[i] = match (ai, bi) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(OpError::ShapeMismatch(format!(
                    "add: shapes {:?} and {:?} are not broadcast-compatible",
                    a, b
                )))
            }
        };
    }
    Ok(out)
}

/// 2-D max pooling.
///
/// Input:  `x` — shape [N, C, H, W]
/// Output: shape [N, C, out_H, out_W]
///   where out_H = (H + 2*padding − kernel_size) / stride + 1
///
/// # Errors
/// Returns [`OpError::Dimensionality`] if the input is not 4-D.
fn maxpool(
    x: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Tensor, OpError> {
    if x.ndim() != 4 {
        return Err(OpError::Dimensionality(format!(
            "maxpool input must be 4-D [N,C,H,W], got {:?}",
            x.shape()
        )));
    }
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;

    let x_arr = x.cpu_array()?;
    let x4 = x_arr.view().into_dimensionality::<Ix4>().map_err(|_| {
        OpError::Dimensionality("maxpool: failed to view input as Ix4".into())
    })?;

    let mut out_data = vec![f32::NEG_INFINITY; n * c * out_h * out_w];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for khi in 0..kernel_size {
                        for kwi in 0..kernel_size {
                            let ih = oh * stride + khi;
                            let iw = ow * stride + kwi;
                            let val = if padding > 0 {
                                let ihi = ih as isize - padding as isize;
                                let iwi = iw as isize - padding as isize;
                                if ihi < 0
                                    || ihi >= h as isize
                                    || iwi < 0
                                    || iwi >= w as isize
                                {
                                    f32::NEG_INFINITY
                                } else {
                                    x4[[ni, ci, ihi as usize, iwi as usize]]
                                }
                            } else {
                                x4[[ni, ci, ih, iw]]
                            };
                            max_val = max_val.max(val);
                        }
                    }
                    let dst = ni * c * out_h * out_w + ci * out_h * out_w + oh * out_w + ow;
                    out_data[dst] = max_val;
                }
            }
        }
    }
    Tensor::from_shape_vec(&[n, c, out_h, out_w], out_data)
        .map_err(|e| OpError::ShapeMismatch(e.to_string()))
}

/// Reshape a tensor, resolving any `-1` dimension from the element count.
///
/// # Errors
/// Returns [`OpError::ShapeMismatch`] if more than one dimension is `-1` or the
/// total element count is inconsistent with the new shape.
fn reshape(x: &Tensor, target_shape: &[i64]) -> Result<Tensor, OpError> {
    let total = x.numel();
    let neg_count = target_shape.iter().filter(|&&d| d < 0).count();
    if neg_count > 1 {
        return Err(OpError::ShapeMismatch(
            "reshape: at most one -1 dimension allowed".into(),
        ));
    }
    let known_product: usize = target_shape
        .iter()
        .filter(|&&d| d >= 0)
        .map(|&d| d as usize)
        .product();
    let shape: Vec<usize> = target_shape
        .iter()
        .map(|&d| {
            if d < 0 {
                if known_product == 0 {
                    0
                } else {
                    total / known_product
                }
            } else {
                d as usize
            }
        })
        .collect();
    let new_total: usize = shape.iter().product();
    if new_total != total {
        return Err(OpError::ShapeMismatch(format!(
            "reshape: {total} elements cannot reshape to {shape:?} ({new_total} elements)"
        )));
    }
    let x_arr = x.cpu_array()?;
    let flat: Vec<f32> = x_arr.iter().copied().collect();
    Tensor::from_shape_vec(&shape, flat).map_err(|e| OpError::ShapeMismatch(e.to_string()))
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{Op, OpKind};
    use crate::tensor::Tensor;

    fn make_op(kind: OpKind) -> Op {
        Op::new(0, kind, vec![], vec![])
    }

    fn exec(kind: OpKind, inputs: &[&Tensor]) -> Result<Tensor, OpError> {
        execute_op(&make_op(kind), inputs, &Device::Cpu)
    }

    // ── matmul ──────────────────────────────────────────────────────────────

    #[test]
    fn matmul_2x2() {
        let a = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![5., 6., 7., 8.]).unwrap();
        let out = exec(OpKind::Matmul { m: 2, n: 2, k: 2 }, &[&a, &b]).unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn matmul_wrong_inner_dims_errors() {
        let a = Tensor::from_shape_vec(&[2, 3], vec![0.; 6]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![0.; 4]).unwrap();
        let result = exec(OpKind::Matmul { m: 2, n: 2, k: 3 }, &[&a, &b]);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ── relu ────────────────────────────────────────────────────────────────

    #[test]
    fn relu_clamps_negatives() {
        let x = Tensor::from_shape_vec(&[4], vec![-2., -0.1, 0., 1.5]).unwrap();
        let out = exec(OpKind::Relu { len: 4 }, &[&x]).unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![0., 0., 0., 1.5]);
    }

    // ── layer_norm ──────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_zero_mean_unit_std() {
        let x = Tensor::from_shape_vec(&[4], vec![1., 2., 3., 4.]).unwrap();
        let out = exec(OpKind::LayerNorm { len: 4 }, &[&x]).unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
        let var: f32 =
            flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
        assert!(mean.abs() < 1e-5, "mean={mean}");
        assert!((var - 1.0).abs() < 1e-3, "var={var}");
    }

    // ── reduce ──────────────────────────────────────────────────────────────

    #[test]
    fn reduce_sum_axis0() {
        let x = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap();
        let out = exec(OpKind::Reduce { axis: 0, len: 4 }, &[&x]).unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![4., 6.]);
    }

    #[test]
    fn reduce_invalid_axis_errors() {
        let x = Tensor::from_shape_vec(&[2, 2], vec![0.; 4]).unwrap();
        let result = exec(OpKind::Reduce { axis: 5, len: 4 }, &[&x]);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ── slow ────────────────────────────────────────────────────────────────

    #[test]
    fn slow_passes_through_input() {
        let x = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).unwrap();
        let out = exec(OpKind::Slow { duration_ms: 0 }, &[&x]).unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![1., 2., 3.]);
    }

    // ── softmax ─────────────────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let x = Tensor::from_shape_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let out = exec(OpKind::Softmax { len: 3 }, &[&x]).unwrap();
        let sum: f32 = out.cpu_array().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn softmax_numerically_stable() {
        let x = Tensor::from_shape_vec(&[1, 3], vec![1000.0, 1001.0, 1002.0]).unwrap();
        let out = exec(OpKind::Softmax { len: 3 }, &[&x]).unwrap();
        assert!(out.cpu_array().unwrap().iter().all(|v| v.is_finite()), "non-finite values");
        let sum: f32 = out.cpu_array().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn softmax_preserves_shape() {
        let x = Tensor::from_shape_vec(&[2, 4], vec![0.0; 8]).unwrap();
        let out = exec(OpKind::Softmax { len: 8 }, &[&x]).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }

    // ── batch_norm ──────────────────────────────────────────────────────────

    #[test]
    fn batch_norm_zero_mean_unit_var() {
        let x = Tensor::from_shape_vec(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scale = Tensor::from_shape_vec(&[4], vec![2.0, 2.0, 2.0, 2.0]).unwrap();
        let bias = Tensor::from_shape_vec(&[4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let mean = Tensor::from_shape_vec(&[4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let var = Tensor::from_shape_vec(&[4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let out = exec(
            OpKind::BatchNorm { epsilon: 1e-5 },
            &[&x, &scale, &bias, &mean, &var],
        )
        .unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        for (i, (&got, expected)) in flat.iter().zip([3.0f32, 5.0, 7.0, 9.0]).enumerate() {
            assert!((got - expected).abs() < 1e-4, "i={i} got={got} expected={expected}");
        }
    }

    #[test]
    fn batch_norm_epsilon_prevents_div_zero() {
        let x = Tensor::from_shape_vec(&[1, 2], vec![1.0, 2.0]).unwrap();
        let scale = Tensor::from_shape_vec(&[2], vec![1.0, 1.0]).unwrap();
        let bias = Tensor::from_shape_vec(&[2], vec![0.0, 0.0]).unwrap();
        let mean = Tensor::from_shape_vec(&[2], vec![0.0, 0.0]).unwrap();
        let var = Tensor::from_shape_vec(&[2], vec![0.0, 0.0]).unwrap();
        let out =
            exec(OpKind::BatchNorm { epsilon: 1e-5 }, &[&x, &scale, &bias, &mean, &var]).unwrap();
        assert!(out.cpu_array().unwrap().iter().all(|v| v.is_finite()));
    }

    // ── conv2d ───────────────────────────────────────────────────────────────

    #[test]
    fn conv2d_3x3_on_5x5_gives_3x3() {
        let x = Tensor::from_shape_vec(&[1, 1, 5, 5], vec![1.0; 25]).unwrap();
        let w = Tensor::from_shape_vec(&[1, 1, 3, 3], vec![1.0; 9]).unwrap();
        let out = exec(
            OpKind::Conv2d { kernel_size: 3, stride: 1, padding: 0 },
            &[&x, &w],
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        assert!(out.cpu_array().unwrap().iter().all(|&v| (v - 9.0).abs() < 1e-5));
    }

    #[test]
    fn conv2d_1x1_equals_pointwise_matmul() {
        let x =
            Tensor::from_shape_vec(&[1, 2, 3, 3], (1..=18).map(|v| v as f32).collect()).unwrap();
        let w = Tensor::from_shape_vec(&[2, 2, 1, 1], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let out = exec(
            OpKind::Conv2d { kernel_size: 1, stride: 1, padding: 0 },
            &[&x, &w],
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 2, 3, 3]);
        let xf: Vec<f32> = x.cpu_array().unwrap().iter().copied().collect();
        let of: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        for (i, (a, b)) in xf.iter().zip(&of).enumerate() {
            assert!((a - b).abs() < 1e-5, "i={i} x={a} out={b}");
        }
    }

    // ── add ──────────────────────────────────────────────────────────────────

    #[test]
    fn add_same_shape() {
        let a = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).unwrap();
        let b = Tensor::from_shape_vec(&[3], vec![4., 5., 6.]).unwrap();
        let out = exec(OpKind::Add, &[&a, &b]).unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![5., 7., 9.]);
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn add_broadcast_scalar() {
        let scalar = Tensor::from_shape_vec(&[1], vec![10.]).unwrap();
        let x = Tensor::from_shape_vec(&[3], vec![1., 2., 3.]).unwrap();
        let out = exec(OpKind::Add, &[&scalar, &x]).unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![11., 12., 13.]);
    }

    #[test]
    fn add_preserves_shape() {
        let a = Tensor::from_shape_vec(&[2, 3], vec![1.; 6]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 3], vec![2.; 6]).unwrap();
        let out = exec(OpKind::Add, &[&a, &b]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }

    // ── maxpool ──────────────────────────────────────────────────────────────

    #[test]
    fn maxpool_2x2_stride2_on_4x4() {
        let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let x = Tensor::from_shape_vec(&[1, 1, 4, 4], data).unwrap();
        let out = exec(
            OpKind::MaxPool { kernel_size: 2, stride: 2, padding: 0 },
            &[&x],
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![6., 8., 14., 16.]);
    }

    #[test]
    fn maxpool_selects_max_not_sum() {
        let x = Tensor::from_shape_vec(&[1, 1, 2, 2], vec![3., 1., 4., 2.]).unwrap();
        let out = exec(
            OpKind::MaxPool { kernel_size: 2, stride: 2, padding: 0 },
            &[&x],
        )
        .unwrap();
        let flat: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![4.]);
    }

    #[test]
    fn maxpool_stride_reduces_spatial() {
        let x = Tensor::from_shape_vec(&[1, 1, 4, 4], vec![1.0; 16]).unwrap();
        let out = exec(
            OpKind::MaxPool { kernel_size: 2, stride: 2, padding: 0 },
            &[&x],
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
    }

    // ── reshape ──────────────────────────────────────────────────────────────

    #[test]
    fn reshape_2x3x4_to_2x12() {
        let x =
            Tensor::from_shape_vec(&[2, 3, 4], (0..24).map(|v| v as f32).collect()).unwrap();
        let out = exec(OpKind::Reshape { target_shape: vec![2, 12] }, &[&x]).unwrap();
        assert_eq!(out.shape(), &[2, 12]);
        let xf: Vec<f32> = x.cpu_array().unwrap().iter().copied().collect();
        let of: Vec<f32> = out.cpu_array().unwrap().iter().copied().collect();
        assert_eq!(xf, of);
    }

    #[test]
    fn reshape_flat_to_3d() {
        let x = Tensor::from_shape_vec(&[24], (0..24).map(|v| v as f32).collect()).unwrap();
        let out = exec(OpKind::Reshape { target_shape: vec![2, 3, 4] }, &[&x]).unwrap();
        assert_eq!(out.shape(), &[2, 3, 4]);
    }

    #[test]
    fn reshape_neg1_inference() {
        let x = Tensor::from_shape_vec(&[2, 3, 4], vec![0.0; 24]).unwrap();
        let out = exec(OpKind::Reshape { target_shape: vec![2, -1] }, &[&x]).unwrap();
        assert_eq!(out.shape(), &[2, 12]);
    }

    // ── arity guard ─────────────────────────────────────────────────────────

    #[test]
    fn wrong_input_count_errors() {
        let x = Tensor::zeros(&[2, 2]);
        let result = exec(OpKind::Relu { len: 4 }, &[&x, &x]);
        assert!(matches!(result, Err(OpError::InputCount { .. })));
    }
}
