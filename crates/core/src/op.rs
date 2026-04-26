use serde::{Deserialize, Serialize};

/// Unique identifier for an [`Op`] within a [`Dag`](crate::Dag).
pub type OpId = usize;

/// The kind of computation performed by an [`Op`], with parameters used for
/// cost estimation and shape inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpKind {
    /// Dense matrix multiplication: C = A (m×k) · B (k×n) → (m×n).
    Matmul { m: usize, n: usize, k: usize },
    /// Element-wise ReLU: out = max(0, x).
    Relu { len: usize },
    /// Layer normalisation over all elements (mean=0, std=1).
    LayerNorm { len: usize },
    /// Sum-reduction along a single axis.
    Reduce { axis: usize, len: usize },
    /// Numerically-stable softmax along the last axis: exp(x−max) / Σexp(x−max).
    Softmax { len: usize },
    /// Batch normalisation (inference mode): scale * (x − mean) / sqrt(var + ε) + bias.
    BatchNorm { epsilon: f32 },
    /// 2-D convolution via im2col + GEMM.
    Conv2d {
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    /// Element-wise addition of two tensors with NumPy-style broadcasting.
    Add,
    /// 2-D max pooling over a sliding window.
    MaxPool {
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    /// Reshape a tensor to a new shape; `-1` in `target_shape` is inferred.
    Reshape { target_shape: Vec<i64> },
    /// Artificial delay: sleeps for `duration_ms` milliseconds then passes the
    /// single input tensor through unchanged.  Used for skew injection in benchmarks.
    Slow { duration_ms: u64 },
    /// Parsed but not yet implemented.  Execution returns an error; `info` reports it.
    Unsupported { name: String },
}

impl OpKind {
    /// Heuristic work-unit estimate used by the scheduler for load balancing.
    ///
    /// Larger value means more expensive. Units are abstract (not nanoseconds).
    pub fn cost_estimate(&self) -> u64 {
        match self {
            OpKind::Matmul { m, n, k } => (m * n * k) as u64,
            OpKind::Relu { len } | OpKind::LayerNorm { len } | OpKind::Softmax { len } => {
                *len as u64
            }
            OpKind::BatchNorm { .. } => 1,
            OpKind::Conv2d { .. } => 1,
            OpKind::Add => 1,
            OpKind::MaxPool { .. } => 1,
            OpKind::Reshape { .. } => 0,
            OpKind::Reduce { len, .. } => *len as u64,
            OpKind::Slow { duration_ms } => *duration_ms,
            OpKind::Unsupported { .. } => 0,
        }
    }
}

/// A single node in the computation [`Dag`](crate::Dag).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Op {
    /// Unique identifier. Must equal the op's index in [`Dag::ops`](crate::Dag::ops).
    pub id: OpId,
    /// The operation to execute.
    pub kind: OpKind,
    /// IDs of ops whose output tensors feed this op (in argument order).
    pub input_ids: Vec<OpId>,
    /// Expected shape of this op's output tensor.
    pub output_shape: Vec<usize>,
}

impl Op {
    /// Creates a new `Op`.
    pub fn new(id: OpId, kind: OpKind, input_ids: Vec<OpId>, output_shape: Vec<usize>) -> Self {
        Self {
            id,
            kind,
            input_ids,
            output_shape,
        }
    }

    /// Delegates to [`OpKind::cost_estimate`].
    pub fn cost_estimate(&self) -> u64 {
        self.kind.cost_estimate()
    }
}
