use ndarray::{Axis, Ix2};
use thiserror::Error;

use crate::op::{Op, OpKind};
use crate::tensor::Tensor;

/// Errors produced when executing a single [`Op`].
#[derive(Debug, Error)]
pub enum OpError {
    /// Wrong number of input tensors supplied.
    #[error("op {op}: expected {expected} input(s), got {got}")]
    InputCount { op: &'static str, expected: usize, got: usize },

    /// Tensor shapes are incompatible for this operation.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// A tensor did not have the required number of dimensions.
    #[error("dimensionality error: {0}")]
    Dimensionality(String),
}

/// Executes `op` given its ordered input `tensors` and returns the output tensor.
///
/// `tensors` must be ordered to match `op.input_ids`.
///
/// # Errors
/// Returns [`OpError`] on shape or arity mismatches.
pub fn execute_op(op: &Op, tensors: &[&Tensor]) -> Result<Tensor, OpError> {
    match &op.kind {
        OpKind::Matmul { .. } => {
            require_inputs("matmul", tensors, 2)?;
            matmul(tensors[0], tensors[1])
        }
        OpKind::Relu { .. } => {
            require_inputs("relu", tensors, 1)?;
            Ok(relu(tensors[0]))
        }
        OpKind::LayerNorm { .. } => {
            require_inputs("layer_norm", tensors, 1)?;
            Ok(layer_norm(tensors[0]))
        }
        OpKind::Reduce { axis, .. } => {
            require_inputs("reduce", tensors, 1)?;
            reduce(tensors[0], *axis)
        }
    }
}

// ── private helpers ──────────────────────────────────────────────────────────

fn require_inputs(op: &'static str, tensors: &[&Tensor], expected: usize) -> Result<(), OpError> {
    if tensors.len() != expected {
        return Err(OpError::InputCount { op, expected, got: tensors.len() });
    }
    Ok(())
}

/// Dense matrix multiply: C = A · B.
///
/// Both inputs must be 2-D. A's column count must equal B's row count.
fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    let a2 = a.data.view().into_dimensionality::<Ix2>().map_err(|_| {
        OpError::Dimensionality(format!(
            "matmul input A must be 2-D, got shape {:?}",
            a.shape()
        ))
    })?;
    let b2 = b.data.view().into_dimensionality::<Ix2>().map_err(|_| {
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
    Ok(Tensor { data: a2.dot(&b2).into_dyn() })
}

/// Element-wise ReLU: out = max(0, x).
fn relu(x: &Tensor) -> Tensor {
    Tensor { data: x.data.map(|&v| v.max(0.0)) }
}

/// Layer normalisation: normalise all elements to mean=0, std≈1.
///
/// Uses a global mean/variance over all elements (flattened). A small ε=1e-5
/// is added to the variance for numerical stability.
fn layer_norm(x: &Tensor) -> Tensor {
    let n = x.data.len() as f32;
    let mean = x.data.sum() / n;
    let var = x.data.map(|&v| (v - mean).powi(2)).sum() / n;
    let std = (var + 1e-5).sqrt();
    Tensor { data: x.data.map(|&v| (v - mean) / std) }
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
    Ok(Tensor { data: x.data.sum_axis(Axis(axis)).into_dyn() })
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{Op, OpKind};
    use crate::tensor::Tensor;

    fn op(kind: OpKind) -> Op {
        Op::new(0, kind, vec![], vec![])
    }

    // ── matmul ──────────────────────────────────────────────────────────────

    #[test]
    fn matmul_2x2() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![5., 6., 7., 8.]).unwrap();
        let out = execute_op(&op(OpKind::Matmul { m: 2, n: 2, k: 2 }), &[&a, &b]).unwrap();
        let flat: Vec<f32> = out.data.iter().copied().collect();
        assert_eq!(flat, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn matmul_wrong_inner_dims_errors() {
        let a = Tensor::from_shape_vec(&[2, 3], vec![0.; 6]).unwrap();
        let b = Tensor::from_shape_vec(&[2, 2], vec![0.; 4]).unwrap();
        let result = execute_op(&op(OpKind::Matmul { m: 2, n: 2, k: 3 }), &[&a, &b]);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ── relu ────────────────────────────────────────────────────────────────

    #[test]
    fn relu_clamps_negatives() {
        let x = Tensor::from_shape_vec(&[4], vec![-2., -0.1, 0., 1.5]).unwrap();
        let out = execute_op(&op(OpKind::Relu { len: 4 }), &[&x]).unwrap();
        let flat: Vec<f32> = out.data.iter().copied().collect();
        assert_eq!(flat, vec![0., 0., 0., 1.5]);
    }

    // ── layer_norm ──────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_zero_mean_unit_std() {
        let x = Tensor::from_shape_vec(&[4], vec![1., 2., 3., 4.]).unwrap();
        let out = execute_op(&op(OpKind::LayerNorm { len: 4 }), &[&x]).unwrap();
        let flat: Vec<f32> = out.data.iter().copied().collect();
        let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
        let var: f32 = flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
        assert!(mean.abs() < 1e-5, "mean={mean}");
        assert!((var - 1.0).abs() < 1e-3, "var={var}");
    }

    // ── reduce ──────────────────────────────────────────────────────────────

    #[test]
    fn reduce_sum_axis0() {
        // [[1,2],[3,4]] summed along axis 0 → [4,6]
        let x = Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap();
        let out = execute_op(&op(OpKind::Reduce { axis: 0, len: 4 }), &[&x]).unwrap();
        let flat: Vec<f32> = out.data.iter().copied().collect();
        assert_eq!(flat, vec![4., 6.]);
    }

    #[test]
    fn reduce_invalid_axis_errors() {
        let x = Tensor::from_shape_vec(&[2, 2], vec![0.; 4]).unwrap();
        let result = execute_op(&op(OpKind::Reduce { axis: 5, len: 4 }), &[&x]);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ── arity guard ─────────────────────────────────────────────────────────

    #[test]
    fn wrong_input_count_errors() {
        let x = Tensor::zeros(&[2, 2]);
        let result = execute_op(&op(OpKind::Relu { len: 4 }), &[&x, &x]);
        assert!(matches!(result, Err(OpError::InputCount { .. })));
    }
}
