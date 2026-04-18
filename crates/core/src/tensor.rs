use ndarray::{Array, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from [`Tensor`] construction.
#[derive(Debug, Error)]
pub enum TensorError {
    /// The flat data length does not match the product of `shape`.
    #[error("data length {got} does not match shape {shape:?} (expected {expected})")]
    ShapeMismatch {
        shape: Vec<usize>,
        expected: usize,
        got: usize,
    },
}

/// A dense f32 tensor backed by an n-dimensional [`ndarray::ArrayD`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}

impl Tensor {
    /// Creates a tensor from a shape and flat data buffer.
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
        Ok(Self {
            data: Array::from_shape_vec(IxDyn(shape), data).map_err(|_| {
                TensorError::ShapeMismatch {
                    shape: shape.to_vec(),
                    expected,
                    got,
                }
            })?,
        })
    }

    /// Creates a zero-filled tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Creates a tensor filled with `value`.
    pub fn full(shape: &[usize], value: f32) -> Self {
        Self {
            data: ArrayD::from_elem(IxDyn(shape), value),
        }
    }

    /// Shape of the tensor (dimension sizes).
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}
