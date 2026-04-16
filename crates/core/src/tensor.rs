use thiserror::Error;
#[derive(Debug,Error)] pub enum TensorError { #[error("shape")] Shape }
pub struct Tensor { pub data: ndarray::ArrayD<f32> }
