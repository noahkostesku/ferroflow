use thiserror::Error;
use crate::{op::Op, tensor::Tensor};
#[derive(Debug,Error)] pub enum OpError { #[error("todo")] Todo }
pub fn execute_op(_op: &Op, _inputs: &[&Tensor]) -> Result<Tensor, OpError> { todo!() }
