pub mod dag;
pub mod op;
pub mod ops;
pub mod tensor;

pub use dag::{Dag, DagError};
pub use op::{Op, OpId, OpKind};
pub use ops::{execute_op, OpError};
pub use tensor::{Tensor, TensorError};
