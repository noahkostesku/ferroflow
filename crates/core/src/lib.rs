pub mod dag;
pub mod dag_gen;
pub mod device;
pub mod metrics;
pub mod op;
pub mod ops;
pub mod tensor;

pub use dag::{Dag, DagError};
pub use dag_gen::{
    gen_imbalanced, gen_large_transformer, gen_large_wide, gen_matmul_chain, gen_matmul_parallel,
    gen_resnet_block, gen_transformer_block, gen_wide_dag, gen_xlarge_transformer, gen_xlarge_wide,
};
pub use device::{gpu_available, Device, DeviceError, DevicePolicy};
pub use metrics::{
    LiveMetrics, RunMetrics, SchedulerMetrics, WorkerLiveSnapshot, WorkerLiveStatus,
};
pub use op::{Op, OpId, OpKind};
pub use ops::{execute_op, execute_op_auto, OpError};
pub use tensor::{Tensor, TensorError};
