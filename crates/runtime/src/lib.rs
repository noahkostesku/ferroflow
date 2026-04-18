pub mod executor;
pub mod static_scheduler;
pub mod work_stealing;
pub mod worker;

#[cfg(feature = "distributed")]
pub mod mpi_worker;

pub use executor::{ExecutorError, SequentialExecutor};
pub use ferroflow_core::{
    LiveMetrics, RunMetrics, SchedulerMetrics, WorkerLiveSnapshot, WorkerLiveStatus,
};
pub use static_scheduler::{SchedulerError, StaticScheduler};
pub use work_stealing::WorkStealingScheduler;
pub use worker::{Message, WorkQueue, WorkerId, WorkerTrait};

#[cfg(feature = "distributed")]
pub use mpi_worker::{MpiSchedulerMode, MpiWorker};
