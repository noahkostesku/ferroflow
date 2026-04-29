use std::fmt;

use serde::{Deserialize, Serialize};

/// Real-time status of a single worker during scheduler execution.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerLiveStatus {
    /// No work queued; waiting for steals or dependency completion.
    Idle,
    /// Currently executing an op.
    Executing,
    /// Scanning peer queues attempting a steal.
    Stealing,
}

/// Per-worker snapshot emitted by the live-metrics ticker.
#[derive(Debug, Clone)]
pub struct WorkerLiveSnapshot {
    /// Zero-based worker index.
    pub id: usize,
    /// Ops completed by this worker so far.
    pub ops_completed: u64,
    /// Cumulative idle time in microseconds.
    pub idle_us: u64,
    /// Current activity status.
    pub status: WorkerLiveStatus,
}

/// Scheduler state snapshot emitted every 100 ms by
/// [`WorkStealingScheduler::execute_with_watch`](crate::WorkStealingScheduler::execute_with_watch).
#[derive(Debug, Clone)]
pub struct LiveMetrics {
    /// Per-worker snapshots.
    pub workers: Vec<WorkerLiveSnapshot>,
    /// Total compute ops in the DAG.
    pub total_ops: u64,
    /// Ops completed so far.
    pub completed_ops: u64,
    /// Wall-clock seconds since execution started.
    pub elapsed_secs: f64,
    /// Cumulative steal attempts across all workers.
    pub steal_attempts: u64,
    /// Cumulative successful steals across all workers.
    pub successful_steals: u64,
}

impl LiveMetrics {
    /// Constructs an empty initial snapshot with all workers idle.
    pub fn empty(n_workers: usize) -> Self {
        Self {
            workers: (0..n_workers)
                .map(|i| WorkerLiveSnapshot {
                    id: i,
                    ops_completed: 0,
                    idle_us: 0,
                    status: WorkerLiveStatus::Idle,
                })
                .collect(),
            total_ops: 0,
            completed_ops: 0,
            elapsed_secs: 0.0,
            steal_attempts: 0,
            successful_steals: 0,
        }
    }
}

/// Execution metrics collected by a scheduler after running a DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    /// Total number of compute ops in the DAG (source ops excluded).
    pub total_ops: u64,
    /// Number of ops that completed successfully.
    pub completed_ops: u64,
    /// Wall-clock time for the full execution in milliseconds.
    pub elapsed_ms: f64,
    /// Derived: `completed_ops / elapsed_sec`.
    pub throughput_ops_per_sec: f64,
    /// Total worker idle time in milliseconds (summed across all workers).
    ///
    /// For the sequential executor this is always 0. For parallel schedulers
    /// it captures both dependency-wait time and backoff-sleep time.
    pub idle_time_ms: f64,
    /// Number of steal attempts made by workers (0 for non-stealing schedulers).
    pub steal_attempts: u64,
    /// Number of steals that successfully transferred an op.
    pub successful_steals: u64,
    /// Derived: `successful_steals / elapsed_sec`.
    pub steal_rate: f64,
    /// Steal threshold in effect at job end (0 when adaptive is disabled or not applicable).
    #[serde(default)]
    pub threshold_final: usize,
    /// Number of times the adaptive threshold changed value during execution.
    #[serde(default)]
    pub threshold_adjustments: u64,
    /// Ops routed to GPU under the `auto` policy (0 for `AllCpu`/`AllGpu` runs).
    #[serde(default)]
    pub gpu_ops: u64,
    /// Ops routed to CPU under the `auto` policy (0 for `AllCpu`/`AllGpu` runs).
    #[serde(default)]
    pub cpu_ops: u64,
    /// Number of GPU batch submissions (0 when CUDA is not enabled or no GPU ops ran).
    #[serde(default)]
    pub gpu_batches: u64,
    /// Average number of ops per GPU batch submission (0.0 when `gpu_batches == 0`).
    #[serde(default)]
    pub gpu_batch_size_avg: f64,
}

impl SchedulerMetrics {
    /// Constructs `SchedulerMetrics`, computing derived fields from raw values.
    ///
    /// `gpu_batch_size_total` is the sum of all batch sizes; `gpu_batch_size_avg`
    /// is derived as `gpu_batch_size_total / gpu_batches` (0.0 when no batches ran).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        total_ops: u64,
        completed_ops: u64,
        elapsed_ms: f64,
        idle_time_ms: f64,
        steal_attempts: u64,
        successful_steals: u64,
        threshold_final: usize,
        threshold_adjustments: u64,
        gpu_ops: u64,
        cpu_ops: u64,
        gpu_batches: u64,
        gpu_batch_size_total: u64,
    ) -> Self {
        let elapsed_sec = elapsed_ms / 1000.0;
        let throughput_ops_per_sec = if elapsed_sec > 0.0 {
            completed_ops as f64 / elapsed_sec
        } else {
            0.0
        };
        let steal_rate = if elapsed_sec > 0.0 {
            successful_steals as f64 / elapsed_sec
        } else {
            0.0
        };
        let gpu_batch_size_avg = if gpu_batches > 0 {
            gpu_batch_size_total as f64 / gpu_batches as f64
        } else {
            0.0
        };
        Self {
            total_ops,
            completed_ops,
            elapsed_ms,
            throughput_ops_per_sec,
            idle_time_ms,
            steal_attempts,
            successful_steals,
            steal_rate,
            threshold_final,
            threshold_adjustments,
            gpu_ops,
            cpu_ops,
            gpu_batches,
            gpu_batch_size_avg,
        }
    }
}

/// Combined metadata + metrics for a single scheduler run.
///
/// Intended for logging to `docs/benchmarks.md` and JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetrics {
    /// Scheduler variant that produced these results.
    pub scheduler: String,
    /// Number of nodes (MPI ranks) involved in the run.
    pub nodes: u32,
    /// Worker tasks per node.
    pub workers_per_node: u32,
    /// Total ops in the DAG (including source ops).
    pub dag_size: u32,
    /// Whether the DAG had skewed op costs.
    pub skew: bool,
    /// Execution metrics.
    pub metrics: SchedulerMetrics,
}

impl fmt::Display for RunMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = &self.metrics;
        writeln!(f, "scheduler         : {}", self.scheduler)?;
        writeln!(f, "nodes             : {}", self.nodes)?;
        writeln!(f, "workers/node      : {}", self.workers_per_node)?;
        writeln!(f, "dag_size          : {}", self.dag_size)?;
        writeln!(f, "skew              : {}", self.skew)?;
        writeln!(f, "total_ops         : {}", m.total_ops)?;
        writeln!(f, "completed_ops     : {}", m.completed_ops)?;
        writeln!(f, "elapsed_ms        : {:.3}", m.elapsed_ms)?;
        writeln!(f, "throughput ops/s  : {:.1}", m.throughput_ops_per_sec)?;
        writeln!(f, "idle_time_ms      : {:.3}", m.idle_time_ms)?;
        writeln!(f, "steal_attempts    : {}", m.steal_attempts)?;
        writeln!(f, "successful_steals : {}", m.successful_steals)?;
        writeln!(f, "steal_rate /s     : {:.2}", m.steal_rate)?;
        writeln!(f, "threshold_final   : {}", m.threshold_final)?;
        writeln!(f, "threshold_adj     : {}", m.threshold_adjustments)?;
        writeln!(f, "gpu_ops           : {}", m.gpu_ops)?;
        writeln!(f, "cpu_ops           : {}", m.cpu_ops)?;
        writeln!(f, "gpu_batches       : {}", m.gpu_batches)?;
        write!(f, "avg_batch_size    : {:.1}", m.gpu_batch_size_avg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics() -> SchedulerMetrics {
        SchedulerMetrics::new(20, 20, 50.0, 5.0, 8, 3, 2, 4, 0, 0, 0, 0)
    }

    fn sample_run() -> RunMetrics {
        RunMetrics {
            scheduler: "work-stealing".into(),
            nodes: 1,
            workers_per_node: 4,
            dag_size: 20,
            skew: false,
            metrics: sample_metrics(),
        }
    }

    #[test]
    fn derived_fields_correct() {
        let m = sample_metrics();
        // elapsed_sec = 0.05  →  20 / 0.05 = 400
        assert!((m.throughput_ops_per_sec - 400.0).abs() < 1e-6);
        // 3 / 0.05 = 60
        assert!((m.steal_rate - 60.0).abs() < 1e-6);
    }

    #[test]
    fn zero_elapsed_no_divide_by_zero() {
        let m = SchedulerMetrics::new(10, 10, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(m.throughput_ops_per_sec, 0.0);
        assert_eq!(m.steal_rate, 0.0);
    }

    #[test]
    fn json_roundtrip_scheduler_metrics() {
        let m = sample_metrics();
        let json = serde_json::to_string(&m).expect("serialize");
        let back: SchedulerMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.total_ops, m.total_ops);
        assert_eq!(back.completed_ops, m.completed_ops);
        assert!((back.elapsed_ms - m.elapsed_ms).abs() < 1e-9);
        assert!((back.throughput_ops_per_sec - m.throughput_ops_per_sec).abs() < 1e-6);
        assert_eq!(back.steal_attempts, m.steal_attempts);
        assert_eq!(back.successful_steals, m.successful_steals);
    }

    #[test]
    fn json_roundtrip_run_metrics() {
        let r = sample_run();
        let json = serde_json::to_string(&r).expect("serialize");
        let back: RunMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.scheduler, r.scheduler);
        assert_eq!(back.nodes, r.nodes);
        assert_eq!(back.skew, r.skew);
        assert_eq!(back.metrics.total_ops, r.metrics.total_ops);
    }

    #[test]
    fn display_contains_key_fields() {
        let r = sample_run();
        let s = format!("{r}");
        assert!(s.contains("work-stealing"));
        assert!(s.contains("steal_attempts"));
        assert!(s.contains("idle_time_ms"));
        assert!(s.contains("threshold_final"));
        assert!(s.contains("threshold_adj"));
        assert!(s.contains("gpu_ops"));
        assert!(s.contains("cpu_ops"));
        assert!(s.contains("gpu_batches"));
        assert!(s.contains("avg_batch_size"));
    }
}
