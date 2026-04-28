use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferroflow_core::{
    gpu_available,
    ops::{execute_op_auto, OpError},
    Dag, Device, DevicePolicy, LiveMetrics, OpId, SchedulerMetrics, Tensor, TensorError,
    WorkerLiveSnapshot, WorkerLiveStatus,
};
use tokio::sync::{watch, Mutex};

use crate::static_scheduler::SchedulerError;
use crate::worker::WorkQueue;

const DEFAULT_STEAL_THRESHOLD: usize = 2;
const STATUS_IDLE: u8 = 0;
const STATUS_EXEC: u8 = 1;
const STATUS_STEAL: u8 = 2;
const STEAL_HISTORY_WINDOW: usize = 20;
const ADAPTIVE_UPDATE_INTERVAL: usize = 10;

/// Rolling window of steal outcomes used to compute the adaptive steal threshold.
struct StealHistory {
    window: VecDeque<bool>,
}

impl StealHistory {
    fn new() -> Self {
        Self {
            window: VecDeque::with_capacity(STEAL_HISTORY_WINDOW),
        }
    }

    /// Records one steal outcome and slides the window forward.
    fn record(&mut self, success: bool) {
        if self.window.len() >= STEAL_HISTORY_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(success);
    }

    /// Fraction of recent steal attempts that succeeded.
    ///
    /// Returns 1.0 when the window is empty (optimistic start — steal freely until
    /// we have evidence to back off).
    fn success_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 1.0;
        }
        self.window.iter().filter(|&&s| s).count() as f64 / STEAL_HISTORY_WINDOW as f64
    }
}

/// Computes the new steal threshold for one worker based on recent steal outcomes.
///
/// Rules:
/// - Base: `max(1, n_workers / 8)` — scales with worker count so large pools don't all steal at once.
/// - `rate > 0.7` → lower threshold by 1 (stealing is productive, be more aggressive).
/// - `rate < 0.3` → raise threshold by 2 (stealing is futile, back off).
/// - Otherwise → keep `current_threshold`.
/// - Result is clamped to `[1, max(4, n_workers / 4)]`.
fn adaptive_threshold(
    current_threshold: usize,
    history: &StealHistory,
    n_workers: usize,
) -> usize {
    let rate = history.success_rate();
    let raw = if rate > 0.7 {
        current_threshold.saturating_sub(1)
    } else if rate < 0.3 {
        current_threshold + 2
    } else {
        current_threshold
    };
    let max_t = (n_workers / 4).max(4);
    raw.clamp(1, max_t)
}

/// Errors returned by [`WorkStealingScheduler::execute`].
///
/// Re-uses [`SchedulerError`] from the static scheduler — the error variants
/// are identical.
pub type WorkStealingError = SchedulerError;

/// High-level device placement strategy for a scheduler run.
///
/// Maps the `--device` CLI flag to a concrete placement rule without exposing
/// raw [`Device`] indices in caller code.
pub enum DeviceStrategy {
    /// All ops execute on the host CPU (default).
    AllCpu,
    /// All ops execute on CUDA device 0; tensors stay on GPU between ops.
    #[cfg(feature = "cuda")]
    AllGpu,
}

impl DeviceStrategy {
    /// Returns the concrete [`Device`] for this strategy.
    pub(crate) fn device(&self) -> Device {
        match self {
            DeviceStrategy::AllCpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            DeviceStrategy::AllGpu => Device::Cuda(0),
        }
    }
}

/// Work-stealing scheduler.
///
/// Workers start with a static round-robin assignment and steal ops from
/// busy peers when their own queue empties.  Stealing respects a threshold:
/// a worker is only stolen from when its queue length exceeds `steal_threshold`.
/// When `adaptive` is enabled (the default) each worker self-tunes its threshold
/// every [`ADAPTIVE_UPDATE_INTERVAL`] steal attempts based on its recent success
/// rate — see [`adaptive_threshold`].  Failed steal attempts back off with a
/// deterministic pseudo-random delay in the range 10–50 ms.
///
/// The device placement policy ([`DevicePolicy`]) controls where each op runs.
/// Under [`DevicePolicy::Auto`] workers make independent per-op routing decisions
/// without any global coordination.
pub struct WorkStealingScheduler {
    n_workers: usize,
    steal_threshold: usize,
    adaptive: bool,
    policy: DevicePolicy,
}

impl WorkStealingScheduler {
    /// Creates a new `WorkStealingScheduler` with `n_workers` concurrent workers,
    /// adaptive threshold enabled, and [`DevicePolicy::AllCpu`].
    pub fn new(n_workers: usize) -> Self {
        assert!(n_workers > 0, "n_workers must be at least 1");
        Self {
            n_workers,
            steal_threshold: DEFAULT_STEAL_THRESHOLD,
            adaptive: true,
            policy: DevicePolicy::AllCpu,
        }
    }

    /// Sets the fixed steal threshold used when adaptive mode is off and returns `self`.
    pub fn with_steal_threshold(mut self, threshold: usize) -> Self {
        self.steal_threshold = threshold;
        self
    }

    /// Enables or disables the adaptive threshold.  When disabled the fixed
    /// `steal_threshold` (default 2, or set via [`with_steal_threshold`](Self::with_steal_threshold))
    /// is used for the entire run.
    pub fn with_adaptive_threshold(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Sets the per-op device placement policy and returns `self`.
    ///
    /// Under [`DevicePolicy::Auto`] each worker independently routes ops to the
    /// optimal device without global coordination.
    pub fn with_policy(mut self, policy: DevicePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets the compute device used for all ops and returns `self`.
    ///
    /// Convenience bridge: converts `device` to the equivalent [`DevicePolicy`]
    /// variant.  Prefer [`with_policy`](Self::with_policy) for new code.
    pub fn with_device(mut self, device: Device) -> Self {
        self.policy = match device {
            Device::Cpu => DevicePolicy::AllCpu,
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => DevicePolicy::AllGpu,
        };
        self
    }

    /// Sets the device placement strategy and returns `self`.
    ///
    /// Convenience bridge for callers that still use the [`DeviceStrategy`] enum.
    /// Prefer [`with_policy`](Self::with_policy) for new code.
    pub fn with_strategy(mut self, strategy: DeviceStrategy) -> Self {
        self.policy = match strategy {
            DeviceStrategy::AllCpu => DevicePolicy::AllCpu,
            #[cfg(feature = "cuda")]
            DeviceStrategy::AllGpu => DevicePolicy::AllGpu,
        };
        self
    }

    /// Executes `dag` returning the complete tensor store and execution metrics.
    ///
    /// Convenience wrapper around [`execute_with_watch`](Self::execute_with_watch)
    /// that discards the live-metrics stream.
    ///
    /// # Errors
    /// Returns [`WorkStealingError`] if any op fails.
    pub async fn execute(
        &self,
        dag: Arc<Dag>,
        source_tensors: HashMap<OpId, Tensor>,
    ) -> Result<(HashMap<OpId, Tensor>, SchedulerMetrics), WorkStealingError> {
        let (tx, _rx) = watch::channel(LiveMetrics::empty(self.n_workers));
        self.execute_with_watch(dag, source_tensors, tx).await
    }

    /// Executes `dag` and streams live metrics snapshots every 100 ms via
    /// `metrics_tx`.
    ///
    /// Initial op assignment is round-robin (`op_id % n_workers`).  Workers
    /// steal from peers when idle, respecting the steal threshold.  A
    /// background ticker task reads per-worker atomics and sends a
    /// [`LiveMetrics`] snapshot to `metrics_tx` at 100 ms intervals.  The
    /// ticker stops when all workers finish or when all receivers are dropped.
    ///
    /// # Errors
    /// Returns [`WorkStealingError`] if any op fails.
    pub async fn execute_with_watch(
        &self,
        dag: Arc<Dag>,
        source_tensors: HashMap<OpId, Tensor>,
        metrics_tx: watch::Sender<LiveMetrics>,
    ) -> Result<(HashMap<OpId, Tensor>, SchedulerMetrics), WorkStealingError> {
        let n = self.n_workers;
        let fixed_threshold = self.steal_threshold;
        let adaptive = self.adaptive;
        let total_ops = dag.ops.iter().filter(|op| !op.input_ids.is_empty()).count() as u64;

        // Per-worker queues with round-robin initial assignment.
        let queues: Arc<Vec<WorkQueue>> = Arc::new((0..n).map(|_| WorkQueue::new()).collect());
        for op in &dag.ops {
            if !op.input_ids.is_empty() {
                queues[op.id % n].push(op.id).await;
            }
        }

        let store: Arc<Mutex<HashMap<OpId, Tensor>>> = Arc::new(Mutex::new(source_tensors));

        let (version_tx, version_rx) = watch::channel(0usize);
        let version_tx = Arc::new(version_tx);

        let completed = Arc::new(AtomicU64::new(0));
        let steal_attempts = Arc::new(AtomicU64::new(0));
        let successful_steals = Arc::new(AtomicU64::new(0));
        let threshold_adjustments = Arc::new(AtomicU64::new(0));
        // Stores the final threshold of any one worker — representative since all converge.
        let threshold_final = Arc::new(AtomicUsize::new(fixed_threshold));
        let gpu_ops = Arc::new(AtomicU64::new(0));
        let cpu_ops = Arc::new(AtomicU64::new(0));

        // Per-worker tracking for the live-metrics ticker.
        let worker_ops: Arc<Vec<AtomicU64>> = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());
        let worker_status: Arc<Vec<AtomicU8>> =
            Arc::new((0..n).map(|_| AtomicU8::new(STATUS_IDLE)).collect());
        let worker_idle_us: Arc<Vec<AtomicU64>> =
            Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());

        let t0 = Instant::now();

        // Wrap in Arc so both the ticker and the post-completion final send can use it.
        let metrics_tx = Arc::new(metrics_tx);

        // Helper closure: snapshot current atomic state into a LiveMetrics.
        let snapshot = |elapsed: f64| LiveMetrics {
            workers: (0..n)
                .map(|i| WorkerLiveSnapshot {
                    id: i,
                    ops_completed: worker_ops[i].load(Ordering::Relaxed),
                    idle_us: worker_idle_us[i].load(Ordering::Relaxed),
                    status: match worker_status[i].load(Ordering::Relaxed) {
                        STATUS_EXEC => WorkerLiveStatus::Executing,
                        STATUS_STEAL => WorkerLiveStatus::Stealing,
                        _ => WorkerLiveStatus::Idle,
                    },
                })
                .collect(),
            total_ops,
            completed_ops: completed.load(Ordering::Relaxed),
            elapsed_secs: elapsed,
            steal_attempts: steal_attempts.load(Ordering::Relaxed),
            successful_steals: successful_steals.load(Ordering::Relaxed),
        };

        // Background metrics ticker — fires every 50 ms.
        let ticker = {
            let completed_t = Arc::clone(&completed);
            let steal_attempts_t = Arc::clone(&steal_attempts);
            let successful_steals_t = Arc::clone(&successful_steals);
            let worker_ops_t = Arc::clone(&worker_ops);
            let worker_status_t = Arc::clone(&worker_status);
            let worker_idle_us_t = Arc::clone(&worker_idle_us);
            let tx = Arc::clone(&metrics_tx);

            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(50));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    let elapsed = t0.elapsed().as_secs_f64();
                    let workers = (0..n)
                        .map(|i| WorkerLiveSnapshot {
                            id: i,
                            ops_completed: worker_ops_t[i].load(Ordering::Relaxed),
                            idle_us: worker_idle_us_t[i].load(Ordering::Relaxed),
                            status: match worker_status_t[i].load(Ordering::Relaxed) {
                                STATUS_EXEC => WorkerLiveStatus::Executing,
                                STATUS_STEAL => WorkerLiveStatus::Stealing,
                                _ => WorkerLiveStatus::Idle,
                            },
                        })
                        .collect();
                    let live = LiveMetrics {
                        workers,
                        total_ops,
                        completed_ops: completed_t.load(Ordering::Relaxed),
                        elapsed_secs: elapsed,
                        steal_attempts: steal_attempts_t.load(Ordering::Relaxed),
                        successful_steals: successful_steals_t.load(Ordering::Relaxed),
                    };
                    if tx.send(live).is_err() {
                        break;
                    }
                }
            })
        };

        let mut handles = Vec::with_capacity(n);

        for worker_id in 0..n {
            let dag = Arc::clone(&dag);
            let store = Arc::clone(&store);
            let queues = Arc::clone(&queues);
            let version_tx = Arc::clone(&version_tx);
            let mut version_rx = version_rx.clone();
            let completed = Arc::clone(&completed);
            let steal_attempts = Arc::clone(&steal_attempts);
            let successful_steals = Arc::clone(&successful_steals);
            let threshold_adjustments = Arc::clone(&threshold_adjustments);
            let threshold_final = Arc::clone(&threshold_final);
            let worker_ops = Arc::clone(&worker_ops);
            let worker_status = Arc::clone(&worker_status);
            let worker_idle_us = Arc::clone(&worker_idle_us);
            let policy = self.policy.clone();
            let gpu_ops = Arc::clone(&gpu_ops);
            let cpu_ops = Arc::clone(&cpu_ops);

            handles.push(tokio::spawn(async move {
                let mut attempt = 0usize;
                // Per-worker adaptive threshold state — no shared mutable state, no locks.
                let base_threshold = (n / 8).max(1);
                let mut current_threshold = if adaptive { base_threshold } else { fixed_threshold };
                let mut steal_history = StealHistory::new();
                let mut attempts_since_update = 0usize;

                loop {
                    if completed.load(Ordering::Acquire) >= total_ops {
                        threshold_final.store(current_threshold, Ordering::Relaxed);
                        break;
                    }

                    worker_status[worker_id].store(STATUS_IDLE, Ordering::Relaxed);

                    let op_id = match queues[worker_id].pop().await {
                        Some(id) => id,
                        None => {
                            worker_status[worker_id].store(STATUS_STEAL, Ordering::Relaxed);
                            let mut stolen = None;
                            for offset in 1..n {
                                let victim = (worker_id + offset) % n;
                                steal_attempts.fetch_add(1, Ordering::Relaxed);
                                let result =
                                    queues[victim].steal_if_above(current_threshold).await;
                                let success = result.is_some();
                                steal_history.record(success);
                                attempts_since_update += 1;

                                if success {
                                    successful_steals.fetch_add(1, Ordering::Relaxed);
                                    stolen = result;
                                }

                                // Recompute threshold every ADAPTIVE_UPDATE_INTERVAL attempts.
                                if adaptive && attempts_since_update >= ADAPTIVE_UPDATE_INTERVAL {
                                    let new_t =
                                        adaptive_threshold(current_threshold, &steal_history, n);
                                    if new_t != current_threshold {
                                        threshold_adjustments.fetch_add(1, Ordering::Relaxed);
                                        current_threshold = new_t;
                                    }
                                    attempts_since_update = 0;
                                }

                                if stolen.is_some() {
                                    break;
                                }
                            }
                            match stolen {
                                Some(id) => {
                                    attempt = 0;
                                    id
                                }
                                None => {
                                    worker_status[worker_id].store(STATUS_IDLE, Ordering::Relaxed);
                                    let backoff_ms =
                                        10 + ((worker_id * 13 + attempt * 7) % 41) as u64;
                                    attempt += 1;
                                    let tw = Instant::now();
                                    tokio::select! {
                                        biased;
                                        _ = version_rx.changed() => {}
                                        _ = tokio::time::sleep(
                                            Duration::from_millis(backoff_ms)
                                        ) => {}
                                    }
                                    worker_idle_us[worker_id].fetch_add(
                                        tw.elapsed().as_micros() as u64,
                                        Ordering::Relaxed,
                                    );
                                    continue;
                                }
                            }
                        }
                    };

                    worker_status[worker_id].store(STATUS_EXEC, Ordering::Relaxed);

                    // Wait for all input dependencies.
                    loop {
                        version_rx.borrow_and_update();
                        let ready = {
                            let s = store.lock().await;
                            let op = dag
                                .get_op(op_id)
                                .ok_or(SchedulerError::Internal("valid op id"))?;
                            op.input_ids.iter().all(|dep| s.contains_key(dep))
                        };
                        if ready {
                            break;
                        }
                        let tw = Instant::now();
                        version_rx.changed().await.ok();
                        worker_idle_us[worker_id]
                            .fetch_add(tw.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }

                    let op = dag
                        .get_op(op_id)
                        .ok_or(SchedulerError::Internal("valid op id"))?;
                    // Resolve the target device once — used for input transfer and
                    // routing-metric tracking before the execute call.
                    let gpu = gpu_available();
                    let target_device = policy.device_for_op(op, gpu);
                    let is_gpu_op = target_device != Device::Cpu;
                    let inputs: Vec<Tensor> = {
                        let s = store.lock().await;
                        op.input_ids
                            .iter()
                            .map(|dep| s[dep].to_device_cached(&target_device))
                            .collect::<Result<Vec<_>, TensorError>>()
                            .map_err(|e| SchedulerError::OpFailed {
                                op_id,
                                source: OpError::Tensor(e),
                            })?
                    };
                    let input_refs: Vec<&Tensor> = inputs.iter().collect();
                    let result = execute_op_auto(op, &input_refs, &policy)
                        .map_err(|source| SchedulerError::OpFailed { op_id, source })?;
                    if is_gpu_op {
                        gpu_ops.fetch_add(1, Ordering::Relaxed);
                    } else {
                        cpu_ops.fetch_add(1, Ordering::Relaxed);
                    }

                    store.lock().await.insert(op_id, result);
                    worker_ops[worker_id].fetch_add(1, Ordering::Relaxed);
                    completed.fetch_add(1, Ordering::Release);
                    version_tx.send_modify(|v| *v += 1);
                }

                Ok::<(), WorkStealingError>(())
            }));
        }

        for handle in handles {
            handle
                .await
                .map_err(|e| SchedulerError::WorkerPanicked(e.to_string()))??;
        }

        // Send one final snapshot so the TUI sees the completed state regardless
        // of whether the 50 ms ticker managed to fire during execution.
        let _ = metrics_tx.send(snapshot(t0.elapsed().as_secs_f64()));

        ticker.abort();

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let total_idle_micros: u64 = worker_idle_us
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .sum();
        let idle_time_ms = total_idle_micros as f64 / 1000.0;
        let sa = steal_attempts.load(Ordering::Relaxed);
        let ss = successful_steals.load(Ordering::Relaxed);
        let ta = threshold_adjustments.load(Ordering::Relaxed);
        let tf = threshold_final.load(Ordering::Relaxed);
        let go = gpu_ops.load(Ordering::Relaxed);
        let co = cpu_ops.load(Ordering::Relaxed);
        let metrics = SchedulerMetrics::new(
            total_ops,
            total_ops,
            elapsed_ms,
            idle_time_ms,
            sa,
            ss,
            tf,
            ta,
            go,
            co,
        );

        let store = Arc::try_unwrap(store)
            .map_err(|_| SchedulerError::Internal("all worker handles dropped"))?
            .into_inner();
        Ok((store, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferroflow_core::{Op, OpKind};

    #[tokio::test]
    async fn ws_chain_matches_sequential() {
        let ops = vec![
            Op::new(0, OpKind::Relu { len: 4 }, vec![], vec![4]),
            Op::new(1, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(2, OpKind::Relu { len: 4 }, vec![1], vec![4]),
            Op::new(3, OpKind::Relu { len: 4 }, vec![2], vec![4]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = WorkStealingScheduler::new(2);

        let mut sources = HashMap::new();
        sources.insert(
            0usize,
            Tensor::from_shape_vec(&[4], vec![-3., -1., 0., 2.]).unwrap(),
        );

        let (results, metrics) = sched.execute(dag, sources).await.unwrap();
        let out: Vec<f32> = results[&3].cpu_array().unwrap().iter().copied().collect();
        assert!(out.iter().all(|&v| v >= 0.0));
        assert_eq!(out[3], 2.0);
        assert_eq!(metrics.total_ops, 3);
        assert!(metrics.elapsed_ms >= 0.0);
    }

    #[tokio::test]
    async fn ws_fan_out_all_ops_present() {
        let ops = vec![
            Op::new(0, OpKind::Relu { len: 4 }, vec![], vec![4]),
            Op::new(1, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(2, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(3, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(4, OpKind::Relu { len: 4 }, vec![0], vec![4]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = WorkStealingScheduler::new(4);

        let mut sources = HashMap::new();
        sources.insert(0usize, Tensor::full(&[4], 1.0));

        let (results, _) = sched.execute(dag, sources).await.unwrap();
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn ws_diamond() {
        let ops = vec![
            Op::new(0, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),
            Op::new(1, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),
            Op::new(
                2,
                OpKind::Matmul { m: 2, n: 2, k: 2 },
                vec![0, 1],
                vec![2, 2],
            ),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = WorkStealingScheduler::new(2);

        let mut sources = HashMap::new();
        sources.insert(
            0usize,
            Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]).unwrap(),
        );
        sources.insert(
            1usize,
            Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap(),
        );

        let (results, _) = sched.execute(dag, sources).await.unwrap();
        let flat: Vec<f32> = results[&2].cpu_array().unwrap().iter().copied().collect();
        assert_eq!(flat, vec![1., 2., 3., 4.]);
    }

    #[tokio::test]
    async fn execute_with_watch_sends_metrics() {
        let ops = vec![
            Op::new(0, OpKind::Relu { len: 4 }, vec![], vec![4]),
            Op::new(1, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(2, OpKind::Relu { len: 4 }, vec![1], vec![4]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = WorkStealingScheduler::new(2);

        let mut sources = HashMap::new();
        sources.insert(0usize, Tensor::full(&[4], 1.0));

        let init = LiveMetrics::empty(2);
        let (tx, rx) = watch::channel(init);

        let (_, metrics) = sched.execute_with_watch(dag, sources, tx).await.unwrap();
        assert_eq!(metrics.total_ops, 2);
        // Receiver should have seen at least one update.
        assert_eq!(rx.borrow().workers.len(), 2);
    }
}
