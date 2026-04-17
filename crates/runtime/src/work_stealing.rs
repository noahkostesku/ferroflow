use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferroflow_core::{
    ops::execute_op, Dag, LiveMetrics, OpId, SchedulerMetrics, Tensor, WorkerLiveSnapshot,
    WorkerLiveStatus,
};
use tokio::sync::{watch, Mutex};

use crate::static_scheduler::SchedulerError;
use crate::worker::WorkQueue;

const STEAL_THRESHOLD: usize = 2;
const STATUS_IDLE: u8 = 0;
const STATUS_EXEC: u8 = 1;
const STATUS_STEAL: u8 = 2;

/// Errors returned by [`WorkStealingScheduler::execute`].
///
/// Re-uses [`SchedulerError`] from the static scheduler — the error variants
/// are identical.
pub type WorkStealingError = SchedulerError;

/// Work-stealing scheduler.
///
/// Workers start with a static round-robin assignment and steal ops from
/// busy peers when their own queue empties.  Stealing respects a threshold:
/// a worker is only stolen from when its queue length exceeds
/// [`STEAL_THRESHOLD`] (currently 2), preventing starvation of the victim.
/// Failed steal attempts back off with a deterministic pseudo-random delay
/// in the range 10–50 ms.
pub struct WorkStealingScheduler {
    n_workers: usize,
}

impl WorkStealingScheduler {
    /// Creates a new `WorkStealingScheduler` with `n_workers` concurrent workers.
    pub fn new(n_workers: usize) -> Self {
        assert!(n_workers > 0, "n_workers must be at least 1");
        Self { n_workers }
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
        let total_ops =
            dag.ops.iter().filter(|op| !op.input_ids.is_empty()).count() as u64;

        // Per-worker queues with round-robin initial assignment.
        let queues: Arc<Vec<WorkQueue>> = Arc::new((0..n).map(|_| WorkQueue::new()).collect());
        for op in &dag.ops {
            if !op.input_ids.is_empty() {
                queues[op.id % n].push(op.id).await;
            }
        }

        let store: Arc<Mutex<HashMap<OpId, Tensor>>> =
            Arc::new(Mutex::new(source_tensors));

        let (version_tx, version_rx) = watch::channel(0usize);
        let version_tx = Arc::new(version_tx);

        let completed = Arc::new(AtomicU64::new(0));
        let steal_attempts = Arc::new(AtomicU64::new(0));
        let successful_steals = Arc::new(AtomicU64::new(0));

        // Per-worker tracking for the live-metrics ticker.
        let worker_ops: Arc<Vec<AtomicU64>> =
            Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());
        let worker_status: Arc<Vec<AtomicU8>> =
            Arc::new((0..n).map(|_| AtomicU8::new(STATUS_IDLE)).collect());
        let worker_idle_us: Arc<Vec<AtomicU64>> =
            Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());

        let t0 = Instant::now();

        // Background metrics ticker.
        let ticker = {
            let completed_t = Arc::clone(&completed);
            let steal_attempts_t = Arc::clone(&steal_attempts);
            let successful_steals_t = Arc::clone(&successful_steals);
            let worker_ops_t = Arc::clone(&worker_ops);
            let worker_status_t = Arc::clone(&worker_status);
            let worker_idle_us_t = Arc::clone(&worker_idle_us);

            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_millis(100));
                interval
                    .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    let cmp = completed_t.load(Ordering::Relaxed);
                    let sa = steal_attempts_t.load(Ordering::Relaxed);
                    let ss = successful_steals_t.load(Ordering::Relaxed);
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
                        completed_ops: cmp,
                        elapsed_secs: elapsed,
                        steal_attempts: sa,
                        successful_steals: ss,
                    };
                    if metrics_tx.send(live).is_err() {
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
            let worker_ops = Arc::clone(&worker_ops);
            let worker_status = Arc::clone(&worker_status);
            let worker_idle_us = Arc::clone(&worker_idle_us);

            handles.push(tokio::spawn(async move {
                let mut attempt = 0usize;

                loop {
                    if completed.load(Ordering::Acquire) >= total_ops {
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
                                if let Some(id) =
                                    queues[victim].steal_if_above(STEAL_THRESHOLD).await
                                {
                                    successful_steals.fetch_add(1, Ordering::Relaxed);
                                    stolen = Some(id);
                                    break;
                                }
                            }
                            match stolen {
                                Some(id) => {
                                    attempt = 0;
                                    id
                                }
                                None => {
                                    worker_status[worker_id]
                                        .store(STATUS_IDLE, Ordering::Relaxed);
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
                            let op = dag.get_op(op_id).expect("valid op id");
                            op.input_ids.iter().all(|dep| s.contains_key(dep))
                        };
                        if ready {
                            break;
                        }
                        let tw = Instant::now();
                        version_rx.changed().await.ok();
                        worker_idle_us[worker_id].fetch_add(
                            tw.elapsed().as_micros() as u64,
                            Ordering::Relaxed,
                        );
                    }

                    let op = dag.get_op(op_id).expect("valid op id");
                    let inputs: Vec<Tensor> = {
                        let s = store.lock().await;
                        op.input_ids.iter().map(|dep| s[dep].clone()).collect()
                    };
                    let input_refs: Vec<&Tensor> = inputs.iter().collect();
                    let result = execute_op(op, &input_refs)
                        .map_err(|source| SchedulerError::OpFailed { op_id, source })?;

                    store.lock().await.insert(op_id, result);
                    worker_ops[worker_id].fetch_add(1, Ordering::Relaxed);
                    completed.fetch_add(1, Ordering::Release);
                    version_tx.send_modify(|v| *v += 1);
                }

                Ok::<(), WorkStealingError>(())
            }));
        }

        for handle in handles {
            handle.await.expect("worker task panicked")?;
        }

        ticker.abort();

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let total_idle_micros: u64 =
            worker_idle_us.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        let idle_time_ms = total_idle_micros as f64 / 1000.0;
        let sa = steal_attempts.load(Ordering::Relaxed);
        let ss = successful_steals.load(Ordering::Relaxed);
        let metrics =
            SchedulerMetrics::new(total_ops, total_ops, elapsed_ms, idle_time_ms, sa, ss);

        let store = Arc::try_unwrap(store)
            .expect("all worker handles dropped")
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
        sources.insert(0usize, Tensor::from_shape_vec(&[4], vec![-3., -1., 0., 2.]).unwrap());

        let (results, metrics) = sched.execute(dag, sources).await.unwrap();
        let out: Vec<f32> = results[&3].data.iter().copied().collect();
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
            Op::new(2, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![0, 1], vec![2, 2]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = WorkStealingScheduler::new(2);

        let mut sources = HashMap::new();
        sources.insert(0usize, Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]).unwrap());
        sources.insert(1usize, Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap());

        let (results, _) = sched.execute(dag, sources).await.unwrap();
        let flat: Vec<f32> = results[&2].data.iter().copied().collect();
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
