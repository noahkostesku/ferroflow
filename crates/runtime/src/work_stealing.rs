use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ferroflow_core::{ops::execute_op, Dag, OpId, Tensor};
use tokio::sync::{watch, Mutex};

use crate::static_scheduler::SchedulerError;
use crate::worker::WorkQueue;

const STEAL_THRESHOLD: usize = 2;

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

    /// Executes `dag` with work-stealing across `n_workers` tokio tasks.
    ///
    /// Initial assignment is identical to [`StaticScheduler`](crate::StaticScheduler)
    /// (round-robin by `op_id % n_workers`).  Workers steal from victims when
    /// idle, choosing the first victim whose queue exceeds the steal threshold.
    ///
    /// Returns the complete `HashMap<OpId, Tensor>` after all ops finish.
    ///
    /// # Errors
    /// Returns [`WorkStealingError`] if any op fails.
    pub async fn execute(
        &self,
        dag: Arc<Dag>,
        source_tensors: HashMap<OpId, Tensor>,
    ) -> Result<HashMap<OpId, Tensor>, WorkStealingError> {
        let n = self.n_workers;

        // Build per-worker queues with round-robin initial assignment.
        let queues: Arc<Vec<WorkQueue>> = Arc::new((0..n).map(|_| WorkQueue::new()).collect());
        let total_compute = {
            let mut count = 0usize;
            for op in &dag.ops {
                if !op.input_ids.is_empty() {
                    queues[op.id % n].push(op.id).await;
                    count += 1;
                }
            }
            count
        };

        let store: Arc<Mutex<HashMap<OpId, Tensor>>> =
            Arc::new(Mutex::new(source_tensors));

        let (version_tx, version_rx) = watch::channel(0usize);
        let version_tx = Arc::new(version_tx);

        // Shared completion counter to detect when all ops are done.
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(n);

        for worker_id in 0..n {
            let dag = Arc::clone(&dag);
            let store = Arc::clone(&store);
            let queues = Arc::clone(&queues);
            let version_tx = Arc::clone(&version_tx);
            let mut version_rx = version_rx.clone();
            let completed = Arc::clone(&completed);

            handles.push(tokio::spawn(async move {
                let mut attempt = 0usize;

                loop {
                    if completed.load(std::sync::atomic::Ordering::Acquire) >= total_compute {
                        break;
                    }

                    // Try own queue first.
                    let op_id = match queues[worker_id].pop().await {
                        Some(id) => id,
                        None => {
                            // Attempt to steal from another worker.
                            let mut stolen = None;
                            for offset in 1..n {
                                let victim = (worker_id + offset) % n;
                                if let Some(id) =
                                    queues[victim].steal_if_above(STEAL_THRESHOLD).await
                                {
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
                                    // Back off: wait for any op to complete (wakes up
                                    // reactively via version channel) or for the maximum
                                    // backoff window to expire (10–50 ms).  The select
                                    // ensures workers are responsive when work becomes
                                    // available while still capping retry frequency.
                                    let backoff_ms =
                                        10 + ((worker_id * 13 + attempt * 7) % 41) as u64;
                                    attempt += 1;
                                    tokio::select! {
                                        biased;
                                        _ = version_rx.changed() => {}
                                        _ = tokio::time::sleep(
                                            Duration::from_millis(backoff_ms)
                                        ) => {}
                                    }
                                    continue;
                                }
                            }
                        }
                    };

                    // Wait until all input dependencies are in the store.
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
                        version_rx.changed().await.ok();
                    }

                    let op = dag.get_op(op_id).expect("valid op id");
                    let inputs: Vec<Tensor> = {
                        let s = store.lock().await;
                        op.input_ids.iter().map(|dep| s[dep].clone()).collect()
                    };
                    let input_refs: Vec<&Tensor> = inputs.iter().collect();
                    let result = execute_op(op, &input_refs)
                        .map_err(|source| SchedulerError::OpFailed { op_id, source })?;

                    {
                        store.lock().await.insert(op_id, result);
                    }
                    completed.fetch_add(1, std::sync::atomic::Ordering::Release);
                    version_tx.send_modify(|v| *v += 1);
                }

                Ok::<(), WorkStealingError>(())
            }));
        }

        for handle in handles {
            handle.await.expect("worker task panicked")?;
        }

        let store = Arc::try_unwrap(store)
            .expect("all worker handles dropped")
            .into_inner();
        Ok(store)
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

        let results = sched.execute(dag, sources).await.unwrap();
        let out: Vec<f32> = results[&3].data.iter().copied().collect();
        assert!(out.iter().all(|&v| v >= 0.0));
        assert_eq!(out[3], 2.0);
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

        let results = sched.execute(dag, sources).await.unwrap();
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

        let results = sched.execute(dag, sources).await.unwrap();
        let flat: Vec<f32> = results[&2].data.iter().copied().collect();
        assert_eq!(flat, vec![1., 2., 3., 4.]);
    }
}
