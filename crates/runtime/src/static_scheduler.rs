use std::collections::HashMap;
use std::sync::Arc;

use ferroflow_core::{ops::execute_op, Dag, DagError, OpError, OpId, Tensor};
use thiserror::Error;
use tokio::sync::{watch, Mutex};

/// Errors returned by [`StaticScheduler::execute`].
#[derive(Debug, Error)]
pub enum SchedulerError {
    /// Topological sort or DAG validation failed.
    #[error("DAG error: {0}")]
    Dag(#[from] DagError),
    /// Execution of a specific op failed.
    #[error("op {op_id} failed: {source}")]
    OpFailed {
        op_id: OpId,
        #[source]
        source: OpError,
    },
}

/// Static round-robin scheduler.
///
/// Assigns each compute op to worker `op_id % n_workers` at construction time.
/// Workers run as concurrent tokio tasks and wait on a shared version counter
/// until their input dependencies are available.
pub struct StaticScheduler {
    n_workers: usize,
    /// `assignments[w]` — op IDs assigned to worker `w`, in op-ID order.
    assignments: Vec<Vec<OpId>>,
}

impl StaticScheduler {
    /// Builds assignment tables for `dag` split across `n_workers` workers.
    ///
    /// Source ops (no `input_ids`) are excluded — they must be supplied via
    /// `source_tensors` in [`execute`](Self::execute).
    pub fn new(dag: &Dag, n_workers: usize) -> Self {
        assert!(n_workers > 0, "n_workers must be at least 1");
        let mut assignments = vec![Vec::new(); n_workers];
        for op in &dag.ops {
            if !op.input_ids.is_empty() {
                assignments[op.id % n_workers].push(op.id);
            }
        }
        Self { n_workers, assignments }
    }

    /// Executes `dag` with `n_workers` concurrent tokio tasks.
    ///
    /// Each worker processes its pre-assigned ops in op-ID order, waiting for
    /// dependency tensors to appear in the shared store before executing.
    ///
    /// Returns the complete `HashMap<OpId, Tensor>` after all ops finish.
    ///
    /// # Errors
    /// Returns [`SchedulerError`] if any op fails.
    pub async fn execute(
        &self,
        dag: Arc<Dag>,
        source_tensors: HashMap<OpId, Tensor>,
    ) -> Result<HashMap<OpId, Tensor>, SchedulerError> {
        let store: Arc<Mutex<HashMap<OpId, Tensor>>> =
            Arc::new(Mutex::new(source_tensors));

        // Shared completion counter — workers bump this when they finish an op.
        // Waiting workers subscribe to `version_rx` and re-check deps on each tick.
        let (version_tx, version_rx) = watch::channel(0usize);
        let version_tx = Arc::new(version_tx);

        let mut handles = Vec::with_capacity(self.n_workers);

        for worker_id in 0..self.n_workers {
            let assigned = self.assignments[worker_id].clone();
            let dag = Arc::clone(&dag);
            let store = Arc::clone(&store);
            let version_tx = Arc::clone(&version_tx);
            let mut version_rx = version_rx.clone();

            handles.push(tokio::spawn(async move {
                for op_id in assigned {
                    // Mark the version we have seen before the readiness check so that
                    // `changed()` below detects any completion that races with the check.
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
                    // Notify all waiting workers that a new tensor is available.
                    version_tx.send_modify(|v| *v += 1);
                }
                Ok::<(), SchedulerError>(())
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
    async fn static_chain_matches_sequential() {
        let ops = vec![
            Op::new(0, OpKind::Relu { len: 4 }, vec![], vec![4]),
            Op::new(1, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(2, OpKind::Relu { len: 4 }, vec![1], vec![4]),
            Op::new(3, OpKind::Relu { len: 4 }, vec![2], vec![4]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = StaticScheduler::new(&dag, 2);

        let mut sources = HashMap::new();
        sources.insert(0usize, Tensor::from_shape_vec(&[4], vec![-3., -1., 0., 2.]).unwrap());

        let results = sched.execute(dag, sources).await.unwrap();
        let out: Vec<f32> = results[&3].data.iter().copied().collect();
        assert!(out.iter().all(|&v| v >= 0.0));
        assert_eq!(out[3], 2.0);
    }

    #[tokio::test]
    async fn static_fan_out_all_ops_present() {
        // src_0 → [relu_1, relu_2, relu_3, relu_4]  (4 independent ops)
        let ops = vec![
            Op::new(0, OpKind::Relu { len: 4 }, vec![], vec![4]),
            Op::new(1, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(2, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(3, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(4, OpKind::Relu { len: 4 }, vec![0], vec![4]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = StaticScheduler::new(&dag, 4);

        let mut sources = HashMap::new();
        sources.insert(0usize, Tensor::full(&[4], 1.0));

        let results = sched.execute(dag, sources).await.unwrap();
        assert_eq!(results.len(), 5);
        for id in 1..=4 {
            let out: Vec<f32> = results[&id].data.iter().copied().collect();
            assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0]);
        }
    }

    #[tokio::test]
    async fn static_diamond() {
        // A ──┐
        //     ├─▶ matmul  (A·B = B when A = I)
        // B ──┘
        let ops = vec![
            Op::new(0, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),
            Op::new(1, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),
            Op::new(2, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![0, 1], vec![2, 2]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());
        let sched = StaticScheduler::new(&dag, 2);

        let mut sources = HashMap::new();
        sources.insert(0usize, Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]).unwrap());
        sources.insert(1usize, Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap());

        let results = sched.execute(dag, sources).await.unwrap();
        let flat: Vec<f32> = results[&2].data.iter().copied().collect();
        assert_eq!(flat, vec![1., 2., 3., 4.]);
    }
}
