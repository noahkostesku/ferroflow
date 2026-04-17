use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use ferroflow_core::{ops::execute_op, Dag, Op, OpId, Tensor};
use mpi::traits::*;
use serde::{Deserialize, Serialize};

/// Messages exchanged over MPI point-to-point between coordinator and workers.
#[derive(Debug, Serialize, Deserialize)]
enum MpiMsg {
    /// Worker requests the next available op.
    StealRequest { rank: i32 },
    /// Coordinator grants an op along with its pre-fetched input tensors.
    StealGrant { op_id: OpId, op: Op, inputs: Vec<Tensor> },
    /// No ready ops are available; worker should back off and retry.
    StealNone,
    /// Worker reports a completed op and its output tensor.
    OpResult { op_id: OpId, tensor: Tensor },
    /// Coordinator instructs the worker to exit cleanly.
    Shutdown,
}

/// MPI-backed distributed work-stealing scheduler.
///
/// Rank 0 is the **coordinator**: it owns the [`Dag`], tracks completions,
/// and dispatches ready ops on demand.  Ranks 1..N are **workers**: they
/// steal ops via point-to-point MPI, execute them using the local rayon-backed
/// tensor ops, and report results back to the coordinator.
///
/// All MPI ranks must construct an `MpiWorker` and call [`run`](Self::run)
/// together — this is the standard SPMD entry point.
pub struct MpiWorker {
    dag: Arc<Dag>,
    source_tensors: HashMap<OpId, Tensor>,
}

impl MpiWorker {
    /// Creates a new `MpiWorker` with the given DAG and pre-supplied source tensors.
    ///
    /// `source_tensors` must contain an output [`Tensor`] for every source op
    /// (an op whose `input_ids` is empty).
    pub fn new(dag: Arc<Dag>, source_tensors: HashMap<OpId, Tensor>) -> Self {
        Self { dag, source_tensors }
    }

    /// Runs the distributed DAG execution.
    ///
    /// Initialises MPI, branches on rank, and returns:
    /// - `Ok(Some(store))` on rank 0 after the full DAG has completed.
    /// - `Ok(None)` on all worker ranks once they receive `Shutdown`.
    ///
    /// # Errors
    /// Returns an error if MPI initialisation fails, there are fewer than 2
    /// ranks, or any op execution fails on a worker.
    pub fn run(self) -> anyhow::Result<Option<HashMap<OpId, Tensor>>> {
        let universe = mpi::initialize().context("MPI initialisation failed")?;
        let world = universe.world();
        let rank = world.rank();
        let n_ranks = world.size() as usize;

        anyhow::ensure!(
            n_ranks >= 2,
            "MpiWorker requires ≥2 MPI ranks (1 coordinator + ≥1 worker); got {n_ranks}"
        );

        if rank == 0 {
            let results = coordinator_loop(&world, self.dag, self.source_tensors, n_ranks)
                .context("coordinator loop failed")?;
            Ok(Some(results))
        } else {
            worker_loop(&world, rank).context("worker loop failed")?;
            Ok(None)
        }
    }
}

// ── point-to-point helpers ────────────────────────────────────────────────────

fn mpi_send<C: Communicator>(world: &C, dest: i32, msg: &MpiMsg) -> anyhow::Result<()> {
    let bytes = bincode::serialize(msg).context("MpiMsg serialize")?;
    world.process_at_rank(dest).send(&bytes[..]);
    Ok(())
}

/// Blocking receive from any rank; returns the message and its source rank.
fn mpi_recv_any<C: Communicator>(world: &C) -> anyhow::Result<(MpiMsg, i32)> {
    let (bytes, status) = world.any_process().receive_vec::<u8>();
    let msg = bincode::deserialize(&bytes).context("MpiMsg deserialize")?;
    Ok((msg, status.source_rank()))
}

/// Blocking receive from a specific rank.
fn mpi_recv_from<C: Communicator>(world: &C, source: i32) -> anyhow::Result<MpiMsg> {
    let (bytes, _) = world.process_at_rank(source).receive_vec::<u8>();
    bincode::deserialize(&bytes).context("MpiMsg deserialize")
}

// ── coordinator (rank 0) ──────────────────────────────────────────────────────

fn coordinator_loop<C: Communicator>(
    world: &C,
    dag: Arc<Dag>,
    source_tensors: HashMap<OpId, Tensor>,
    n_ranks: usize,
) -> anyhow::Result<HashMap<OpId, Tensor>> {
    let n_workers = n_ranks - 1;
    let mut tensor_store = source_tensors;
    let mut completed: HashSet<OpId> = HashSet::new();
    let mut dispatched: HashSet<OpId> = HashSet::new();
    let mut ready_queue: VecDeque<OpId> = VecDeque::new();

    // Source ops (no input_ids) are considered pre-completed.
    for op in &dag.ops {
        if op.input_ids.is_empty() {
            completed.insert(op.id);
        }
    }

    let total_compute = dag.ops.iter().filter(|op| !op.input_ids.is_empty()).count();
    let mut completed_compute = 0usize;
    // Ops whose StealGrant has been sent but whose OpResult has not arrived.
    let mut in_flight = 0usize;

    enqueue_ready(&dag, &completed, &dispatched, &mut ready_queue);

    // Main event loop — runs until every compute op has reported a result.
    loop {
        let (msg, source_rank) = mpi_recv_any(world)?;

        match msg {
            MpiMsg::StealRequest { .. } => {
                // Refresh the ready queue with any ops newly unblocked by recent completions.
                enqueue_ready(&dag, &completed, &dispatched, &mut ready_queue);

                let response = match ready_queue.pop_front() {
                    Some(op_id) => {
                        dispatched.insert(op_id);
                        in_flight += 1;
                        let op = dag.get_op(op_id).expect("op_id from ready_ops is valid").clone();
                        let inputs = op
                            .input_ids
                            .iter()
                            .map(|&dep| tensor_store[&dep].clone())
                            .collect();
                        MpiMsg::StealGrant { op_id, op, inputs }
                    }
                    None => MpiMsg::StealNone,
                };

                mpi_send(world, source_rank, &response)?;
            }

            MpiMsg::OpResult { op_id, tensor } => {
                tensor_store.insert(op_id, tensor);
                completed.insert(op_id);
                in_flight -= 1;
                completed_compute += 1;

                if completed_compute >= total_compute && in_flight == 0 {
                    break;
                }
            }

            _ => {}
        }
    }

    // All compute ops are done and no ops are in-flight.  Each worker is either
    // in backoff or has just sent a StealRequest.  Drain them one by one.
    let mut shut = 0;
    while shut < n_workers {
        let (msg, source_rank) = mpi_recv_any(world)?;
        if matches!(msg, MpiMsg::StealRequest { .. }) {
            mpi_send(world, source_rank, &MpiMsg::Shutdown)?;
            shut += 1;
        }
    }

    Ok(tensor_store)
}

/// Appends newly ready, un-dispatched ops to `queue`.
fn enqueue_ready(
    dag: &Dag,
    completed: &HashSet<OpId>,
    dispatched: &HashSet<OpId>,
    queue: &mut VecDeque<OpId>,
) {
    for op_id in dag.ready_ops(completed) {
        if !dispatched.contains(&op_id) && !queue.contains(&op_id) {
            queue.push_back(op_id);
        }
    }
}

// ── worker (ranks 1..N) ───────────────────────────────────────────────────────

fn worker_loop<C: Communicator>(world: &C, rank: i32) -> anyhow::Result<()> {
    let mut backoff = Duration::from_millis(10);

    loop {
        mpi_send(world, 0, &MpiMsg::StealRequest { rank })?;

        match mpi_recv_from(world, 0)? {
            MpiMsg::StealGrant { op_id, op, inputs } => {
                backoff = Duration::from_millis(10);
                let input_refs: Vec<&Tensor> = inputs.iter().collect();
                // execute_op may invoke rayon internally for CPU-bound work.
                let tensor = execute_op(&op, &input_refs)
                    .map_err(|e| anyhow::anyhow!("rank {rank}: op {op_id} failed: {e}"))?;
                mpi_send(world, 0, &MpiMsg::OpResult { op_id, tensor })?;
            }

            MpiMsg::StealNone => {
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_millis(100));
            }

            MpiMsg::Shutdown => break,

            _ => {}
        }
    }

    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ferroflow_core::{Op, OpKind};

    /// 4-op diamond DAG executed across 2 MPI ranks.
    ///
    /// DAG shape:
    ///   op0 (I₂, src) ──┬──▶ op2 = op0·op1 ──▶ op3 = op0·op2
    ///   op1 (A,  src) ──┘
    ///
    /// Expected: op2 = I₂·A = A, op3 = I₂·A = A.
    ///
    /// **Run with:** `mpirun -n 2 cargo test -p ferroflow-runtime \
    ///     --features distributed -- --test-threads=1`
    ///
    /// When invoked without MPI (plain `cargo test`), the `ensure!` in
    /// `MpiWorker::run()` returns an error because only 1 rank is present;
    /// the test treats this as a skip and passes.
    #[test]
    fn diamond_dag_two_ranks() {
        let ops = vec![
            Op::new(0, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),
            Op::new(1, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),
            Op::new(2, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![0, 1], vec![2, 2]),
            Op::new(3, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![0, 2], vec![2, 2]),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());

        let mut sources = HashMap::new();
        // op0 = I₂, op1 = A = [[1,2],[3,4]]
        sources.insert(0usize, Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]).unwrap());
        sources.insert(1usize, Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap());

        match MpiWorker::new(dag, sources).run() {
            // Single-rank run (plain `cargo test`) — skip gracefully.
            Err(e) if e.to_string().contains("≥2 MPI ranks") => {}

            // Rank 0 with 2+ ranks: verify correctness.
            Ok(Some(store)) => {
                let op2: Vec<f32> = store[&2].data.iter().copied().collect();
                assert_eq!(op2, vec![1., 2., 3., 4.], "op2 should be A");
                let op3: Vec<f32> = store[&3].data.iter().copied().collect();
                assert_eq!(op3, vec![1., 2., 3., 4.], "op3 should be I·A = A");
            }

            // Worker rank — no return value expected.
            Ok(None) => {}

            Err(e) => panic!("unexpected MpiWorker error: {e}"),
        }
    }
}
