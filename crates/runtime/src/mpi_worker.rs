use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use ferroflow_core::{ops::execute_op, Dag, Op, OpId, SchedulerMetrics, Tensor};
use mpi::traits::*;
use serde::{Deserialize, Serialize};

// ── scheduler mode ────────────────────────────────────────────────────────────

/// Controls how the coordinator dispatches ready ops to worker ranks.
///
/// - `Static`: ops are pre-assigned to workers by `(op_id − 1) % n_workers`.
///   A worker only receives ops from its own slice; starvation is visible when
///   one slice is systematically heavier than others.
/// - `WorkStealing`: any idle worker receives the next available op from the
///   global ready queue — demand-driven, no pre-assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpiSchedulerMode {
    Static,
    WorkStealing,
}

// ── message protocol ──────────────────────────────────────────────────────────

/// Messages exchanged over MPI point-to-point between coordinator and workers.
#[derive(Debug, Serialize, Deserialize)]
enum MpiMsg {
    /// Worker requests the next available op.
    StealRequest { rank: i32 },
    /// Coordinator grants an op along with its pre-fetched input tensors.
    StealGrant {
        op_id: OpId,
        op: Op,
        inputs: Vec<Tensor>,
    },
    /// No ready ops are available; worker should back off and retry.
    StealNone,
    /// Worker reports a completed op and its output tensor.
    OpResult { op_id: OpId, tensor: Tensor },
    /// Coordinator instructs the worker to exit cleanly.
    Shutdown,
    /// Worker sends its local counters after receiving Shutdown.
    Metrics {
        idle_time_ms: f64,
        steal_attempts: u64,
        successful_steals: u64,
    },
}

// ── public API ─────────────────────────────────────────────────────────────────

/// MPI-backed distributed work-stealing (or static) scheduler.
///
/// Rank 0 is the **coordinator**: it owns the [`Dag`], tracks completions,
/// and dispatches ready ops on demand.  Ranks 1..N are **workers**: they
/// steal ops via point-to-point MPI, execute them, and report results back.
///
/// All MPI ranks must construct an `MpiWorker` and call [`run`](Self::run)
/// together — this is the standard SPMD entry point.
pub struct MpiWorker {
    dag: Arc<Dag>,
    source_tensors: HashMap<OpId, Tensor>,
    mode: MpiSchedulerMode,
}

impl MpiWorker {
    /// Creates a new `MpiWorker` with work-stealing dispatch (default).
    pub fn new(dag: Arc<Dag>, source_tensors: HashMap<OpId, Tensor>) -> Self {
        Self {
            dag,
            source_tensors,
            mode: MpiSchedulerMode::WorkStealing,
        }
    }

    /// Overrides the dispatch strategy.
    pub fn with_mode(mut self, mode: MpiSchedulerMode) -> Self {
        self.mode = mode;
        self
    }

    /// Runs the distributed DAG execution.
    ///
    /// Returns:
    /// - `Ok(Some((store, metrics)))` on rank 0 after the full DAG completes.
    /// - `Ok(None)` on all worker ranks once they receive `Shutdown`.
    ///
    /// # Errors
    /// Returns an error if MPI initialisation fails, there are fewer than 2
    /// ranks, or any op execution fails on a worker.
    pub fn run(self) -> anyhow::Result<Option<(HashMap<OpId, Tensor>, SchedulerMetrics)>> {
        let universe = mpi::initialize().context("MPI initialisation failed")?;
        let world = universe.world();
        let rank = world.rank();
        let n_ranks = world.size() as usize;

        anyhow::ensure!(
            n_ranks >= 2,
            "MpiWorker requires ≥2 MPI ranks (1 coordinator + ≥1 worker); got {n_ranks}"
        );

        if rank == 0 {
            let (results, metrics) =
                coordinator_loop(&world, self.dag, self.source_tensors, n_ranks, self.mode)
                    .context("coordinator loop failed")?;
            Ok(Some((results, metrics)))
        } else {
            worker_loop(&world, rank, self.mode).context("worker loop failed")?;
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

fn worker_for_op(op_id: OpId, n_workers: usize) -> usize {
    op_id.saturating_sub(1) % n_workers
}

fn coordinator_loop<C: Communicator>(
    world: &C,
    dag: Arc<Dag>,
    source_tensors: HashMap<OpId, Tensor>,
    n_ranks: usize,
    mode: MpiSchedulerMode,
) -> anyhow::Result<(HashMap<OpId, Tensor>, SchedulerMetrics)> {
    let n_workers = n_ranks - 1;
    let mut tensor_store = source_tensors;
    let mut completed: HashSet<OpId> = HashSet::new();
    let mut dispatched: HashSet<OpId> = HashSet::new();
    let mut ready_queue: VecDeque<OpId> = VecDeque::new();

    for op in &dag.ops {
        if op.input_ids.is_empty() {
            completed.insert(op.id);
        }
    }

    let total_compute = dag.ops.iter().filter(|op| !op.input_ids.is_empty()).count();
    let mut completed_compute = 0usize;
    let mut in_flight = 0usize;

    enqueue_ready(&dag, &completed, &dispatched, &mut ready_queue);

    let t0 = Instant::now();

    loop {
        let (msg, source_rank) = mpi_recv_any(world)?;

        match msg {
            MpiMsg::StealRequest { .. } => {
                enqueue_ready(&dag, &completed, &dispatched, &mut ready_queue);

                let worker_idx = (source_rank - 1) as usize;

                let maybe_op_id = match mode {
                    MpiSchedulerMode::WorkStealing => ready_queue.pop_front(),
                    MpiSchedulerMode::Static => {
                        // Find the first op in the ready queue assigned to this worker.
                        let pos = ready_queue
                            .iter()
                            .position(|&id| worker_for_op(id, n_workers) == worker_idx);
                        pos.and_then(|i| ready_queue.remove(i))
                    }
                };

                let response = match maybe_op_id {
                    Some(op_id) => {
                        dispatched.insert(op_id);
                        in_flight += 1;
                        let op = dag
                            .get_op(op_id)
                            .ok_or_else(|| {
                                anyhow::anyhow!("op_id {} from ready_ops is not in DAG", op_id)
                            })?
                            .clone();
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

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Drain remaining StealRequests, send Shutdown, collect per-worker Metrics.
    let mut agg_idle_ms = 0.0f64;
    let mut agg_steal_attempts = 0u64;
    let mut agg_successful_steals = 0u64;
    let mut shut = 0;

    while shut < n_workers {
        let (msg, source_rank) = mpi_recv_any(world)?;
        if matches!(msg, MpiMsg::StealRequest { .. }) {
            mpi_send(world, source_rank, &MpiMsg::Shutdown)?;
            match mpi_recv_from(world, source_rank)? {
                MpiMsg::Metrics {
                    idle_time_ms,
                    steal_attempts,
                    successful_steals,
                } => {
                    agg_idle_ms += idle_time_ms;
                    agg_steal_attempts += steal_attempts;
                    agg_successful_steals += successful_steals;
                }
                _ => {}
            }
            shut += 1;
        }
    }

    let metrics = SchedulerMetrics::new(
        total_compute as u64,
        total_compute as u64,
        elapsed_ms,
        agg_idle_ms,
        agg_steal_attempts,
        agg_successful_steals,
        0,
        0,
    );

    Ok((tensor_store, metrics))
}

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

fn worker_loop<C: Communicator>(
    world: &C,
    rank: i32,
    mode: MpiSchedulerMode,
) -> anyhow::Result<()> {
    let mut backoff = Duration::from_millis(10);
    let mut idle_time_ms = 0.0f64;
    let mut steal_attempts = 0u64;
    let mut successful_steals = 0u64;
    let is_ws = mode == MpiSchedulerMode::WorkStealing;

    loop {
        if is_ws {
            steal_attempts += 1;
        }
        mpi_send(world, 0, &MpiMsg::StealRequest { rank })?;

        match mpi_recv_from(world, 0)? {
            MpiMsg::StealGrant { op_id, op, inputs } => {
                backoff = Duration::from_millis(10);
                if is_ws {
                    successful_steals += 1;
                }
                let input_refs: Vec<&Tensor> = inputs.iter().collect();
                let tensor = execute_op(&op, &input_refs)
                    .map_err(|e| anyhow::anyhow!("rank {rank}: op {op_id} failed: {e}"))?;
                mpi_send(world, 0, &MpiMsg::OpResult { op_id, tensor })?;
            }

            MpiMsg::StealNone => {
                let tw = Instant::now();
                std::thread::sleep(backoff);
                idle_time_ms += tw.elapsed().as_secs_f64() * 1000.0;
                backoff = (backoff * 2).min(Duration::from_millis(100));
            }

            MpiMsg::Shutdown => {
                mpi_send(
                    world,
                    0,
                    &MpiMsg::Metrics {
                        idle_time_ms,
                        steal_attempts,
                        successful_steals,
                    },
                )?;
                break;
            }

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
            Op::new(
                2,
                OpKind::Matmul { m: 2, n: 2, k: 2 },
                vec![0, 1],
                vec![2, 2],
            ),
            Op::new(
                3,
                OpKind::Matmul { m: 2, n: 2, k: 2 },
                vec![0, 2],
                vec![2, 2],
            ),
        ];
        let dag = Arc::new(Dag::new(ops).unwrap());

        let mut sources = HashMap::new();
        sources.insert(
            0usize,
            Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]).unwrap(),
        );
        sources.insert(
            1usize,
            Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap(),
        );

        match MpiWorker::new(dag, sources).run() {
            // Single-rank run (plain `cargo test`) — skip gracefully.
            Err(e) if e.to_string().contains("≥2 MPI ranks") => {}

            // Rank 0 with 2+ ranks: verify correctness.
            Ok(Some((store, _metrics))) => {
                let op2: Vec<f32> = store[&2].data.iter().copied().collect();
                assert_eq!(op2, vec![1., 2., 3., 4.], "op2 should be A");
                let op3: Vec<f32> = store[&3].data.iter().copied().collect();
                assert_eq!(op3, vec![1., 2., 3., 4.], "op3 should be I·A = A");
            }

            Ok(None) => {}

            Err(e) => panic!("unexpected MpiWorker error: {e}"),
        }
    }
}
