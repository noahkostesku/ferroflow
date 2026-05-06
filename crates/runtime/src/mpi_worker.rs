use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use ferroflow_core::{ops::execute_op, Dag, Device, Op, OpId, SchedulerMetrics, Tensor};
use mpi::traits::*;
use serde::{Deserialize, Serialize};

// ── constants ─────────────────────────────────────────────────────────────────

const DEFAULT_STEAL_THRESHOLD: usize = 2;
const RECENT_VICTIM_WINDOW: usize = 3;
const P2P_MAX_BACKOFF_MS: u64 = 50;
const P2P_INITIAL_BACKOFF_MS: u64 = 1;
/// Maximum time to wait for a steal response before abandoning the attempt.
const STEAL_RESPONSE_TIMEOUT_MS: u64 = 100;
/// Maximum ops dispatched per worker before coordinator waits for results.
///
/// Prevents the coordinator from flooding all initially-ready ops to workers in
/// one synchronous burst (which serialises on MPI rendezvous for large tensors).
const MAX_WORKER_INFLIGHT: usize = 4;

// ── scheduler mode ────────────────────────────────────────────────────────────

/// Controls how ops are dispatched to and between worker ranks.
///
/// - `Static`: ops pre-assigned by `(op_id − 1) % n_workers`.
/// - `WorkStealing`: demand-driven; idle workers pull from coordinator's global ready queue.
/// - `P2PWorkStealing`: push-assign from coordinator (capped window); idle workers steal
///   directly from peers via two-way MPI handshake, bypassing the coordinator hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpiSchedulerMode {
    Static,
    WorkStealing,
    P2PWorkStealing,
}

// ── message protocol ──────────────────────────────────────────────────────────

/// All messages exchanged over MPI point-to-point.
///
/// Coordinator ↔ worker messages are used by the `Static` and `WorkStealing`
/// modes.  The `P2P*` variants are used exclusively by `P2PWorkStealing`.
#[derive(Debug, Serialize, Deserialize)]
enum MpiMsg {
    // ── coordinator-mediated (Static / WorkStealing) ──────────────────────────
    /// Worker requests the next available op from the coordinator.
    StealRequest { rank: i32 },
    /// Coordinator grants an op with pre-fetched input tensors.
    StealGrant { op_id: OpId, op: Op, inputs: Vec<Tensor> },
    /// No ready ops are currently available; worker should back off and retry.
    StealNone,
    /// Worker reports a completed op and its output tensor.
    OpResult { op_id: OpId, tensor: Tensor },
    /// Coordinator instructs the worker to exit cleanly.
    Shutdown,
    /// Worker sends local counters after receiving `Shutdown`.
    Metrics { idle_time_ms: f64, steal_attempts: u64, successful_steals: u64 },

    // ── P2P work-stealing ─────────────────────────────────────────────────────
    /// Coordinator pushes a newly-ready op directly to a worker (P2P mode only).
    P2POpAssignment { op_id: OpId, op: Op, inputs: Vec<Tensor> },
    /// Worker has exhausted P2P stealing for longer than the idle threshold and
    /// is now blocking on rank 0, waiting for a direct op assignment or Shutdown.
    P2PIdleNotification { rank: i32 },
    /// Step 1 of the two-way handshake: thief asks victim for an op.
    P2PStealRequest { from_rank: i32, request_id: u64 },
    /// Step 2: victim replies with an op (or `None` if its queue is too short).
    ///
    /// Two-way protocol: victim pops the op before sending the response so no
    /// Ack round-trip is required.  The op belongs to the thief on receipt.
    P2PStealResponse {
        request_id: u64,
        op_id: Option<OpId>,
        op: Option<Op>,
        inputs: Option<Vec<Tensor>>,
    },
    /// Worker sends P2P-specific counters to coordinator at shutdown.
    P2PMetrics {
        idle_time_ms: f64,
        p2p_steals_attempted: u64,
        p2p_steals_successful: u64,
        p2p_steal_latency_ms: f64,
    },
}

// ── public API ─────────────────────────────────────────────────────────────────

/// MPI-backed distributed scheduler supporting three dispatch strategies.
///
/// Rank 0 is the **coordinator**; ranks 1..N are **workers**.  All ranks must
/// construct an `MpiWorker` and call [`run`](Self::run) simultaneously (SPMD
/// entry point).
pub struct MpiWorker {
    dag: Arc<Dag>,
    source_tensors: HashMap<OpId, Tensor>,
    mode: MpiSchedulerMode,
    p2p_debug: bool,
    /// Idle duration after which a P2P worker falls back to coordinator dispatch.
    p2p_idle_threshold_ms: u64,
}

impl MpiWorker {
    /// Creates a new `MpiWorker` defaulting to coordinator-mediated work-stealing.
    pub fn new(dag: Arc<Dag>, source_tensors: HashMap<OpId, Tensor>) -> Self {
        Self {
            dag,
            source_tensors,
            mode: MpiSchedulerMode::WorkStealing,
            p2p_debug: false,
            p2p_idle_threshold_ms: 1000,
        }
    }

    /// Sets how long a P2P worker stays idle before falling back to coordinator dispatch.
    ///
    /// When a worker's queue is empty and all P2P steal attempts have failed for
    /// longer than `ms` milliseconds, it sends a `P2PIdleNotification` to rank 0
    /// and blocks until the coordinator pushes a ready op directly.  This prevents
    /// livelock on serial DAGs where no peer ever holds a stealable op.
    ///
    /// Default: 1000 ms.  Corresponds to `--p2p-idle-threshold` on the CLI.
    pub fn with_p2p_idle_threshold(mut self, ms: u64) -> Self {
        self.p2p_idle_threshold_ms = ms;
        self
    }

    /// Overrides the dispatch strategy.
    pub fn with_mode(mut self, mode: MpiSchedulerMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enables per-steal timing output on stderr (P2P mode only).
    ///
    /// Each successful or failed steal prints one line:
    /// `[p2p-debug] rank=N stole from=M latency=X.XXms`
    pub fn with_p2p_debug(mut self, debug: bool) -> Self {
        self.p2p_debug = debug;
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
            let (results, metrics) = match self.mode {
                MpiSchedulerMode::Static | MpiSchedulerMode::WorkStealing => {
                    coordinator_loop(&world, self.dag, self.source_tensors, n_ranks, self.mode)
                        .context("coordinator loop failed")?
                }
                MpiSchedulerMode::P2PWorkStealing => {
                    p2p_coordinator_loop(
                        &world,
                        self.dag,
                        self.source_tensors,
                        n_ranks,
                        self.p2p_debug,
                    )
                    .context("P2P coordinator loop failed")?
                }
            };
            Ok(Some((results, metrics)))
        } else {
            match self.mode {
                MpiSchedulerMode::Static | MpiSchedulerMode::WorkStealing => {
                    worker_loop(&world, rank, self.mode).context("worker loop failed")?;
                }
                MpiSchedulerMode::P2PWorkStealing => {
                    p2p_worker_loop(
                        &world,
                        rank,
                        n_ranks,
                        self.p2p_debug,
                        Duration::from_millis(self.p2p_idle_threshold_ms),
                    )
                    .context("P2P worker loop failed")?;
                }
            }
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

// ── peer directory (P2P) ──────────────────────────────────────────────────────

/// Per-worker directory of peers for P2P victim selection.
///
/// Maintains a weighted score per peer based on historical steal success rate,
/// and avoids recently-stolen-from peers for `RECENT_VICTIM_WINDOW` steals.
struct PeerDirectory {
    ranks: Vec<i32>,
    recent_victims: VecDeque<i32>,
    steal_attempts: HashMap<i32, u64>,
    steal_successes: HashMap<i32, u64>,
}

impl PeerDirectory {
    /// Builds a peer list containing all worker ranks except `self_rank`.
    fn new(self_rank: i32, n_ranks: usize) -> Self {
        let ranks: Vec<i32> = (1..n_ranks as i32).filter(|&r| r != self_rank).collect();
        Self {
            ranks,
            recent_victims: VecDeque::with_capacity(RECENT_VICTIM_WINDOW),
            steal_attempts: HashMap::new(),
            steal_successes: HashMap::new(),
        }
    }

    /// Chooses a victim rank using weighted scoring.
    ///
    /// Scoring rules (higher = preferred):
    /// - Optimistic prior: 0.8 success rate until any history exists.
    /// - Peer with highest `successes / attempts` ratio wins.
    /// - 0.3 penalty for peers seen in the last `RECENT_VICTIM_WINDOW` steals.
    fn choose_victim(&self) -> i32 {
        let mut best_rank = self.ranks[0];
        let mut best_score = f64::NEG_INFINITY;

        for &r in &self.ranks {
            let attempts = self.steal_attempts.get(&r).copied().unwrap_or(0);
            let successes = self.steal_successes.get(&r).copied().unwrap_or(0);
            let rate = if attempts == 0 { 0.8 } else { successes as f64 / attempts as f64 };
            let penalty = if self.recent_victims.contains(&r) { 0.3 } else { 0.0 };
            let score = rate - penalty;
            if score > best_score {
                best_score = score;
                best_rank = r;
            }
        }
        best_rank
    }

    /// Records the outcome of one steal attempt against `victim`.
    fn record_outcome(&mut self, victim: i32, success: bool) {
        *self.steal_attempts.entry(victim).or_default() += 1;
        if success {
            *self.steal_successes.entry(victim).or_default() += 1;
        }
        if self.recent_victims.len() >= RECENT_VICTIM_WINDOW {
            self.recent_victims.pop_front();
        }
        self.recent_victims.push_back(victim);
    }
}

// ── P2P coordinator (rank 0) ──────────────────────────────────────────────────

/// Push-based coordinator for P2P work-stealing.
///
/// Dispatches ops to workers in round-robin order with a per-worker-inflight
/// window of `MAX_WORKER_INFLIGHT`.  The cap prevents flooding all initially-
/// ready ops at once, which would serialise on MPI rendezvous for large tensors.
///
/// Also handles `P2PIdleNotification`: when a worker has failed to steal for
/// longer than the idle threshold it parks here and the coordinator wakes it
/// directly once a dependency unlocks a new op.  This prevents the livelock that
/// occurs on serial DAGs when all workers exhaust P2P stealing simultaneously.
fn p2p_coordinator_loop<C: Communicator>(
    world: &C,
    dag: Arc<Dag>,
    source_tensors: HashMap<OpId, Tensor>,
    n_ranks: usize,
    p2p_debug: bool,
) -> anyhow::Result<(HashMap<OpId, Tensor>, SchedulerMetrics)> {
    let n_workers = n_ranks - 1;
    let dispatch_cap = n_workers * MAX_WORKER_INFLIGHT;
    let mut tensor_store = source_tensors;
    let mut completed: HashSet<OpId> = HashSet::new();
    let mut dispatched: HashSet<OpId> = HashSet::new();
    let mut ready_queue: VecDeque<OpId> = VecDeque::new();
    let mut coordinator_messages = 0u64;
    let mut worker_cursor = 0usize;
    let mut total_inflight = 0usize;

    for op in &dag.ops {
        if op.input_ids.is_empty() {
            completed.insert(op.id);
        }
    }

    let total_compute = dag.ops.iter().filter(|op| !op.input_ids.is_empty()).count();
    let mut completed_compute = 0usize;

    enqueue_ready(&dag, &completed, &dispatched, &mut ready_queue);

    // Ranks of workers that sent P2PIdleNotification and are blocking on us.
    let mut idle_workers: VecDeque<i32> = VecDeque::new();

    let t0 = Instant::now();

    // Initial dispatch: fill workers up to `dispatch_cap` total in-flight.
    while total_inflight < dispatch_cap {
        if let Some(op_id) = ready_queue.pop_front() {
            let target_rank = (worker_cursor % n_workers) as i32 + 1;
            worker_cursor += 1;
            let op = dag
                .get_op(op_id)
                .ok_or_else(|| anyhow::anyhow!("op {op_id} missing from DAG"))?
                .clone();
            let inputs = op.input_ids.iter().map(|&dep| tensor_store[&dep].clone()).collect();
            mpi_send(world, target_rank, &MpiMsg::P2POpAssignment { op_id, op, inputs })?;
            dispatched.insert(op_id);
            total_inflight += 1;
            coordinator_messages += 1;
        } else {
            break;
        }
    }

    loop {
        let (msg, _src) = mpi_recv_any(world)?;
        coordinator_messages += 1;

        match msg {
            MpiMsg::OpResult { op_id, tensor } => {
                tensor_store.insert(op_id, tensor);
                completed.insert(op_id);
                if total_inflight > 0 {
                    total_inflight -= 1;
                }
                completed_compute += 1;

                enqueue_ready(&dag, &completed, &dispatched, &mut ready_queue);

                // Fill inflight cap — wake parked idle workers before using round-robin.
                while total_inflight < dispatch_cap {
                    let op_id = match ready_queue.pop_front() {
                        Some(id) => id,
                        None => break,
                    };
                    let target_rank = if let Some(idle) = idle_workers.pop_front() {
                        if p2p_debug {
                            eprintln!(
                                "[p2p-debug] coordinator waking rank={idle} with op={op_id}"
                            );
                        }
                        idle
                    } else {
                        let r = (worker_cursor % n_workers) as i32 + 1;
                        worker_cursor += 1;
                        r
                    };
                    let op = dag
                        .get_op(op_id)
                        .ok_or_else(|| anyhow::anyhow!("op {op_id} missing from DAG"))?
                        .clone();
                    let inputs =
                        op.input_ids.iter().map(|&dep| tensor_store[&dep].clone()).collect();
                    mpi_send(world, target_rank, &MpiMsg::P2POpAssignment { op_id, op, inputs })?;
                    dispatched.insert(op_id);
                    total_inflight += 1;
                    coordinator_messages += 1;
                }

                if completed_compute >= total_compute && total_inflight == 0 {
                    break;
                }
            }

            // Worker exhausted P2P stealing — it is blocking, waiting for a direct
            // assignment.  Try to satisfy it immediately from the ready queue; if no
            // op is ready yet, park the rank in `idle_workers` so it gets the next
            // op that unlocks.
            MpiMsg::P2PIdleNotification { rank } => {
                if p2p_debug {
                    eprintln!(
                        "[p2p-debug] coordinator received idle from rank={rank}, \
                         ready_ops={} queued={}",
                        ready_queue.len(),
                        idle_workers.len()
                    );
                }
                idle_workers.push_back(rank);

                while total_inflight < dispatch_cap {
                    let op_id = match ready_queue.pop_front() {
                        Some(id) => id,
                        None => break,
                    };
                    let target_rank = match idle_workers.pop_front() {
                        Some(r) => {
                            if p2p_debug {
                                eprintln!(
                                    "[p2p-debug] coordinator waking rank={r} with op={op_id}"
                                );
                            }
                            r
                        }
                        None => {
                            ready_queue.push_front(op_id);
                            break;
                        }
                    };
                    let op = dag
                        .get_op(op_id)
                        .ok_or_else(|| anyhow::anyhow!("op {op_id} missing from DAG"))?
                        .clone();
                    let inputs =
                        op.input_ids.iter().map(|&dep| tensor_store[&dep].clone()).collect();
                    mpi_send(world, target_rank, &MpiMsg::P2POpAssignment { op_id, op, inputs })?;
                    dispatched.insert(op_id);
                    total_inflight += 1;
                    coordinator_messages += 1;
                }
            }

            _ => {}
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Broadcast Shutdown to all workers and collect P2P metrics.
    for rank in 1..n_ranks as i32 {
        mpi_send(world, rank, &MpiMsg::Shutdown)?;
        coordinator_messages += 1;
    }

    let mut agg_idle_ms = 0.0f64;
    let mut agg_p2p_attempted = 0u64;
    let mut agg_p2p_successful = 0u64;
    let mut agg_p2p_latency_sum = 0.0f64;
    let mut agg_p2p_latency_count = 0u64;
    let mut shut = 0;

    while shut < n_workers {
        let (msg, _) = mpi_recv_any(world)?;
        if let MpiMsg::P2PMetrics {
            idle_time_ms,
            p2p_steals_attempted,
            p2p_steals_successful,
            p2p_steal_latency_ms,
        } = msg
        {
            agg_idle_ms += idle_time_ms;
            agg_p2p_attempted += p2p_steals_attempted;
            agg_p2p_successful += p2p_steals_successful;
            if p2p_steal_latency_ms > 0.0 {
                agg_p2p_latency_sum += p2p_steal_latency_ms;
                agg_p2p_latency_count += 1;
            }
            shut += 1;
        }
    }

    let avg_latency = if agg_p2p_latency_count > 0 {
        agg_p2p_latency_sum / agg_p2p_latency_count as f64
    } else {
        0.0
    };

    let metrics = SchedulerMetrics::new(
        total_compute as u64,
        total_compute as u64,
        elapsed_ms,
        agg_idle_ms,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    .with_p2p(agg_p2p_attempted, agg_p2p_successful, avg_latency, coordinator_messages);

    Ok((tensor_store, metrics))
}

// ── P2P worker (ranks 1..N) ───────────────────────────────────────────────────

/// Non-blocking poll for an MPI message, giving up once `deadline` passes.
///
/// Uses `immediate_probe` in a 1 ms sleep loop so the caller thread is not
/// pinned to 100 % CPU while waiting.  Returns `None` on timeout so the
/// caller can take corrective action (abandon a pending steal, retry after
/// backoff, etc.) rather than blocking forever.
///
/// # Errors
/// Returns an error if MPI receive or deserialisation fails.
fn poll_message_until<C: Communicator>(
    world: &C,
    deadline: Instant,
) -> anyhow::Result<Option<(MpiMsg, i32)>> {
    loop {
        if let Some(status) = world.any_process().immediate_probe() {
            let src = status.source_rank();
            let (bytes, _) = world.process_at_rank(src).receive_vec::<u8>();
            let msg = bincode::deserialize(&bytes).context("MpiMsg deserialize")?;
            return Ok(Some((msg, src)));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Pending steal this worker initiated (thief role), awaiting the victim's response.
struct PendingSteal {
    request_id: u64,
    victim_rank: i32,
    sent_at: Instant,
}

/// P2P worker loop — handles coordinator assignments, serves steal requests from
/// peers, and steals from peers when its own queue is empty.
///
/// **Startup phase**: the worker blocks waiting for the coordinator's first
/// `P2POpAssignment` before it ever enters the steal loop.  This eliminates the
/// spurious steal-failure burst visible in `--p2p-debug` output when all workers
/// race to steal from each other before the coordinator's initial dispatch arrives.
///
/// **Two-way handshake**: victim pops the op and sends it in one message.
/// No Ack round-trip is needed, halving steal latency vs the original three-way design.
///
/// **Timestamp backoff**: failed steals set `next_steal_at` instead of sleeping,
/// so the worker remains in the poll loop and processes incoming assignments
/// immediately during the backoff window.
///
/// **Coordinator fallback**: if the worker has been idle for longer than
/// `idle_threshold` it sends `P2PIdleNotification` to rank 0 and blocks on a
/// `mpi_recv_any`.  The coordinator wakes it with a direct `P2POpAssignment`
/// once a dependency resolves.  This breaks the livelock on serial DAGs where
/// all workers exhaust stealing simultaneously.
fn p2p_worker_loop<C: Communicator>(
    world: &C,
    rank: i32,
    n_ranks: usize,
    p2p_debug: bool,
    idle_threshold: Duration,
) -> anyhow::Result<()> {
    let mut local_queue: VecDeque<(OpId, Op, Vec<Tensor>)> = VecDeque::new();
    let mut peer_dir = PeerDirectory::new(rank, n_ranks);
    let steal_threshold = DEFAULT_STEAL_THRESHOLD;

    let mut pending_steal: Option<PendingSteal> = None;
    let mut next_steal_at: Option<Instant> = None;
    let mut backoff = Duration::from_millis(P2P_INITIAL_BACKOFF_MS);

    let mut idle_time_ms = 0.0f64;
    let mut idle_start: Option<Instant> = None;
    let mut p2p_steals_attempted = 0u64;
    let mut p2p_steals_successful = 0u64;
    let mut steal_latency_sum_ms = 0.0f64;
    let mut steal_latency_count = 0u64;
    let mut next_request_id: u64 = (rank as u64) << 32; // rank-scoped IDs; no collisions

    // ── Phase 1: startup — block until coordinator sends first op ─────────────
    //
    // Workers must not steal from peers before receiving initial work.  Without
    // this guard every worker immediately tries to steal from every other worker,
    // all fail (queues are universally empty), and the 100 ms steal timeout plus
    // the 1000 ms idle threshold fire before the coordinator's first dispatch even
    // lands — producing the "steal failed" lines at t≈2 ms seen in --p2p-debug.
    'startup: loop {
        let (msg, _) = mpi_recv_any(world)?;
        match msg {
            MpiMsg::P2POpAssignment { op_id, op, inputs } => {
                if p2p_debug {
                    eprintln!("[p2p-debug] rank={rank} received initial op from coordinator");
                }
                local_queue.push_back((op_id, op, inputs));
                break 'startup;
            }
            MpiMsg::P2PStealRequest { from_rank, request_id } => {
                // Queue is empty during startup — respond None immediately so
                // the requesting peer is not left waiting.
                mpi_send(
                    world,
                    from_rank,
                    &MpiMsg::P2PStealResponse {
                        request_id,
                        op_id: None,
                        op: None,
                        inputs: None,
                    },
                )?;
            }
            MpiMsg::Shutdown => {
                // DAG completed before this worker received any work (possible
                // when n_workers > number of initially ready ops and the DAG
                // finishes before round-robin dispatch reaches this rank).
                mpi_send(
                    world,
                    0,
                    &MpiMsg::P2PMetrics {
                        idle_time_ms: 0.0,
                        p2p_steals_attempted: 0,
                        p2p_steals_successful: 0,
                        p2p_steal_latency_ms: 0.0,
                    },
                )?;
                return Ok(());
            }
            _ => {}
        }
    }

    // ── Phase 2: normal P2P steal loop ───────────────────────────────────────
    loop {
        // ── execute local work ────────────────────────────────────────────────
        if let Some((op_id, op, inputs)) = local_queue.pop_front() {
            if let Some(start) = idle_start.take() {
                idle_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            }
            let input_refs: Vec<&Tensor> = inputs.iter().collect();
            let tensor = execute_op(&op, &input_refs, &Device::Cpu)
                .map_err(|e| anyhow::anyhow!("rank {rank}: op {op_id} failed: {e}"))?;
            mpi_send(world, 0, &MpiMsg::OpResult { op_id, tensor })?;
            backoff = Duration::from_millis(P2P_INITIAL_BACKOFF_MS);
            continue;
        }

        // ── queue empty — track idle ──────────────────────────────────────────
        if idle_start.is_none() {
            idle_start = Some(Instant::now());
        }

        // ── eager coordinator probe ───────────────────────────────────────────
        // On serial/deep DAGs the coordinator resolves dependencies faster than
        // peers can yield work.  Checking rank 0 on every idle iteration catches
        // a pending P2POpAssignment before we spend a steal RTT + backoff only
        // to spin until the idle threshold fires.
        if let Some(status) = world.process_at_rank(0).immediate_probe() {
            let src = status.source_rank();
            let (bytes, _) = world.process_at_rank(src).receive_vec::<u8>();
            let coord_msg: MpiMsg =
                bincode::deserialize(&bytes).context("MpiMsg deserialize")?;
            match coord_msg {
                MpiMsg::P2POpAssignment { op_id, op, inputs } => {
                    if p2p_debug {
                        eprintln!(
                            "[p2p-debug] rank={rank} coordinator probe found op={op_id}"
                        );
                    }
                    if let Some(start) = idle_start.take() {
                        idle_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                    }
                    local_queue.push_back((op_id, op, inputs));
                    pending_steal = None;
                    next_steal_at = None;
                    backoff = Duration::from_millis(P2P_INITIAL_BACKOFF_MS);
                    continue;
                }
                MpiMsg::Shutdown => {
                    if let Some(start) = idle_start.take() {
                        idle_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                    }
                    let avg_latency = if steal_latency_count > 0 {
                        steal_latency_sum_ms / steal_latency_count as f64
                    } else {
                        0.0
                    };
                    mpi_send(
                        world,
                        0,
                        &MpiMsg::P2PMetrics {
                            idle_time_ms,
                            p2p_steals_attempted,
                            p2p_steals_successful,
                            p2p_steal_latency_ms: avg_latency,
                        },
                    )?;
                    return Ok(());
                }
                _ => {}
            }
        }

        // Check idle threshold on every iteration, not only in the poll-timeout
        // arm.  Steal responses arrive as Some(m) via poll_message_until and
        // never hit the timeout path, so the threshold must be evaluated here.
        if idle_start.map_or(false, |t| t.elapsed() >= idle_threshold) {
            if p2p_debug {
                let ms = idle_start.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
                eprintln!(
                    "[p2p-debug] rank={rank} idle for {ms:.1}ms, sending idle notification"
                );
            }
            pending_steal = None;
            next_steal_at = None;
            backoff = Duration::from_millis(P2P_INITIAL_BACKOFF_MS);
            mpi_send(world, 0, &MpiMsg::P2PIdleNotification { rank })?;
            if p2p_debug {
                eprintln!("[p2p-debug] rank={rank} sent idle notification to coordinator");
            }
            'coordinator_wait: loop {
                let (park_msg, _) = mpi_recv_any(world)?;
                match park_msg {
                    MpiMsg::P2POpAssignment { op_id, op, inputs } => {
                        local_queue.push_back((op_id, op, inputs));
                        if let Some(start) = idle_start.take() {
                            idle_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                        }
                        break 'coordinator_wait;
                    }
                    MpiMsg::P2PStealRequest { from_rank, request_id } => {
                        mpi_send(
                            world,
                            from_rank,
                            &MpiMsg::P2PStealResponse {
                                request_id,
                                op_id: None,
                                op: None,
                                inputs: None,
                            },
                        )?;
                    }
                    MpiMsg::Shutdown => {
                        if let Some(start) = idle_start.take() {
                            idle_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                        }
                        let avg_latency = if steal_latency_count > 0 {
                            steal_latency_sum_ms / steal_latency_count as f64
                        } else {
                            0.0
                        };
                        mpi_send(
                            world,
                            0,
                            &MpiMsg::P2PMetrics {
                                idle_time_ms,
                                p2p_steals_attempted,
                                p2p_steals_successful,
                                p2p_steal_latency_ms: avg_latency,
                            },
                        )?;
                        return Ok(());
                    }
                    _ => {}
                }
            }
            continue;
        }

        // ── initiate steal if no pending steal and backoff has expired ────────
        if pending_steal.is_none() && !peer_dir.ranks.is_empty() {
            let now = Instant::now();
            if next_steal_at.map_or(true, |t| now >= t) {
                let victim = peer_dir.choose_victim();
                let req_id = next_request_id;
                next_request_id += 1;
                mpi_send(
                    world,
                    victim,
                    &MpiMsg::P2PStealRequest { from_rank: rank, request_id: req_id },
                )?;
                p2p_steals_attempted += 1;
                pending_steal = Some(PendingSteal {
                    request_id: req_id,
                    victim_rank: victim,
                    sent_at: now,
                });
                next_steal_at = None;
            }
        }

        // ── wait for next message — with a timeout when either a steal response
        // is pending or a backoff window is active ──────────────────────────────
        //
        // Uses non-blocking probe so that the worker wakes on deadline expiry and
        // can re-initiate stealing or fall back to coordinator dispatch.  A fully
        // blocking mpi_recv_any() here would stall once all backoff timers fire
        // and nobody has messages left to send — the coordinator is also blocked
        // waiting for an OpResult, completing the deadlock.
        let poll_deadline: Option<Instant> = {
            let steal_timeout = pending_steal
                .as_ref()
                .map(|ps| ps.sent_at + Duration::from_millis(STEAL_RESPONSE_TIMEOUT_MS));
            [steal_timeout, next_steal_at].into_iter().flatten().min()
        };

        // Shared timeout handler: clears pending steal and accumulates backoff.
        // Returns true if the coordinator-fallback path should be entered.
        // Defined as a labeled block so we can `break` to the outer match arm.
        let (msg, _src_rank) = match poll_deadline {
            // No deadline — safe to block; coordinator or a peer will send soon.
            None => mpi_recv_any(world)?,

            Some(deadline) => {
                let poll_result = if Instant::now() >= deadline {
                    None // already expired — skip the probe
                } else {
                    poll_message_until(world, deadline)?
                };

                match poll_result {
                    // Got a message — hand it to the match below.
                    Some(m) => m,

                    // Timed out: clean up pending steal, check idle threshold.
                    None => {
                        if let Some(ps) = pending_steal.take() {
                            peer_dir.record_outcome(ps.victim_rank, false);
                            next_steal_at = Some(Instant::now() + backoff);
                            backoff =
                                (backoff * 2).min(Duration::from_millis(P2P_MAX_BACKOFF_MS));
                        }
                        continue;
                    }
                }
            }
        };

        match msg {
            // ── coordinator pushes a new op ───────────────────────────────────
            MpiMsg::P2POpAssignment { op_id, op, inputs } => {
                local_queue.push_back((op_id, op, inputs));
                next_steal_at = None;
                backoff = Duration::from_millis(P2P_INITIAL_BACKOFF_MS);
            }

            // ── peer wants to steal from us (victim role) ─────────────────────
            // Two-way: pop the op now and send it; no Ack needed.
            // Always respond immediately — even when the queue is too short —
            // so the thief is never left waiting for a response that won't come.
            MpiMsg::P2PStealRequest { from_rank, request_id } => {
                let stolen = if local_queue.len() > steal_threshold {
                    local_queue.pop_back()
                } else {
                    None
                };
                let response = match stolen {
                    Some((oid, op, inputs)) => MpiMsg::P2PStealResponse {
                        request_id,
                        op_id: Some(oid),
                        op: Some(op),
                        inputs: Some(inputs),
                    },
                    None => MpiMsg::P2PStealResponse {
                        request_id,
                        op_id: None,
                        op: None,
                        inputs: None,
                    },
                };
                mpi_send(world, from_rank, &response)?;
            }

            // ── victim replied to our steal (thief role) ──────────────────────
            // Two-way: op is ours on receipt; no Ack sent.
            MpiMsg::P2PStealResponse { request_id, op_id, op, inputs } => {
                if let Some(ps) = pending_steal.take() {
                    if ps.request_id == request_id {
                        let latency_ms = ps.sent_at.elapsed().as_secs_f64() * 1000.0;
                        steal_latency_sum_ms += latency_ms;
                        steal_latency_count += 1;

                        match (op_id, op, inputs) {
                            (Some(oid), Some(o), Some(inp)) => {
                                if p2p_debug {
                                    eprintln!(
                                        "[p2p-debug] rank={rank} stole from={} latency={latency_ms:.2}ms",
                                        ps.victim_rank
                                    );
                                }
                                local_queue.push_back((oid, o, inp));
                                peer_dir.record_outcome(ps.victim_rank, true);
                                p2p_steals_successful += 1;
                                next_steal_at = None;
                                backoff = Duration::from_millis(P2P_INITIAL_BACKOFF_MS);
                            }
                            _ => {
                                if p2p_debug {
                                    eprintln!(
                                        "[p2p-debug] rank={rank} steal failed from={} \
                                         latency={latency_ms:.2}ms backoff={}ms",
                                        ps.victim_rank,
                                        backoff.as_millis()
                                    );
                                }
                                peer_dir.record_outcome(ps.victim_rank, false);
                                next_steal_at = Some(Instant::now() + backoff);
                                backoff =
                                    (backoff * 2).min(Duration::from_millis(P2P_MAX_BACKOFF_MS));
                            }
                        }
                    } else {
                        // Stale response for a previous request — restore pending.
                        pending_steal = Some(ps);
                    }
                }
            }

            // ── coordinator signals end of DAG ────────────────────────────────
            MpiMsg::Shutdown => {
                if let Some(start) = idle_start.take() {
                    idle_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                }
                let avg_latency = if steal_latency_count > 0 {
                    steal_latency_sum_ms / steal_latency_count as f64
                } else {
                    0.0
                };
                mpi_send(
                    world,
                    0,
                    &MpiMsg::P2PMetrics {
                        idle_time_ms,
                        p2p_steals_attempted,
                        p2p_steals_successful,
                        p2p_steal_latency_ms: avg_latency,
                    },
                )?;
                break;
            }

            _ => {}
        }
    }

    Ok(())
}

// ── coordinator (rank 0) — coordinator-mediated modes ─────────────────────────

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
                        let pos = ready_queue
                            .iter()
                            .position(|&id| worker_for_op(id, n_workers) == worker_idx);
                        pos.and_then(|i| ready_queue.remove(i))
                    }
                    MpiSchedulerMode::P2PWorkStealing => {
                        unreachable!("routed to p2p_coordinator_loop")
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

    let mut agg_idle_ms = 0.0f64;
    let mut agg_steal_attempts = 0u64;
    let mut agg_successful_steals = 0u64;
    let mut shut = 0;

    while shut < n_workers {
        let (msg, source_rank) = mpi_recv_any(world)?;
        if matches!(msg, MpiMsg::StealRequest { .. }) {
            mpi_send(world, source_rank, &MpiMsg::Shutdown)?;
            match mpi_recv_from(world, source_rank)? {
                MpiMsg::Metrics { idle_time_ms, steal_attempts, successful_steals } => {
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
        0,
        0,
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

// ── worker (ranks 1..N) — coordinator-mediated modes ─────────────────────────

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
                let tensor = execute_op(&op, &input_refs, &Device::Cpu)
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
                    &MpiMsg::Metrics { idle_time_ms, steal_attempts, successful_steals },
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

    /// Same diamond DAG, P2P mode — verifies the P2P coordinator and worker
    /// loops produce the same correct output as the coordinator-mediated path.
    #[test]
    fn diamond_dag_p2p_two_ranks() {
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

        match MpiWorker::new(dag, sources)
            .with_mode(MpiSchedulerMode::P2PWorkStealing)
            .run()
        {
            Err(e) if e.to_string().contains("≥2 MPI ranks") => {}

            Ok(Some((store, metrics))) => {
                let op2: Vec<f32> = store[&2].data.iter().copied().collect();
                assert_eq!(op2, vec![1., 2., 3., 4.], "op2 P2P");
                let op3: Vec<f32> = store[&3].data.iter().copied().collect();
                assert_eq!(op3, vec![1., 2., 3., 4.], "op3 P2P");
                assert!(metrics.coordinator_messages > 0);
            }

            Ok(None) => {}

            Err(e) => panic!("unexpected P2P MpiWorker error: {e}"),
        }
    }

    /// Smoke-test PeerDirectory: choose_victim never returns self_rank,
    /// and record_outcome adjusts scoring so a repeated-success peer stays
    /// preferred (modulo the recent-victim penalty).
    #[test]
    fn peer_directory_choose_and_record() {
        let self_rank = 1i32;
        let mut dir = PeerDirectory::new(self_rank, 4); // peers: 2, 3

        // Initially both peers have equal optimistic score; choose_victim must
        // return one of the valid peers.
        let v = dir.choose_victim();
        assert!(dir.ranks.contains(&v), "victim must be a peer");
        assert_ne!(v, self_rank, "must not choose self");

        // After many successes on rank 2, it should score higher than rank 3.
        for _ in 0..10 {
            dir.record_outcome(2, true);
        }
        // Flush recent_victim window so penalty doesn't dominate.
        for _ in 0..RECENT_VICTIM_WINDOW {
            dir.record_outcome(3, false);
        }
        assert_eq!(dir.choose_victim(), 2, "rank 2 should be preferred after repeated successes");
    }
}
