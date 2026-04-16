# ferroflow — Architecture

## System Overview and Goals

ferroflow is a distributed tensor computation scheduler that represents a
computation as a directed acyclic graph (DAG) of tensor operations and
dynamically distributes that work across a cluster of nodes using a
work-stealing protocol over MPI.

**Goals:**
1. Execute tensor DAGs faster than static round-robin MPI partitioning on
   skewed workloads (where op costs are unequal).
2. Scale to 256 nodes on Narval with near-linear throughput on balanced DAGs.
3. Provide a reproducible benchmark suite comparing static vs. dynamic
   scheduling strategies.

**Non-goals (v1):** fault tolerance, heterogeneous accelerator support,
dynamic graph mutation at runtime.

---

## DAG Representation

### Graph Model
A computation is expressed as a `Dag<Op>` — a directed acyclic graph where:
- **Nodes** are tensor operations (`Op` enum).
- **Edges** encode data dependencies: an edge `A → B` means B consumes the
  output tensor produced by A.

```
src/dag/
├── graph.rs    # Dag<T> adjacency structure, topological sort
├── op.rs       # Op enum and cost model
└── tensor.rs   # Tensor descriptor (shape, dtype, buffer handle)
```

### Op Types
```rust
pub enum Op {
    MatMul { m: usize, n: usize, k: usize },
    Elementwise { kind: ElemKind, len: usize },
    Reduction  { axis: usize, len: usize },
    Transpose  { shape: Vec<usize> },
    Noop,                        // source/sink sentinels
}
```

Each `Op` has an associated `cost_estimate() -> u64` used by the scheduler
for load-balancing decisions.

---

## Scheduler Design

### Overview
The scheduler lives on **rank 0** (coordinator node). It:
1. Accepts a `Dag<Op>` and topologically sorts it.
2. Maintains a **global ready queue** of ops whose dependencies are satisfied.
3. Pushes ops to worker nodes using a pull-based protocol.

### Work-Stealing Protocol (Pull-Based)

Workers never receive ops unsolicited. The protocol is:

```
Worker → Coordinator: STEAL_REQUEST { worker_id }
Coordinator → Worker: STEAL_GRANT  { op_id, op_payload }
                    | STEAL_NONE   (queue empty or no ready ops)
Worker → Coordinator: OP_COMPLETE  { op_id, output_handle }
```

On `OP_COMPLETE`, the coordinator:
1. Marks the op done in the DAG.
2. Checks if any successor ops become ready (all dependencies done).
3. Enqueues newly ready ops on the global ready queue.

This design keeps the coordinator simple (single-threaded event loop) and
avoids push-based overload. Workers block on their tokio runtime waiting for
grants, then execute locally and report back.

### Coordinator State Machine
```
IDLE → RUNNING (on submit_dag)
RUNNING → DRAINING (all ops enqueued, waiting for completions)
DRAINING → DONE (completion count == op count)
DONE → IDLE (result tensors available)
```

---

## Worker Node Design

Each non-zero MPI rank runs a **worker process** with:
- A **tokio multi-thread runtime** for async coordination tasks.
- A **local work deque** (`VecDeque<ReadyOp>`) protected by a mutex.
- A **rayon thread pool** for CPU-bound tensor execution.

### Worker Loop
```
loop {
    if local_deque.is_empty() {
        send STEAL_REQUEST to coordinator
        match recv():
            STEAL_GRANT(op) => local_deque.push_back(op)
            STEAL_NONE      => backoff (exponential, max 10ms)
    }
    if let Some(op) = local_deque.pop_front() {
        spawn_blocking(|| execute_op(op))
            .await
            .map(|result| send OP_COMPLETE to coordinator)
    }
}
```

Workers do **not** steal from each other in v1. Inter-worker stealing is a
planned v2 feature — the current protocol uses coordinator-mediated stealing
only.

---

## MPI Communication Layer

```
src/transport/
├── mpi_transport.rs   # MPI send/recv wrappers (rsmpi bindings)
├── messages.rs        # Message enum + serde serialization (bincode)
└── coordinator.rs     # Coordinator event loop
```

### Message Serialization
Messages are serialized with `bincode` (compact binary, no schema overhead).
The `Message` enum is tagged so both sides can dispatch on type without a
separate header.

### Blocking vs. Async MPI
MPI calls in `rsmpi` are inherently blocking. They are wrapped in
`tokio::task::spawn_blocking` so they don't stall the async runtime.
A dedicated OS thread per rank handles MPI polling to avoid head-of-line
blocking on the tokio thread pool.

---

## Benchmark Plan

### Benchmark Suite (`benches/`)
| Benchmark | Description |
|-----------|-------------|
| `dag_static` | Static round-robin assignment, no stealing |
| `dag_workstealing` | Full work-stealing protocol |
| `dag_balanced` | Uniform op costs — measures overhead |
| `dag_skewed` | Power-law op cost distribution — measures stealing benefit |

### Skew Injection
The `dag_skewed` benchmark artificially inflates a random 10% of ops to 10×
their normal cost. This simulates real-world tensor workloads where a few
large matmuls dominate.

### Metrics
- **Total wall time** (primary)
- **Worker utilization** = (active compute time) / (wall time × num_workers)
- **Steal rate** = steal grants / total steal requests (proxy for imbalance)

### Target Node Counts
Strong-scaling runs: 4, 8, 16, 32, 64, 128, 256 nodes (2 ranks/node on Narval).

Results logged to `docs/benchmarks.md` per run.
