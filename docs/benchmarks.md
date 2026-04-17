# ferroflow — Benchmark Results

Results are logged here after each run. Include date, git commit, node count, and key metrics.

## Format

```
### YYYY-MM-DD — <short description>
- Commit: <git hash>
- Nodes: N  |  Ranks: M  |  CPUs/rank: K
- Problem: <DAG type, op count, problem size>
- Scheduler: static | work-stealing
- Wall time: X.Xs
- Worker utilization: X%
- Steal rate: X% (work-stealing only)
- Notes: ...
```

---

### 2026-04-17 — Local single-node scheduler comparison (Week 2, Session 2)

- **Machine:** Apple M-series dev laptop (local, single node, no MPI)
- **Commit:** pre-tag (Week 2 session 2, before git commit)
- **Nodes:** 1  |  **Workers (tokio tasks):** 4  |  **Rust:** 1.91 (local)
- **Benchmark harness:** Criterion 0.5, 100 samples each

#### Topology A — Balanced matmul chain (20 ops, 128×128)

Linear chain: `src₀·src₁ → mm₁ → mm₂ → … → mm₂₀` — inherently sequential.
No op can run until the previous completes, so schedulers provide no parallelism benefit.

| Scheduler | Median | Notes |
|-----------|--------|-------|
| Sequential (1 worker) | **1.06 ms** | baseline |
| Static (4 workers) | 1.45 ms | +37% vs sequential — coordination overhead, zero parallelism gain |
| Work-stealing (4 workers) | 1.43 ms | ~same as static — reactive backoff, still serialised by deps |

**Takeaway:** For purely sequential DAGs, multi-worker schedulers add tokio task + mutex overhead (~380 µs/iter) with no benefit. Expected — confirmed correct behaviour.

#### Topology B — Skewed fan-out (16 independent ops, 8 heavy + 8 light)

4 sources → 16 independent compute ops (all fan-out from sources, no inter-op deps).

- **Heavy ops (ids ≡ 0,1 mod 4):** 192×192 matmul — assigned to workers 0 and 1 by round-robin
- **Light ops (ids ≡ 2,3 mod 4):** 48×48 matmul — assigned to workers 2 and 3

This deliberately loads workers 0 and 1 with 4× the computational work.

| Scheduler | Median | Notes |
|-----------|--------|-------|
| Static (4 workers) | 773 µs | workers 2,3 sit idle after finishing light ops |
| Work-stealing (4 workers) | **745 µs** | workers 2,3 steal heavy ops — **~3.6% faster** |

**Takeaway:** Work-stealing shows a small but consistent win (~28 µs) on this deliberately skewed local workload. The gain is modest because 192×192 matmuls are ~170 µs each on this hardware — short enough that stealing overhead is visible. The steal threshold of 2 also caps how aggressively workers redistribute. The real benefit emerges at Narval scale where ops take 50–500 ms and steal threshold effects are negligible.

#### Design note — backoff strategy

The work-stealing scheduler uses a `tokio::select!` that wakes on whichever fires first:
- A version-channel notification (any op completes → immediate wake), or
- A deterministic pseudo-random timeout capped at 10–50 ms

This makes the scheduler reactive under load (fast wake on completion) while still capping retry frequency when the system is genuinely idle. The 10–50 ms cap is sized for the distributed case where MPI steal-request round trips cost ~1 ms; on Narval the cap will rarely trigger during active computation.

---

### 2026-04-17 — Skew injection + criterion comparison table (Week 3, Session 2)

- **Machine:** Apple M-series dev laptop (local, single node, no MPI)
- **Commit:** Week 3 Session 2 (skew injection + new bench groups)
- **Nodes:** 1  |  **Workers:** 4 tokio tasks  |  **Rust:** 1.91 (local)
- **Benchmark harness:** Criterion 0.5, 100 samples each; one-shot RunMetrics pass for table
- **JSON results saved to:** `docs/benchmark_results.json`

#### DAG topologies

- **Uniform:** 20-op linear matmul chain (128×128).  Inherently sequential — no parallelism gain expected.
- **Skewed:** 22-op two-branch DAG (`Dag::with_skew(20, 5)`).
  - Branch A (fast): 10 independent `Slow(1 ms)` ops, all fan-out from source 0.
  - Branch B (slow): 10 independent `Slow(5 ms)` ops, all fan-out from source 1.
  - Sequential total ≈ 60–75 ms; round-robin distributes ~5 ops/worker.

#### Criterion timing (median, 100 samples)

| Benchmark | Median |
|-----------|--------|
| sequential_uniform_20op_128x128 | **1.10 ms** |
| static_4w_uniform_20op_128x128 | 1.39 ms |
| ws_4w_uniform_20op_128x128 | 1.46 ms |
| sequential_skewed_20op_factor5 | **74.0 ms** |
| static_4w_skewed_20op_factor5 | 21.3 ms |
| ws_4w_skewed_20op_factor5 | 21.2 ms |

#### One-shot RunMetrics comparison table

| Scheduler     | DAG     | Throughput  | Idle%  | Steal Rate |
|---------------|---------|-------------|--------|------------|
| sequential    | uniform | 17 145 ops/s |  0.0% |      0.0/s |
| static        | uniform | 14 099 ops/s | 64.1% |      0.0/s |
| work-stealing | uniform | 16 087 ops/s | 66.3% |      0.0/s |
| sequential    | skewed  |    268 ops/s |  0.0% |      0.0/s |
| static        | skewed  |    935 ops/s |  0.0% |      0.0/s |
| work-stealing | skewed  |    935 ops/s | 11.5% |      0.0/s |

#### Key observations

- **Uniform DAG:** All schedulers produce ~1–1.5 ms because the chain is inherently serial (each op depends on the previous). Static and WS add coordination overhead (~30–40%) with no parallelism gain. Expected.
- **Skewed DAG:** Parallel schedulers achieve ~3.5× speedup over sequential (74 ms → 21 ms). Both static and work-stealing reach the same wall time on this DAG because round-robin assignment already distributes the `Slow` ops across all four workers without a gross imbalance. The steal threshold of 2 prevents stealing from queues with ≤ 2 remaining ops, which is sufficient here.
- **No successful steals on skewed DAG:** The two-branch fan-out structure means all ops are immediately ready (all depend only on pre-populated sources). Workers exhaust their own queues before the queues of others fall below the steal threshold. The 15 steal attempts but 0 successes confirm the threshold is the bottleneck. Lowering `STEAL_THRESHOLD` or using a more extreme imbalance would surface work-stealing's benefit — planned for the Narval multi-node runs.

_Full multi-node scaling results (4 / 8 / 16 / 32 nodes) pending Narval runs._

## 2026-04-16T21:34:44-04:00  job=59467408
- nodes: 2
- ranks: 2  (1 per node)
- threads/rank: 32
- exit: 0
- git: f3f0ccc

## Criterion Timing - Local

  | Scheduler     | DAG     |     Throughput | Idle%  | Steal Rate |      
  |---------------|---------|----------------|--------|------------|      
  | sequential    | uniform |    17145 ops/s |   0.0% |      0.0/s |
  | static        | uniform |    14099 ops/s |  64.1% |      0.0/s |      
  | work-stealing | uniform |    16087 ops/s |  66.3% |      0.0/s |      
  | sequential    | skewed  |      268 ops/s |   0.0% |      0.0/s |      
  | static        | skewed  |      935 ops/s |   0.0% |      0.0/s |      
  | work-stealing | skewed  |      935 ops/s |  11.5% |      0.0/s | 


---

### 2026-04-16T22:42:26-04:00 — Narval multi-node benchmark (job 59471496)

- **Machine:** Narval (Alliance Canada)  |  **Job:** 59471496
- **Nodes allocated:** 4  |  **CPUs/task:** 32
- **Commit:** 10f3d58
- **DAG:** 20-op uniform (128×128 matmul) and skewed (Slow × 5)
- **Configs tested:** 2-node and 4-node worker counts
- **Full RunMetrics:** `docs/benchmark_results.json`


## Scheduler Comparison

| Scheduler     | Nodes | DAG     |     Throughput |  Idle% | Steal Rate |
|---------------|-------|---------|----------------|--------|------------|
| sequential    |     1 | uniform |         17145 /s |   0.0% |       0.0/s |
| static        |     1 | uniform |         14099 /s |  64.1% |       0.0/s |
| work-stealing |     1 | uniform |         16087 /s |  66.3% |       0.0/s |
| sequential    |     1 | skewed  |           268 /s |   0.0% |       0.0/s |
| static        |     1 | skewed  |           935 /s |   0.0% |       0.0/s |
| work-stealing |     1 | skewed  |           935 /s |  11.5% |       0.0/s |
| sequential    |     2 | uniform |         11747 /s |   0.0% |       0.0/s |
| static        |     2 | uniform |          8924 /s |  43.5% |       0.0/s |
| work-stealing |     2 | uniform |         10513 /s |  49.5% |       0.0/s |
| sequential    |     2 | skewed  |           327 /s |   0.0% |       0.0/s |
| static        |     2 | skewed  |           652 /s |   0.0% |       0.0/s |
| work-stealing |     2 | skewed  |           652 /s |   0.0% |       0.0/s |
| sequential    |     4 | uniform |         11484 /s |   0.0% |       0.0/s |
| static        |     4 | uniform |          9542 /s |  60.9% |       0.0/s |
| work-stealing |     4 | uniform |          9552 /s |  73.2% |       0.0/s |
| sequential    |     4 | skewed  |           327 /s |   0.0% |       0.0/s |
| static        |     4 | skewed  |          1151 /s |   0.0% |       0.0/s |
| work-stealing |     4 | skewed  |          1152 /s |  11.6% |       0.0/s |
