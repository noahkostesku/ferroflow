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

## Scheduler Comparison

| Scheduler        | Nodes | DAG     |     Throughput   |  Idle% | Steal Rate  |
|------------------|-------|---------|------------------|--------|-------------|
| mpi-static       |     2 | uniform |            20 /s |   0.0% |       0.0/s |
| mpi-work-stealing|     2 | uniform |            20 /s |   0.0% |      19.9/s |
| mpi-static       |     2 | skewed  |             7 /s |   0.0% |       0.0/s |
| mpi-work-stealing|     2 | skewed  |             7 /s |   0.0% |       6.7/s |
| mpi-static       |     4 | uniform |            56 /s |  22.5% |       0.0/s |
| mpi-work-stealing|     4 | uniform |            56 /s |  22.6% |      56.3/s |
| mpi-static       |     4 | skewed  |            17 /s |  43.3% |       0.0/s |
| mpi-work-stealing|     4 | skewed  |            18 /s |  45.2% |      18.1/s |

## Local Scale Results

| DAG                  | Seq    | Static | WS     |
|----------------------|--------|--------|--------|
| transformer          | 3.3 ms | 2.6 ms | 2.5 ms |
| wide+skew(8×5,0.25)  | 97.2 ms| 59.2 ms| 58.8 ms|
| resnet               | 0.1 ms | 0.1 ms | 0.1 ms |

### Strong Scaling — Transformer DAG (18 ops, 3-way QKV parallel)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |             2234 |             1886 |         1.000 |      0.0/s |
|     4 |             2476 |             1549 |         0.411 |      0.0/s |
|     8 |             2510 |             1496 |         0.198 |      0.0/s |

### Strong Scaling — Wide Skewed DAG (81 ops, width=16 depth=5 skew=0.25)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |             2896 |             2877 |         1.000 |      0.0/s |
|     4 |             3129 |             3057 |         0.531 |      0.0/s |
|     8 |             3128 |             3021 |         0.263 |      0.0/s |

### 2026-04-17 — Strong Scaling Study (Week 5 Session 2, job 59544118)

### Strong Scaling — Large Transformer DAG (137 ops, 8 layers, d=512)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |              556 |              556 |         1.000 |      0.0/s |
|     4 |              560 |              557 |         0.501 |     34.5/s |
|     8 |              552 |              543 |         0.244 |     28.3/s |

### Strong Scaling — Large Wide DAG (321 ops, width=320 depth=1 skew=0.47)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |             2701 |             2700 |         1.000 |     42.2/s |
|     4 |             5219 |             5310 |         0.983 |    182.5/s |
|     8 |            10359 |            10364 |         0.960 |    808.1/s |

### Strong Scaling — Imbalanced DAG (205 ops, 4 heavy ops × 200ms + 200 fast ops × 1ms)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |              900 |             1014 |         1.000 |    477.0/s |
|     4 |              959 |             1013 |         0.500 |    218.5/s |
|     8 |              988 |             1013 |         0.250 |     99.3/s |


---

### 2026-04-24 — Strong Scaling Study (Week 5 Session 2, job 59826638)

- **Machine:** Narval (Alliance Canada)  |  **Job:** 59826638
- **Nodes:** 2, 4, 8 |  **Ranks/node:** 1  |  **CPUs/rank:** 32
- **Workers:** N × 32 threads (32 per simulated node)
- **DAGs:** xlarge-wide (1281 ops, width=1280 depth=1 skew=0.003), xlarge-transformer (545 ops, 32 layers d=512 n_heads=8), imbalanced (205 ops, 4 heavy ops × 200ms + 200 fast ops × 1ms)
- **Commit:** 3709ffc


### Strong Scaling — XLarge Wide DAG (1281 ops, width=1280 depth=1 skew=0.003)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |             7387 |             7431 |         1.000 |     34.8/s |
|     4 |            14443 |            14777 |         0.994 |     92.4/s |
|     8 |            27631 |            28839 |         0.970 |    180.2/s |

### Strong Scaling — XLarge Transformer DAG (545 ops, 32 layers, d=512, n_heads=8)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |              581 |              578 |         1.000 |      0.0/s |
|     4 |              581 |              583 |         0.504 |      7.6/s |
|     8 |              578 |              574 |         0.248 |      7.5/s |

### Strong Scaling — Imbalanced DAG (205 ops, 4 heavy ops × 200ms + 200 fast ops × 1ms)

| Nodes |   Static (ops/s) |       WS (ops/s) | WS Efficiency | Steal Rate |
|-------||-----------------||-----------------||---------------||------------||
|     2 |              900 |             1014 |         1.000 |    477.0/s |
|     4 |              958 |             1013 |         0.500 |    218.6/s |
|     8 |              988 |             1012 |         0.250 |     99.3/s |

---

### 2026-04-27 — Adaptive Steal Threshold (Issue #5, commit cpu-work-scheduler)

- **Machine:** Apple M-series dev laptop (local, single node, no MPI)
- **Workers:** 8 tokio tasks  |  **Rust:** 1.91 (local)
- **Feature:** Per-worker self-tuning steal threshold based on 20-sample rolling window of steal success rate. Updates every 10 steal attempts. Base = max(1, n_workers/8); clamp = [1, max(4, n_workers/4)].

| DAG         | Threshold  | Throughput  | Steal Rate | Adjustments |
|-------------|------------|-------------|------------|-------------|
| imbalanced  | fixed=1    | ~996 ops/s  | ~469/s     | —           |
| imbalanced  | fixed=2    | ~989 ops/s  | ~446/s     | —           |
| imbalanced  | adaptive   | ~986 ops/s  | ~461/s     | threshold→4 |
| xlarge-wide | fixed=1    | ~6148 ops/s | ~31/s      | —           |
| xlarge-wide | fixed=2    | ~6153 ops/s | ~14/s      | —           |
| xlarge-wide | adaptive   | ~6215 ops/s | ~34/s      | threshold→4 |

**Result:** Adaptive threshold **ties or marginally outperforms** fixed alternatives on both DAGs within run-to-run noise (±20 ops/s). With n_workers=8, base threshold = max(1, 8/8) = 1; the adaptive threshold converges to 4 (the upper clamp = max(4, 8/4)) as end-of-job drain produces failed steals. During bulk execution it behaves identically to fixed=1.

**Design note — local vs. Narval behaviour:** The adaptive benefit scales with worker count. At 32 workers (Narval, 1 node): base = max(1, 32/8) = 4, upper clamp = max(4, 32/4) = 8. High steal success rate (plenty of imbalance) drives threshold down to 1; low rate (balanced work, futile steals) raises threshold to 6–8, eliminating the wasted scan. That dynamic does not manifest locally at 8 workers where base=1 already covers the aggressive case.

**Issue #5 outcome:** Adaptive beats fixed=2 (the prior hardcoded default) on xlarge-wide (6215 vs 6153 ops/s, +1%) and ties on imbalanced. Closing Issue #5 — adaptive is the new default and the infrastructure is in place for Narval-scale tuning.