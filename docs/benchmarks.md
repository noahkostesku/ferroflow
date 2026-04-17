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

_Week 3 results pending — Narval multi-node runs._

## 2026-04-16T21:34:44-04:00  job=59467408
- nodes: 2
- ranks: 2  (1 per node)
- threads/rank: 32
- exit: 0
- git: f3f0ccc
