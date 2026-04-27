# ferroflow — Progress Tracker

> Check this file at the start of every session to orient on current status.

Last updated: 2026-04-27 (Week 7 Session 3)

---

## Week 1 — Scaffold, Tensor Ops, DAG, Single-Node Baseline
_Target: 2026-04-20_

- [x] `cargo init` + Cargo workspace with core dependencies (ndarray, rayon, tokio, thiserror, anyhow, criterion)
      — workspace: `.` + `crates/core` + `crates/runtime`
- [x] Define `Tensor` type (`ArrayD<f32>` wrapper with `from_shape_vec`, `zeros`, `full`)
- [x] Implement core tensor ops: `matmul` (ndarray dot), `relu` (elementwise max), `layer_norm` (mean/var), `reduce` (sum_axis)
- [x] Unit tests for all tensor ops (7 tests: correctness, error paths, statistical)
- [x] Define `OpKind` enum (Matmul/Relu/LayerNorm/Reduce) + `Op` struct with `cost_estimate()`
- [x] Implement `Dag` (adjacency list, Kahn's topo sort, `ready_ops`)
- [x] Unit tests for DAG (7 tests: ordering, cycle detection, invalid IDs, ready-op progression)
- [x] Single-node `SequentialExecutor` — walks topo order, executes ops, returns `HashMap<OpId, Tensor>`
- [x] Baseline benchmark: 20-op matmul chain (128×128) → **~1.02 ms / iteration** (local dev machine)
- [x] CI setup (GitHub Actions or local `cargo test && cargo clippy`)

## Week 2 — MPI Setup, Static Scheduler, Work-Stealing Protocol
_Target: 2026-04-27_

- [x] Add `distributed` feature flag to `ferroflow-core` and `ferroflow-runtime`; `mpi` workspace dep declared, gated behind feature; `mpi-hello` excluded from workspace (Narval-only)
- [x] Worker abstractions: `WorkerId`, `WorkQueue` (Arc+Mutex), `Message` enum (Serialize/Deserialize + bincode roundtrip), `WorkerTrait` — in `crates/runtime/src/worker.rs`
- [ ] Verify `rsmpi`/MPI environment on Narval (`cargo test -p ferroflow-runtime --features distributed`)
- [ ] Verify `rsmpi`/MPI environment on Narval (`cargo test -p ferroflow-runtime --features distributed`)
- [x] MPI worker bridge: `crates/runtime/src/mpi_worker.rs` (rank 0 = coordinator, ranks 1..N = workers; pull-based steal protocol over MPI point-to-point; bincode serialization; shutdown drain)
- [x] Integration test: 4-op diamond DAG — run with `mpirun -n 2 cargo test -p ferroflow-runtime --features distributed -- --test-threads=1`
- [x] SLURM script: `slurm/ferroflow.sh` (parameterized, $SLURM_ACCOUNT, 30 min, openmpi 5.0.8, appends to benchmarks.md)
- [ ] End-to-end run on Narval: `cargo build --release --features distributed && sbatch slurm/ferroflow.sh`
- [ ] Second run: `sbatch --nodes=4 slurm/ferroflow.sh`
- [ ] Static scheduler: round-robin op assignment across ranks
- [ ] Multi-node integration test with static scheduler (4-rank smoke test)
- [ ] SLURM script: `slurm/bench-static.sh`
- [ ] SLURM script: `slurm/bench-workstealing.sh`

## Week 3 — Benchmarking, Skew Injection, Tuning
_Target: 2026-05-04_

- [x] Metrics types: `SchedulerMetrics` + `RunMetrics` in `crates/core/src/metrics.rs` (serde, Display, 5 unit tests)
- [x] Instrument `SequentialExecutor` — returns `(HashMap, SchedulerMetrics)`; elapsed_ms timing
- [x] Instrument `StaticScheduler` — per-worker idle time via `std::time::Instant`; returns `(HashMap, SchedulerMetrics)`
- [x] Instrument `WorkStealingScheduler` — `AtomicU64` steal_attempts/successful_steals; per-worker idle time; returns `(HashMap, SchedulerMetrics)`
- [x] Implement skew injection: `OpKind::Slow { duration_ms }` + `Dag::with_skew(n_ops, factor)`
- [x] Criterion bench groups: `sequential/static/ws × uniform/skewed` (6 benches) with custom main
- [x] Collect worker utilization and steal-rate metrics via `RunMetrics` in one-shot pass
- [x] Log results to `docs/benchmarks.md`; JSON saved to `docs/benchmark_results.json`
- [x] `slurm/scaling.sh`: strong scaling harness — 2/4/8/16 nodes × static/WS × transformer/wide × 3 runs; per-rank output via `$SCRATCH`; JSON append; prints comparison tables + 4-sentence analysis; appends to `docs/benchmarks.md`
- [x] `gen_imbalanced`: new DAG generator (4 heavy ops × 200ms + 200 fast ops × 1ms, all independent fan-out from root); `--dag imbalanced` CLI flag; added to `slurm/scaling.sh`; local validation (8 workers): **static=862 ops/s vs WS=994 ops/s (+15%), 465 steals/s** — gap is compelling
- [x] `slurm/scaling.sh` upgraded: 5 runs/config (was 3), median5, stddev via `statistics.stdev`, `imbalanced` added to DAGS list
- [ ] Run static vs. work-stealing on 4, 8, 16, 32 nodes (balanced DAG) — pending Narval `sbatch slurm/scaling.sh`
- [ ] Run static vs. work-stealing on 4, 8, 16, 32 nodes (skewed DAG) — pending Narval `sbatch slurm/scaling.sh`
- [ ] Profile coordinator bottleneck (is it the bottleneck at 32+ nodes?)
- [x] **Issue #5 — adaptive steal threshold:** `StealHistory` (20-sample window), `adaptive_threshold` fn (base = max(1,n/8), clamp = [1,max(4,n/4)], updates every 10 attempts), per-worker local state (no locks), `--no-adaptive-threshold` CLI flag, `threshold_final` + `threshold_adjustments` in `SchedulerMetrics`. Local 8w: adaptive ties fixed=1 on imbalanced, marginally beats fixed=2 on xlarge-wide. Adaptive is new default. Issue closed.
- [ ] Tune backoff strategy in worker steal loop
- [ ] Tune rayon thread pool size vs. `$SLURM_CPUS_PER_TASK`
- [ ] Run 64-node and 128-node scaling experiments

## Week 4 — Polish, CLI, README, Optional PyO3 Bindings
_Target: 2026-05-11_

- [x] ONNX model import: `crates/onnx` (`ferroflow-onnx`) — `parse_onnx`, `load_model`, `dag_summary`; maps MatMul/Gemm→Matmul, Relu, LayerNorm, Reduce; transB weight transposition; 4 unit tests
- [x] CLI extended: `ferroflow info --model <path>` (dag_summary) and `ferroflow run --model <path> --workers <n> --scheduler <seq|static|work-stealing>`
- [x] `scripts/export_mlp.py` exports a 784→256→128→10 MLP; `models/mlp.onnx` verified
- [x] End-to-end: `ferroflow info` prints 12 ops / 5 compute / 8 edges; `ferroflow run --workers 4` executes and prints RunMetrics; 37/37 tests pass
- [x] Synthetic DAG generators: `crates/core/src/dag_gen.rs` — `gen_transformer_block(seq_len, d_model, n_heads)` (18 ops, 3-way QKV parallel), `gen_wide_dag(width, depth, skew_factor)` (width×depth ops, configurable skew), `gen_resnet_block(channels)` (8 ops, fork-join); 12 unit tests; 55/55 tests pass
- [x] CLI extended: `ferroflow info/run --dag <transformer|wide|resnet>` alongside `--model`; wide supports `--width`, `--depth`, `--skew`; transformer supports `--seq-len`, `--d-model`, `--n-heads`; resnet supports `--channels`
- [ ] DAG spec file format (JSON or TOML)
- [x] README.md with project description, build instructions, benchmark summary
- [x] Clean up all `clippy` warnings
- [x] Ensure all public APIs have doc comments (`cargo doc --no-deps` clean)
- [ ] Final 256-node benchmark run on Narval
- [ ] Update `docs/benchmarks.md` with final results
- [ ] Tag v0.1.0 release
- [x] (Optional) PyO3 bindings: `crates/python` — `DAG` + `run` via `WorkStealingScheduler`; `maturin develop` + `simple_mlp.py` verified
- [x] (Optional) TUI dashboard: `crates/tui` (`ferroflow-tui`) — `DashboardState`, `WorkerStatus`, four-panel ratatui layout (workers/throughput/steal-activity/summary); `WorkStealingScheduler::execute_with_watch` streams `LiveMetrics` via `tokio::sync::watch` at 100 ms; final `RunMetrics` table printed on completion; `ferroflow-tui --workers <n> --dag-size <n> --skew <n>`; 38/38 tests pass
- [ ] (Optional) Visualizer: DOT/graphviz output for DAG structure

## Week 6 — Open Source Release Prep
_Target: 2026-04-17_

- [x] CHANGELOG.md — v0.1.0 entry with full feature list
- [x] CONTRIBUTING.md — dev setup, MPI build, PR checklist, contribution areas
- [x] LICENSE — MIT, Copyright 2026 Noah Kostesku
- [x] Cargo.toml — homepage, repository, license, keywords, categories, readme fields
- [x] README.md — fix placeholder repo URL, update status line, update license section
- [x] Tag v0.1.0 and push to GitHub

---

## Notes / Blockers

- **2026-04-16:** `cargo bench` requires `-j2` flag on this dev machine (build scripts OOM at default parallelism). On Narval this should not be an issue.
- **2026-04-16:** `mpi-hello` excluded from workspace — it requires an MPI library at link time and is Narval-only. Build it standalone on the cluster. `tokio/sync` feature added to workspace dep.
- **2026-04-16:** `MpiWorker` in `crates/runtime/src/mpi_worker.rs` (Week 2 Session 3). Coordinator event loop + worker steal loop over MPI point-to-point, bincode messages, exponential backoff on StealNone, shutdown drain. Diamond DAG integration test skips gracefully without mpirun.
- **Benchmark baseline (local, unoptimised HW):** 20-op 128×128 matmul chain ≈ 1.02 ms. Re-run on Narval compute node for the canonical result.
- **Week 1 remaining:** CI setup (clippy in GH Actions). All code complete.
- **2026-04-17:** Week 3 Session 1. Metrics layer added to `ferroflow-core`. All three schedulers now return `(HashMap<OpId, Tensor>, SchedulerMetrics)`. Benches updated to `.unwrap().0`. 32/32 tests pass.
- **2026-04-17:** Week 3 Session 2. `OpKind::Slow { duration_ms }` added to core; `Dag::with_skew(n_ops, factor)` constructor builds two-branch skewed DAG. Bench rewritten with 6 functions across 2 groups + custom `main` that prints RunMetrics comparison table and saves `docs/benchmark_results.json`. 33/33 tests pass. Key local results: uniform DAG ~1.1 ms sequential / ~1.4 ms parallel (chain serial, no parallelism); skewed DAG ~74 ms sequential / ~21 ms static+WS (3.5× speedup). Steal threshold prevents stealing on this fan-out topology — real WS benefit expected at Narval scale.
- **2026-04-17:** Week 4 Session 3. Live TUI dashboard in `crates/tui`. Added `LiveMetrics` / `WorkerLiveSnapshot` / `WorkerLiveStatus` to `ferroflow-core`. Refactored `WorkStealingScheduler` to track per-worker atomics (`worker_ops`, `worker_status`, `worker_idle_us`); `execute_with_watch` spawns a 100 ms ticker that sends snapshots via `watch::Sender<LiveMetrics>`; `execute` delegates to it. `ferroflow-tui` binary: 4-panel ratatui dashboard (worker gauges + throughput sparkline + steal activity + summary), prints final `RunMetrics` on completion. 38/38 tests pass.
- **2026-04-17:** Week 5 Session 3. Fixed zero steal-rate by adding `gen_large_transformer` (8 layers, d=512, 137 ops) and `gen_large_wide` (width=320, depth=1, skew=0.47, 321 ops) to `crates/core/src/dag_gen.rs`. Root cause: round-robin assignment + chained DAGs kept all workers pipeline-busy; fix is flat fan-out (depth=1) with a 150/170 slow/fast op split that is not divisible by 4, creating exploitable load imbalance. Added `--steal-threshold` CLI flag to `ferroflow run` (default 2); `WorkStealingScheduler::with_steal_threshold` builder. Updated `slurm/scaling.sh`: `WORKERS=N×4` (was N×32), `DAGS=(large-transformer large-wide)`, `--steal-threshold 1` on all srun calls. Local validation (4 workers): large-wide WS=1150 ops/s steal=14.4/s vs static=1124 ops/s ✓; large-transformer WS loses to static locally (sequential transformer backbone, 4 workers too few for QKV parallelism to help — expected at Narval scale with more workers). 58/58 tests pass.
- **2026-04-17:** Week 5 Session 3. `gen_imbalanced` in `crates/core/src/dag_gen.rs`: 4 heavy ops (200ms each) + 200 fast ops (1ms each), all independent fan-out from root. Round-robin assigns the 4 heavy ops to workers 0-3; workers 4-7 exhaust fast ops and steal the remaining fast ops from workers 0-3, reducing wall time from ~236ms to ~206ms. Local 8-worker result: static=862 ops/s, WS=994 ops/s (+15%), 465 steals/s. `slurm/scaling.sh` upgraded to 5 runs/config, median5, stddev. Root cause of original zero-gap design: sequential chain dependencies prevent mid-chain stealing; fix is independent heavy ops.
- **2026-04-17:** Week 5 Session 2. `slurm/scaling.sh` — strong scaling harness (2/4/8/16 nodes, transformer+wide-skew, static+WS, 3 runs/config, median). srun per-rank output to $SCRATCH. JSON append, comparison tables, analysis paragraph auto-generated. Ready to rsync and sbatch.
- **2026-04-17:** Week 5 Session 1. Synthetic DAG generators in `crates/core/src/dag_gen.rs`. Local scaling preview (release build, 4 workers): transformer WS=2.5 ms vs static=2.6 ms vs seq=3.3 ms; resnet WS=0.1 ms (fork-join too small for steal benefit); wide+skew(8×5,0.25) WS=58.8 ms vs static=59.2 ms vs seq=97.2 ms (1.65× speedup over sequential). Steal rate is 0 locally — coordinator-mediated stealing is the key differentiator at Narval scale. 55/55 tests pass.
- **2026-04-17:** Week 4 Session 1. PyO3 bindings in `crates/python` (new workspace member). `#[pyclass(name = "DAG")]` wraps a `Vec<Op>` builder; `matmul`/`relu`/`layer_norm`/`reduce` each append an `Op` and return its `u32` ID. `run(dag, workers)` constructs the `Dag`, pre-populates source tensors, and blocks a new `tokio::Runtime` on `WorkStealingScheduler::execute`. `pyproject.toml` uses maturin 1.x. `maturin develop` + `python ferroflow/examples/simple_mlp.py` verified — 33/33 tests pass.
- **2026-04-24:** Week 7 Session 1. GitHub Issue #4: additional ONNX op support. Added `OpKind::Softmax`, `OpKind::BatchNorm { epsilon }`, `OpKind::Conv2d { kernel_size, stride, padding }` to `crates/core/src/op.rs`. Implemented all three in `crates/core/src/ops.rs`: softmax uses numerically-stable max-subtraction along last axis; batch_norm is inference-mode only with per-channel broadcast for [N,C,H,W] and [N,C] inputs; conv2d uses im2col + explicit GEMM (loops with comment — correctness over cleverness). Wired `Softmax`, `BatchNormalization` (with epsilon attribute), and `Conv` (kernel_shape/strides/pads attrs, dilation≠1 errors) into `crates/onnx/src/lib.rs`. Fixed exhaustive match in `crates/runtime/src/executor.rs` and `src/main.rs`. Added 9 unit tests (3 per op). Updated README supported ops table. Added `scripts/export_resnet.py`. 76/76 tests pass.
- **2026-04-18:** Week 6 Session 2. Code quality + CI pass. Fixed all clippy warnings (`manual_is_multiple_of`, `too_many_arguments` via `SyntheticDagParams` struct, `if_same_then_else` in python crate). Replaced all `.unwrap()`/`.expect()` in library code with proper `?` propagation; added `WorkerPanicked` and `Internal` variants to `SchedulerError` and `ExecutorError`. `cargo doc --no-deps` emits zero warnings. Created `.github/workflows/ci.yml` (test/clippy/fmt jobs, `release/*` + `main` triggers). CI badge already present in README. 63/63 tests pass locally.
- **2026-04-27:** Week 7 Session 3. GPU benchmark on Narval A100 (job 59957529). matmul-parallel 16×8 branches, 2048×2048: CPU 8w = 25 ops/s (5,101ms), A100 = 62 ops/s (2,049ms), 2.5× speedup. 4096×4096 GPU: 17 ops/s (3,689ms). Bottleneck is per-op CPU↔GPU tensor transfer. Issue #8 closed. Next: issue #3 persistent GPU tensor storage.
- **2026-04-27:** Week 7 Session 3. Pure matmul DAG generators for GPU benchmarking. `gen_matmul_chain(n_ops, matrix_size)` (2 + n_ops ops, fully sequential matmul chain, no Slow ops) and `gen_matmul_parallel(n_branches, ops_per_branch, matrix_size)` (2 + n_branches×ops_per_branch ops, all branches independent, no Slow ops) added to `crates/core/src/dag_gen.rs`. Both exported from `ferroflow-core`. CLI: `--dag matmul-chain` / `--dag matmul-parallel` with `--matrix-size`, `--n-ops`, `--n-branches`, `--ops-per-branch` override flags. `slurm/gpu_bench.sh` updated: replaced `--dag imbalanced` and ResNet-18 ONNX runs with three `matmul-parallel` runs (CPU 32×4 m=256, GPU 32×4 m=256, GPU 16×4 m=512). Local validation (4w, m=512): `matmul-chain` static=315 ops/s, WS=340 ops/s, steal=0/s (sequential — expected); `matmul-parallel` static=949 ops/s, WS=1146 ops/s, steal=35.8/s (32 independent branches actively stolen — confirms WS benefit). 75/75 tests pass.
- **2026-04-27:** Week 7 Session 2. Issue #5: adaptive steal threshold. `StealHistory` (20-sample sliding window, optimistic start 1.0, `record`/`success_rate`/`reset`) in `work_stealing.rs`. `adaptive_threshold(current, history, n_workers)` fn: base=max(1,n/8), rate>0.7→decrease by 1, rate<0.3→increase by 2, rate 0.3–0.7→keep, clamp [1, max(4,n/4)], updates every 10 steal attempts. Per-worker local state (no locks, no shared mutable state). Added `threshold_final: usize` + `threshold_adjustments: u64` to `SchedulerMetrics` (serde default=0 for backward compat). `WorkStealingScheduler::with_adaptive_threshold(bool)` builder; `--no-adaptive-threshold` CLI flag; `[run]` output now shows `threshold=N`. Local 8w benchmark: adaptive ties/margially beats fixed=2 (prior hardcoded default). Issue #5 closed. 85/85 tests pass.
