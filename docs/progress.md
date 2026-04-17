# ferroflow — Progress Tracker

> Check this file at the start of every session to orient on current status.

Last updated: 2026-04-16

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
- [ ] CI setup (GitHub Actions or local `cargo test && cargo clippy`)

## Week 2 — MPI Setup, Static Scheduler, Work-Stealing Protocol
_Target: 2026-04-27_

- [x] Add `distributed` feature flag to `ferroflow-core` and `ferroflow-runtime`; `mpi` workspace dep declared, gated behind feature; `mpi-hello` excluded from workspace (Narval-only)
- [x] Worker abstractions: `WorkerId`, `WorkQueue` (Arc+Mutex), `Message` enum (Serialize/Deserialize + bincode roundtrip), `WorkerTrait` — in `crates/runtime/src/worker.rs`
- [ ] Verify `rsmpi`/MPI environment on Narval (`cargo test -p ferroflow-runtime --features distributed`)
- [ ] Verify `rsmpi`/MPI environment on Narval (`cargo test -p ferroflow-runtime --features distributed`)
- [x] MPI worker bridge: `crates/runtime/src/mpi_worker.rs` (rank 0 = coordinator, ranks 1..N = workers; pull-based steal protocol over MPI point-to-point; bincode serialization; shutdown drain)
- [x] Integration test: 4-op diamond DAG — run with `mpirun -n 2 cargo test -p ferroflow-runtime --features distributed -- --test-threads=1`
- [x] SLURM script: `slurm/ferroflow.sh` (parameterized, def-cbravo, 30 min, openmpi 5.0.8, appends to benchmarks.md)
- [ ] End-to-end run on Narval: `cargo build --release --features distributed && sbatch slurm/ferroflow.sh`
- [ ] Second run: `sbatch --nodes=4 slurm/ferroflow.sh`
- [ ] Static scheduler: round-robin op assignment across ranks
- [ ] Multi-node integration test with static scheduler (4-rank smoke test)
- [ ] SLURM script: `slurm/bench-static.sh`
- [ ] SLURM script: `slurm/bench-workstealing.sh`

## Week 3 — Benchmarking, Skew Injection, Tuning
_Target: 2026-05-04_

- [ ] Implement skew injection in benchmark harness (10% of ops at 10× cost)
- [ ] Run static vs. work-stealing on 4, 8, 16, 32 nodes (balanced DAG)
- [ ] Run static vs. work-stealing on 4, 8, 16, 32 nodes (skewed DAG)
- [ ] Collect worker utilization and steal-rate metrics
- [ ] Log results to `docs/benchmarks.md`
- [ ] Profile coordinator bottleneck (is it the bottleneck at 32+ nodes?)
- [ ] Tune backoff strategy in worker steal loop
- [ ] Tune rayon thread pool size vs. `$SLURM_CPUS_PER_TASK`
- [ ] Run 64-node and 128-node scaling experiments

## Week 4 — Polish, CLI, README, Optional PyO3 Bindings
_Target: 2026-05-11_

- [ ] CLI (`src/main.rs`): accept DAG spec file, node count, scheduler type
- [ ] DAG spec file format (JSON or TOML)
- [ ] README.md with project description, build instructions, benchmark summary
- [ ] Clean up all `clippy` warnings
- [ ] Ensure all public APIs have doc comments (`cargo doc --no-deps` clean)
- [ ] Final 256-node benchmark run on Narval
- [ ] Update `docs/benchmarks.md` with final results
- [ ] Tag v0.1.0 release
- [ ] (Optional) PyO3 bindings: expose `submit_dag` to Python
- [ ] (Optional) Visualizer: DOT/graphviz output for DAG structure

---

## Notes / Blockers

- **2026-04-16:** `cargo bench` requires `-j2` flag on this dev machine (build scripts OOM at default parallelism). On Narval this should not be an issue.
- **2026-04-16:** `mpi-hello` excluded from workspace — it requires an MPI library at link time and is Narval-only. Build it standalone on the cluster. `tokio/sync` feature added to workspace dep.
- **2026-04-16:** `MpiWorker` in `crates/runtime/src/mpi_worker.rs` (Week 2 Session 3). Coordinator event loop + worker steal loop over MPI point-to-point, bincode messages, exponential backoff on StealNone, shutdown drain. Diamond DAG integration test skips gracefully without mpirun.
- **Benchmark baseline (local, unoptimised HW):** 20-op 128×128 matmul chain ≈ 1.02 ms. Re-run on Narval compute node for the canonical result.
- **Week 1 remaining:** CI setup (clippy in GH Actions). All code complete.
