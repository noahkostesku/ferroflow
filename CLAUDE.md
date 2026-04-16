# ferroflow — Project Memory

## What & Why
- Distributed work-stealing tensor computation scheduler targeting the Narval supercomputer (Alliance Canada)
- Designed to outperform static MPI scheduling on skewed DAG workloads by dynamically rebalancing tensor ops across nodes
- Research vehicle for comparing work-stealing vs. static partitioning strategies at scale (32–256 nodes)

## Repo Structure
```
ferroflow/
├── src/               # Main Rust source (lib.rs + modules)
│   ├── dag/           # DAG representation and op types
│   ├── scheduler/     # Work-stealing scheduler core
│   ├── worker/        # Per-node worker runtime (tokio)
│   └── transport/     # MPI communication layer
├── benches/           # Criterion benchmarks
├── slurm/             # SLURM job scripts (sbatch)
├── docs/              # Architecture, progress, benchmarks, setup
│   ├── architecture.md
│   ├── progress.md       ← CHECK THIS AT SESSION START
│   ├── benchmarks.md
│   └── narval-setup.md
└── tests/             # Integration tests
```

## Key Commands
```bash
cargo build --release          # Release build
cargo test                     # Run all tests
cargo bench                    # Run Criterion benchmarks
cargo doc --open               # Build and open API docs

# Submit a SLURM job on Narval
sbatch slurm/ferroflow.sh

# Check job status
squeue -u $USER
```

## Rust Conventions
- Error handling: use `thiserror` for library error types; propagate with `?`; no `unwrap`/`expect` in `src/lib.rs` or any library module
- `unwrap`/`expect` are acceptable only in `tests/`, `benches/`, and `src/main.rs`
- Intra-node parallelism: `rayon`
- Async networking / coordination: `tokio`
- All public API items (`pub fn`, `pub struct`, `pub trait`) must have doc comments (`///`)
- Keep modules small — split when a file exceeds ~300 lines

## Deeper Context
- See `docs/architecture.md` for system design and scheduler protocol
- See `docs/narval-setup.md` for cluster environment setup
- See `docs/benchmarks.md` for benchmark results (date-stamped)

> **Session start checklist:** Read `docs/progress.md` to see current task status before doing any work.
