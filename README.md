# ferroflow

Distributed **tensor DAG** execution and scheduling experiments: compare **static** partitioning vs **work-stealing** when operator costs are uneven (“skew”). The primary target environment is **[Narval](https://docs.alliancecan.ca/wiki/Narval)** (Alliance Canada); local development uses a single-process runtime and Criterion benchmarks.

> **Status:** active research codebase — see [`docs/progress.md`](docs/progress.md) for what is implemented vs planned.

## What exists today

- **`ferroflow-core`** — `Tensor` helpers, tensor ops (`matmul`, `relu`, `layer_norm`, `reduce`), `Dag` + `Op` model, topological order and ready-set logic.
- **`ferroflow-runtime`** — sequential execution, static round-robin scheduling, work-stealing scheduler (local / async prototype), Criterion DAG benchmarks.
- **Root binary** — `ferroflow` CLI (scheduler flags; full DAG-from-file workflow is still evolving).
- **`mpi-hello`** — small MPI sanity check, **excluded** from the Cargo workspace (`Cargo.toml` `exclude`) so local `cargo build` does not require an MPI toolchain.

MPI-backed multi-node execution and SLURM job templates are documented in [`docs/narval-setup.md`](docs/narval-setup.md) and [`docs/architecture.md`](docs/architecture.md); treat those as the source of truth for the next milestones.

## Requirements

| Item | Notes |
|------|--------|
| **Rust** | `1.74+` recommended (project uses **Edition 2021**); pin your own `rust-toolchain.toml` if you need reproducibility. |
| **MPI** | Optional until you enable the `distributed` feature on `ferroflow-runtime` / `ferroflow-core`. Use your site’s MPI module (e.g. on Narval: `<YOUR_PREREQ_MODULE>`, `<YOUR_OPENMPI_OR_INTEL_MPI_MODULE>` — replace with cluster docs). |

## Quick start (local)

```bash
git clone <YOUR_REPO_URL>
cd ferroflow

# Build workspace (root crate + crates/core + crates/runtime)
cargo build --release

cargo test
cargo bench -p ferroflow-runtime
```

If `cargo bench` OOMs on a small machine, try lower parallelism, e.g. `cargo bench -p ferroflow-runtime -j 2`.

### Optional: MPI / `distributed` feature

```bash
# After loading MPI in your environment (see docs/narval-setup.md)
cargo test -p ferroflow-runtime --features distributed
```

## Repository layout

```
ferroflow/
├── crates/
│   ├── core/       # DAG, ops, tensors
│   └── runtime/    # executors, schedulers, benches
├── docs/           # architecture, progress, benchmarks, Narval notes
├── mpi-hello/      # standalone MPI smoke test (workspace exclude)
├── slurm/          # example job scripts (placeholders / site-specific)
└── src/            # ferroflow binary
```

## Benchmarks

Results and methodology live in [`docs/benchmarks.md`](docs/benchmarks.md). Re-run locally after changes:

```bash
cargo bench -p ferroflow-runtime
```

## Running on Narval (placeholder checklist)

1. SSH: `ssh <YOUR_USER>@narval.alliancecan.ca` (or your site’s login host).
2. Clone this repo under `$PROJECT` (e.g. `$SCRATCH/ferroflow`).
3. Load compiler + MPI modules per [`docs/narval-setup.md`](docs/narval-setup.md).
4. Submit a job: `sbatch slurm/<YOUR_JOB_SCRIPT>.sh` (adjust account, partition, and walltime in the script).

Replace angle-bracket placeholders with your Alliance username, project allocation, and module names from current cluster documentation.

## Documentation

| Doc | Purpose |
|-----|---------|
| [`docs/architecture.md`](docs/architecture.md) | Intended multi-node design (coordinator, workers, messages). |
| [`docs/progress.md`](docs/progress.md) | Milestones and checkboxes. |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Dated benchmark log. |
| [`docs/narval-setup.md`](docs/narval-setup.md) | Environment and job hints for Narval. |
| [`CLAUDE.md`](CLAUDE.md) | Contributor / agent orientation for this repo. |

## License

**TBD** — add a `LICENSE` file (e.g. MIT or Apache-2.0) when you are ready to publish terms.
