# ferroflow

Distributed work-stealing tensor computation scheduler for HPC clusters, written in Rust.

[![CI](https://github.com/noahkostesku/ferroflow/actions/workflows/ci.yml/badge.svg)](https://github.com/noahkostesku/ferroflow/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Static scheduling concentrates slow ops on specific workers, leaving others idle. ferroflow represents a tensor computation as a DAG and distributes it across nodes, dynamically stealing ops from overloaded workers when others run dry. It supports DAG execution, ONNX model import, a Python API via PyO3, a live terminal dashboard, and SLURM integration for Alliance Canada clusters.

---

## Results

| Workload | Nodes | Static | ferroflow | Advantage |
|---|---|---|---|---|
| Imbalanced DAG | 2 | 900 ops/s | 1,014 ops/s | +12.7% |
| Imbalanced DAG | 8 | 988 ops/s | 1,013 ops/s | +2.5% |
| Parallel DAG | 8 | 10,359 ops/s | 10,364 ops/s | 96% efficiency |
| Transformer DAG | 8 | ~550 ops/s | ~550 ops/s | DAG-bounded |

Narval supercomputer (Alliance Canada), AMD EPYC Milan, 100Gb/s interconnect. 5-run medians.

---

## How it works

A `Dag<Op>` encodes tensor operations as nodes and data dependencies as directed edges. The coordinator (rank 0) topologically sorts the DAG and maintains a global ready queue. Worker ranks loop on a pull protocol: send `STEAL_REQUEST`, receive `STEAL_GRANT(op)` or `STEAL_NONE`, execute the op locally via a rayon thread pool, then send `OP_COMPLETE`. On completion, the coordinator marks successors ready and enqueues them. Messages are serialized with bincode over rsmpi point-to-point; MPI calls run in `spawn_blocking` to avoid stalling the tokio runtime.

```
┌──────────────────────────────────────────┐
│  Python API (PyO3)                        │
│  ┌─────────┐ steal  ┌─────────┐          │
│  │Worker 0 │◄──────►│Worker N │          │
│  │ deque   │        │ deque   │          │
│  └─────────┘        └─────────┘          │
├──────────────────────────────────────────┤
│      MPI transport (rsmpi)               │
│      100Gb/s Narval fabric               │
└──────────────────────────────────────────┘
```

---

## Quick start

**Local (no MPI):**

```bash
git clone https://github.com/noahkostesku/ferroflow
cd ferroflow
cargo build --release
./target/release/ferroflow run --dag imbalanced --workers 8
./target/release/ferroflow run --model models/mlp.onnx --workers 4
./target/release/ferroflow run --dag imbalanced --workers 8 --dashboard
```

---

## Built-in DAGs

| Flag | Description | Ops |
|---|---|---|
| `--dag transformer` | 8-layer attention block, serial deps | 137 |
| `--dag large-transformer` | same, larger d_model | 137 |
| `--dag wide` | parallel branches, optional skew | varies |
| `--dag large-wide` | wide with more ops | 321 |
| `--dag imbalanced` | 4 slow chains + 200 fast ops | 205 |

---

## ONNX support

Supported ops: `MatMul`, `Gemm`, `Relu`, `LayerNormalization`, `ReduceMean`, `GlobalAveragePool`

```bash
# Export from PyTorch
python scripts/export_mlp.py

# Run
./target/release/ferroflow run --model models/mlp.onnx --workers 4
```

---

## Scaling results

![Throughput scaling](docs/plots/scaling_throughput.png)
![Parallel efficiency](docs/plots/scaling_efficiency.png)

The work-stealing scheduler showed lower throughput than static across all configurations, with steal rate of 0/s in all runs. This indicates that with 32 threads per worker and DAGs of 18–81 ops, no worker queue ran dry during execution — the steal threshold was never triggered. The efficiency degradation in work-stealing vs static reflects coordination overhead without the compensating benefit of load rebalancing. This suggests the current DAG sizes are too small relative to worker count to demonstrate work-stealing benefits at this scale. Larger DAGs (500+ ops) or reduced thread counts would create the queue exhaustion needed to trigger stealing.

---

## Narval setup

Load the verified module stack before building or submitting jobs:

```bash
module load StdEnv/2023 llvm openmpi rust/1.91.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/llvm21/openmpi/5.0.8/lib:$LD_LIBRARY_PATH
export CARGO_TARGET_DIR=$SCRATCH/ferroflow-target
cargo build --release --features distributed
```

Point `CARGO_TARGET_DIR` at `$SCRATCH` to avoid filling the `$HOME` quota with build artifacts. Submit jobs with `sbatch slurm/ferroflow.sh`, check status with `squeue -u $USER`, and check efficiency with `seff <job_id>` after completion.

See [docs/narval-setup.md](docs/narval-setup.md) for full environment notes and allocation guidance.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `cargo test --workspace` and `cargo clippy --workspace -- -D warnings` before opening a PR.

---

## License

MIT — see [LICENSE](LICENSE).
