# ferroflow

Distributed work-stealing tensor computation scheduler for HPC clusters, written in Rust.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Static scheduling concentrates slow ops on specific workers, leaving others idle. ferroflow represents a tensor computation as a DAG and distributes it across nodes, dynamically stealing ops from overloaded workers when others run dry. It supports DAG execution, ONNX model import, a Python API via PyO3, a live terminal dashboard, and SLURM integration for Alliance Canada clusters.

---

## Results

| Workload | Device | Scheduler | Throughput | Advantage |
|---|---|---|---|---|
| Imbalanced DAG (2 nodes) | CPU | work-stealing | 1,014 ops/s | +12.7% vs static |
| XLarge Wide DAG (8 nodes) | CPU | work-stealing | 28,839 ops/s | 97% efficiency |
| Matmul-parallel 2048×2048 | CPU (8 workers) | work-stealing | 25 ops/s | baseline |
| Matmul-parallel 2048×2048 | A100 GPU | work-stealing | 62 ops/s | 2.5× faster than CPU |

Narval supercomputer (Alliance Canada).
AMD EPYC Milan (CPU) / A100-SXM4-40GB (GPU).
5-run medians for scaling study, single run for GPU benchmark.

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

| ONNX Op | ferroflow OpKind | Status |
|---|---|---|
| MatMul | Matmul | ✓ |
| Gemm | Matmul | ✓ |
| Relu | Relu | ✓ |
| LayerNormalization | LayerNorm | ✓ |
| ReduceMean | Reduce | ✓ |
| GlobalAveragePool | Reduce | ✓ |
| Softmax | Softmax | ✓ |
| BatchNormalization | BatchNorm | ✓ |
| Conv | Conv2d | ✓ |
| MaxPool | — | planned |
| Add | — | planned |
| Flatten | — | planned |

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
![Steal rate](docs/plots/steal_rate.png)

Narval, job 59826638. Commit 3709ffc.

### XLarge Wide DAG (1281 ops, width=1280 depth=1 skew=0.003)

| Nodes | Static (ops/s) | WS (ops/s) | WS Efficiency | Steal Rate |
|-------|----------------|------------|---------------|------------|
| 2 | 7,387 | 7,431 | 1.000 | 34.8/s |
| 4 | 14,443 | 14,777 | 0.994 | 92.4/s |
| 8 | 27,631 | 28,839 | 0.970 | 180.2/s |

### XLarge Transformer DAG (545 ops, 32 layers, d=512)

| Nodes | Static (ops/s) | WS (ops/s) | WS Efficiency | Steal Rate |
|-------|----------------|------------|---------------|------------|
| 2 | 581 | 578 | 1.000 | 0.0/s |
| 4 | 581 | 583 | 0.504 | 7.6/s |
| 8 | 578 | 574 | 0.248 | 7.5/s |

### Imbalanced DAG (205 ops, 4 heavy×200ms + 200 fast×1ms)

| Nodes | Static (ops/s) | WS (ops/s) | WS Efficiency | Steal Rate |
|-------|----------------|------------|---------------|------------|
| 2 | 900 | 1,014 | 1.000 | 477.0/s |
| 4 | 958 | 1,013 | 0.500 | 218.6/s |
| 8 | 988 | 1,012 | 0.250 | 99.3/s |

On parallel skewed workloads (xlarge-wide, 1281 ops), work-stealing maintains 97% parallel efficiency at 8 nodes with steal rate scaling from 34/s at 2 nodes to 180/s at 8 nodes, outperforming static at every node count. On imbalanced workloads where round-robin concentrates slow chains on specific workers, work-stealing recovers dynamically with 477 steals/sec and 12.7% higher throughput than static at 2 nodes. Serial dependency chains (transformer DAG) plateau at ~580 ops/s regardless of scheduler or node count, correctly identifying DAG depth as the throughput ceiling.

---

### GPU Acceleration (A100)

ferroflow routes matmul ops to GPU via cuBLAS when `--device cuda` is set. Other op types execute on CPU with automatic tensor transfer.

| Matrix Size | CPU (8 workers) | A100 GPU | Speedup |
|---|---|---|---|
| 2048×2048 | 25 ops/s | 62 ops/s | 2.5× |
| 4096×4096 | — | 17 ops/s | — |

The 2.5× speedup reflects per-op CPU↔GPU transfer overhead — each op moves tensors to GPU, computes via cuBLAS, and returns results to CPU for dependency tracking. Eliminating this transfer overhead through persistent GPU tensor storage is the next optimization target (see roadmap).

To run with GPU acceleration:

```bash
# Build with CUDA support
module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0
cargo build --release --features cuda

# Run on GPU
ferroflow run --dag matmul-parallel \
  --n-branches 16 --ops-per-branch 8 --matrix-size 2048 \
  --workers 8 --device cuda
```

Requires: CUDA 12.x, A100 or compatible GPU.

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

## Roadmap

| Issue | Description | Status |
|---|---|---|
| #1 | Larger DAGs (500+ ops) to validate work-stealing at scale | ✅ closed |
| #2 | Adaptive steal threshold | ✅ closed |
| #4 | Additional ONNX ops (Conv2d, BatchNorm, Softmax, Add, MaxPool, Reshape) | ✅ closed |
| #8 | CUDA support via cudarc (Narval A100s) | ✅ closed |
| #3 | Persistent GPU tensor storage (eliminate CPU↔GPU transfer overhead) | open |
| #5 | Peer-to-peer work stealing (remove coordinator bottleneck) | open |
| #6 | Heterogeneous CPU+GPU scheduling | open |
| #7 | Multi-cluster benchmarks (Fir H100s, Rorqual) | open |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `cargo test --workspace` and `cargo clippy --workspace -- -D warnings` before opening a PR.

---

## License

MIT — see [LICENSE](LICENSE).
