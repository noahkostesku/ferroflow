# Contributing to ferroflow

## Development setup

```bash
cargo build --workspace
cargo test --workspace
```

## MPI development

Requires OpenMPI. On Narval:

```bash
module load StdEnv/2023 llvm openmpi rust/1.91.0
cargo build --features distributed
```

## Running benchmarks locally

```bash
ferroflow run --dag imbalanced --workers 8 --scheduler work-stealing
ferroflow run --dag imbalanced --workers 8 --scheduler static
```

## Submitting changes

- Run `cargo clippy --workspace -- -D warnings` (must pass)
- Run `cargo fmt --all` (must be clean)
- Run `cargo test --workspace` (must pass)
- Open a PR with a clear description of what changed and why

## Areas for contribution

- GPU support via cudarc (A100 on Narval)
- Adaptive steal threshold
- Additional ONNX op support
- Larger scaling study (16+ nodes)
