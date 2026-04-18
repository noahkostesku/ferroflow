# Changelog

## [0.1.0] - 2026-04-17

### Added
- DAG representation with topological sort and dependency tracking
- Op types: Matmul, Relu, LayerNorm, Reduce, Slow (for benchmarking)
- SequentialExecutor: single-node baseline
- StaticScheduler: round-robin work distribution
- WorkStealingScheduler: pull-based dynamic load balancing
- MPI worker bridge for multi-node distributed execution
- DAG generators: transformer, large-transformer, wide, large-wide, imbalanced
- ONNX model import (MatMul, Gemm, Relu, LayerNorm, ReduceMean)
- Python API via PyO3/maturin
- Live terminal dashboard via ratatui
- SLURM job scripts for Narval (Alliance Canada)
- Strong scaling study: 2–8 nodes, 5-run medians
- Benchmark: 12.7% WS throughput advantage on imbalanced workloads
