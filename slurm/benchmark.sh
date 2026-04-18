#!/bin/bash
# ferroflow MPI distributed benchmark job.
#
# Runs static and work-stealing schedulers across four configurations
# (2-node/4-node × uniform/skewed DAG) using real MPI ranks and records
# RunMetrics to docs/benchmark_results_mpi.json.  Prints a markdown
# comparison table when all runs complete.
#
# Build requirement: cargo build --release --features distributed
#
# Usage:
#   export SLURM_ACCOUNT=def-yourpi
#   sbatch slurm/benchmark.sh
#
#SBATCH --job-name=ferroflow-bench
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=ferroflow-bench-%j.out
#SBATCH --error=ferroflow-bench-%j.err

set -euo pipefail

module load StdEnv/2023 llvm openmpi rust/1.91.0

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/llvm21/openmpi/5.0.8/lib:${LD_LIBRARY_PATH}
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CARGO_TARGET_DIR=${SCRATCH}/ferroflow-target

SUBMIT_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_JSON="${SUBMIT_DIR}/docs/benchmark_results_mpi.json"

cd "${SUBMIT_DIR}"

echo "================================================================"
echo "[ferroflow-bench] job=${SLURM_JOB_ID}"
echo "[ferroflow-bench] nodes=${SLURM_NNODES}  cpus/task=${SLURM_CPUS_PER_TASK}"
echo "[ferroflow-bench] started: $(date -Iseconds)"
echo "================================================================"

# ── Build ─────────────────────────────────────────────────────────────────────

echo "[ferroflow-bench] building release binary (--features distributed)..."
cargo build --release --features distributed -j"${SLURM_CPUS_PER_TASK}" 2>&1 | tail -5

BINARY="${CARGO_TARGET_DIR}/release/ferroflow"
echo "[ferroflow-bench] binary: ${BINARY}"

# ── MPI benchmark parameters ──────────────────────────────────────────────────

N_OPS=20
OP_DURATION_MS=50   # wall-clock ms per op on the fast branch
SKEW_FACTOR=5       # slow branch = 50 ms * 5 = 250 ms per op

# ── Helper: run one srun invocation ──────────────────────────────────────────
# Usage: run_mpi_bench <n_nodes> <dag_kind> <scheduler>
run_mpi_bench() {
    local n_nodes=$1
    local dag_kind=$2
    local scheduler=$3

    echo ""
    echo "[ferroflow-bench] srun: ${n_nodes} nodes / ${dag_kind} DAG / ${scheduler}"

    srun \
        --nodes="${n_nodes}" \
        --ntasks="${n_nodes}" \
        --ntasks-per-node=1 \
        "${BINARY}" mpi-bench \
            --dag           "${dag_kind}" \
            --scheduler     "${scheduler}" \
            --n-ops         "${N_OPS}" \
            --op-duration-ms "${OP_DURATION_MS}" \
            --skew-factor   "${SKEW_FACTOR}" \
            --nodes         "${n_nodes}" \
            --output        "${RESULTS_JSON}"

    echo "[ferroflow-bench] done: ${n_nodes}n/${dag_kind}/${scheduler}"
}

echo ""
echo "── MPI benchmark runs ──────────────────────────────────────────"

# Config 1: 2 nodes, uniform DAG — static then work-stealing
run_mpi_bench 2 uniform static
run_mpi_bench 2 uniform work-stealing

# Config 2: 2 nodes, skewed DAG — static then work-stealing
run_mpi_bench 2 skewed static
run_mpi_bench 2 skewed work-stealing

# Config 3: 4 nodes, uniform DAG — static then work-stealing
run_mpi_bench 4 uniform static
run_mpi_bench 4 uniform work-stealing

# Config 4: 4 nodes, skewed DAG — static then work-stealing
run_mpi_bench 4 skewed static
run_mpi_bench 4 skewed work-stealing

echo ""
echo "── Comparison table ────────────────────────────────────────────"
"${BINARY}" report "${RESULTS_JSON}"

echo ""
echo "── Updating benchmarks.md ──────────────────────────────────────"
FINISH=$(date -Iseconds)
GIT_SHA=$(git -C "${SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)

cat >> "${SUBMIT_DIR}/docs/benchmarks.md" <<EOF

---

### ${FINISH} — Narval MPI distributed benchmark (job ${SLURM_JOB_ID})

- **Machine:** Narval (Alliance Canada)  |  **Job:** ${SLURM_JOB_ID}
- **Nodes allocated:** ${SLURM_NNODES}  |  **CPUs/task:** ${SLURM_CPUS_PER_TASK}
- **Commit:** ${GIT_SHA}
- **DAG:** ${N_OPS}-op uniform (Slow(${OP_DURATION_MS} ms)) and skewed (Slow × ${SKEW_FACTOR})
- **Configs tested:** 2-node and 4-node × static and work-stealing
- **Full RunMetrics:** \`docs/benchmark_results_mpi.json\`

$(${BINARY} report "${RESULTS_JSON}" 2>/dev/null)
EOF

echo "[ferroflow-bench] benchmarks.md updated"
echo "[ferroflow-bench] finished: ${FINISH}  exit=0"
echo "================================================================"
