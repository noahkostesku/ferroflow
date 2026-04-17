#!/bin/bash
# ferroflow multi-configuration benchmark job.
#
# Runs three schedulers (sequential, static, work-stealing) across four
# configurations (2-node/4-node × uniform/skewed DAG) and records RunMetrics
# to docs/benchmark_results.json.  Prints a markdown comparison table when done.
#
# Usage:
#   sbatch slurm/benchmark.sh
#
#SBATCH --job-name=ferroflow-bench
#SBATCH --account=def-cbravo
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
RESULTS_JSON="${SUBMIT_DIR}/docs/benchmark_results.json"

cd "${SUBMIT_DIR}"

echo "================================================================"
echo "[ferroflow-bench] job=${SLURM_JOB_ID}"
echo "[ferroflow-bench] nodes=${SLURM_NNODES}  cpus/task=${SLURM_CPUS_PER_TASK}"
echo "[ferroflow-bench] started: $(date -Iseconds)"
echo "================================================================"

# ── Build ─────────────────────────────────────────────────────────────────────

echo "[ferroflow-bench] building release binary..."
cargo build --release -j"${SLURM_CPUS_PER_TASK}" 2>&1 | tail -5

BINARY="${CARGO_TARGET_DIR}/release/ferroflow"
echo "[ferroflow-bench] binary: ${BINARY}"

# ── Configurations ────────────────────────────────────────────────────────────
# Each entry: "NODES DAG_KIND"
# NODES is used for both --workers and --nodes so the RunMetrics records the
# simulated node count.  (Full MPI distribution pending Week 2 Narval runs.)

CONFIGS=(
    "2 uniform"
    "2 skewed"
    "4 uniform"
    "4 skewed"
)

DAG_OPS=20
SKEW_FACTOR=5
CHAIN_DIM=128

echo ""
echo "── Benchmark runs ──────────────────────────────────────────────"

for config in "${CONFIGS[@]}"; do
    N_NODES=$(echo "${config}" | awk '{print $1}')
    DAG_KIND=$(echo "${config}" | awk '{print $2}')

    echo ""
    echo "[ferroflow-bench] config: ${N_NODES} nodes / ${DAG_KIND} DAG"

    "${BINARY}" bench \
        --dag         "${DAG_KIND}" \
        --workers     "${N_NODES}" \
        --nodes       "${N_NODES}" \
        --dag-ops     "${DAG_OPS}" \
        --skew-factor "${SKEW_FACTOR}" \
        --chain-dim   "${CHAIN_DIM}" \
        --output      "${RESULTS_JSON}"

    echo "[ferroflow-bench] config ${N_NODES}n/${DAG_KIND} done"
done

echo ""
echo "── Comparison table ────────────────────────────────────────────"
"${BINARY}" report "${RESULTS_JSON}"

echo ""
echo "── Updating benchmarks.md ──────────────────────────────────────"
FINISH=$(date -Iseconds)
GIT_SHA=$(git -C "${SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)

cat >> "${SUBMIT_DIR}/docs/benchmarks.md" <<EOF

---

### ${FINISH} — Narval multi-node benchmark (job ${SLURM_JOB_ID})

- **Machine:** Narval (Alliance Canada)  |  **Job:** ${SLURM_JOB_ID}
- **Nodes allocated:** ${SLURM_NNODES}  |  **CPUs/task:** ${SLURM_CPUS_PER_TASK}
- **Commit:** ${GIT_SHA}
- **DAG:** ${DAG_OPS}-op uniform (128×128 matmul) and skewed (Slow × ${SKEW_FACTOR})
- **Configs tested:** 2-node and 4-node worker counts
- **Full RunMetrics:** \`docs/benchmark_results.json\`

$(${BINARY} report "${RESULTS_JSON}" 2>/dev/null)
EOF

echo "[ferroflow-bench] benchmarks.md updated"
echo "[ferroflow-bench] finished: ${FINISH}  exit=0"
echo "================================================================"
