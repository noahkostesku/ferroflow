#!/bin/bash
# GPU benchmark: CPU baseline vs A100 GPU for pure matmul DAGs.
# Uses matmul-parallel to guarantee real compute ops with no Slow placeholders,
# so the A100 has actual matmuls to accelerate.

#SBATCH --job-name=ferroflow-gpu
#SBATCH --account=def-cbravo
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gpu-bench-%j.out
#SBATCH --error=gpu-bench-%j.err

module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0

BINARY=/lustre06/project/6040457/noahkost/ferroflow/target/release/ferroflow
RESULTS=/lustre06/project/6040457/noahkost/ferroflow/docs/gpu_results.txt

echo "[gpu-bench] job ${SLURM_JOB_ID} started: $(date -Iseconds)"
echo "[gpu-bench] node: $(hostname)"
echo "[gpu-bench] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "[gpu-bench] CPUs per task: ${SLURM_CPUS_PER_TASK}"

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "$(dirname "${RESULTS}")"

# ── CPU baseline ──────────────────────────────────────────────────────────────
echo "" | tee -a "${RESULTS}"
echo "=== CPU baseline (job ${SLURM_JOB_ID}) ===" | tee -a "${RESULTS}"

"${BINARY}" run --dag matmul-parallel \
    --n-branches 32 --ops-per-branch 4 --matrix-size 256 \
    --workers 8 --scheduler work-stealing --device cpu \
    | tee -a "${RESULTS}"

# ── GPU (A100) ────────────────────────────────────────────────────────────────
echo "" | tee -a "${RESULTS}"
echo "=== GPU A100 (job ${SLURM_JOB_ID}) ===" | tee -a "${RESULTS}"

"${BINARY}" run --dag matmul-parallel \
    --n-branches 32 --ops-per-branch 4 --matrix-size 256 \
    --workers 8 --scheduler work-stealing --device cuda \
    | tee -a "${RESULTS}"

# ── GPU A100 larger matrices ──────────────────────────────────────────────────
echo "" | tee -a "${RESULTS}"
echo "=== GPU A100 larger matrices (job ${SLURM_JOB_ID}) ===" | tee -a "${RESULTS}"

"${BINARY}" run --dag matmul-parallel \
    --n-branches 16 --ops-per-branch 4 --matrix-size 512 \
    --workers 8 --scheduler work-stealing --device cuda \
    | tee -a "${RESULTS}"

# ── Auto routing — large matmul (should prefer GPU) ──────────────────────────
echo "" | tee -a "${RESULTS}"
echo "=== Auto routing - 2048x2048 (job ${SLURM_JOB_ID}) ===" | tee -a "${RESULTS}"

"${BINARY}" run --dag matmul-parallel \
    --n-branches 16 --ops-per-branch 8 --matrix-size 2048 \
    --workers 8 --scheduler work-stealing --device auto \
    | tee -a "${RESULTS}"

# ── Auto routing — ResNet-18 (conv2d→GPU, relu/add→CPU) ──────────────────────
echo "" | tee -a "${RESULTS}"
echo "=== Auto routing - ResNet-18 (job ${SLURM_JOB_ID}) ===" | tee -a "${RESULTS}"

"${BINARY}" run --model models/resnet18.onnx \
    --workers 8 --scheduler work-stealing --device auto \
    | tee -a "${RESULTS}"

echo "" | tee -a "${RESULTS}"
echo "[gpu-bench] finished: $(date -Iseconds)"
