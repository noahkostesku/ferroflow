#!/bin/bash
#SBATCH --job-name=ferroflow-nibi-gpu
#SBATCH --account=def-cbravo
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=nibi-gpu-bench-%j.out
#SBATCH --error=nibi-gpu-bench-%j.err

module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/lib64:${LD_LIBRARY_PATH}

BINARY=/home/noahkost/ferroflow/target/release/ferroflow
RESULTS=/home/noahkost/ferroflow/docs/nibi_gpu_results.txt

echo "[nibi-gpu] job ${SLURM_JOB_ID} started: $(date -Iseconds)"
echo "[nibi-gpu] node: $(hostname)"
echo "[nibi-gpu] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p "$(dirname "${RESULTS}")"

echo "=== CPU baseline 2048x2048 ===" | tee -a "${RESULTS}"
"${BINARY}" run --dag matmul-parallel \
    --n-branches 16 --ops-per-branch 8 --matrix-size 2048 \
    --workers 8 --scheduler work-stealing --device cpu \
    | tee -a "${RESULTS}"

echo "=== H100 GPU 256x256 ===" | tee -a "${RESULTS}"
"${BINARY}" run --dag matmul-parallel \
    --n-branches 32 --ops-per-branch 4 --matrix-size 256 \
    --workers 8 --scheduler work-stealing --device cuda \
    | tee -a "${RESULTS}"

echo "=== H100 GPU 2048x2048 ===" | tee -a "${RESULTS}"
"${BINARY}" run --dag matmul-parallel \
    --n-branches 16 --ops-per-branch 8 --matrix-size 2048 \
    --workers 8 --scheduler work-stealing --device cuda \
    | tee -a "${RESULTS}"

echo "=== H100 GPU auto routing 2048x2048 ===" | tee -a "${RESULTS}"
"${BINARY}" run --dag matmul-parallel \
    --n-branches 16 --ops-per-branch 8 --matrix-size 2048 \
    --workers 8 --scheduler work-stealing --device auto \
    | tee -a "${RESULTS}"

echo "[nibi-gpu] finished: $(date -Iseconds)"