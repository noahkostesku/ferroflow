#!/bin/bash
# ferroflow distributed work-stealing benchmark job.
# Default: 2 nodes, 1 rank/node.  Override node count at submission time:
#   sbatch --nodes=4 slurm/ferroflow.sh
# Or export FERROFLOW_NODES before calling this wrapper script.
#
# Set your Alliance Canada account before submitting:
#   export SLURM_ACCOUNT=def-yourpi
#
#SBATCH --job-name=ferroflow
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load StdEnv/2023 llvm openmpi rust/1.91.0

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/llvm21/openmpi/5.0.8/lib:${LD_LIBRARY_PATH}
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CARGO_TARGET_DIR=${SCRATCH}/ferroflow-target

cd "${SLURM_SUBMIT_DIR}"

echo "[ferroflow] job ${SLURM_JOB_ID}: ${SLURM_NNODES} nodes, ${SLURM_NTASKS} ranks, ${SLURM_CPUS_PER_TASK} threads/rank"
echo "[ferroflow] started: $(date -Iseconds)"

cargo build --release --features distributed 2>&1 | tail -5

BINARY="${CARGO_TARGET_DIR}/release/ferroflow"

srun --ntasks="${SLURM_NTASKS}" \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     "${BINARY}"

STATUS=$?
FINISH=$(date -Iseconds)

echo "[ferroflow] finished: ${FINISH}  exit=${STATUS}"

# Append a summary line to the benchmark log.
cat >> "${SLURM_SUBMIT_DIR}/docs/benchmarks.md" <<EOF

## ${FINISH}  job=${SLURM_JOB_ID}
- nodes: ${SLURM_NNODES}
- ranks: ${SLURM_NTASKS}  (${SLURM_NTASKS_PER_NODE} per node)
- threads/rank: ${SLURM_CPUS_PER_TASK}
- exit: ${STATUS}
- git: $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)
EOF
