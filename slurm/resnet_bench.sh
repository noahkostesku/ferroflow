#!/bin/bash
#SBATCH --job-name=ferroflow-resnet
#SBATCH --account=def-cbravo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=resnet-bench-%j.out
#SBATCH --error=resnet-bench-%j.err

module load StdEnv/2023 llvm openmpi rust/1.91.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/llvm21/openmpi/5.0.8/lib:${LD_LIBRARY_PATH}

BINARY=/lustre06/project/6040457/noahkost/ferroflow/target/release/ferroflow
RESULTS=/lustre06/project/6040457/noahkost/ferroflow/docs/resnet18_results.txt

echo "[resnet-bench] job ${SLURM_JOB_ID} started: $(date -Iseconds)"
echo "[resnet-bench] node: $(hostname)"

echo "=== ferroflow info ===" | tee ${RESULTS}
${BINARY} info --model models/resnet18.onnx | tee -a ${RESULTS}

echo "=== static scheduler ===" | tee -a ${RESULTS}
${BINARY} run --model models/resnet18.onnx \
    --workers 8 --scheduler static | tee -a ${RESULTS}

echo "=== work-stealing scheduler ===" | tee -a ${RESULTS}
${BINARY} run --model models/resnet18.onnx \
    --workers 8 --scheduler work-stealing | tee -a ${RESULTS}

echo "[resnet-bench] finished: $(date -Iseconds)"