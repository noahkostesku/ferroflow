#!/bin/bash
# Set your Alliance Canada account before submitting:
#   export SLURM_ACCOUNT=def-yourpi
#SBATCH --job-name=mpi-hello
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --account=${SLURM_ACCOUNT}

module load StdEnv/2023 llvm openmpi rust/1.91.0

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/llvm21/openmpi/5.0.8/lib:$LD_LIBRARY_PATH
export CARGO_TARGET_DIR=${SCRATCH}/ferroflow-target

srun "${CARGO_TARGET_DIR}/release/mpi-hello"
