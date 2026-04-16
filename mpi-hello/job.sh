#!/bin/bash
#SBATCH --job-name=mpi-hello
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --account=def-cbravo

module load StdEnv/2023 llvm openmpi rust/1.91.0

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/llvm21/openmpi/5.0.8/lib:$LD_LIBRARY_PATH

srun /lustre06/project/6040457/noahkost/ferroflow/target/release/mpi-hello
