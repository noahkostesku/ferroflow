#!/bin/bash
#SBATCH --job-name=ferroflow-bench
#SBATCH --account=<your-account>        # replace with your Alliance account
#SBATCH --time=01:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=%x-%j-out.txt
#SBATCH --error=%x-%j-err.txt

module load StdEnv/2023 gcc openmpi rust

export CARGO_TARGET_DIR=$SCRATCH/ferroflow-target
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
cargo build --release 2>&1 | tail -5

BINARY=$CARGO_TARGET_DIR/release/ferroflow

echo "Job $SLURM_JOB_ID: $SLURM_NNODES nodes, $SLURM_NTASKS ranks"
echo "Started: $(date)"

mpirun -n $SLURM_NTASKS \
    --map-by ppr:$SLURM_NTASKS_PER_NODE:node:pe=$SLURM_CPUS_PER_TASK \
    $BINARY

echo "Finished: $(date)"
