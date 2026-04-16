# Narval Setup Guide — ferroflow

Narval is an Alliance Canada (formerly Compute Canada) cluster at ÉTS Montréal.
This guide covers everything needed to build and run ferroflow on the login/compute nodes.

---

## 1. Loading Rust and MPI Modules

Narval uses the `StdEnv/2023` software stack. Load the required modules in
your `~/.bashrc` or at the top of every job script:

```bash
module load StdEnv/2023 gcc openmpi rust
```

Verify the loaded versions:
```bash
rustc --version       # should be 1.70+ (check available: module spider rust)
mpirun --version      # OpenMPI
```

If a newer Rust toolchain is needed, install via rustup after loading the base
module — Alliance clusters allow user-space rustup installs in `$HOME`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
```

> **Note:** `$HOME` has a small quota. Keep the rustup toolchain in
> `$HOME/.cargo` but point `CARGO_TARGET_DIR` at `$SCRATCH` to avoid filling
> your home quota with build artifacts:
> ```bash
> export CARGO_TARGET_DIR=$SCRATCH/ferroflow-target
> ```

---

## 2. Installing Claude Code on the Login Node

Claude Code runs as a Node.js CLI. Alliance login nodes have internet access,
so a user-space install is straightforward:

```bash
# Load Node.js (or use a user install)
module load nodejs

# Install Claude Code globally in user prefix
npm install -g @anthropic-ai/claude-code

# Verify
claude --version
```

If `npm` global installs go to a non-writable path, set a user prefix first:
```bash
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH="$HOME/.npm-global/bin:$PATH"   # add to ~/.bashrc
npm install -g @anthropic-ai/claude-code
```

Set your API key in `~/.bashrc`:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 3. tmux Workflow for Persistent Sessions

SSH connections to Narval drop when your local machine sleeps or you lose
network. Use `tmux` to keep sessions alive across disconnects.

### Start a named session
```bash
ssh narval.alliancecan.ca
tmux new -s ferroflow
```

### Detach and reattach
```bash
# Detach from current session (stays running on the server)
Ctrl-b d

# Later, reattach
ssh narval.alliancecan.ca
tmux attach -t ferroflow
```

### Suggested tmux layout
```
Window 0: editor / claude code session
Window 1: cargo build / test output
Window 2: squeue / job monitoring
Window 3: $SCRATCH output inspection
```

Create windows with `Ctrl-b c`, switch with `Ctrl-b <number>`.

### Persistent tmux config (`~/.tmux.conf`)
```
set -g mouse on
set -g history-limit 50000
set -g default-terminal "screen-256color"
bind r source-file ~/.tmux.conf \; display "Config reloaded"
```

---

## 4. Sample sbatch Script — Multi-Node ferroflow Job

Save as `slurm/ferroflow.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ferroflow-bench
#SBATCH --account=<your-account>         # replace with your Alliance account
#SBATCH --time=01:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2              # 2 MPI ranks per node
#SBATCH --cpus-per-task=32              # 32 cores per rank (64-core Narval nodes)
#SBATCH --mem=0                          # use all available memory
#SBATCH --output=$SCRATCH/ferroflow-logs/%j-out.txt
#SBATCH --error=$SCRATCH/ferroflow-logs/%j-err.txt

# --- Environment setup ---
module load StdEnv/2023 gcc openmpi rust
export CARGO_TARGET_DIR=$SCRATCH/ferroflow-target
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Build (if not pre-built) ---
cd $SLURM_SUBMIT_DIR
cargo build --release 2>&1 | tail -5

BINARY=$CARGO_TARGET_DIR/release/ferroflow

# --- Run ---
mkdir -p $SCRATCH/ferroflow-logs

echo "Job $SLURM_JOB_ID: $SLURM_NNODES nodes, $SLURM_NTASKS ranks"
echo "Started: $(date)"

mpirun -n $SLURM_NTASKS \
    --map-by ppr:$SLURM_NTASKS_PER_NODE:node:pe=$SLURM_CPUS_PER_TASK \
    $BINARY \
    --scheduler work-stealing \
    --dag-file $SLURM_SUBMIT_DIR/data/bench-skewed.json \
    --output $SCRATCH/ferroflow-logs/$SLURM_JOB_ID-results.json

echo "Finished: $(date)"
```

### Submitting and monitoring
```bash
sbatch slurm/ferroflow.sh              # submit
squeue -u $USER                        # check queue
seff $SLURM_JOB_ID                     # check efficiency after completion
tail -f $SCRATCH/ferroflow-logs/<job-id>-out.txt  # stream live output
```

### Development job (interactive, small scale)
```bash
salloc --account=<account> --time=0:30:00 --nodes=2 --ntasks-per-node=2 \
       --cpus-per-task=4 --mem=16G
# Once allocated, you're on a compute node — run mpirun directly:
module load StdEnv/2023 gcc openmpi rust
mpirun -n 4 ./target/release/ferroflow --scheduler work-stealing
```
