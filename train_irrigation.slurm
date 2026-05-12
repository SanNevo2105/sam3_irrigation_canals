#!/bin/bash
# ==============================================================================
# Generic SLURM job script for SAM3 irrigation canal fine-tuning
# Edit every line marked with  # USER:  before submitting.
# Submit with:  sbatch train_irrigation.slurm
# ==============================================================================

# --- Resource requests --------------------------------------------------------
#SBATCH -p cornell                  # USER: replace with your cluster partition
#SBATCH --gres=gpu:1                # USER: number of GPUs (match launcher.gpus_per_node in config)
#SBATCH --cpus-per-task=8           # USER: CPUs per task (match submitit.cpus_per_task in config)
#SBATCH --mem=32G                   # USER: RAM per node
#SBATCH --time=72:00:00             # USER: wall-clock limit (HH:MM:SS)
#SBATCH -J sam3_irrigation          # job name shown in squeue
#SBATCH -o logs/%x-%j.out           # stdout → logs/<jobname>-<jobid>.out
#SBATCH -e logs/%x-%j.err           # stderr → logs/<jobname>-<jobid>.err

# --- Environment --------------------------------------------------------------
echo "Starting job on $(hostname)"
echo "Time: $(date)"

# USER: activate your virtual environment
source ~/.bashrc
source sam3_env/bin/activate        # USER: adjust path if your venv is elsewhere

# USER: cd to the repository root (where you cloned sam3_irrigation_canals)
cd sam3_irrigation_canals           # USER: replace with the actual path if needed

# --- Sanity checks ------------------------------------------------------------
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

# --- Distributed setup (single-node, single-GPU) ------------------------------
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Optional: verbose distributed debug output
export TORCH_DISTRIBUTED_DEBUG=INFO

# --- Create log directory if it does not exist --------------------------------
mkdir -p logs

# --- Run training -------------------------------------------------------------
python sam3/train/train.py \
    -c configs/irrigation/irrigation_finetune

echo "Finished at: $(date)"
