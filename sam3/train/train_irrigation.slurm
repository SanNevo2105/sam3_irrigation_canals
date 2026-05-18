#!/bin/bash
#SBATCH -p cornell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -J sam3_irrigation
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ---- Setup ----
echo "Starting job on $(hostname)"
echo "Time: $(date)"

# Load your environment
# Load python39 module FIRST to ensure /usr/bin/python3 → Python 3.9 on all nodes.
# The venv was created with Python 3.9; without this, nodes where python3 defaults
# to a newer version (e.g. 3.12) will ignore the venv's site-packages entirely.
# module load python39
source /mnt/home/tkomolkhetruck/sam3_env/bin/activate

# Go to project directory
cd /mnt/home/tkomolkhetruck/sam3_irrigation_canals

# ---- Debug info (useful) ----
PYTHON=/mnt/home/tkomolkhetruck/sam3_env/bin/python
echo "Python binary: $PYTHON"
echo "Python version: $($PYTHON --version)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

# ---- Important: avoid DDP issues ----
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# ---- Optional: better error messages ----
export TORCH_DISTRIBUTED_DEBUG=INFO

# ---- Run training ----
$PYTHON sam3/train/train.py \
    --config configs/irrigation_canal/irrigation_canal_finetune

echo "Finished at: $(date)"
