#!/bin/bash
#SBATCH -p cornell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -J sam3_irrigation
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -e

echo "Starting job on $(hostname)"
echo "Time: $(date)"

REPO_DIR="/mnt/home/tkomolkhetruck/sam3_irrigation_canals"
PYTHON="/mnt/home/tkomolkhetruck/sam3_env/bin/python3.9"
CONFIG_NAME="configs/irrigation_canal/irrigation_canal_finetune"

source /mnt/home/tkomolkhetruck/sam3_env/bin/activate

cd "$REPO_DIR"
mkdir -p logs

echo "Python binary: $PYTHON"
ls -l "$PYTHON"
readlink -f "$PYTHON" || true

"$PYTHON" --version
"$PYTHON" -c "import sys; print('sys.executable:', sys.executable); print('sys.version:', sys.version)"
"$PYTHON" -m pip show torch
"$PYTHON" -m pip show submitit
"$PYTHON" -c "import torch; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
"$PYTHON" -c "import submitit; print('submitit:', submitit.__version__)"

nvidia-smi

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONUNBUFFERED=1

"$PYTHON" -m sam3.train.train \
    --config "$CONFIG_NAME"

echo "Finished at: $(date)"