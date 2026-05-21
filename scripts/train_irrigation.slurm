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


# CHANGE THIS TO YOUR REPO PATH
REPO_DIR="/path/to/sam3_irrigation_canals"



module load python39
# export LD_LIBRARY_PATH=/cm/images/default-image/usr/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH="$REPO_DIR/local_lib:/cm/images/default-image/usr/lib64:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Checking libcrypt path:"
ls -lh /cm/images/default-image/usr/lib64/libcrypt.so.2* || true

PYTHON="$REPO_DIR/.venv/bin/python"
CONFIG_NAME="configs/irrigation_canal/irrigation_canal_finetune copy"

cd "$REPO_DIR"
mkdir -p logs

echo "Python binary: $PYTHON"
ldd "$PYTHON" || true

"$PYTHON" --version
"$PYTHON" -c "import sys; print('sys.executable:', sys.executable); print('sys.version:', sys.version)"
"$PYTHON" -m pip show torch
"$PYTHON" -m pip show submitit
"$PYTHON" -m pip show opencv-python-headless || true
"$PYTHON" -c "import torch; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
"$PYTHON" -c "import submitit; print('submitit:', submitit.__version__)"
"$PYTHON" -c "import cv2, einops, decord; print('extra imports OK')"

nvidia-smi

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m sam3.train.train \
    --config "$CONFIG_NAME"

echo "Finished at: $(date)"
