#!/bin/bash
set -e

ENV_DIR="${1:-sam3_env}"
PYTHON_BIN="${PYTHON_BIN:-python3.9}"

"$PYTHON_BIN" -m venv --copies "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train]"

python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import submitit; print('submitit:', submitit.__version__)"