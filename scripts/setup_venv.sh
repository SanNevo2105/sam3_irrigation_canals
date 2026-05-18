#!/usr/bin/env bash
# Create a local virtual environment for SAM3 training.
#
# Usage:
#   bash scripts/setup_venv.sh
#   bash scripts/setup_venv.sh .venv
#
# Optional environment variables:
#   PYTHON_BIN=python3.9
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#
# Example:
#   PYTHON_BIN=python3.9 bash scripts/setup_venv.sh .venv

set -euo pipefail

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.9}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "============================================================"
echo "SAM3 venv setup"
echo "============================================================"
echo "Workspace : $(pwd)"
echo "ENV_DIR   : ${ENV_DIR}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "Torch URL : ${TORCH_INDEX_URL}"
echo "============================================================"

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found."
    echo "Run this script from the repository root."
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} not found in PATH."
    echo "Try one of:"
    echo "  module load python39"
    echo "  PYTHON_BIN=/full/path/to/python3.9 bash scripts/setup_venv.sh .venv"
    exit 1
fi

echo "[1/6] Python used to create venv:"
"${PYTHON_BIN}" -c 'import sys; print(sys.executable); print(sys.version)'

if [ -d "${ENV_DIR}" ]; then
    echo "ERROR: ${ENV_DIR} already exists."
    echo "Remove it first if you want a fresh environment:"
    echo "  rm -rf ${ENV_DIR}"
    exit 1
fi

echo "[2/6] Creating venv with copied binaries..."
"${PYTHON_BIN}" -m venv --copies "${ENV_DIR}"

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"

echo "[3/6] Venv Python:"
python -c 'import sys; print(sys.executable); print(sys.version)'

echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "[5/6] Installing PyTorch..."
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

echo "[6/6] Installing SAM3 training dependencies..."
python -m pip install -e ".[train]"

echo "============================================================"
echo "Verification"
echo "============================================================"
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'torch CUDA:', torch.version.cuda)"
python -c "import submitit; print('submitit:', submitit.__version__)"
python -c "import sam3; print('sam3:', sam3.__file__)"

echo "============================================================"
echo "Venv binary checks"
echo "============================================================"
ls -l "${ENV_DIR}/bin/python"* || true
echo "Resolved python:"
readlink -f "${ENV_DIR}/bin/python" || true

echo "Shared-library check:"
if command -v ldd >/dev/null 2>&1; then
    ldd "${ENV_DIR}/bin/python" || true
else
    echo "ldd not available on this system."
fi

cat > "${ENV_DIR}/ENVIRONMENT_NOTES.txt" <<EOF
This virtual environment was created with:

  PYTHON_BIN=${PYTHON_BIN}
  TORCH_INDEX_URL=${TORCH_INDEX_URL}
  created_at=$(date)
  repo=$(pwd)

Important cluster note:
  This venv is tied to the Python major/minor version used to create it.
  If this Python binary needs libpythonX.Y.so and Slurm compute nodes cannot
  find that shared library, load the matching Python module in your Slurm file
  before running ${ENV_DIR}/bin/python.

Recommended Slurm pattern:
  module load python39   # or the module matching PYTHON_BIN, if required
  PYTHON="\$SLURM_SUBMIT_DIR/${ENV_DIR}/bin/python"
  "\$PYTHON" -m sam3.train.train --config configs/irrigation_canal/irrigation_canal_finetune
EOF

echo "============================================================"
echo "Done."
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
echo
echo "For Slurm, prefer:"
echo "  PYTHON=\"/absolute/path/to/${ENV_DIR}/bin/python\""
echo "  \"\$PYTHON\" -m sam3.train.train --config configs/irrigation_canal/irrigation_canal_finetune"
echo "============================================================"
