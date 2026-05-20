#!/usr/bin/env bash
# setup_venv.sh
#
# Create a local virtual environment for SAM3 training.
#
# Usage:
#   bash scripts/setup_venv.sh
#   bash scripts/setup_venv.sh .venv
#
# Optional environment variables:
#   PYTHON_BIN=python3.9
#   PYTHON_MODULE=python39
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#
# Examples:
#   bash scripts/setup_venv.sh .venv
#   PYTHON_BIN=python3.9 bash scripts/setup_venv.sh .venv
#   PYTHON_MODULE=python39 PYTHON_BIN=python3.9 bash scripts/setup_venv.sh .venv
#
# Notes:
# - Run this from the repository root.
# - Do not copy .venv between machines.
# - On clusters/cloud VMs, create the venv on the same machine/filesystem used for training.
# - Uses `venv --copies` to avoid fragile symlinks to system Python.
# - Uses --no-cache-dir to avoid filling small disks with CUDA wheel caches.
# - Pins setuptools<80 because SAM3 imports pkg_resources.

set -euo pipefail

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.9}"
PYTHON_MODULE="${PYTHON_MODULE:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
SETUPTOOLS_VERSION="${SETUPTOOLS_VERSION:-setuptools<80}"

# Optional module loading. Do not hardcode a module name because clusters differ.
if [ -n "${PYTHON_MODULE}" ]; then
    if command -v module >/dev/null 2>&1; then
        echo "Loading Python module: ${PYTHON_MODULE}"
        module load "${PYTHON_MODULE}"
    else
        echo "ERROR: PYTHON_MODULE=${PYTHON_MODULE}, but the 'module' command is not available."
        exit 1
    fi
fi

echo "============================================================"
echo "SAM3 virtual environment setup"
echo "============================================================"
echo "Repository     : $(pwd)"
echo "ENV_DIR        : ${ENV_DIR}"
echo "PYTHON_BIN     : ${PYTHON_BIN}"
echo "PYTHON_MODULE  : ${PYTHON_MODULE:-none}"
echo "Torch URL      : ${TORCH_INDEX_URL}"
echo "Setuptools     : ${SETUPTOOLS_VERSION}"
echo "============================================================"

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found."
    echo "Run this script from the repository root."
    exit 1
fi

if [ -d "${ENV_DIR}" ]; then
    echo "ERROR: ${ENV_DIR} already exists."
    echo "Remove it first if you want a fresh environment:"
    echo "  rm -rf ${ENV_DIR}"
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Could not find Python executable: ${PYTHON_BIN}"
    echo
    echo "Try one of:"
    echo "  PYTHON_MODULE=python39 PYTHON_BIN=python3.9 bash scripts/setup_venv.sh .venv"
    echo "  PYTHON_BIN=/full/path/to/python3.9 bash scripts/setup_venv.sh .venv"
    exit 1
fi

echo "[1/8] Checking Python version..."
"${PYTHON_BIN}" - <<'PY'
import sys
major, minor = sys.version_info[:2]

print("Python executable:", sys.executable)
print("Python version   :", sys.version)

if major != 3:
    raise SystemExit("ERROR: Python 3 is required.")

if minor < 9:
    raise SystemExit(f"ERROR: Python 3.{minor} is too old. Use Python 3.9 or newer.")

if minor >= 13:
    print("WARNING: Python 3.13+ may not be supported by all ML dependencies yet. Python 3.9-3.12 is safer.")
PY

echo "[2/8] Creating venv with copied Python binaries..."
"${PYTHON_BIN}" -m venv --copies "${ENV_DIR}"

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"

echo "[3/8] Verifying venv Python..."
python - <<'PY'
import sys
print("Venv executable:", sys.executable)
print("Venv version   :", sys.version)
PY

echo "[4/8] Upgrading pip/wheel and installing pkg_resources-compatible setuptools..."
python -m pip install --upgrade --no-cache-dir pip wheel
python -m pip install --upgrade --no-cache-dir "${SETUPTOOLS_VERSION}"

echo "[5/8] Checking pkg_resources..."
python - <<'PY'
import pkg_resources
print("pkg_resources:", pkg_resources.__file__)
PY

echo "[6/8] Installing PyTorch..."
python -m pip install --no-cache-dir torch torchvision --index-url "${TORCH_INDEX_URL}"

echo "[7/8] Installing SAM3 training + runtime dependencies..."
python -m pip install --no-cache-dir -e ".[train]"

# Extra packages that SAM3 training/evaluation paths may import but may not be pulled
# reliably by the train extra in this fork.
python -m pip install --no-cache-dir \
  "${SETUPTOOLS_VERSION}" \
  "numpy>=1.26,<2" \
  "opencv-python-headless==4.10.0.84" \
  einops \
  decord \
  pycocotools \
  psutil

echo "[8/8] Running import checks..."
python - <<'PY'
import sys
import pkg_resources
import torch
import submitit
import sam3
import cv2
import einops
import decord
import numpy as np
import pycocotools
import psutil

print("Python executable:", sys.executable)
print("pkg_resources    :", pkg_resources.__file__)
print("torch            :", torch.__version__)
print("torch CUDA build :", torch.version.cuda)
print("CUDA available   :", torch.cuda.is_available())
print("submitit         :", submitit.__version__)
print("sam3             :", sam3.__file__)
print("cv2              :", cv2.__version__)
print("einops           :", einops.__version__)
print("decord           :", decord.__version__)
print("numpy            :", np.__version__)
print("pycocotools      :", getattr(pycocotools, "__version__", "installed"))
print("psutil           :", psutil.__version__)
PY

echo "[extra] Installing Hugging Face Hub..."
python -m pip install --no-cache-dir -U huggingface_hub

echo "[extra] Checking Hugging Face authentication..."
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is set. Logging in to Hugging Face..."
    python - <<'PY'
import os
from huggingface_hub import login

token = os.environ["HF_TOKEN"]
login(token=token, add_to_git_credential=False)
print("Hugging Face login complete.")
PY
else
    echo "HF_TOKEN is not set."
    echo "If using gated SAM3 weights, run one of:"
    echo "  export HF_TOKEN=your_huggingface_token"
    echo "  huggingface-cli login"
fi

echo "[extra] Testing access to facebook/sam3..."
python - <<'PY'
from huggingface_hub import hf_hub_download

try:
    path = hf_hub_download(
        repo_id="facebook/sam3",
        filename="config.json",
    )
    print("facebook/sam3 access OK:", path)
except Exception as e:
    print("WARNING: Could not access facebook/sam3.")
    print("You may need to accept the model license on Hugging Face and set HF_TOKEN.")
    print("Error:", repr(e))
PY

echo "============================================================"
echo "Venv binary check"
echo "============================================================"
ls -l "${ENV_DIR}/bin/python"* || true

echo
echo "Resolved Python path:"
readlink -f "${ENV_DIR}/bin/python" || true

if command -v file >/dev/null 2>&1; then
    echo
    echo "Binary type:"
    file "${ENV_DIR}/bin/python" || true
fi

if command -v ldd >/dev/null 2>&1; then
    echo
    echo "Shared-library check:"
    ldd "${ENV_DIR}/bin/python" || true
fi

cat > "${ENV_DIR}/ENVIRONMENT_NOTES.txt" <<EOF
SAM3 environment notes
======================

Created at:
  $(date)

Repository:
  $(pwd)

Environment:
  ${ENV_DIR}

Python used:
  ${PYTHON_BIN}

Python module loaded:
  ${PYTHON_MODULE:-none}

Torch index:
  ${TORCH_INDEX_URL}

Setuptools:
  ${SETUPTOOLS_VERSION}

Important:
  This venv is tied to the Python major/minor version used to create it.
  Do not copy this venv between machines, operating systems, or architectures.

Cluster note:
  If the venv Python fails in Slurm with a missing libpython error, load the
  same Python module used to create the venv before running it.

Recommended Slurm pattern:
  module load ${PYTHON_MODULE:-<matching-python-module-if-needed>}
  PYTHON="/absolute/path/to/repo/${ENV_DIR}/bin/python"

  "\$PYTHON" -m sam3.train.train \\
      --config configs/irrigation_canal/irrigation_canal_finetune
EOF

echo "============================================================"
echo "Setup complete."
echo "============================================================"
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
echo
echo "Verify later with:"
echo "  ${ENV_DIR}/bin/python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo
echo "For Slurm, use the venv Python directly:"
echo "  PYTHON=\"/absolute/path/to/repo/${ENV_DIR}/bin/python\""
echo "  \"\$PYTHON\" -m sam3.train.train --config configs/irrigation_canal/irrigation_canal_finetune"
echo "============================================================"
