# SAM3 Irrigation Canal Fine-tuning

Fine-tune SAM3 on satellite imagery for **binary irrigation-canal segmentation**. The pipeline takes paired images and binary masks, converts them to COCO annotations for SAM3 training, then evaluates the trained checkpoint with segmentation metrics such as IoU and pixel accuracy.

## What this repo does

- Converts image/mask datasets into COCO JSON.
- Fine-tunes SAM3 using box + mask supervision.
- Supports local, container, and Slurm-based training.
- Evaluates checkpoints on binary segmentation metrics:
  - mean IoU
  - global IoU
  - pixel accuracy
  - approximate BCE loss
- Predicts binary masks of irrigation canals for a provided dataset and checkpoint.
- Provides utilities for splitting layered RGBA TIFF chips into RGB images and binary masks.

## Index

- [Clone the repository](#clone-the-repository)
- [Hugging Face access](#hugging-face-access)
- [Environment setup](#environment-setup)
  - [HPC clusters, for example Empire AI](#hpc-clusters-for-example-empire-ai)
  - [Containers, for example RunPod](#containers-for-example-runpod)
- [Dataset layout](#dataset-layout)
- [Optional: split layered TIFF chips](#optional-split-layered-tiff-chips)
- [Convert masks to COCO](#convert-masks-to-coco)
- [Training directly](#training-directly)
- [Slurm training](#slurm-training)
- [Evaluation](#evaluation)
- [Plot training logs](#plot-training-logs)
- [Training Outputs](#training-outputs)
- [Inference](#inference)
- [Multi-GPU training status](#multi-gpu-training-status)
- [Troubleshooting](#troubleshooting)
  - [Hugging Face 401 Unauthorized](#hugging-face-401-unauthorized)
  - [Slurm cannot find `torch`, `submitit`, or other packages](#slurm-cannot-find-torch-submitit-or-other-packages)
  - [`libcrypt.so.2` missing on Empire AI](#libcryptso2-missing-on-empire-ai)
  - [CUDA out of memory](#cuda-out-of-memory)
  - [Disk quota exceeded during checkpoint saving](#disk-quota-exceeded-during-checkpoint-saving)
  - [NaN or unstable training](#nan-or-unstable-training)
  - [No matching image/mask pairs](#no-matching-imagemask-pairs)
- [Notes](#notes)

## Clone the repository

```bash
git clone https://github.com/SanNevo2105/sam3_irrigation_canals.git
cd sam3_irrigation_canals
```

## Hugging Face access

SAM3 weights are hosted behind a gated Hugging Face repository. Before setup or training, make sure your Hugging Face account has access to `facebook/sam3`.

Set your token in the shell:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

You can also log in manually:

```bash
hf auth login
```

## Environment setup

Use the provided setup script. It creates the virtual environment with copied Python binaries, installs PyTorch, installs this repo in editable mode, and checks key dependencies.

### HPC clusters, for example Empire AI

On Empire AI, use the cluster Python module:

```bash
export HF_TOKEN="your_huggingface_token_here"

PYTHON_MODULE=python39 PYTHON_BIN=python3.9 bash scripts/setup_venv.sh .venv
source .venv/bin/activate
```

Verify:

```bash
.venv/bin/python --version
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -c "import submitit, cv2, einops, decord; print('extra imports OK')"
```

Expected behavior:

```text
CUDA available: True
extra imports OK
```

### Containers, for example RunPod

Containers usually do not have the `module` command. Do not set `PYTHON_MODULE`.

```bash
export HF_TOKEN="your_huggingface_token_here"

PYTHON_BIN=python3 bash scripts/setup_venv.sh .venv
source .venv/bin/activate
```

Verify:

```bash
.venv/bin/python --version
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -c "import submitit, cv2, einops, decord; print('extra imports OK')"
```

For RunPod, keep the repository, dataset, checkpoints, and logs under `/workspace`, not `/root`, if possible.

## Dataset layout

Put the dataset under:

```text
sam3/train/data/irrigation_canal/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Masks should be single-channel binary images:

```text
0   = background
> 0 = irrigation canal
```

Images and masks must share the same filename stem:

```text
images/Canal_ML_Chip_0000.png
masks/Canal_ML_Chip_0000.png
```

or:

```text
images/Canal_ML_Chip_0000.jpg
masks/Canal_ML_Chip_0000.png
```

The link to the dataset used for preliminary finetuning: https://drive.google.com/drive/folders/1WIyMazltltBUvg19kxCtEgHauGRWrOI6?usp=sharing

## Optional: split layered TIFF chips

If your raw `.tif` files contain RGB image channels plus a final binary-mask channel, split them first:

```bash
python scripts/split_tif_rgba_masks.py \
  --input-dir sam3/train/data/irrigation_layered \
  --output-dir sam3/train/data/irrigation_canal/train \
  --workers 4 \
  --image-format png \
  --png-compress-level 1
```

For smaller RGB images, use JPEG for images while keeping masks as PNG:

```bash
python scripts/split_tif_rgba_masks.py \
  --input-dir sam3/train/data/irrigation_layered \
  --output-dir sam3/train/data/irrigation_canal/train \
  --workers 4 \
  --image-format jpg \
  --jpg-quality 95
```

Masks are always saved as PNG.

## Convert masks to COCO

SAM3 training expects COCO-style annotations. Convert each split with:

```bash
python scripts/convert_masks_to_coco.py \
  --dataset_path sam3/train/data/irrigation_canal \
  --category_name "irrigation canal" \
  --image_extensions jpg png \
  --mask_extensions png \
  --per-component \
  --min-component-area 100
```

This writes:

```text
sam3/train/data/irrigation_canal/train/_annotations.coco.json
sam3/train/data/irrigation_canal/val/_annotations.coco.json
sam3/train/data/irrigation_canal/test/_annotations.coco.json
```

Use `--per-component` when the canal mask contains many disconnected canal segments. It produces one annotation per connected component, which gives SAM3 more localized box/mask targets.

## Training directly

The main config is:

```text
sam3/train/configs/irrigation_canal/irrigation_canal_finetune.yaml
```

Run from the repository root:

```bash
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python -m sam3.train.train \
  --config configs/irrigation_canal/irrigation_canal_finetune
```

The config name is relative to `sam3/train/`, so do not include `sam3/train/` or `.yaml`.

Important config fields:

```yaml
scratch:
  train_batch_size: 4
  val_batch_size: 2
  lr_scale: 0.02

trainer:
  max_epochs: 20
```

Reduce `train_batch_size` if you hit CUDA OOM. Reduce `lr_scale` if training becomes unstable.

## Slurm training

Edit `REPO_DIR` in `train_irrigation.slurm` to point to your cloned repository:

```bash
REPO_DIR="/path/to/sam3_irrigation_canals"
```

For Empire AI, the Slurm script should include:

```bash
module load python39
export LD_LIBRARY_PATH="$REPO_DIR/local_lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Submit from the repository root:

```bash
mkdir -p logs
sbatch scripts/train_irrigation.slurm
```

The Slurm script should call the venv Python directly:

```bash
"$PYTHON" -m sam3.train.train \
  --config "$CONFIG_NAME"
```

Calling `.venv/bin/python` directly is more reliable than relying on `source .venv/bin/activate` inside batch jobs.

## Evaluation

Evaluate a fine-tuned checkpoint on a split:

```bash
python scripts/evaluate_test.py \
  --dataset-root sam3/train/data/irrigation_canal \
  --split test \
  --text-prompt "irrigation canal" \
  --checkpoint-path experiments/irrigation_canal/checkpoints/checkpoint.pt \
  --num-vis 10 \
  --save-path canal_predictions.png
```

The evaluator reports:

```text
test_loss
test_mean_iou
test_global_iou
test_pixel_accuracy
```

For canal segmentation, **IoU is the main metric to emphasize**. Pixel accuracy can be misleading because most pixels are background.

Evaluate fresh SAM3 weights instead of a fine-tuned checkpoint:

```bash
python scripts/evaluate_test.py \
  --dataset-root sam3/train/data/irrigation_canal \
  --split test \
  --text-prompt "irrigation canal" \
  --load-from-hf
```

## Plot training logs

```bash
python scripts/plot_logs.py \
  --log-dir experiments/irrigation_canal/logs
```

This reads:

```text
experiments/irrigation_canal/logs/train_stats.json
experiments/irrigation_canal/logs/val_stats.json
```

and saves loss/validation curves.

## Training Outputs

A training run writes to:

```text
experiments/irrigation_canal/
├── config.yaml
├── config_resolved.yaml
├── checkpoints/
│   ├── checkpoint.pt
│   ├── checkpoint_1.pt
│   ├── checkpoint_2.pt
│   └── ...
├── logs/
│   ├── train_stats.json
│   ├── val_stats.json
│   └── log.txt
└── tensorboard/
```

Use the checkpoint with the best validation metric, not necessarily the final checkpoint.

For detection-style validation, the main metric is usually:

```text
Meters_train/val_irrigation canal/detection/coco_eval_bbox_AP
```

For the final canal segmentation report, emphasize:

```text
test_mean_iou
test_global_iou
test_pixel_accuracy
```

## Inference

Use `scripts/inference.py` to run a trained SAM3 checkpoint on a folder of images and save the predicted canal masks.

The script loads a local checkpoint, runs text-prompt inference with the default prompt:

```text
irrigation canal
```

and writes two output folders:

```text
<output-dir>/masks/      predicted binary masks
<output-dir>/overlays/   predicted masks overlaid on the original images
```

Example:

```bash
python scripts/inference.py \
  --dataset-root sam3/train/data/irrigation_canal \
  --split test \
  --checkpoint-path experiments/irrigation_canal/checkpoints/checkpoint.pt \
  --output-dir predictions
```

This expects images at:

```text
sam3/train/data/irrigation_canal/test/images/
```

You can also pass an image folder directly:

```bash
python scripts/inference.py \
  --image-dir sam3/train/data/irrigation_canal/test/images \
  --checkpoint-path experiments/irrigation_canal/checkpoints/checkpoint.pt \
  --output-dir predictions
```

Useful options:

```bash
--text-prompt "irrigation canal"   # prompt sent to SAM3
--detection-threshold 0.5          # confidence threshold for predicted masks
--alpha 0.45                       # opacity of overlay masks
```

The script does **not** load SAM3 from Hugging Face. It requires a local fine-tuned checkpoint specified with `--checkpoint-path`.

## Multi-GPU training status

Use **single-GPU training** for this repository.

Multi-GPU training with PyTorch DistributedDataParallel was tested but is not currently reliable for this SAM3 fine-tuning setup. The run was able to start on multiple GPUs, but it crashed during training with a DDP reduction error:

```text
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
```

## Troubleshooting

### Hugging Face 401 Unauthorized

If training fails with:

```text
Cannot access gated repo for url https://huggingface.co/facebook/sam3
Access to model facebook/sam3 is restricted
```

make sure you have accepted access to `facebook/sam3` on Hugging Face and set:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Then test:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="facebook/sam3", filename="config.json"))
PY
```

### Slurm cannot find `torch`, `submitit`, or other packages

If the login node works but the Slurm job fails with:

```text
ModuleNotFoundError: No module named 'torch'
```

then the job is probably using the wrong Python. Make sure the Slurm file uses the venv Python directly:

```bash
PYTHON="/path/to/repo/.venv/bin/python"
```

Check the venv Python:

```bash
ls -l .venv/bin/python*
readlink -f .venv/bin/python
cat .venv/pyvenv.cfg
```

Recreate the venv if needed:

```bash
rm -rf .venv
PYTHON_MODULE=python39 PYTHON_BIN=python3.9 bash scripts/setup_venv.sh .venv
```

### `libcrypt.so.2` missing on Empire AI

If the GPU job fails with:

```text
error while loading shared libraries: libcrypt.so.2
```

copy `libcrypt.so.2` into `local_lib`:

```bash
mkdir -p local_lib
cp -L /cm/images/default-image/usr/lib64/libcrypt.so.2 local_lib/
cp -L /cm/images/default-image/usr/lib64/libcrypt.so.2.0.0 local_lib/ 2>/dev/null || true
```

Then add this to the Slurm job before running Python:

```bash
export LD_LIBRARY_PATH="$REPO_DIR/local_lib:${LD_LIBRARY_PATH:-}"
```

### CUDA out of memory

Reduce batch size:

```yaml
scratch:
  train_batch_size: 2
```

You can also keep:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Disk quota exceeded during checkpoint saving

If training fails while saving a checkpoint:

```text
OSError: [Errno 122] Disk quota exceeded
```

free space or increase the volume/disk size. On RunPod, increase the **volume disk** if the repo is under `/workspace`.

Check disk usage:

```bash
df -h / /root /workspace
du -h --max-depth=1 /workspace | sort -h
```

Delete large archives after extraction:

```bash
rm -f sam3/train/data/irrigation_canal.tar
rm -f sam3/train/data/irrigation_canal.tar.zst
```

### NaN or unstable training

Lower the learning-rate scale:

```yaml
scratch:
  lr_scale: 0.01
```

Also use the best validation checkpoint rather than assuming the final checkpoint is best.

### No matching image/mask pairs

Make sure image and mask stems match:

```text
images/Canal_ML_Chip_0000.jpg
masks/Canal_ML_Chip_0000.png
```

Then rerun COCO conversion.

## Notes

- Keep raw datasets, checkpoints, archives, and virtual environments out of Git.
- Do not commit Hugging Face tokens or any API keys.
- Use `.gitignore` for `.venv/`, `experiments/`, data folders, checkpoints, and large archives.
- For reports, include IoU metrics in addition to pixel accuracy.