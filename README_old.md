# SAM3 – Irrigation Canal Fine-tuning

Fine-tune [SAM 3 (Segment Anything Model 3)](https://ai.meta.com/sam3) on satellite imagery to perform **semantic segmentation of irrigation canals** using binary masks.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Project Structure](#project-structure)
5. [Workflow at a Glance](#workflow-at-a-glance)
6. [Step 1 – Populate the Dataset Folder](#step-1--populate-the-dataset-folder)
7. [Step 2 – Convert Masks to COCO JSON](#step-2--convert-masks-to-coco-json)
8. [Step 3 – Review the Training Config](#step-3--review-the-training-config)
9. [Step 4 – Run Training](#step-4--run-training)
10. [Step 5 – Evaluate on the Test Set](#step-5--evaluate-on-the-test-set)
11. [Step 6 – Plot Training Logs](#step-6--plot-training-logs)
12. [Output Directory Layout](#output-directory-layout)
13. [Troubleshooting](#troubleshooting)

---

## Overview

This repository adapts SAM 3 for binary semantic segmentation of irrigation canals in satellite imagery. The pipeline converts paired (image, binary-mask) datasets into COCO JSON format, fine-tunes SAM 3's detection head with box + mask supervision, and evaluates the result with standard IoU / pixel-accuracy metrics.

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 / 3.12 also supported |
| CUDA | 11.8 | 12.x recommended for H100 |
| GPU VRAM | 40 GB | A100 / H100 80 GB tested |
| Disk space | ~10 GB | For SAM 3 weights + dataset |

> **Note – HuggingFace download:** The first training run downloads the SAM 3 weights (~5 GB) from `facebook/sam3` automatically. After the first run the weights are cached at `~/.cache/huggingface/hub/models--facebook--sam3/`. You can set `checkpoint_path` in the config to that cached `.pt` file to skip re-downloading on future runs.

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/SanNevo2105/sam3_irrigation_canals.git
cd sam3_irrigation_canals
```

All subsequent commands are run from **inside this directory** (the repository root).

### 2. Create and activate a virtual environment

```bash
python -m venv sam3_env
source sam3_env/bin/activate       # Linux / macOS
# sam3_env\Scripts\activate.bat   # Windows
```

Or with conda:

```bash
conda create -n sam3_env python=3.11 -y
conda activate sam3_env
```

### 3. Install PyTorch with CUDA support

Follow the [official PyTorch install page](https://pytorch.org/get-started/locally/) for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install the SAM 3 package with training dependencies

```bash
pip install -e ".[train]"
```

This installs the `sam3` package in editable mode together with all training dependencies declared in [`pyproject.toml`](pyproject.toml):

```
hydra-core  submitit  tensorboard  scipy  torchmetrics
fvcore  fairscale  scikit-image  scikit-learn  zstandard
```

### 5. Install script dependencies

```bash
pip install pycocotools matplotlib pillow tqdm scipy
```

---

## Project Structure

```
sam3_irrigation_canals/          ← repository root (run all commands from here)
├── scripts/
│   ├── convert_masks_to_coco.py   ← Step 2: dataset conversion
│   ├── evaluate_test.py           ← Step 5: test-set evaluation
│   └── plot_logs.py               ← Step 6: training curve plots
├── sam3/
│   └── train/
│       ├── train.py               ← Step 4: main training entry point
│       ├── data/
│       │   └── irrigation_canal/  ← YOUR DATA GOES HERE (see Step 1)
│       └── configs/
│           └── irrigation/
│               └── irrigation_canal_finetune.yaml  ← training config (Step 3)
├── train_irrigation.slurm         ← SLURM job template (cluster users)
└── experiments/
    └── irrigation/                ← checkpoints & logs (created automatically)
```

---

## Workflow at a Glance

```
Populate sam3/train/data/irrigation_canal/ with images + masks
         │
         ▼
scripts/convert_masks_to_coco.py   →  train/, val/, test/ _annotations.coco.json
         │
         ▼
(optional) review sam3/train/configs/irrigation_canal/irrigation_canal_finetune.yaml
         │
         ├─── Local ──► python sam3/train/train.py -c configs/irrigation_canal/irrigation_canal_finetune
         └─── SLURM ──► sbatch train_irrigation.slurm
                              │
                              ▼
                  experiments/irrigation/checkpoints/checkpoint.pt
                              │
               ┌──────────────┴──────────────────┐
               ▼                                  ▼
  scripts/evaluate_test.py             scripts/plot_logs.py
  (IoU, pixel accuracy, PNG)           (loss & AP curves)
```

---

## Step 1 – Populate the Dataset Folder

Place your satellite images and binary masks inside [`sam3/train/data/irrigation_canal/`](sam3/train/data/irrigation_canal/) following this layout:

```
sam3/train/data/irrigation_canal/
├── train/
│   ├── images/     ← image_001.png, image_002.png, …
│   └── masks/      ← mask_001.png,  mask_002.png,  …
├── val/
│   ├── images/
│   └── masks/
└── test/            (optional — skip if you have no held-out test split)
    ├── images/
    └── masks/
```

**Mask format:** Single-channel (grayscale) PNG where pixel value `0` = background and any value `> 0` = irrigation canal.

**Filename matching:** Images and masks are paired by their shared numeric suffix:
`image_1001.png` ↔ `mask_1001.png` → both resolve to key `1001`.

---

## Step 2 – Convert Masks to COCO JSON

Run [`scripts/convert_masks_to_coco.py`](scripts/convert_masks_to_coco.py) from the repository root to generate the COCO annotation files the trainer requires.

### Standard mode (one annotation per image)

```bash
python scripts/convert_masks_to_coco.py \
    --dataset_path sam3/train/data/irrigation_canal \
    --category_name "irrigation canal" \
    --splits train val test
```

This writes `_annotations.coco.json` inside each split folder:

```
sam3/train/data/irrigation_canal/
├── train/_annotations.coco.json
├── val/_annotations.coco.json
└── test/_annotations.coco.json
```

### Per-connected-component mode (recommended for canal networks)

Irrigation canals form spatially sparse networks. `--per-component` decomposes each mask into individual connected canal segments and emits one COCO annotation per segment, giving SAM 3's box head spatially diverse targets:

```bash
python scripts/convert_masks_to_coco.py \
    --dataset_path sam3/train/data/irrigation_canal \
    --category_name "irrigation canal" \
    --per-component \
    --min-component-area 100
```

`--min-component-area 100` discards noise components smaller than 100 pixels.

### All CLI options

| Argument | Default | Description |
|---|---|---|
| `--dataset_path` | *(required)* | Root directory of the dataset |
| `--category_name` | `object` | Class name written into the COCO `categories` list |
| `--splits` | `train val test` | Which splits to process |
| `--image_extensions` | `png` | File extension(s) for images (no leading dot) |
| `--mask_extensions` | `png` | File extension(s) for masks (no leading dot) |
| `--image_subdir` | `images` | Sub-folder inside each split holding the images; pass `.` if images live directly in the split folder |
| `--mask_dir_template` | `{split}/masks` | Template for the mask directory; `{split}` is replaced by the current split name |
| `--output_dir` | *(writes next to split folder)* | Separate directory for COCO JSON output |
| `--per-component` | `False` | Decompose each mask into connected components |
| `--min-component-area` | `100` | Minimum component size in pixels (used with `--per-component`) |

---

## Step 3 – Review the Training Config

The training config is at [`sam3/train/configs/irrigation_canal/irrigation_canal_finetune.yaml`](sam3/train/configs/irrigation_canal/irrigation_canal_finetune.yaml).

All `paths:` values already point to the `sam3/train/data/irrigation_canal/` data folder relative to the repository root — no edits are needed unless you stored your data elsewhere.

Key tunable parameters:

| Parameter | Location in YAML | Default | When to change |
|---|---|---|---|
| `max_epochs` | `trainer.max_epochs` | `20` | Increase for larger datasets |
| `train_batch_size` | `scratch.train_batch_size` | `4` | Reduce if GPU runs out of memory |
| `lr_scale` | `scratch.lr_scale` | `0.02` | Lower if loss diverges or NaNs appear |
| `num_train_workers` | `scratch.num_train_workers` | `4` | Set `0` when debugging |
| `gpus_per_node` | `launcher.gpus_per_node` | `1` | Increase for multi-GPU training |
| `use_cluster` | `submitit.use_cluster` | `False` | Set `True` when submitting to SLURM |

---

## Step 4 – Run Training

Run all commands from the **repository root** with your virtual environment active.

### Local (single GPU)

```bash
python sam3/train/train.py \
    -c configs/irrigation_canal/irrigation_canal_finetune
```

> The `-c` / `--config` argument is a config name relative to `sam3/train/` — no `.yaml` extension, no `sam3/train/` prefix.

### Local (multi-GPU, single node)

```bash
python sam3/train/train.py \
    -c configs/irrigation_canal/irrigation_canal_finetune \
    --num-gpus 4
```

### SLURM cluster

```bash
# 1. Edit the USER-marked lines at the top of train_irrigation.slurm
nano train_irrigation.slurm

# 2. Create the logs directory if it does not exist
mkdir -p logs

# 3. Submit
sbatch train_irrigation.slurm
```

Or pass SLURM settings directly to the training script (set `submitit.use_cluster: True` in the config first):

```bash
python sam3/train/train.py \
    -c configs/irrigation_canal/irrigation_canal_finetune \
    --use-cluster 1 \
    --partition <your_partition> \
    --account <your_account> \
    --num-gpus 1
```

### All training CLI arguments

| Argument | Default | Description |
|---|---|---|
| `-c / --config` | *(required)* | Config name relative to `sam3/train/` (no `.yaml`) |
| `--use-cluster` | *(from config)* | `0` = local, `1` = submit via SLURM |
| `--partition` | *(from config)* | SLURM partition name |
| `--account` | *(from config)* | SLURM account |
| `--qos` | *(from config)* | SLURM Quality of Service |
| `--num-gpus` | *(from config)* | GPUs per node |
| `--num-nodes` | *(from config)* | Number of nodes |

---

## Step 5 – Evaluate on the Test Set

[`scripts/evaluate_test.py`](scripts/evaluate_test.py) loads a fine-tuned checkpoint (or fresh HuggingFace weights) and evaluates on a directory of paired images and binary masks.

It reports four metrics and saves a side-by-side prediction visualisation:

| Metric | Description |
|---|---|
| `test_loss` | Approximate binary cross-entropy between the confidence probability map and the GT mask |
| `test_mean_iou` | Per-image foreground IoU averaged over all test images |
| `test_global_iou` | Globally accumulated IoU across the entire test set |
| `test_pixel_accuracy` | Fraction of pixels correctly classified |

### Evaluate with your fine-tuned checkpoint

```bash
python scripts/evaluate_test.py \
    --dataset-root sam3/train/data/irrigation_canal \
    --split test \
    --text-prompt "irrigation canal" \
    --checkpoint-path experiments/irrigation/checkpoints/checkpoint.pt \
    --num-vis 10 \
    --save-path canal_predictions.png
```

### Evaluate with fresh HuggingFace weights (no local checkpoint needed)

```bash
python scripts/evaluate_test.py \
    --dataset-root sam3/train/data/irrigation_canal \
    --split test \
    --text-prompt "irrigation canal" \
    --load-from-hf
```

### Direct image / mask directory override

```bash
python scripts/evaluate_test.py \
    --image-dir sam3/train/data/irrigation_canal/test/images \
    --mask-dir  sam3/train/data/irrigation_canal/test/masks \
    --text-prompt "irrigation canal" \
    --checkpoint-path experiments/irrigation/checkpoints/checkpoint.pt
```

### All CLI options

| Argument | Default | Description |
|---|---|---|
| `--dataset-root` | `assets/landslide_dataset` | Root dataset folder (`<root>/<split>/images/` and `masks/`) |
| `--split` | `test` | Sub-folder name, e.g. `test` or `val` |
| `--image-dir` | *(from dataset-root)* | Direct path to image folder (overrides `--dataset-root`/`--split`) |
| `--mask-dir` | *(from dataset-root)* | Direct path to mask folder (overrides `--dataset-root`/`--split`) |
| `--text-prompt` | `landslide` | Text query string sent to SAM 3 for every image — use `"irrigation canal"` |
| `--bpe-path` | *(auto-resolved)* | Path to BPE vocabulary file |
| `--checkpoint-path` | `experiments/landslide/checkpoints/checkpoint.pt` | Path to a fine-tuned trainer checkpoint |
| `--load-from-hf` | `False` | Load pre-trained weights from HuggingFace instead of a local checkpoint |
| `--num-vis` | `10` | Number of images to include in the prediction visualisation |
| `--save-path` | `test_predictions.png` | Output path for the prediction visualisation PNG |

---

## Step 6 – Plot Training Logs

[`scripts/plot_logs.py`](scripts/plot_logs.py) reads the `train_stats.json` and `val_stats.json` files produced by the trainer and saves two PNG plots:

- **`train_loss_runs.png`** – training loss by epoch (multiple resumed runs overlaid)
- **`val_ap_runs.png`** – validation AP by epoch

```bash
python scripts/plot_logs.py \
    --log-dir experiments/irrigation/logs
```

| Argument | Default | Description |
|---|---|---|
| `--log-dir` | `experiments/road/logs` | Directory containing `train_stats.json` and `val_stats.json` |

The log directory is always at `<experiment_log_dir>/logs/` — for the default config that is `experiments/irrigation/logs/`.

---

## Output Directory Layout

After a successful training run:

```
experiments/irrigation/
├── config.yaml              ← original config used for this run
├── config_resolved.yaml     ← config with all Hydra variables expanded
├── checkpoints/
│   ├── checkpoint.pt        ← latest checkpoint (overwritten each epoch)
│   └── checkpoint_ep*.pt    ← per-epoch checkpoints
├── logs/
│   ├── train_stats.json     ← one JSON line per training step (input to plot_logs.py)
│   ├── val_stats.json       ← one JSON line per validation epoch
│   └── log.txt              ← human-readable log
├── tensorboard/
│   └── events.out.tfevents.*
├── dumps/
│   └── irrigation/          ← raw COCO prediction files for offline eval
└── submitit_logs/           ← SLURM job logs (cluster only)
```

Monitor live training with TensorBoard:

```bash
tensorboard --logdir experiments/irrigation/tensorboard
```

---

## Troubleshooting

### `FileNotFoundError: Fine-tuned checkpoint not found`

The checkpoint path in `evaluate_test.py` defaults to a landslide path. Always pass `--checkpoint-path` explicitly:

```bash
python scripts/evaluate_test.py \
    --checkpoint-path experiments/irrigation/checkpoints/checkpoint.pt …
```

### CUDA out of memory during training

Reduce `train_batch_size` in the config (e.g. from `4` to `2`). Increase `gradient_accumulation_steps` proportionally to maintain the effective batch size:

```yaml
scratch:
  train_batch_size: 2
  gradient_accumulation_steps: 2
```

### NaN / loss explosion

Lower the learning-rate scale in the config:

```yaml
scratch:
  lr_scale: 0.01   # default 0.02; halve if NaNs appear in training
```

Also ensure `gradient_clip.max_norm` remains at `0.1` (the default conservative value).

### `No matching (image, mask) pairs found`

Verify that image and mask filenames share a common numeric suffix. For example:
- ✅ `sat_0042.png` ↔ `label_0042.png` (both key to `0042`)
- ❌ `imageA.png` ↔ `maskB.png` (no numeric suffix — no match)

Run the conversion script on a single split to inspect diagnostic output:

```bash
python scripts/convert_masks_to_coco.py \
    --dataset_path sam3/train/data/irrigation_canal \
    --splits train
```

### HuggingFace download fails or is slow

After a successful download the weights are cached at:

```
~/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt
```

To skip re-downloading, note that cached path and set these two fields in [`sam3/train/configs/irrigation_canal/irrigation_canal_finetune.yaml`](sam3/train/configs/irrigation_canal/irrigation_canal_finetune.yaml):

```yaml
paths:
  checkpoint_path: ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt

trainer:
  model:
    load_from_HF: false
```
