# SAM3 Irrigation Canal Fine-tuning

Fine-tune SAM3 on satellite imagery for **binary irrigation-canal segmentation**. The pipeline takes paired images and binary masks, converts them to COCO annotations for SAM3 training, then evaluates the trained checkpoint with segmentation metrics such as IoU and pixel accuracy.

## What this repo does

- Converts image/mask datasets into COCO JSON.
- Fine-tunes SAM3 using box + mask supervision.
- Supports local or Slurm-based training.
- Evaluates checkpoints on binary segmentation metrics:
  - mean IoU
  - global IoU
  - pixel accuracy
  - approximate BCE loss
- Provides utilities for splitting layered RGBA TIFF chips into RGB images and binary masks.

## Recommended environment setup

Clone the repository:

```bash
git clone https://github.com/SanNevo2105/sam3_irrigation_canals.git
cd sam3_irrigation_canals
```

Use the provided setup script:

```bash
bash scripts/setup_venv.sh sam3_env
source sam3_env/bin/activate
```

The setup script should create the virtual environment with copied Python binaries, not symlinked system Python. This matters on clusters where login nodes and GPU compute nodes may have different `/usr/bin/python3` versions.

After setup, verify:

```bash
which python
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import submitit; print(submitit.__version__)"
```

Expected behavior:

```text
.../sam3_env/bin/python
CUDA available: True
```

If you are creating the venv manually on a cluster, use:

```bash
python3.9 -m venv --copies sam3_env
source sam3_env/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train]"
```

Do not create a symlinked venv with plain `python -m venv sam3_env` on clusters where compute nodes may use a different system Python.

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

Images and masks must share the same stem:

```text
images/Canal_ML_Chip_0000.png
masks/Canal_ML_Chip_0000.png
```

or:

```text
images/Canal_ML_Chip_0000.jpg
masks/Canal_ML_Chip_0000.png
```

## Optional: split layered TIFF chips

If your raw `.tif` files contain RGB image channels plus a final binary-mask channel, split them first:

```bash
python scripts/split_tif_rgba_masks_fast.py \
  --input-dir sam3/train/data/irrigation_layered \
  --output-dir sam3/train/data/irrigation_canal/train \
  --workers 4 \
  --image-format png \
  --png-compress-level 1
```

For faster/smaller RGB images, use JPEG for images while keeping masks as PNG:

```bash
python scripts/split_tif_rgba_masks_fast.py \
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

## Training

The main config is:

```text
sam3/train/configs/irrigation_canal/irrigation_canal_finetune.yaml
```

Run from the repository root:

```bash
python -m sam3.train.train \
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

Edit the user settings in `train_irrigation.slurm`, especially the Python path:

```bash
PYTHON="/path/to/sam3_env/bin/python"
REPO_DIR="/path/to/sam3_irrigation_canals"
CONFIG_NAME="configs/irrigation_canal/irrigation_canal_finetune"
```

Submit from the repository root:

```bash
mkdir -p logs
sbatch train_irrigation.slurm
```

The Slurm script should call the venv Python directly:

```bash
"$PYTHON" -m sam3.train.train \
  --config "$CONFIG_NAME"
```

This is more reliable than relying on `source sam3_env/bin/activate` inside batch jobs.

## Evaluation

Evaluate a checkpoint on a split:

```bash
python scripts/evaluate_test.py \
  --dataset-root sam3/train/data/irrigation_canal \
  --split test \
  --text-prompt "irrigation canal" \
  --checkpoint-path experiments/irrigation/checkpoints/checkpoint.pt \
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

Use fresh SAM3 weights instead of a fine-tuned checkpoint:

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
  --log-dir experiments/irrigation/logs
```

This reads:

```text
experiments/irrigation/logs/train_stats.json
experiments/irrigation/logs/val_stats.json
```

and saves loss/validation curves.

## Outputs

A training run writes to:

```text
experiments/irrigation/
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

## Troubleshooting

### Slurm cannot find `torch` or `submitit`

If the login node works but the Slurm job fails with:

```text
ModuleNotFoundError: No module named 'torch'
```

then the job is probably using the wrong Python. Check whether your venv points to system Python:

```bash
ls -l sam3_env/bin/python*
readlink -f sam3_env/bin/python
cat sam3_env/pyvenv.cfg
```

Fix by recreating the environment with copied binaries:

```bash
rm -rf sam3_env
python3.9 -m venv --copies sam3_env
source sam3_env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train]"
```

Then make the Slurm script use:

```bash
PYTHON="/path/to/sam3_env/bin/python"
```

### CUDA out of memory

Reduce batch size:

```yaml
scratch:
  train_batch_size: 2
```

### NaN or unstable training

Lower the learning-rate scale:

```yaml
scratch:
  lr_scale: 0.01
```

Also use the best validation checkpoint rather than training for more epochs by default.

### No matching image/mask pairs

Make sure image and mask stems match:

```text
images/Canal_ML_Chip_0000.jpg
masks/Canal_ML_Chip_0000.png
```

Then rerun COCO conversion.

## Notes

- The first run may download SAM3 weights from Hugging Face.
- For public benchmarks, report IoU metrics in addition to pixel accuracy.
- Keep raw datasets, checkpoints, and large archives out of Git.
