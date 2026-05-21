"""
Text-Only SAM3 Inference with Fresh HuggingFace Weights
=======================================================

Loads the pre-trained SAM3 model directly from HuggingFace (facebook/sam3),
runs **text-only** binary semantic segmentation on a configurable dataset
folder, prints the standard four evaluation metrics, and writes JSONL log
records that are immediately consumable by:

    scripts/evaluate_test.py  – for qualitative predictions
    scripts/plot_training_curves.py  – for curve visualisation

No fine-tuned checkpoint is required.  The HuggingFace weights are downloaded
automatically the first time and cached under
``~/.cache/huggingface/hub/models--facebook--sam3/``.

Dataset folder layout (default):
    <dataset-root>/<split>/images/image_N.png
    <dataset-root>/<split>/masks/mask_N.png

Both ``--image-dir`` and ``--mask-dir`` can be supplied directly to override
the auto-derived subdirectory layout, e.g. to point at an arbitrary folder of
images and masks.

Usage
-----
    # Minimal – uses HF weights, test split, "landslide" prompt:
    python scripts/text_only_hf_inference.py \\
        --dataset-root /home/rocky/sam3/sam3/train/data/landslide_dataset

    # Full override:
    python scripts/text_only_hf_inference.py \\
        --image-dir /path/to/images \\
        --mask-dir  /path/to/masks \\
        --text-prompt "landslide" \\
        --log-dir   scripts/logs \\
        --save-predictions

Outputs
-------
    <log-dir>/train_stats.json   – single JSONL record (plot_training_curves.py)
    <log-dir>/val_stats.json     – single JSONL record (plot_training_curves.py)
    <log-dir>/inference_results.json  – human-readable metric summary
    [optional] test_predictions_hf.png  – visualisation of the first N images
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import Dataset

# ── SAM3 public API ───────────────────────────────────────────────────────────
import sam3
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_SAM3_ROOT  = Path(sam3.__file__).resolve().parent.parent

# Fixed query-ID used for single-image inference batches
_INFERENCE_ID = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Dataset
# ==============================================================================

class BinarySegDataset(Dataset):
    """
    Paired (image, binary-mask) dataset.

    **Inference-only mode** (``mask_dir=None``)
        Collects ALL PNG and JPG/JPEG files in ``image_dir`` sorted
        alphabetically.  Any filename is accepted – no numeric-ID requirement.

    **Evaluation mode** (``mask_dir`` provided)
        Scans ``image_dir`` for PNG/JPG files whose stem contains a numeric
        suffix *N* and pairs each with the corresponding
        ``mask_dir/mask_N.png``.  Only pairs where both files exist are kept.

    Returns
    -------
    pil_image : PIL.Image.Image  – RGB image
    gt_mask   : np.ndarray | None  – binary uint8 array {0, 1}, shape (H, W)
    img_path  : str  – absolute path to the image file
    """

    _IMG_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")

    def __init__(self, image_dir: Path, mask_dir: Optional[Path] = None) -> None:
        image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None

        # ── Collect all images in the directory ───────────────────────────────
        all_images: List[Path] = []
        for glob in self._IMG_GLOBS:
            all_images.extend(image_dir.glob(glob))
        all_images = sorted(set(all_images))  # deduplicate (case-insensitive FS)

        if not all_images:
            raise RuntimeError(
                f"No PNG/JPG images found in: {image_dir}"
            )

        if self.mask_dir is None:
            # Inference-only: accept every image regardless of filename
            self.samples: List[Tuple[Path, Optional[Path]]] = [
                (p, None) for p in all_images
            ]
        else:
            # Evaluation: pair by numeric ID extracted from the stem
            def _numeric_id(p: Path) -> Optional[int]:
                m = re.search(r"(\d+)", p.stem)
                return int(m.group(1)) if m else None

            images_by_id = {
                _numeric_id(p): p
                for p in all_images
                if _numeric_id(p) is not None
            }
            masks_by_id = {
                _numeric_id(p): p
                for p in sorted(self.mask_dir.glob("*.png"))
                if _numeric_id(p) is not None
            }
            common = sorted(set(images_by_id) & set(masks_by_id))
            if not common:
                raise RuntimeError(
                    f"No matching (image, mask) pairs found.\n"
                    f"  images: {image_dir}\n"
                    f"  masks:  {self.mask_dir}\n"
                    "  Tip: use --image-dir / --mask-dir for non-standard layouts."
                )
            self.samples = [
                (images_by_id[i], masks_by_id[i]) for i in common
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, Optional[np.ndarray], str]:
        img_path, mask_path = self.samples[idx]
        image = PILImage.open(img_path).convert("RGB")
        if mask_path is not None:
            mask = np.array(PILImage.open(mask_path).convert("L"))
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = None
        return image, mask, str(img_path)


# ==============================================================================
# Model loading from HuggingFace
# ==============================================================================

def load_model_from_hf(
    bpe_path: Optional[str] = None,
    device: torch.device = DEVICE,
) -> torch.nn.Module:
    """
    Download (or use cache) the pre-trained SAM3 weights from HuggingFace
    (``facebook/sam3``) and return an eval-mode image model.

    The checkpoint is cached at:
        ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt
    after the first download.
    """
    print("[load_model_from_hf] Loading SAM3 from HuggingFace (facebook/sam3) …")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cpu",          # move after loading
        eval_mode=True,
        checkpoint_path=None,  # let load_from_HF handle download
        load_from_HF=True,     # download / use cache
        enable_segmentation=True,
    )
    model.to(device)
    model.eval()
    print("[load_model_from_hf] Model ready.")
    return model


# ==============================================================================
# Transform & postprocessor
# ==============================================================================

def build_transform() -> ComposeAPI:
    """Validation transform pipeline (resize → tensor → normalise)."""
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_postprocessor() -> PostProcessImage:
    """PostProcessImage configured for binary semantic segmentation."""
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=True,
        use_presence=True,
    )


# ==============================================================================
# Single-image inference
# ==============================================================================

def run_inference_on_image(
    model:          torch.nn.Module,
    transform:      ComposeAPI,
    postprocessor:  PostProcessImage,
    pil_image:      PILImage.Image,
    text_prompt:    str,
    device:         torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SAM3 text-only inference on a single PIL image.

    Returns
    -------
    pred_binary : np.ndarray  shape (H, W)  dtype uint8  values {0, 1}
    prob_map    : np.ndarray  shape (H, W)  dtype float32  range [0, 1]
    """
    w_orig, h_orig = pil_image.size

    # 1. Build Datapoint with a single text query
    datapoint = Datapoint(find_queries=[], images=[])
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h_orig, w_orig])]
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=_INFERENCE_ID,
                original_image_id=_INFERENCE_ID,
                original_category_id=1,
                original_size=[w_orig, h_orig],
                object_id=0,
                frame_index=0,
            ),
        )
    )

    # 2. Apply val transforms
    datapoint = transform(datapoint)

    # 3. Collate (batch size = 1) and move to device
    batch = collate([datapoint], dict_key="eval")["eval"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    # 4. Forward pass
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        output = model(batch)

    # 5. Post-process → binary per-instance masks + confidence scores
    processed = postprocessor.process_results(output, batch.find_metadatas)

    # 6. Merge instance predictions into a single semantic mask
    if _INFERENCE_ID in processed and "masks" in processed[_INFERENCE_ID]:
        result     = processed[_INFERENCE_ID]
        inst_masks = result["masks"]   # bool tensor [N, 1, H, W]
        scores     = result["scores"]  # float tensor [N]

        if inst_masks.numel() > 0 and inst_masks.shape[0] > 0:
            pred_binary = (
                inst_masks.squeeze(1)
                .any(dim=0)
                .numpy()
                .astype(np.uint8)
            )
            masks_f  = inst_masks.squeeze(1).float()
            s_exp    = scores[:, None, None].expand_as(masks_f)
            prob_map = (masks_f * s_exp).max(dim=0).values.numpy()
            return pred_binary, prob_map

    return (
        np.zeros((h_orig, w_orig), dtype=np.uint8),
        np.zeros((h_orig, w_orig), dtype=np.float32),
    )


# ==============================================================================
# Metric helpers
# ==============================================================================

def compute_mean_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = int(((pred == 1) & (gt == 1)).sum())
    union        = int(((pred == 1) | (gt == 1)).sum())
    if union == 0:
        return 1.0
    return intersection / union


# ==============================================================================
# Evaluation loop
# ==============================================================================

@torch.no_grad()
def evaluate_dataset(
    model:         torch.nn.Module,
    transform:     ComposeAPI,
    postprocessor: PostProcessImage,
    dataset:       BinarySegDataset,
    text_prompt:   str,
    device:        torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Iterate over the dataset and compute:
        test_loss, test_mean_iou, test_global_iou, test_pixel_accuracy

    When ground-truth masks are unavailable (mask is None), loss and IoU
    metrics are skipped and only the count of processed images is reported.
    """
    model.eval()

    total_loss       = 0.0
    total_iou        = 0.0
    total_pixels     = 0
    total_correct    = 0
    all_intersection = 0.0
    all_union        = 0.0
    n_with_gt        = 0

    n = len(dataset)
    for idx in range(n):
        pil_image, gt_mask, img_path = dataset[idx]
        fname = os.path.basename(img_path)
        print(f"  [{idx + 1:>3}/{n}] {fname}", end="\r", flush=True)

        pred_binary, prob_map = run_inference_on_image(
            model, transform, postprocessor, pil_image, text_prompt, device
        )

        if gt_mask is not None:
            n_with_gt += 1

            # Approximate BCE loss
            prob_t = torch.tensor(prob_map, dtype=torch.float32)
            gt_t   = torch.tensor(gt_mask,  dtype=torch.float32)
            loss   = F.binary_cross_entropy(
                prob_t.clamp(1e-6, 1.0 - 1e-6), gt_t, reduction="mean"
            )
            total_loss += loss.item()

            # Per-image IoU
            total_iou += compute_mean_iou(pred_binary, gt_mask)

            # Pixel accuracy
            total_correct += int((pred_binary == gt_mask).sum())
            total_pixels  += int(gt_mask.size)

            # Global IoU accumulators
            all_intersection += float(((pred_binary == 1) & (gt_mask == 1)).sum())
            all_union        += float(((pred_binary == 1) | (gt_mask == 1)).sum())

    print()

    denom = max(n_with_gt, 1)
    return {
        "test_loss":           total_loss    / denom,
        "test_mean_iou":       total_iou     / denom,
        "test_global_iou":     (all_intersection + 1e-6) / (all_union + 1e-6),
        "test_pixel_accuracy": total_correct / max(total_pixels, 1),
        "n_images":            n,
        "n_with_gt":           n_with_gt,
    }


# ==============================================================================
# Visualisation
# ==============================================================================

def save_predictions_figure(
    dataset:       BinarySegDataset,
    model:         torch.nn.Module,
    transform:     ComposeAPI,
    postprocessor: PostProcessImage,
    text_prompt:   str,
    device:        torch.device,
    num_images:    int = 10,
    save_path:     str = "test_predictions_hf.png",
) -> None:
    """Save a side-by-side (image | GT mask | prediction) figure."""
    n = min(num_images, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for row in range(n):
        pil_image, gt_mask, img_path = dataset[row]
        img_np = np.array(pil_image)

        pred_binary, _ = run_inference_on_image(
            model, transform, postprocessor, pil_image, text_prompt, device
        )
        print(f"  Visualising [{row + 1}/{n}] {os.path.basename(img_path)}")

        axes[row][0].imshow(img_np)
        axes[row][0].set_title(Path(img_path).name, fontsize=8)
        axes[row][0].axis("off")

        axes[row][1].imshow(img_np)
        if gt_mask is not None:
            gt_overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
            gt_overlay[gt_mask == 1] = [0.0, 1.0, 0.0, 0.5]
            axes[row][1].imshow(gt_overlay)
        axes[row][1].set_title("Ground Truth", fontsize=8)
        axes[row][1].axis("off")

        axes[row][2].imshow(img_np)
        pred_overlay = np.zeros((*pred_binary.shape, 4), dtype=np.float32)
        pred_overlay[pred_binary == 1] = [1.0, 0.0, 0.0, 0.5]
        axes[row][2].imshow(pred_overlay)
        axes[row][2].set_title(f"Prediction ({text_prompt!r})", fontsize=8)
        axes[row][2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPrediction visualisation saved to: {os.path.abspath(save_path)}")


# ==============================================================================
# Log writing  (compatible with plot_training_curves.py)
# ==============================================================================

def write_log_records(
    metrics:  Dict[str, float],
    log_dir:  Path,
) -> None:
    """
    Write JSONL records into ``<log_dir>/train_stats.json`` and
    ``<log_dir>/val_stats.json`` using the exact metric-key names expected by
    ``scripts/plot_training_curves.py``.

    Both files receive a single epoch-0 record so that ``plot_training_curves.py``
    can render a single-point curve even without any fine-tuning history.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Map our metric names to the keys expected by plot_training_curves.py
    # test_mean_iou is used as a proxy for CE-F1 (train) and AP@50 (val)
    iou   = metrics.get("test_mean_iou",       0.0)
    loss  = metrics.get("test_loss",            0.0)
    ap50  = metrics.get("test_mean_iou",        0.0)  # best available proxy
    ap    = metrics.get("test_global_iou",      0.0)

    train_record = {
        "Trainer/epoch":             0,
        "Losses/train_all_loss":     loss,
        "Losses/train_all_ce_f1":    iou,
        # extra metadata
        "test_pixel_accuracy":       metrics.get("test_pixel_accuracy", 0.0),
        "test_global_iou":           metrics.get("test_global_iou",     0.0),
        "inference_mode":            "text_only_hf",
    }

    val_record = {
        "Trainer/epoch": 0,
        "Losses/val_all_loss": loss,
        "Meters_train/val_landslide/detection/coco_eval_bbox_AP_50": ap50,
        "Meters_train/val_landslide/detection/coco_eval_bbox_AP":    ap,
        "inference_mode": "text_only_hf",
    }

    train_path = log_dir / "train_stats.json"
    val_path   = log_dir / "val_stats.json"

    with open(train_path, "w") as fh:
        fh.write(json.dumps(train_record) + "\n")

    with open(val_path, "w") as fh:
        fh.write(json.dumps(val_record) + "\n")

    # Also write a human-readable summary
    summary_path = log_dir / "inference_results.json"
    with open(summary_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nLog records written to : {log_dir}")
    print(f"  train_stats.json     : {train_path}")
    print(f"  val_stats.json       : {val_path}")
    print(f"  inference_results.json: {summary_path}")


# ==============================================================================
# CLI
# ==============================================================================

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Text-only SAM3 binary semantic segmentation using fresh "
            "HuggingFace weights (facebook/sam3).  No fine-tuned checkpoint "
            "required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset location ──────────────────────────────────────────────────────
    ds = p.add_argument_group("Dataset")
    ds.add_argument(
        "--dataset-root",
        default=str(_SAM3_ROOT / "sam3" / "train" / "data" / "landslide_dataset"),
        metavar="DIR",
        help=(
            "Root of the dataset.  Images are expected at "
            "<dataset-root>/<split>/images/ and masks at "
            "<dataset-root>/<split>/masks/ unless --image-dir/--mask-dir override."
        ),
    )
    ds.add_argument(
        "--split",
        default="test",
        metavar="SPLIT",
        help=(
            "Dataset split sub-folder name (e.g. 'test', 'validation', 'train').  "
            "Ignored when --image-dir is supplied."
        ),
    )
    ds.add_argument(
        "--image-dir",
        default=None,
        metavar="DIR",
        help=(
            "Direct path to the image folder.  Overrides "
            "--dataset-root / --split auto-resolution.  "
            "Pair with --mask-dir for evaluation metrics."
        ),
    )
    ds.add_argument(
        "--mask-dir",
        default=None,
        metavar="DIR",
        help=(
            "Direct path to the mask folder.  Overrides "
            "--dataset-root / --split auto-resolution.  "
            "Optional; when absent, metrics are skipped."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    m = p.add_argument_group("Model")
    m.add_argument(
        "--bpe-path",
        default=None,
        metavar="FILE",
        help=(
            "Path to the BPE tokeniser vocabulary "
            "(bpe_simple_vocab_16e6.txt.gz).  Auto-resolved from the SAM3 "
            "package if not supplied."
        ),
    )
    m.add_argument(
        "--text-prompt",
        default="landslide",
        metavar="TEXT",
        help="Text query sent to SAM3 for every image.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    o = p.add_argument_group("Output")
    o.add_argument(
        "--log-dir",
        default=str(_SCRIPT_DIR / "logs"),
        metavar="DIR",
        help=(
            "Directory where train_stats.json, val_stats.json and "
            "inference_results.json are written.  Created if absent."
        ),
    )
    o.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save a side-by-side prediction visualisation PNG.",
    )
    o.add_argument(
        "--num-vis",
        type=int,
        default=10,
        metavar="N",
        help="Number of images to include in the prediction visualisation.",
    )
    o.add_argument(
        "--pred-save-path",
        default="test_predictions_hf.png",
        metavar="FILE",
        help="Output path for the prediction visualisation PNG.",
    )

    return p.parse_args(argv)


# ==============================================================================
# Entry point
# ==============================================================================

def main(argv=None) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    args = _parse_args(argv)

    # ── Resolve image / mask directories ─────────────────────────────────────
    if args.image_dir is not None:
        image_dir = Path(args.image_dir)
        mask_dir  = Path(args.mask_dir) if args.mask_dir else None
    else:
        dataset_root     = Path(args.dataset_root)
        candidate_images = dataset_root / args.split / "images"
        candidate_masks  = dataset_root / args.split / "masks"

        if candidate_images.exists():
            # Standard layout: <dataset-root>/<split>/images/
            image_dir = candidate_images
            mask_dir  = candidate_masks if candidate_masks.exists() else None
            if mask_dir is None:
                print(
                    f"[warn] Mask directory not found: {candidate_masks}\n"
                    "       Evaluation metrics will be skipped."
                )
        else:
            # Flat layout fallback: images live directly in <dataset-root>
            image_dir = dataset_root
            mask_dir  = None
            print(
                f"[info] Standard layout not found ({candidate_images}).\n"
                f"       Falling back to flat layout: {image_dir}\n"
                "       No masks available; evaluation metrics will be skipped."
            )

    if not image_dir.exists():
        print(
            f"[error] Image directory not found: {image_dir}\n"
            "        Check --dataset-root / --split / --image-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    log_dir = Path(args.log_dir)

    print("=" * 65)
    print("Text-Only SAM3 HuggingFace Inference")
    print("=" * 65)
    print(f"Device       : {DEVICE}")
    print(f"Image dir    : {image_dir}")
    print(f"Mask dir     : {mask_dir or '(none – metrics skipped)'}")
    print(f"Text prompt  : {args.text_prompt!r}")
    print(f"Log dir      : {log_dir}")
    print()

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = BinarySegDataset(image_dir=image_dir, mask_dir=mask_dir)
    print(f"Dataset      : {len(dataset)} image(s)\n")

    # ── Load model from HuggingFace ───────────────────────────────────────────
    model         = load_model_from_hf(bpe_path=args.bpe_path, device=DEVICE)
    transform     = build_transform()
    postprocessor = build_postprocessor()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Running text-only inference …")
    metrics = evaluate_dataset(
        model, transform, postprocessor, dataset,
        text_prompt=args.text_prompt,
        device=DEVICE,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Results")
    print("=" * 65)
    print(f"  Images processed     : {metrics['n_images']}")
    print(f"  Images with GT masks : {metrics['n_with_gt']}")
    if metrics["n_with_gt"] > 0:
        print(f"  Test loss (approx BCE): {metrics['test_loss']:.4f}")
        print(f"  Test mean IoU         : {metrics['test_mean_iou']:.4f}")
        print(f"  Test global IoU       : {metrics['test_global_iou']:.4f}")
        print(f"  Test pixel accuracy   : {metrics['test_pixel_accuracy']:.4f}")

    # ── Write JSONL log records for plot_training_curves.py ──────────────────
    write_log_records(metrics, log_dir)

    # ── Optional visualisation ────────────────────────────────────────────────
    if args.save_predictions:
        print("\nGenerating prediction visualisation …")
        save_predictions_figure(
            dataset       = dataset,
            model         = model,
            transform     = transform,
            postprocessor = postprocessor,
            text_prompt   = args.text_prompt,
            device        = DEVICE,
            num_images    = args.num_vis,
            save_path     = args.pred_save_path,
        )


if __name__ == "__main__":
    main()
