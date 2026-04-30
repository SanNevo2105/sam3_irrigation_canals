"""
Road Val/Test Evaluation using the Fine-tuned SAM3 Model
=========================================================

Adapted from scripts/evaluate_test.py for the road segmentation task.

Key differences from the landslide version
-------------------------------------------
* **Dataset** – reads a COCO JSON + TIFF image directory instead of flat
  PNG image/mask pairs.  Segmentation annotations (polygon or RLE) are
  rasterised into a single binary GT mask per image.
* **Normalization** – ImageNet mean/std ``[0.485, 0.456, 0.406]`` /
  ``[0.229, 0.224, 0.225]`` matching ``road_finetune.yaml``.
* **Text prompt** – defaults to ``"road"`` instead of ``"landslide"``.
* **Checkpoint** – defaults to ``experiments/road/checkpoints/checkpoint.pt``.
* **Dataset root** – defaults to the road COCO split used at val time
  (``sam3/train/data/road/val_merged`` + ``road_coco/val_merged``).

The four reported metrics are identical to the landslide script:
    test_loss           – approximate BCE between probability map and GT mask
    test_mean_iou       – per-image IoU averaged over all images
    test_global_iou     – globally accumulated IoU across the whole split
    test_pixel_accuracy – fraction of pixels correctly classified

Usage
-----
    # Default: fine-tuned road checkpoint, val_merged split
    python scripts/evaluate_test_road.py

    # Custom split / checkpoint
    python scripts/evaluate_test_road.py \\
        --coco-json  sam3/train/data/road_coco/test/_annotations.coco.json \\
        --image-dir  sam3/train/data/road/test \\
        --checkpoint-path experiments/road/checkpoints/checkpoint.pt \\
        --text-prompt road \\
        --num-vis 20 \\
        --save-path road_predictions.png

    # Fresh HuggingFace weights (no local checkpoint needed)
    python scripts/evaluate_test_road.py --load-from-hf

CLI arguments
-------------
    --coco-json       Path to COCO annotation JSON  (default: val_merged)
    --image-dir       Directory containing TIFF images (default: val_merged)
    --text-prompt     Text query string  (default: "road")
    --bpe-path        Path to BPE vocabulary file (auto-resolved if absent)
    --checkpoint-path Path to fine-tuned trainer checkpoint
    --load-from-hf    Load fresh HuggingFace weights instead of local ckpt
    --num-vis         Number of images to visualise  (default: 10)
    --save-path       Output path for prediction visualisation PNG
    --split           Convenience flag: resolves --coco-json and --image-dir
                      from the standard directory layout when neither is given.
                      Values: val | test | val_merged  (default: val_merged)

Dependencies (already satisfied by the SAM3 training environment):
    torch, torchvision, matplotlib, numpy, pillow, pycocotools
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── PIL resampling constants (Pillow 9 vs 10 compatibility) ───────────────────
try:
    from PIL.Image import Resampling as _Resampling
    _NEAREST  = _Resampling.NEAREST
    _BILINEAR = _Resampling.BILINEAR
except ImportError:                         # Pillow < 9.1
    import PIL.Image as _pil_compat
    _NEAREST  = _pil_compat.NEAREST   # type: ignore[attr-defined]
    _BILINEAR = _pil_compat.BILINEAR  # type: ignore[attr-defined]

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

# ── Default paths (relative to repository root) ───────────────────────────────
_SAM3_ROOT = Path(sam3.__file__).resolve().parent.parent   # …/sam3_irrigation_canals/

_ROAD_IMAGES_ROOT = _SAM3_ROOT / "sam3" / "train" / "data" / "road"
_ROAD_COCO_ROOT   = _SAM3_ROOT / "sam3" / "train" / "data" / "road_coco"

# Default split: val_merged combines original val (14 imgs) + test (49 imgs)
# — this matches what road_finetune.yaml uses for validation at training time.
DEFAULT_SPLIT       = "val_merged"
DEFAULT_IMAGE_DIR   = _ROAD_IMAGES_ROOT / DEFAULT_SPLIT
DEFAULT_COCO_JSON   = _ROAD_COCO_ROOT   / DEFAULT_SPLIT / "_annotations.coco.json"

BPE_PATH        = _SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = _SAM3_ROOT / "experiments" / "road" / "checkpoints" / "checkpoint.pt"

TEXT_PROMPT = "road"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed query-ID used per single-image inference call
_INFERENCE_ID = 1


# ==============================================================================
# Mask rasterisation helpers
# ==============================================================================

def _polygon_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    """
    Rasterise a single COCO polygon annotation into a binary (H, W) mask.

    Uses PIL's ImageDraw for zero-extra-dependency rasterisation; falls back
    gracefully to an empty mask if the polygon is degenerate.
    """
    from PIL import Image as _PIL, ImageDraw as _Draw

    mask = _PIL.new("L", (width, height), 0)
    draw = _Draw.Draw(mask)
    for poly in segmentation:
        # COCO polygon: flat list [x0,y0, x1,y1, …]
        if len(poly) < 6:  # need at least 3 vertices
            continue
        coords = list(zip(poly[0::2], poly[1::2]))
        draw.polygon(coords, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def _rle_to_mask(rle: dict, height: int, width: int) -> np.ndarray:
    """
    Decode a COCO RLE annotation (compressed or uncompressed) into a binary mask.

    Tries ``pycocotools`` first; if unavailable falls back to a pure-Python
    uncompressed-RLE decoder.
    """
    try:
        from pycocotools import mask as coco_mask  # type: ignore
        m = coco_mask.decode(rle)
        return (m > 0).astype(np.uint8)
    except ImportError:
        pass

    # Pure-Python fallback for uncompressed RLE
    # (pycocotools is strongly recommended for compressed RLE)
    counts = rle.get("counts", [])
    if isinstance(counts, str):
        raise RuntimeError(
            "pycocotools is required to decode compressed (COCO string) RLE annotations. "
            "Install it with:  pip install pycocotools"
        )
    flat = np.zeros(height * width, dtype=np.uint8)
    pos, value = 0, 0
    for n in counts:
        flat[pos : pos + n] = value
        pos += n
        value = 1 - value
    return flat.reshape(height, width, order="F")


# ==============================================================================
# Dataset
# ==============================================================================

class RoadCocoDataset(Dataset):
    """
    Road dataset backed by a COCO JSON annotation file + TIFF image directory.

    All segmentation annotations in the JSON are rasterised and unioned into a
    single binary foreground mask per image, matching the semantic-segmentation
    evaluation style used in ``evaluate_test.py``.

    Returns
    -------
    pil_image  : PIL.Image.Image  – RGB image, original size
    gt_mask    : np.ndarray       – binary uint8 array {0, 1}, shape (H, W)
    img_path   : str              – absolute path to the image file
    """

    def __init__(
        self,
        coco_json: Path | str,
        image_dir: Path | str,
    ) -> None:
        coco_json = Path(coco_json)
        image_dir = Path(image_dir)

        if not coco_json.exists():
            raise FileNotFoundError(f"COCO JSON not found: {coco_json}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        with open(coco_json, "r") as f:
            coco: dict = json.load(f)

        # Build image-id → metadata mapping
        id_to_info: dict = {img["id"]: img for img in coco.get("images", [])}

        # Build image-id → list[annotation] mapping
        id_to_anns: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            id_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Keep only images where the file physically exists; support both .tif
        # and .tiff extensions transparently.
        self.samples: List[Tuple[Path, int]] = []  # (image_path, image_id)
        for img_id, info in id_to_info.items():
            filename = info["file_name"]
            candidate = image_dir / filename
            if not candidate.exists():
                # Try swapping extension to the other TIFF variant
                stem = Path(filename).stem
                for ext in (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".jpg"):
                    alt = image_dir / (stem + ext)
                    if alt.exists():
                        candidate = alt
                        break
                else:
                    # Image file missing — skip silently
                    continue
            self.samples.append((candidate, img_id))

        if not self.samples:
            raise RuntimeError(
                f"No valid (image, annotation) pairs found.\n"
                f"  COCO JSON : {coco_json}\n"
                f"  Image dir : {image_dir}"
            )

        # Sort by image-id for reproducibility
        self.samples.sort(key=lambda x: x[1])
        self._id_to_info  = id_to_info
        self._id_to_anns  = id_to_anns

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, np.ndarray, str]:
        img_path, img_id = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────
        image = PILImage.open(img_path).convert("RGB")
        w, h  = image.size   # PIL: (width, height)

        # ── Build GT binary mask from all annotations for this image ──
        info  = self._id_to_info[img_id]
        h_ann = info.get("height", h)
        w_ann = info.get("width",  w)

        gt_mask = np.zeros((h_ann, w_ann), dtype=np.uint8)

        for ann in self._id_to_anns.get(img_id, []):
            seg = ann.get("segmentation", [])
            if not seg:
                # Fall back to bbox if no segmentation
                x, y, bw, bh = (int(v) for v in ann["bbox"])
                gt_mask[y : y + bh, x : x + bw] = 1
                continue

            if isinstance(seg, dict):
                # RLE format
                instance_mask = _rle_to_mask(seg, h_ann, w_ann)
            else:
                # Polygon format (list of lists)
                instance_mask = _polygon_to_mask(seg, h_ann, w_ann)

            gt_mask = np.maximum(gt_mask, instance_mask)

        return image, gt_mask, str(img_path)


# ==============================================================================
# Model loading  (identical to evaluate_test.py)
# ==============================================================================

def load_model(
    bpe_path:        str | Path = BPE_PATH,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device:          torch.device = DEVICE,
    load_from_hf:    bool = False,
) -> torch.nn.Module:
    """
    Build the SAM3 image model and load weights.

    Supports both HuggingFace pre-trained weights (``load_from_hf=True``) and
    local fine-tuned trainer checkpoints (default).  DDP ``"module."`` prefixes
    are stripped automatically.
    """
    bpe_path = str(bpe_path) if bpe_path is not None else None

    if load_from_hf:
        print("[load_model] Loading SAM3 from HuggingFace (facebook/sam3) …")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=True,
            enable_segmentation=True,
        )
        model.to(device)
        model.eval()
        return model

    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found:\n  {checkpoint_path}\n\n"
            "Either run training first:\n"
            "  python sam3/train/train.py -c configs/road/road_finetune.yaml\n\n"
            "Or load pre-trained weights from HuggingFace with --load-from-hf."
        )

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cpu",
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
    )

    ckpt: dict       = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict = ckpt["model"]

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] {len(missing)} missing key(s), first 5: {missing[:5]}")
    if unexpected:
        print(
            f"[load_model] {len(unexpected)} unexpected key(s), "
            f"first 5: {unexpected[:5]}"
        )

    model.to(device)
    model.eval()
    return model


# ==============================================================================
# Transform  (road_finetune.yaml → ImageNet stats)
# ==============================================================================

def build_transform() -> ComposeAPI:
    """
    Validation transform pipeline matching ``road_finetune.yaml``::

        val_transforms:
          - ComposeAPI:
              - RandomResizeAPI(sizes=1008, max_size=1008, square=True)
              - ToTensorAPI
              - NormalizeAPI(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    The ImageNet normalisation stats differ from the landslide script's
    ``[0.5, 0.5, 0.5]`` / ``[0.5, 0.5, 0.5]`` and must match the values used
    during road fine-tuning.
    """
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ==============================================================================
# Postprocessor
# ==============================================================================

def build_postprocessor() -> PostProcessImage:
    """
    PostProcessImage configured to match ``road_finetune.yaml``
    (``use_presence=True``, segmentation masks, original-size resize).
    """
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
    device:         torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SAM3 inference on a single PIL image with ``TEXT_PROMPT``.

    Returns
    -------
    pred_binary : np.ndarray  shape (H, W)  dtype uint8  values {0, 1}
    prob_map    : np.ndarray  shape (H, W)  dtype float32 range [0, 1]
    """
    w_orig, h_orig = pil_image.size

    datapoint = Datapoint(find_queries=[], images=[])
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h_orig, w_orig])]
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=TEXT_PROMPT,
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

    datapoint = transform(datapoint)
    batch = collate([datapoint], dict_key="eval")["eval"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        output = model(batch)

    processed = postprocessor.process_results(output, batch.find_metadatas)

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
    """Per-image binary IoU on the foreground class (label = 1)."""
    intersection = int(((pred == 1) & (gt == 1)).sum())
    union        = int(((pred == 1) | (gt == 1)).sum())
    if union == 0:
        return 1.0
    return intersection / union


# ==============================================================================
# Main evaluation loop
# ==============================================================================

@torch.no_grad()
def evaluate_test_set(
    model:         torch.nn.Module,
    transform:     ComposeAPI,
    postprocessor: PostProcessImage,
    dataset:       RoadCocoDataset,
    device:        torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Evaluate the fine-tuned SAM3 road model on the given split.

    Returns
    -------
    dict with keys:
        "test_loss"           – mean approximate BCE loss
        "test_mean_iou"       – mean per-image foreground IoU
        "test_global_iou"     – globally accumulated IoU
        "test_pixel_accuracy" – fraction of correctly classified pixels
    """
    model.eval()

    total_loss       = 0.0
    total_iou        = 0.0
    total_pixels     = 0
    total_correct    = 0
    all_intersection = 0.0
    all_union        = 0.0

    n = len(dataset)
    for idx in range(n):
        pil_image, gt_mask, img_path = dataset[idx]
        fname = os.path.basename(img_path)
        print(f"  [{idx + 1:>3}/{n}] {fname}", end="\r", flush=True)

        pred_binary, prob_map = run_inference_on_image(
            model, transform, postprocessor, pil_image, device
        )

        # Resize prediction to match GT mask dimensions (COCO may differ from
        # the raw image size after rasterisation uses the annotated w/h).
        if pred_binary.shape != gt_mask.shape:
            gt_h, gt_w = gt_mask.shape
            pred_pil   = PILImage.fromarray(pred_binary * 255)
            pred_pil   = pred_pil.resize((gt_w, gt_h), _NEAREST)
            pred_binary = (np.array(pred_pil) > 127).astype(np.uint8)

            prob_pil  = PILImage.fromarray((prob_map * 255).clip(0, 255).astype(np.uint8))
            prob_pil  = prob_pil.resize((gt_w, gt_h), _BILINEAR)
            prob_map  = np.array(prob_pil).astype(np.float32) / 255.0

        # ── Loss  (approximate BCE) ────────────────────────────────────────
        prob_t = torch.tensor(prob_map, dtype=torch.float32)
        gt_t   = torch.tensor(gt_mask,  dtype=torch.float32)
        loss   = F.binary_cross_entropy(
            prob_t.clamp(1e-6, 1.0 - 1e-6),
            gt_t,
            reduction="mean",
        )
        total_loss += loss.item()

        total_iou        += compute_mean_iou(pred_binary, gt_mask)
        total_correct    += int((pred_binary == gt_mask).sum())
        total_pixels     += int(gt_mask.size)
        all_intersection += float(((pred_binary == 1) & (gt_mask == 1)).sum())
        all_union        += float(((pred_binary == 1) | (gt_mask == 1)).sum())

    print()

    return {
        "test_loss":           total_loss    / max(n, 1),
        "test_mean_iou":       total_iou     / max(n, 1),
        "test_global_iou":     (all_intersection + 1e-6) / (all_union + 1e-6),
        "test_pixel_accuracy": total_correct / max(total_pixels, 1),
    }


# ==============================================================================
# Visualisation
# ==============================================================================

def show_predictions(
    dataset:       RoadCocoDataset,
    num_images:    int = 10,
    model:         Optional[torch.nn.Module]  = None,
    transform:     Optional[ComposeAPI]       = None,
    postprocessor: Optional[PostProcessImage] = None,
    device:        torch.device               = DEVICE,
    save_path:     str                        = "road_predictions.png",
) -> None:
    """
    Visualise ground-truth and predicted masks for the first ``num_images``
    images in ``dataset``.

    Each row shows three panels:

    * **Left**   – original RGB image
    * **Centre** – GT binary mask overlaid in **green** (α = 0.5)
    * **Right**  – predicted binary mask overlaid in **red**  (α = 0.5)

    The figure is saved to ``save_path`` and displayed interactively.
    """
    if model is None or transform is None or postprocessor is None:
        print("[show_predictions] Building model / transform / postprocessor ...")
        if model        is None: model        = load_model(device=device)
        if transform    is None: transform    = build_transform()
        if postprocessor is None: postprocessor = build_postprocessor()

    n = min(num_images, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for row in range(n):
        pil_image, gt_mask, img_path = dataset[row]
        img_np = np.array(pil_image)

        pred_binary, _ = run_inference_on_image(
            model, transform, postprocessor, pil_image, device
        )
        print(f"  Visualising [{row + 1}/{n}] {os.path.basename(img_path)}")

        # Resize pred to match GT if needed
        if pred_binary.shape != gt_mask.shape:
            gt_h, gt_w = gt_mask.shape
            pred_pil   = PILImage.fromarray(pred_binary * 255)
            pred_pil   = pred_pil.resize((gt_w, gt_h), _NEAREST)
            pred_binary = (np.array(pred_pil) > 127).astype(np.uint8)

        # Panel 0: original image
        axes[row][0].imshow(img_np)
        axes[row][0].set_title(Path(img_path).name, fontsize=8)
        axes[row][0].axis("off")

        # Panel 1: GT mask (green overlay)
        axes[row][1].imshow(img_np)
        gt_overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
        gt_overlay[gt_mask == 1] = [0.0, 1.0, 0.0, 0.5]
        axes[row][1].imshow(gt_overlay)
        axes[row][1].set_title("Ground Truth", fontsize=8)
        axes[row][1].axis("off")

        # Panel 2: Predicted mask (red overlay)
        axes[row][2].imshow(img_np)
        pred_overlay = np.zeros((*pred_binary.shape, 4), dtype=np.float32)
        pred_overlay[pred_binary == 1] = [1.0, 0.0, 0.0, 0.5]
        axes[row][2].imshow(pred_overlay)
        axes[row][2].set_title("Prediction", fontsize=8)
        axes[row][2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPrediction visualisations saved to: {os.path.abspath(save_path)}")


# ==============================================================================
# Entry point
# ==============================================================================

def _parse_args(argv=None) -> argparse.Namespace:
    _SAM3_ROOT_local  = Path(sam3.__file__).resolve().parent.parent
    _road_images_root = _SAM3_ROOT_local / "sam3" / "train" / "data" / "road"
    _road_coco_root   = _SAM3_ROOT_local / "sam3" / "train" / "data" / "road_coco"

    p = argparse.ArgumentParser(
        description=(
            "Road Val/Test Evaluation using the fine-tuned SAM3 model.  "
            "Reads COCO JSON annotations + TIFF images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = p.add_argument_group("Dataset")
    ds.add_argument(
        "--split",
        default="val_merged",
        choices=["val", "test", "val_merged"],
        help=(
            "Convenience split name.  Resolves --coco-json and --image-dir from "
            "the standard road directory layout when neither is explicitly given."
        ),
    )
    ds.add_argument(
        "--coco-json",
        default=None,
        metavar="FILE",
        help=(
            "Explicit path to the COCO annotation JSON.  "
            "Overrides --split-based resolution."
        ),
    )
    ds.add_argument(
        "--image-dir",
        default=None,
        metavar="DIR",
        help=(
            "Explicit path to the TIFF image directory.  "
            "Overrides --split-based resolution."
        ),
    )

    # ── Prompt ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--text-prompt",
        default=TEXT_PROMPT,
        metavar="TEXT",
        help="Text query string sent to SAM3 for every image.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    m = p.add_argument_group("Model")
    m.add_argument(
        "--bpe-path",
        default=None,
        metavar="FILE",
        help="Path to BPE vocabulary file (auto-resolved from package if absent).",
    )
    m.add_argument(
        "--checkpoint-path",
        default=str(_SAM3_ROOT_local / "experiments" / "road" / "checkpoints" / "checkpoint.pt"),
        metavar="FILE",
        help="Path to fine-tuned road trainer checkpoint (.pt).",
    )
    m.add_argument(
        "--load-from-hf",
        action="store_true",
        help=(
            "Load pre-trained weights from HuggingFace (facebook/sam3) instead "
            "of a local checkpoint.  When set, --checkpoint-path is ignored."
        ),
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    v = p.add_argument_group("Visualisation")
    v.add_argument(
        "--num-vis",
        type=int,
        default=10,
        metavar="N",
        help="Number of images to include in the prediction visualisation.",
    )
    v.add_argument(
        "--save-path",
        default="road_predictions.png",
        metavar="FILE",
        help="Output path for the prediction visualisation PNG.",
    )

    args = p.parse_args(argv)

    # ── Resolve coco-json / image-dir from split when not given explicitly ────
    if args.coco_json is None:
        args.coco_json = str(
            _road_coco_root / args.split / "_annotations.coco.json"
        )
    if args.image_dir is None:
        args.image_dir = str(_road_images_root / args.split)

    return args


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    args = _parse_args()

    # Override module-level prompt so run_inference_on_image() picks it up
    TEXT_PROMPT = args.text_prompt

    _coco_json = Path(args.coco_json)
    _image_dir = Path(args.image_dir)

    if not _coco_json.exists():
        print(
            f"[error] COCO JSON not found: {_coco_json}\n"
            "        Check --coco-json / --split.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _image_dir.exists():
        print(
            f"[error] Image directory not found: {_image_dir}\n"
            "        Check --image-dir / --split.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 65)
    print("Road Val/Test Evaluation")
    print("=" * 65)
    print(f"Device          : {DEVICE}")
    if args.load_from_hf:
        print("Weights         : HuggingFace (facebook/sam3)")
    else:
        print(f"Checkpoint      : {args.checkpoint_path}")
    print(f"COCO JSON       : {_coco_json}")
    print(f"Image dir       : {_image_dir}")
    print(f"Text prompt     : {TEXT_PROMPT!r}")
    print()

    # ── Build components ──────────────────────────────────────────────────────
    model = load_model(
        bpe_path        = args.bpe_path or BPE_PATH,
        checkpoint_path = args.checkpoint_path,
        device          = DEVICE,
        load_from_hf    = args.load_from_hf,
    )
    transform     = build_transform()
    postprocessor = build_postprocessor()

    dataset = RoadCocoDataset(coco_json=_coco_json, image_dir=_image_dir)
    print(f"Split '{args.split}': {len(dataset)} image(s)\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Running evaluation (one image at a time) ...")
    results = evaluate_test_set(model, transform, postprocessor, dataset, DEVICE)

    print()
    print("Results")
    print("-" * 40)
    print(f"Loss              : {results['test_loss']:.4f}")
    print(f"Mean IoU          : {results['test_mean_iou']:.4f}")
    print(f"Global IoU        : {results['test_global_iou']:.4f}")
    print(f"Pixel accuracy    : {results['test_pixel_accuracy']:.4f}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    print("\nGenerating prediction visualisations ...")
    show_predictions(
        dataset       = dataset,
        num_images    = args.num_vis,
        model         = model,
        transform     = transform,
        postprocessor = postprocessor,
        device        = DEVICE,
        save_path     = args.save_path,
    )
