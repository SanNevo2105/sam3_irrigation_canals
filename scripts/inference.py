#!/usr/bin/env python3
"""
SAM3 checkpoint inference for irrigation-canal mask prediction
=============================================================

Loads a fine-tuned SAM3 checkpoint, runs text-prompt inference on a dataset
folder, and writes:

    1. predicted binary masks
    2. predicted masks overlaid on the original RGB images

No Hugging Face model loading is used. A local checkpoint is required.

Expected dataset layouts
------------------------

Standard split layout:

    <dataset-root>/<split>/images/*.png|*.jpg|*.jpeg|*.tif|*.tiff

Example:

    python scripts/predict_masks_from_checkpoint.py \
        --dataset-root sam3/train/data/irrigation_canal \
        --split test \
        --checkpoint-path experiments/irrigation_canal/checkpoints/checkpoint.pt \
        --output-dir predictions

Direct image-folder layout:

    python scripts/predict_masks_from_checkpoint.py \
        --image-dir sam3/train/data/irrigation_canal/test/images \
        --checkpoint-path experiments/irrigation_canal/checkpoints/checkpoint.pt \
        --output-dir predictions

Outputs
-------

    <output-dir>/masks/<image_stem>_mask.png
    <output-dir>/overlays/<image_stem>_overlay.png
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Pillow 9/10 compatibility
try:
    from PIL.Image import Resampling as _Resampling
    _NEAREST = _Resampling.NEAREST
except ImportError:
    import PIL.Image as _pil_compat
    _NEAREST = _pil_compat.NEAREST  # type: ignore[attr-defined]

import numpy as np
import torch
from PIL import Image as PILImage

# SAM3 public API
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

_SAM3_ROOT = Path(sam3.__file__).resolve().parent.parent
BPE_PATH = _SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INFERENCE_ID = 1

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF")


def collect_images(image_dir: Path) -> List[Path]:
    """Collect all supported image files in image_dir, sorted by filename."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))

    images = sorted(set(images))
    if not images:
        raise RuntimeError(f"No supported image files found in: {image_dir}")

    return images


def resolve_image_dir(dataset_root: Path | None, split: str, image_dir: Path | None) -> Path:
    """
    Resolve where images are stored.

    Priority:
      1. explicit --image-dir
      2. <dataset-root>/<split>/images
      3. <dataset-root>/images
      4. <dataset-root>
    """
    if image_dir is not None:
        return Path(image_dir)

    if dataset_root is None:
        raise ValueError("Either --dataset-root or --image-dir must be provided.")

    dataset_root = Path(dataset_root)

    candidates = [
        dataset_root / split / "images",
        dataset_root / "images",
        dataset_root,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve image directory. Tried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    bpe_path: Path | str | None = BPE_PATH,
    device: torch.device = DEVICE,
) -> torch.nn.Module:
    """Build SAM3 and load a local trainer checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    bpe_path = str(bpe_path) if bpe_path is not None else None

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cpu",
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
    )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(
            f"Checkpoint does not contain key 'model'. Available keys: {list(ckpt.keys())}"
        )

    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] {len(missing)} missing key(s), first 5: {missing[:5]}")
    if unexpected:
        print(f"[load_model] {len(unexpected)} unexpected key(s), first 5: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


def build_transform() -> ComposeAPI:
    """Validation/inference transform matching the irrigation-canal fine-tuning setup."""
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


def build_postprocessor(detection_threshold: float = 0.5) -> PostProcessImage:
    """Postprocess SAM3 detections into binary masks at original image size."""
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=detection_threshold,
        to_cpu=True,
        use_presence=True,
    )


@torch.no_grad()
def run_inference_on_image(
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    pil_image: PILImage.Image,
    text_prompt: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run text-prompt inference on one PIL image.

    Returns:
        pred_binary: uint8 array in {0, 1}
        prob_map: float32 array in [0, 1]
    """
    w_orig, h_orig = pil_image.size

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

    datapoint = transform(datapoint)
    batch = collate([datapoint], dict_key="eval")["eval"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with amp_ctx:
        output = model(batch)

    processed = postprocessor.process_results(output, batch.find_metadatas)

    if _INFERENCE_ID in processed and "masks" in processed[_INFERENCE_ID]:
        result = processed[_INFERENCE_ID]
        inst_masks = result["masks"]   # bool tensor [N, 1, H, W]
        scores = result["scores"]      # float tensor [N]

        if inst_masks.numel() > 0 and inst_masks.shape[0] > 0:
            masks = inst_masks.squeeze(1)  # [N, H, W]

            pred_binary = (
                masks.any(dim=0)
                .numpy()
                .astype(np.uint8)
            )

            masks_f = masks.float()
            score_map = scores[:, None, None].expand_as(masks_f)
            prob_map = (masks_f * score_map).max(dim=0).values.numpy().astype(np.float32)

            return pred_binary, prob_map

    return (
        np.zeros((h_orig, w_orig), dtype=np.uint8),
        np.zeros((h_orig, w_orig), dtype=np.float32),
    )


def resize_prediction_to_image(
    pred_binary: np.ndarray,
    prob_map: np.ndarray,
    image_size_wh: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure prediction arrays match the original image size.

    Uses the same resize-to-target logic as the road evaluation script:
    binary masks use nearest-neighbor; probability maps use image resizing
    after conversion to uint8.
    """
    w, h = image_size_wh
    target_shape = (h, w)

    if pred_binary.shape != target_shape:
        pred_pil = PILImage.fromarray((pred_binary * 255).astype(np.uint8))
        pred_pil = pred_pil.resize((w, h), _NEAREST)
        pred_binary = (np.array(pred_pil) > 127).astype(np.uint8)

    if prob_map.shape != target_shape:
        prob_pil = PILImage.fromarray((prob_map * 255).clip(0, 255).astype(np.uint8))
        prob_pil = prob_pil.resize((w, h), _NEAREST)
        prob_map = np.array(prob_pil).astype(np.float32) / 255.0

    return pred_binary, prob_map


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a binary {0,1} mask as an 8-bit PNG with values {0,255}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


def save_overlay(
    image: PILImage.Image,
    mask: np.ndarray,
    path: Path,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Save a red transparent mask overlay on top of the original RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)

    base = image.convert("RGBA")
    mask_bool = mask.astype(bool)

    overlay_arr = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay_arr[mask_bool, 0] = color[0]
    overlay_arr[mask_bool, 1] = color[1]
    overlay_arr[mask_bool, 2] = color[2]
    overlay_arr[mask_bool, 3] = int(255 * alpha)

    overlay = PILImage.fromarray(overlay_arr, mode="RGBA")
    composed = PILImage.alpha_composite(base, overlay).convert("RGB")
    composed.save(path)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 checkpoint inference and save predicted masks + overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-root",
        default=None,
        metavar="DIR",
        help=(
            "Dataset root. The script looks for <dataset-root>/<split>/images, "
            "then <dataset-root>/images, then <dataset-root>."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        metavar="SPLIT",
        help="Split name used with --dataset-root, for example train/val/test.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        metavar="DIR",
        help="Direct image directory. Overrides --dataset-root and --split.",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        metavar="FILE",
        help="Path to local fine-tuned SAM3 checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="predicted_masks",
        metavar="DIR",
        help="Output directory containing masks/ and overlays/ subfolders.",
    )
    parser.add_argument(
        "--text-prompt",
        default="irrigation canal",
        metavar="TEXT",
        help="Text prompt sent to SAM3 for every image.",
    )
    parser.add_argument(
        "--bpe-path",
        default=str(BPE_PATH),
        metavar="FILE",
        help="Path to bpe_simple_vocab_16e6.txt.gz.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="Detection threshold used by the SAM3 postprocessor.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity for predicted masks.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = _parse_args(argv)

    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else None
    image_dir = resolve_image_dir(
        dataset_root=dataset_root,
        split=args.split,
        image_dir=Path(args.image_dir) if args.image_dir is not None else None,
    )

    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"

    images = collect_images(image_dir)

    print("=" * 65)
    print("SAM3 checkpoint mask prediction")
    print("=" * 65)
    print(f"Device       : {DEVICE}")
    print(f"Image dir    : {image_dir}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Text prompt  : {args.text_prompt!r}")
    print(f"Output dir   : {output_dir}")
    print(f"Images       : {len(images)}")
    print()

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        bpe_path=args.bpe_path,
        device=DEVICE,
    )
    transform = build_transform()
    postprocessor = build_postprocessor(detection_threshold=args.detection_threshold)

    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx:>4}/{len(images)}] {image_path.name}", flush=True)

        image = PILImage.open(image_path).convert("RGB")
        pred_binary, prob_map = run_inference_on_image(
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            pil_image=image,
            text_prompt=args.text_prompt,
            device=DEVICE,
        )

        pred_binary, prob_map = resize_prediction_to_image(
            pred_binary=pred_binary,
            prob_map=prob_map,
            image_size_wh=image.size,
        )

        stem = image_path.stem
        mask_path = masks_dir / f"{stem}_mask.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"

        save_mask(pred_binary, mask_path)
        save_overlay(image, pred_binary, overlay_path, alpha=args.alpha)

    print()
    print("Done.")
    print(f"Predicted masks : {masks_dir.resolve()}")
    print(f"Overlays        : {overlays_dir.resolve()}")


if __name__ == "__main__":
    main()
