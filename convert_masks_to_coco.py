#!/usr/bin/env python3
"""
Convert a dataset of PNG images and binary mask pairs into COCO JSON format
for SAM3 finetuning.

Expected dataset structure:
    dataset_root/
    ├── train/
    │   ├── images/     <- image_1.png, image_2.png, ...
    │   └── masks/      <- mask_1.png,  mask_2.png,  ...
    ├── validation/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

Images and masks are matched by the numeric suffix in their filenames
(e.g., image_42.png ↔ mask_42.png). The output COCO JSON is written to
<split>/_annotations.coco.json alongside the images/ and masks/ folders.

Usage:
    python convert_masks_to_coco.py \\
        --dataset_path /path/to/your/dataset \\
        --category_name my_class \\
        [--splits train validation test]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def load_binary_mask(mask_path: str) -> np.ndarray:
    """Load a PNG mask and return a uint8 binary array (0 / 1)."""
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.uint8)


def mask_to_rle(mask: np.ndarray) -> Dict:
    """Encode a binary uint8 mask as COCO RLE (Fortran column-major order)."""
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")   # JSON-serialisable
    return rle


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0.0, 0.0, 0.0, 0.0]
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y_min, y_max = int(y_idx[0]), int(y_idx[-1])
    x_min, x_max = int(x_idx[0]), int(x_idx[-1])
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


# ---------------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------------

def _numeric_id(path: Path) -> Optional[int]:
    """Extract the first integer found in a filename stem, or None."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def build_image_mask_pairs(
    images_dir: Path,
    masks_dir: Path,
) -> List[Tuple[Path, Path]]:
    """
    Match images to masks by their numeric suffix.
    Returns a list of (image_path, mask_path) sorted by numeric id.
    """
    image_by_id = {_numeric_id(p): p for p in images_dir.glob("*.png") if _numeric_id(p) is not None}
    mask_by_id  = {_numeric_id(p): p for p in masks_dir.glob("*.png")  if _numeric_id(p) is not None}

    common_ids = sorted(set(image_by_id) & set(mask_by_id))

    only_images = set(image_by_id) - set(mask_by_id)
    only_masks  = set(mask_by_id)  - set(image_by_id)
    if only_images:
        print(f"  ⚠  {len(only_images)} image(s) with no matching mask – skipped.")
    if only_masks:
        print(f"  ⚠  {len(only_masks)} mask(s) with no matching image – skipped.")

    return [(image_by_id[i], mask_by_id[i]) for i in common_ids]


def process_split(
    split_name: str,
    images_dir: Path,
    masks_dir: Path,
    category_id: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Build COCO images and annotations lists for one split."""
    pairs = build_image_mask_pairs(images_dir, masks_dir)
    print(f"  {split_name}: {len(pairs)} matched pairs found.")

    images: List[Dict] = []
    annotations: List[Dict] = []

    for img_id, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"  Encoding {split_name}")):
        with Image.open(img_path) as im:
            w, h = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        mask = load_binary_mask(str(mask_path))
        # Skip images whose mask is completely empty
        if mask.sum() == 0:
            continue
        rle  = mask_to_rle(mask)
        bbox = mask_to_bbox(mask)
        area = float(np.sum(mask))

        annotations.append({
            "id": img_id,           # one annotation per image
            "image_id": img_id,
            "category_id": category_id,
            "segmentation": rle,    # COCO RLE – ready for pycocotools
            "area": area,
            "bbox": bbox,           # [x, y, w, h]
            "iscrowd": 0,
        })

    return images, annotations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a paired image/binary-mask dataset to COCO JSON for SAM3."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Root directory of the dataset (contains train/, validation/, test/ subdirs).",
    )
    parser.add_argument(
        "--category_name",
        default="object",
        help="Class name to use in the COCO JSON categories list (default: 'object').",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Which splits to process (default: train validation test).",
    )
    args = parser.parse_args()

    root = Path(args.dataset_path)

    categories = [
        {"id": 1, "name": args.category_name, "supercategory": args.category_name}
    ]

    for split in args.splits:
        images_dir = root / split / "images"
        masks_dir  = root / split / "masks"
        out_json   = root / split / "_annotations.coco.json"

        if not images_dir.is_dir() or not masks_dir.is_dir():
            print(f"[skip] {split}: images/ or masks/ subdirectory not found at {root / split}")
            continue

        print(f"\n── {split} ──────────────────────────────")
        images, annotations = process_split(split, images_dir, masks_dir, category_id=1)

        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(out_json, "w") as f:
            json.dump(coco_dict, f)

        print(f"  ✓ Wrote {out_json}")
        print(f"      images: {len(images)}   annotations: {len(annotations)}")

    print("\nDone. COCO JSON files are ready for SAM3 training.")


if __name__ == "__main__":
    main()
