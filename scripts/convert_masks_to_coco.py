#!/usr/bin/env python3
"""
Convert a dataset of images and binary mask pairs into COCO JSON format
for SAM3 finetuning.  Supports PNG, TIFF, and any other Pillow-readable
format.

Default (PNG) dataset structure:
    dataset_root/
    ├── train/
    │   ├── images/     <- image_1.png, image_2.png, ...
    │   └── masks/      <- mask_1.png,  mask_2.png,  ...
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

Road / TIFF dataset structure (images live directly in the split folder;
masks live in a sibling <split>_labels/ folder):
    dataset_root/
    ├── train/          <- 10078660_15.tiff, ...
    ├── train_labels/   <- 10078660_15.tif,  ...
    ├── val/
    ├── val_labels/
    ├── test/
    └── test_labels/

Images and masks are matched by their full filename stem
(e.g., 10078660_15.tiff ↔ 10078660_15.tif).
The output COCO JSON is written to <split>/_annotations.coco.json.

Usage – default PNG layout:
    python convert_masks_to_coco.py \\
        --dataset_path /path/to/your/dataset \\
        --category_name my_class \\
        [--splits train val test]

Usage – road TIFF layout:
    python convert_masks_to_coco.py \\
        --dataset_path sam3/train/data/road \\
        --category_name road \\
        --splits train val test \\
        --image_extensions tiff \\
        --mask_extensions tif \\
        --image_subdir . \\
        --mask_dir_template {split}_labels \\
        --output_dir /path/to/coco_output

Usage – per-connected-component mode (road_global / network masks):
    python convert_masks_to_coco.py \\
        --dataset_path sam3/train/data/road_global \\
        --category_name road \\
        --per-component \\
        --min-component-area 100

    Each binary mask is split into individual connected components via
    scipy.ndimage.label.  Components with fewer than --min-component-area
    pixels are discarded as noise.  One COCO annotation (RLE mask + tight
    bounding box) is emitted per surviving component, giving the SAM3 box
    head spatially diverse targets instead of a single near-full-image box.

If --output_dir is given, each split's JSON is written to
    <output_dir>/<split>/_annotations.coco.json
and the directory is created automatically if it does not exist.
If --output_dir is omitted, the JSON is written next to the split
folder inside --dataset_path (original default behaviour).
"""

import argparse
import json
import os
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
    """Load a mask (PNG, TIFF, or any Pillow-supported format) and return
    a uint8 binary array (0 / 1).  Multi-band TIFFs are collapsed to
    grayscale via Pillow's 'L' conversion before thresholding."""
    with Image.open(mask_path) as im:
        mask = np.array(im.convert("L"))
    return (mask > 0).astype(np.uint8)


def mask_to_rle(mask: np.ndarray) -> Dict:
    """Encode a binary uint8 mask as COCO RLE (Fortran column-major order)."""
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")   # JSON-serialisable
    return rle


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """Return the tight [x, y, w, h] bounding box over all foreground pixels.

    Uses the inclusive-pixel convention: a mask whose rightmost foreground
    pixel is at column c has bbox width = c - x_min + 1, which equals the
    number of columns covered (inclusive).  This matches the COCO standard.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0.0, 0.0, 0.0, 0.0]
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y_min, y_max = int(y_idx[0]), int(y_idx[-1])
    x_min, x_max = int(x_idx[0]), int(x_idx[-1])
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def mask_to_components(mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """Label connected components in a binary mask and return a list of
    per-component binary uint8 masks, keeping only those whose pixel area
    is >= *min_area*.

    Uses ``scipy.ndimage.label`` with default 4-connectivity (conservative:
    diagonally-adjacent pixels are treated as separate components, which is
    appropriate for road networks).  Returns an empty list when the mask is
    entirely background or no component survives the area filter.
    """
    from scipy.ndimage import label as nd_label  # lazy import – not always needed
    labeled, n_components = nd_label(mask)
    components: List[np.ndarray] = []
    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled == comp_id).astype(np.uint8)
        if int(comp_mask.sum()) >= min_area:
            components.append(comp_mask)
    return components


# ---------------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------------

def _stem_key(path: Path) -> str:
    """Return a normalised matching key for a given file path.

    When image and mask filenames share the same numeric suffix but differ
    in their word prefix (e.g. ``image_1001.png`` vs ``mask_1001.png``),
    matching on the full stem produces an empty intersection.  Instead we
    strip any leading alphabetic prefix up to and including the first
    underscore, so both files key to ``1001``.

    For filenames that contain no underscore (e.g. road TIFFs such as
    ``10078660_15``) the full stem is preserved, which keeps the existing
    road-dataset behaviour intact.
    """
    stem = path.stem
    # Strip a leading word-prefix (e.g. "image_" / "mask_") so that
    # image_1001.png and mask_1001.png share the common key "1001".
    # Only strip when the prefix before the first "_" is entirely
    # alphabetic, so purely-numeric stems like "10078660_15" are
    # left unchanged.
    if "_" in stem:
        prefix, rest = stem.split("_", 1)
        if prefix.isalpha():
            # print("rest", rest)
            return rest
        elif rest.isalpha():
            # print("prefix", prefix)
            return prefix
        # print("stem", stem)
    return stem


def build_image_mask_pairs(
    images_dir: Path,
    masks_dir: Path,
    image_exts: Tuple[str, ...] = ("png","jpg", "jpeg"),
    mask_exts: Tuple[str, ...]  = ("png","jpg", "jpeg"),
) -> List[Tuple[Path, Path]]:
    """
    Match images to masks by their full filename stem.

    All extensions in *image_exts* / *mask_exts* are globbed so that
    mixed-extension directories (e.g. both .tif and .tiff) are handled
    correctly.  Returns a list of (image_path, mask_path) sorted
    lexicographically by stem.
    """
    image_by_stem: Dict[str, Path] = {}
    for ext in image_exts:
        for p in images_dir.glob(f"*.{ext}"):
            image_by_stem[_stem_key(p)] = p

    mask_by_stem: Dict[str, Path] = {}
    for ext in mask_exts:
        for p in masks_dir.glob(f"*.{ext}"):
            mask_by_stem[_stem_key(p)] = p

    common_stems = sorted(set(image_by_stem) & set(mask_by_stem))

    only_images = set(image_by_stem) - set(mask_by_stem)
    only_masks  = set(mask_by_stem)  - set(image_by_stem)
    if only_images:
        print(f"  ⚠  {len(only_images)} image(s) with no matching mask – skipped.")
    if only_masks:
        print(f"  ⚠  {len(only_masks)} mask(s) with no matching image – skipped.")

    return [(image_by_stem[s], mask_by_stem[s]) for s in common_stems]


def process_split(
    split_name: str,
    images_dir: Path,
    masks_dir: Path,
    category_id: int,
    image_exts: Tuple[str, ...] = ("png","jpg", "jpeg"),
    mask_exts: Tuple[str, ...]  = ("png","jpg", "jpeg"),
    per_component: bool = False,
    min_component_area: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """Build COCO images and annotations lists for one split.

    When *per_component* is True each binary mask is decomposed into its
    connected components (via :func:`mask_to_components`) and one annotation
    is emitted per surviving component.  Components smaller than
    *min_component_area* pixels are discarded.  When *per_component* is False
    (default) the original behaviour is preserved: one annotation per image
    covering the whole binary mask.
    """
    pairs = build_image_mask_pairs(images_dir, masks_dir, image_exts, mask_exts)
    print(f"  {split_name}: {len(pairs)} matched pairs found.")

    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id_counter = 0  # global annotation id (unique across all images)

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

        # Decompose into connected components or treat the whole mask as one.
        if per_component:
            component_masks = mask_to_components(mask, min_area=min_component_area)
        else:
            component_masks = [mask]

        for comp_mask in component_masks:
            rle  = mask_to_rle(comp_mask)
            bbox = mask_to_bbox(comp_mask)
            area = float(np.sum(comp_mask))
            annotations.append({
                "id": ann_id_counter,
                "image_id": img_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,       # [x, y, w, h]
                "iscrowd": 0,
            })
            ann_id_counter += 1

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
        help="Root directory of the dataset (contains split subdirs).",
    )
    parser.add_argument(
        "--category_name",
        default="object",
        help="Class name to use in the COCO JSON categories list (default: 'object').",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process (default: train val test).",
    )
    parser.add_argument(
        "--image_extensions",
        nargs="+",
        default=["png"],
        metavar="EXT",
        help=(
            "File extension(s) for images, without leading dot "
            "(default: png).  Example: --image_extensions tiff"
        ),
    )
    parser.add_argument(
        "--mask_extensions",
        nargs="+",
        default=["png"],
        metavar="EXT",
        help=(
            "File extension(s) for masks, without leading dot "
            "(default: png).  Example: --mask_extensions tif"
        ),
    )
    parser.add_argument(
        "--image_subdir",
        default="images",
        help=(
            "Subdirectory inside each split folder that holds the images "
            "(default: 'images').  Pass '.' to use the split folder itself, "
            "e.g. for the road TIFF dataset where images live in road/train/."
        ),
    )
    parser.add_argument(
        "--mask_dir_template",
        default="{split}/masks",
        help=(
            "Template for the mask directory relative to --dataset_path. "
            "The literal string '{split}' is replaced by the current split name. "
            "Default: '{split}/masks'.  "
            "Road TIFF example: '{split}_labels'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory where the _annotations.coco.json files will be written. "
            "Each split produces <output_dir>/<split>/_annotations.coco.json. "
            "The directory (and split subdirectory) are created if they do not exist. "
            "If omitted, the JSON is written into <dataset_path>/<split>/ "
            "(original default behaviour)."
        ),
    )
    parser.add_argument(
        "--per-component",
        action="store_true",
        default=False,
        help=(
            "Split each binary mask into connected components and emit one "
            "COCO annotation per component.  Useful for road/network masks "
            "where a single mask covers the entire road network.  Components "
            "smaller than --min-component-area pixels are discarded."
        ),
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=100,
        metavar="PIXELS",
        help=(
            "Minimum pixel area for a connected component to be included as "
            "an annotation.  Components smaller than this threshold are "
            "discarded as noise.  Only used when --per-component is set "
            "(default: 100)."
        ),
    )
    args = parser.parse_args()

    root        = Path(args.dataset_path)
    image_exts  = tuple(args.image_extensions)
    mask_exts   = tuple(args.mask_extensions)
    out_root    = Path(args.output_dir) if args.output_dir else root

    categories = [
        {"id": 1, "name": args.category_name, "supercategory": args.category_name}
    ]

    for split in args.splits:
        # Resolve image directory
        if args.image_subdir == ".":
            images_dir = root / split
        else:
            images_dir = root / split / args.image_subdir

        # Resolve mask directory (template supports sibling folders like train_labels/)
        masks_dir = root / args.mask_dir_template.format(split=split)

        out_dir  = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "_annotations.coco.json"

        if not images_dir.is_dir() or not masks_dir.is_dir():
            print(
                f"[skip] {split}: images dir '{images_dir}' or "
                f"masks dir '{masks_dir}' not found."
            )
            continue

        print(f"\n── {split} ──────────────────────────────")
        images, annotations = process_split(
            split, images_dir, masks_dir,
            category_id=1,
            image_exts=image_exts,
            mask_exts=mask_exts,
            per_component=args.per_component,
            min_component_area=args.min_component_area,
        )

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
