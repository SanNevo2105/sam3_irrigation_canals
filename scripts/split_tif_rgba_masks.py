#!/usr/bin/env python3
"""
Split layered RGBA TIFF chips into RGB images and binary masks.

Expected input:
    Canal_ML_Chip_0000.tif
        channels 0,1,2 -> RGB satellite image
        channel 3      -> binary mask, usually values {0,1}

Output:
    <output-dir>/images/Canal_ML_Chip_0000.png or .jpg
    <output-dir>/masks/Canal_ML_Chip_0000.png

Examples:
    # Lossless image + mask PNG, faster low-compression PNG
    python scripts/split_tif_rgba_masks_fast.py \
        --input-dir sam3/train/data/irrigation_layered \
        --output-dir sam3/train/data/irrigation_canal/train \
        --workers 4 \
        --image-format png \
        --png-compress-level 1

    # Faster/smaller RGB images as JPEG, masks still lossless PNG
    python scripts/split_tif_rgba_masks_fast.py \
        --input-dir sam3/train/data/irrigation_layered \
        --output-dir sam3/train/data/irrigation_canal/train \
        --workers 4 \
        --image-format jpg \
        --jpg-quality 95
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array to uint8 safely."""
    if rgb.dtype == np.uint8:
        return rgb

    # Common case: float image in [0, 1]
    if np.issubdtype(rgb.dtype, np.floating):
        max_val = float(np.nanmax(rgb)) if rgb.size else 1.0
        if max_val <= 1.0:
            rgb = rgb * 255.0
        return np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)

    # Integer types wider than uint8
    return np.asarray(rgb).clip(0, 255).astype(np.uint8)


def split_one(
    tif_path: Path,
    image_out_dir: Path,
    mask_out_dir: Path,
    image_format: str,
    jpg_quality: int,
    png_compress_level: int,
    overwrite: bool,
) -> Tuple[str, str]:
    """
    Split a single layered TIFF.

    Returns:
        (image_output_path, mask_output_path)
    """
    stem = tif_path.stem

    image_ext = "jpg" if image_format == "jpg" else "png"
    image_out = image_out_dir / f"{stem}.{image_ext}"
    mask_out = mask_out_dir / f"{stem}.png"

    if not overwrite and image_out.exists() and mask_out.exists():
        return str(image_out), str(mask_out)

    with Image.open(tif_path) as im:
        arr = np.array(im)

    if arr.ndim != 3 or arr.shape[-1] < 4:
        raise ValueError(
            f"{tif_path} does not look like a layered RGBA/multichannel TIFF. "
            f"Expected shape (H, W, >=4), got {arr.shape}."
        )

    # First 3 channels are the model input image.
    rgb = _to_uint8_rgb(arr[:, :, :3])

    # Last channel is the binary target mask.
    mask_raw = arr[:, :, -1]
    mask = (mask_raw > 0).astype(np.uint8) * 255

    if image_format == "jpg":
        # JPEG is lossy; use high quality if you want faster/smaller RGB files.
        Image.fromarray(rgb, mode="RGB").save(
            image_out,
            quality=jpg_quality,
            optimize=False,
            subsampling=0,
        )
    else:
        # PNG is lossless; compress_level only changes speed/file size.
        Image.fromarray(rgb, mode="RGB").save(
            image_out,
            compress_level=png_compress_level,
        )

    # Masks should always be lossless.
    Image.fromarray(mask, mode="L").save(
        mask_out,
        compress_level=png_compress_level,
    )

    return str(image_out), str(mask_out)


def find_tifs(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    paths: list[Path] = []

    for pat in patterns:
        paths.extend(input_dir.rglob(pat) if recursive else input_dir.glob(pat))

    # Exclude macOS AppleDouble metadata files like ._Canal_ML_Chip_0000.tif
    paths = [p for p in paths if not p.name.startswith("._")]

    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split layered TIFF files into RGB images and binary masks. "
            "Channels 0-2 are saved as the image; the final channel is saved as the mask."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing layered .tif/.tiff files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output split directory. Creates <output-dir>/images and <output-dir>/masks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Start with 4 on shared filesystems.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help=(
            "Output format for RGB images. "
            "png is lossless and safest for benchmarking; jpg is faster/smaller but lossy. "
            "Masks are always saved as PNG."
        ),
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality for --image-format jpg. Higher is better quality/larger file.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="[0-9]",
        help=(
            "PNG compression level for RGB PNG and mask PNG. "
            "0 is fastest/largest, 9 is slowest/smallest. PNG remains lossless."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under --input-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image/mask outputs. By default existing pairs are skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging, e.g. --limit 10.",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if not (0 <= args.jpg_quality <= 100):
        raise ValueError("--jpg-quality must be between 0 and 100.")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    return args


def main() -> None:
    args = parse_args()

    image_out_dir = args.output_dir / "images"
    mask_out_dir = args.output_dir / "masks"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = find_tifs(args.input_dir, args.recursive)

    if args.limit is not None:
        tif_paths = tif_paths[: args.limit]

    if not tif_paths:
        raise FileNotFoundError(
            f"No .tif/.tiff files found in {args.input_dir} "
            f"({'recursive' if args.recursive else 'non-recursive'} search)."
        )

    print(f"Found {len(tif_paths)} TIFF file(s).")
    print(f"Image output: {image_out_dir}")
    print(f"Mask output : {mask_out_dir}")
    print(f"Image format: {args.image_format}")
    print(f"Workers     : {args.workers}")

    common_kwargs = dict(
        image_out_dir=image_out_dir,
        mask_out_dir=mask_out_dir,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
        png_compress_level=args.png_compress_level,
        overwrite=args.overwrite,
    )

    if args.workers == 1:
        for tif_path in tqdm(tif_paths, desc="Splitting TIFFs"):
            split_one(tif_path=tif_path, **common_kwargs)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(split_one, tif_path=tif_path, **common_kwargs)
                for tif_path in tif_paths
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Splitting TIFFs"):
                # Re-raise worker exceptions immediately.
                fut.result()

    print("Done.")
    print(f"Images written to: {image_out_dir}")
    print(f"Masks written to : {mask_out_dir}")


if __name__ == "__main__":
    main()
