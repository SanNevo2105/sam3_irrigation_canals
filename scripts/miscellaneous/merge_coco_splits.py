"""
merge_coco_splits.py
--------------------
Merge two COCO JSON annotation files (val + test) into a single combined
val_merged split, with globally unique image IDs and annotation IDs.
Also copies (or symlinks) the corresponding TIFF images into a unified
image directory.

Usage
-----
    python scripts/merge_coco_splits.py \
        --ann_a  sam3/train/data/road_coco/val/_annotations.coco.json \
        --ann_b  sam3/train/data/road_coco/test/_annotations.coco.json \
        --img_a  sam3/train/data/road/val \
        --img_b  sam3/train/data/road/test \
        --out_ann  sam3/train/data/road_coco/val_merged/_annotations.coco.json \
        --out_img  sam3/train/data/road/val_merged \
        [--symlink]   # use symlinks instead of copies (saves disk space)
"""

import argparse
import json
import shutil
from pathlib import Path


def load_coco(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def merge_coco(coco_a: dict, coco_b: dict) -> dict:
    """
    Merge two COCO dicts.  IDs from coco_b are re-indexed so they are
    globally unique with respect to coco_a.

    Returns a new COCO dict with merged images, annotations, and categories.
    Categories are taken from coco_a; coco_b categories are assumed compatible
    (same class names / IDs — they originate from the same labelling job).
    """
    # --- images ---
    max_img_id = max((img["id"] for img in coco_a["images"]), default=-1)
    img_id_remap: dict[int, int] = {}  # old coco_b img id  ->  new global id
    merged_images = list(coco_a["images"])
    for img in coco_b["images"]:
        new_id = max_img_id + 1
        max_img_id = new_id
        img_id_remap[img["id"]] = new_id
        merged_images.append({**img, "id": new_id})

    # --- annotations ---
    max_ann_id = max((ann["id"] for ann in coco_a["annotations"]), default=-1)
    merged_annotations = list(coco_a["annotations"])
    for ann in coco_b["annotations"]:
        new_ann_id = max_ann_id + 1
        max_ann_id = new_ann_id
        merged_annotations.append(
            {
                **ann,
                "id": new_ann_id,
                "image_id": img_id_remap[ann["image_id"]],
            }
        )

    # --- categories ---
    # Use coco_a's categories; raise if the sets differ in any meaningful way
    cats_a = {c["id"]: c["name"] for c in coco_a.get("categories", [])}
    cats_b = {c["id"]: c["name"] for c in coco_b.get("categories", [])}
    if cats_a != cats_b:
        print(
            f"WARNING: category sets differ between the two splits.\n"
            f"  coco_a: {cats_a}\n"
            f"  coco_b: {cats_b}\n"
            f"Using coco_a categories. Verify this is intentional."
        )

    return {
        "info": coco_a.get("info", {}),
        "licenses": coco_a.get("licenses", []),
        "categories": coco_a.get("categories", []),
        "images": merged_images,
        "annotations": merged_annotations,
    }


def copy_images(src_dir: Path, dst_dir: Path, *, symlink: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.iterdir():
        if not src_file.is_file():
            continue
        dst_file = dst_dir / src_file.name
        if dst_file.exists():
            print(f"  skip (already exists): {dst_file.name}")
            continue
        if symlink:
            dst_file.symlink_to(src_file.resolve())
            print(f"  symlink: {dst_file.name}")
        else:
            shutil.copy2(src_file, dst_file)
            print(f"  copy:    {dst_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two COCO annotation splits.")
    parser.add_argument("--ann_a", required=True, help="Path to first COCO JSON (e.g. val)")
    parser.add_argument("--ann_b", required=True, help="Path to second COCO JSON (e.g. test)")
    parser.add_argument("--img_a", required=True, help="Directory of images for split A")
    parser.add_argument("--img_b", required=True, help="Directory of images for split B")
    parser.add_argument("--out_ann", required=True, help="Output path for merged COCO JSON")
    parser.add_argument("--out_img", required=True, help="Output directory for merged images")
    parser.add_argument(
        "--symlink",
        action="store_true",
        default=False,
        help="Create symlinks instead of copying image files (saves disk space)",
    )
    args = parser.parse_args()

    ann_a_path = Path(args.ann_a)
    ann_b_path = Path(args.ann_b)
    img_a_dir = Path(args.img_a)
    img_b_dir = Path(args.img_b)
    out_ann_path = Path(args.out_ann)
    out_img_dir = Path(args.out_img)

    # Validate inputs
    for p in (ann_a_path, ann_b_path, img_a_dir, img_b_dir):
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {p}")

    print(f"Loading  {ann_a_path}")
    coco_a = load_coco(ann_a_path)
    print(f"Loading  {ann_b_path}")
    coco_b = load_coco(ann_b_path)

    print(
        f"\nSplit A:  {len(coco_a['images'])} images, "
        f"{len(coco_a['annotations'])} annotations"
    )
    print(
        f"Split B:  {len(coco_b['images'])} images, "
        f"{len(coco_b['annotations'])} annotations"
    )

    merged = merge_coco(coco_a, coco_b)

    print(
        f"\nMerged:   {len(merged['images'])} images, "
        f"{len(merged['annotations'])} annotations"
    )

    # Write merged annotation JSON
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with out_ann_path.open("w") as f:
        json.dump(merged, f)
    print(f"\nWrote annotation JSON -> {out_ann_path}")

    # Copy / symlink images
    print(f"\nCopying images from {img_a_dir} -> {out_img_dir}")
    copy_images(img_a_dir, out_img_dir, symlink=args.symlink)
    print(f"\nCopying images from {img_b_dir} -> {out_img_dir}")
    copy_images(img_b_dir, out_img_dir, symlink=args.symlink)

    print(f"\nDone. Merged image directory: {out_img_dir}")


if __name__ == "__main__":
    main()
