from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def split_tif_file(tif_path: Path, image_out_dir: Path, mask_out_dir: Path):
    img = Image.open(tif_path)
    arr = np.array(img)

    if arr.ndim != 3 or arr.shape[-1] < 4:
        raise ValueError(
            f"{tif_path} does not look like RGBA/multichannel TIFF. "
            f"Got shape {arr.shape}"
        )

    # First 3 channels = RGB image
    rgb = arr[:, :, :3].astype(np.uint8)

    # Last channel = binary mask
    mask = arr[:, :, -1]

    # Convert 0/1 mask to 0/255 mask
    mask = (mask > 0).astype(np.uint8) * 255

    stem = tif_path.stem

    Image.fromarray(rgb).save(image_out_dir / f"{stem}.png")
    Image.fromarray(mask, mode="L").save(mask_out_dir / f"{stem}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Folder containing .tif files")
    parser.add_argument("--output-dir", required=True, help="Output dataset folder")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern, e.g. *.tif or *.tiff")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    image_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"

    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(input_dir.glob(args.pattern))

    if not tif_paths:
        raise FileNotFoundError(f"No files found in {input_dir} matching {args.pattern}")

    for tif_path in tqdm(tif_paths, desc="Splitting TIFFs"):
        split_tif_file(tif_path, image_out_dir, mask_out_dir)

    print(f"Done.")
    print(f"Images saved to: {image_out_dir}")
    print(f"Masks saved to:  {mask_out_dir}")


if __name__ == "__main__":
    main()