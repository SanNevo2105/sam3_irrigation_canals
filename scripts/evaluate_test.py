"""
Landslide Test-Set Evaluation using the Fine-tuned SAM3 Model
=============================================================

Reproduces the 4 metrics from the reference test snippet:

    test_loss           – binary cross-entropy between predicted probability
                          map and binary GT mask (see note below)
    test_mean_iou       – per-image IoU, averaged over all test images
    test_global_iou     – globally accumulated IoU across the whole test set
    test_pixel_accuracy – fraction of pixels correctly classified

Also provides show_predictions() for qualitative side-by-side visualisation.

Note on test_loss
-----------------
SAM3 is an instance-detection / segmentation model that does not compute a
scalar loss during inference.  The closest analog to the semantic-segmentation
BCE loss used in the original snippet is computed here as:

    prob_map[pixel] = max over confident detections of  (score_i × mask_i[pixel])

    test_loss = mean_over_images( BCE(prob_map, gt_binary_mask) )

where score_i is the confidence (sigmoid score × presence score) already
produced by PostProcessImage and mask_i is the binary instance mask.

Usage
-----
    # Fine-tuned checkpoint (default behaviour, backward-compatible):
    python scripts/evaluate_test.py

    # Fresh HuggingFace weights, custom dataset folder:
    python scripts/evaluate_test.py \\
        --dataset-root /home/rocky/sam3/sam3/train/data/landslide_dataset \\
        --split test \\
        --load-from-hf \\
        --text-prompt "landslide"

    # Direct image / mask directory override:
    python scripts/evaluate_test.py \\
        --image-dir /path/to/images \\
        --mask-dir  /path/to/masks \\
        --load-from-hf

CLI arguments
-------------
    --dataset-root   Root dataset folder (default: assets/landslide_dataset)
    --split          Sub-folder name, e.g. "test" or "validation" (default: test)
    --image-dir      Direct path to image folder (overrides dataset-root/split)
    --mask-dir       Direct path to mask folder  (overrides dataset-root/split)
    --text-prompt    Text query string (default: "landslide")
    --bpe-path       Path to BPE vocabulary file (auto-resolved if not given)
    --checkpoint-path  Path to a fine-tuned trainer checkpoint
    --load-from-hf   Load fresh weights from HuggingFace instead of a local ckpt
    --num-vis        Number of images to visualise (default: 10)
    --save-path      Output path for the prediction visualisation PNG

Dependencies (already satisfied by the SAM3 training environment):
    torch, torchvision, matplotlib, numpy, pillow
"""

from __future__ import annotations

import argparse
import contextlib
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

# ── Default paths (relative to repository root) ───────────────────────────────
_SAM3_ROOT = Path(sam3.__file__).resolve().parent.parent   # …/sam3/

DATASET_ROOT    = _SAM3_ROOT / "assets" / "landslide_dataset"
TEST_IMAGE_DIR  = DATASET_ROOT / "test" / "images"
TEST_MASK_DIR   = DATASET_ROOT / "test" / "masks"
BPE_PATH        = _SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = (
    _SAM3_ROOT / "experiments" / "landslide" / "checkpoints" / "checkpoint.pt"
)

TEXT_PROMPT = "landslide"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Fixed query-ID used for single-image inference batches ────────────────────
# process_results() keys results by original_image_id; using a constant is fine
# because each call processes exactly one image.
_INFERENCE_ID = 1


# ==============================================================================
# Dataset
# ==============================================================================

class LandslideTestDataset(Dataset):
    """
    Paired (image, binary-mask) test dataset.

    Scans ``TEST_IMAGE_DIR`` for files matching ``image_N.png`` and pairs each
    with the corresponding ``TEST_MASK_DIR/mask_N.png`` by extracting the
    integer suffix *N*.  Only pairs where both files exist are kept.

    Returns
    -------
    pil_image : PIL.Image.Image   – RGB image, original size (128 × 128)
    gt_mask   : np.ndarray        – binary uint8 array {0, 1}, shape (H, W)
    img_path  : str               – absolute path to the image file
    """

    def __init__(
        self,
        image_dir: Path = TEST_IMAGE_DIR,
        mask_dir:  Path = TEST_MASK_DIR,
    ) -> None:
        image_dir = Path(image_dir)
        mask_dir  = Path(mask_dir)

        def _numeric_id(p: Path) -> Optional[int]:
            m = re.search(r"(\d+)", p.stem)
            return int(m.group(1)) if m else None

        images = {
            _numeric_id(p): p
            for p in sorted(image_dir.glob("*.png"))
            if _numeric_id(p) is not None
        }
        masks = {
            _numeric_id(p): p
            for p in sorted(mask_dir.glob("*.png"))
            if _numeric_id(p) is not None
        }

        common = sorted(set(images) & set(masks))
        if not common:
            raise RuntimeError(
                f"No matching (image, mask) pairs found.\n"
                f"  images: {image_dir}\n"
                f"  masks:  {mask_dir}"
            )
        self.samples: List[Tuple[Path, Path]] = [
            (images[i], masks[i]) for i in common
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, np.ndarray, str]:
        img_path, mask_path = self.samples[idx]
        image = PILImage.open(img_path).convert("RGB")
        mask  = np.array(PILImage.open(mask_path).convert("L"))
        mask  = (mask > 0).astype(np.uint8)   # convert to binary 0 / 1
        return image, mask, str(img_path)


# ==============================================================================
# Model loading
# ==============================================================================

def load_model(
    bpe_path:        str | Path = BPE_PATH,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device:          torch.device = DEVICE,
    load_from_hf:    bool = False,
) -> torch.nn.Module:
    """
    Build the SAM3 image model and load weights.

    Two modes are supported:

    **HuggingFace mode** (``load_from_hf=True``)
        Downloads (or uses the cache at
        ``~/.cache/huggingface/hub/models--facebook--sam3/``) the official
        pre-trained weights.  No local checkpoint is required.

    **Local checkpoint mode** (``load_from_hf=False``, default)
        Loads a fine-tuned trainer checkpoint from ``checkpoint_path``.
        The checkpoint structure is::

            {
                "model":     <state_dict>,
                "optimizer": ...,
                "epoch":     ...,
            }

        ``build_sam3_image_model``'s internal ``_load_checkpoint`` strips keys
        by searching for ``"detector."`` which is wrong for trainer-saved state
        dicts; we therefore bypass it and load the weights manually.
        DDP ``"module."`` prefixes are stripped automatically.
    """
    bpe_path = str(bpe_path) if bpe_path is not None else None

    # ── HuggingFace path ──────────────────────────────────────────────────────
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

    # ── Local checkpoint path ─────────────────────────────────────────────────
    checkpoint_path = str(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found:\n  {checkpoint_path}\n\n"
            "Either run training first:\n"
            "  python sam3/train/train.py "
            "-c configs/landslide/landslide_2.yaml\n\n"
            "Or load pre-trained weights from HuggingFace with --load-from-hf."
        )

    # 1. Build model architecture (no weights loaded)
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cpu",          # we move to the target device after loading
        eval_mode=True,
        checkpoint_path=None,  # skip built-in checkpoint loading
        load_from_HF=False,    # skip HuggingFace download
        enable_segmentation=True,
    )

    # 2. Load trainer checkpoint
    ckpt: dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict = ckpt["model"]

    # 3. Strip DDP 'module.' prefix when the model was saved via
    #    DistributedDataParallel (common in multi-GPU training)
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
# Transform  (mirrors landslide_finetune.yaml → landslide_train.val_transforms)
# ==============================================================================

def build_transform() -> ComposeAPI:
    """
    Validation transform pipeline matching ``landslide_finetune.yaml``::

        val_transforms:
          - ComposeAPI:
              - RandomResizeAPI(sizes=1008, max_size=1008, square=True)
              - ToTensorAPI
              - NormalizeAPI(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
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
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# ==============================================================================
# Postprocessor  (mirrors landslide_finetune.yaml → scratch.original_box_postprocessor)
# ==============================================================================

def build_postprocessor() -> PostProcessImage:
    """
    PostProcessImage configured to:

    * return segmentation masks (``iou_type="segm"``)
    * resize masks back to the original 128 × 128 image size
    * use presence scoring (``use_presence=True``) – matches the training config
    * keep only detections with confidence > 0.5; lower-confidence query slots
      (background predictions) are discarded before building the probability map
    """
    return PostProcessImage(
        max_dets_per_img=-1,          # no global cap on detections
        iou_type="segm",              # return binary instance masks
        use_original_sizes_box=True,
        use_original_sizes_mask=True, # resize masks to original resolution
        convert_mask_to_rle=False,    # keep binary tensors, not RLE
        detection_threshold=0.5,      # filter: only confident detections
        to_cpu=True,                  # move results off GPU before returning
        use_presence=True,            # matches use_presence_eval=True in config
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
    Run SAM3 inference on a single PIL image with the text prompt
    ``TEXT_PROMPT = "landslide"``.

    Follows the batched-inference API demonstrated in
    ``examples/sam3_image_batched_inference.ipynb``.

    Parameters
    ----------
    model, transform, postprocessor
        Components built by the helper functions above.
    pil_image
        A single RGB PIL image (any resolution; 128 × 128 for landslide data).
    device
        Target torch device.

    Returns
    -------
    pred_binary : np.ndarray  shape (H, W)  dtype uint8  values {0, 1}
        Semantic prediction mask – 1 where the model detects landslide.
        Built as the logical OR of all confident instance masks.

    prob_map : np.ndarray  shape (H, W)  dtype float32  range [0, 1]
        Per-pixel confidence probability.  Each pixel's value is the
        maximum confidence score of any detected instance that covers it.
        Used to compute the approximate BCE loss.
    """
    w_orig, h_orig = pil_image.size  # PIL gives (width, height)

    # ── 1. Build a Datapoint with a single text query ─────────────────────────
    datapoint = Datapoint(find_queries=[], images=[])
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h_orig, w_orig])]
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=TEXT_PROMPT,
            image_id=0,
            object_ids_output=[],   # unused at inference time
            is_exhaustive=True,     # unused at inference time
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=_INFERENCE_ID,
                original_image_id=_INFERENCE_ID,
                original_category_id=1,
                # Convention from the inference notebook: [width, height].
                # For the 128×128 landslide images this is symmetric, so the
                # order does not affect correctness.
                original_size=[w_orig, h_orig],
                object_id=0,
                frame_index=0,
            ),
        )
    )

    # ── 2. Apply val transforms ───────────────────────────────────────────────
    datapoint = transform(datapoint)

    # ── 3. Collate (batch size = 1) and move to device ───────────────────────
    batch = collate([datapoint], dict_key="eval")["eval"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    # ── 4. Forward pass with BF16 autocast on CUDA ───────────────────────────
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        output = model(batch)

    # ── 5. Post-process → binary per-instance masks + confidence scores ───────
    processed = postprocessor.process_results(output, batch.find_metadatas)

    # ── 6. Merge instance predictions into a single semantic mask ─────────────
    if _INFERENCE_ID in processed and "masks" in processed[_INFERENCE_ID]:
        result     = processed[_INFERENCE_ID]
        inst_masks = result["masks"]   # bool tensor [N, 1, H, W]  (cpu)
        scores     = result["scores"]  # float tensor [N]           (cpu)

        if inst_masks.numel() > 0 and inst_masks.shape[0] > 0:
            # -- Binary prediction: any pixel covered by at least one instance
            pred_binary = (
                inst_masks.squeeze(1)           # [N, H, W]
                .any(dim=0)                     # [H, W]
                .numpy()
                .astype(np.uint8)
            )

            # -- Probability map: max confidence score at each pixel
            masks_f = inst_masks.squeeze(1).float()              # [N, H, W]
            s_exp   = scores[:, None, None].expand_as(masks_f)   # [N, H, W]
            prob_map = (masks_f * s_exp).max(dim=0).values.numpy()  # [H, W]

            return pred_binary, prob_map

    # No confident detections
    return (
        np.zeros((h_orig, w_orig), dtype=np.uint8),
        np.zeros((h_orig, w_orig), dtype=np.float32),
    )


# ==============================================================================
# Metric helpers
# ==============================================================================

def compute_mean_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Per-image binary IoU on the foreground class (label = 1).

    Returns 1.0 when both masks are entirely background (no landslide),
    which is the semantically correct answer: the model correctly identified
    the absence of landslide.
    """
    intersection = int(((pred == 1) & (gt == 1)).sum())
    union        = int(((pred == 1) | (gt == 1)).sum())
    if union == 0:
        return 1.0   # both masks are empty → perfect agreement
    return intersection / union


# ==============================================================================
# Main evaluation loop
# ==============================================================================

@torch.no_grad()
def evaluate_test_set(
    model:         torch.nn.Module,
    transform:     ComposeAPI,
    postprocessor: PostProcessImage,
    dataset:       LandslideTestDataset,
    device:        torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Evaluate the fine-tuned SAM3 model on the landslide test set.

    Mirrors the structure of the reference test snippet::

        for batch in test_loader:
            outputs = model(...)
            preds   = postprocess(outputs)
            loss   += BCE(preds, labels)
            iou    += compute_mean_iou(preds, labels)
            ...

    Parameters
    ----------
    model, transform, postprocessor, dataset, device
        Components built by the helper functions in this module.

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

        # ── Loss  (approximate BCE) ────────────────────────────────────────
        prob_t = torch.tensor(prob_map, dtype=torch.float32)
        gt_t   = torch.tensor(gt_mask,  dtype=torch.float32)
        loss   = F.binary_cross_entropy(
            prob_t.clamp(1e-6, 1.0 - 1e-6),
            gt_t,
            reduction="mean",
        )
        total_loss += loss.item()

        # ── Mean IoU (per image) ───────────────────────────────────────────
        total_iou += compute_mean_iou(pred_binary, gt_mask)

        # ── Pixel accuracy ─────────────────────────────────────────────────
        total_correct += int((pred_binary == gt_mask).sum())
        total_pixels  += int(gt_mask.size)

        # ── Global IoU  (accumulated numerator / denominator) ─────────────
        all_intersection += float(((pred_binary == 1) & (gt_mask == 1)).sum())
        all_union        += float(((pred_binary == 1) | (gt_mask == 1)).sum())

    print()  # newline after the in-place progress line

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
    dataset:       LandslideTestDataset,
    num_images:    int = 10,
    model:         Optional[torch.nn.Module]  = None,
    transform:     Optional[ComposeAPI]       = None,
    postprocessor: Optional[PostProcessImage] = None,
    device:        torch.device               = DEVICE,
    save_path:     str                        = "test_predictions.png",
) -> None:
    """
    Visualise ground-truth and predicted masks for the first ``num_images``
    test images.

    Each row in the output figure shows three panels:

    * **Left**   – original RGB image
    * **Centre** – GT binary mask overlaid in **green** (α = 0.5)
    * **Right**  – predicted binary mask overlaid in **red**  (α = 0.5)

    The figure is saved to ``save_path`` (default: ``test_predictions.png``
    in the current working directory) and displayed interactively.

    Parameters
    ----------
    dataset
        A ``LandslideTestDataset`` instance.
    num_images
        Number of test images to visualise (default 10).
    model, transform, postprocessor, device
        If not ``None``, these are used to generate predictions.  If any
        is ``None`` the function will attempt to use module-level defaults
        (i.e. build them from the default paths/device).  Passing them in
        avoids rebuilding when they are already available in the caller.
    save_path
        Destination path for the saved figure.
    """
    # Build components lazily if not provided
    if model is None or transform is None or postprocessor is None:
        print("[show_predictions] Building model / transform / postprocessor ...")
        if model is None:
            model = load_model(device=device)
        if transform is None:
            transform = build_transform()
        if postprocessor is None:
            postprocessor = build_postprocessor()

    n = min(num_images, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    # Ensure axes is always a 2-D list even when n == 1
    if n == 1:
        axes = [axes]

    for row in range(n):
        pil_image, gt_mask, img_path = dataset[row]
        img_np = np.array(pil_image)

        pred_binary, _ = run_inference_on_image(
            model, transform, postprocessor, pil_image, device
        )
        print(f"  Visualising [{row + 1}/{n}] {os.path.basename(img_path)}")

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
# Entry point  –  mirrors the user's reference test snippet
# ==============================================================================

def _parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments; all have backward-compatible defaults."""
    _SAM3_ROOT_local = Path(sam3.__file__).resolve().parent.parent

    p = argparse.ArgumentParser(
        description=(
            "Landslide Test-Set Evaluation using the SAM3 Model.  "
            "Supports both fine-tuned local checkpoints and fresh "
            "HuggingFace weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = p.add_argument_group("Dataset")
    ds.add_argument(
        "--dataset-root",
        default=str(_SAM3_ROOT_local / "assets" / "landslide_dataset"),
        metavar="DIR",
        help=(
            "Root dataset folder.  Images are expected at "
            "<dataset-root>/<split>/images/ and masks at "
            "<dataset-root>/<split>/masks/ unless --image-dir/--mask-dir override."
        ),
    )
    ds.add_argument(
        "--split",
        default="test",
        metavar="SPLIT",
        help="Dataset split sub-folder (e.g. 'test', 'validation').  "
             "Ignored when --image-dir is given.",
    )
    ds.add_argument(
        "--image-dir",
        default=None,
        metavar="DIR",
        help="Direct path to image folder (overrides --dataset-root/--split).",
    )
    ds.add_argument(
        "--mask-dir",
        default=None,
        metavar="DIR",
        help="Direct path to mask folder (overrides --dataset-root/--split).",
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
        default=str(CHECKPOINT_PATH),
        metavar="FILE",
        help="Path to a fine-tuned trainer checkpoint (.pt file).",
    )
    m.add_argument(
        "--load-from-hf",
        action="store_true",
        help=(
            "Load pre-trained weights from HuggingFace (facebook/sam3) instead "
            "of a local fine-tuned checkpoint.  When set, --checkpoint-path is "
            "ignored."
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
        default="test_predictions.png",
        metavar="FILE",
        help="Output path for the prediction visualisation PNG.",
    )

    return p.parse_args(argv)


if __name__ == "__main__":
    # Enable TF32 for faster matmuls on Ampere+ GPUs (no accuracy impact)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = _parse_args()

    # ── Override the module-level text prompt so run_inference_on_image() uses it
    TEXT_PROMPT = args.text_prompt

    # ── Resolve dataset directories ───────────────────────────────────────────
    if args.image_dir is not None:
        _image_dir = Path(args.image_dir)
        _mask_dir  = Path(args.mask_dir) if args.mask_dir else None
        if _mask_dir is None:
            print(
                "[warn] --mask-dir not provided; evaluation metrics will be skipped.",
                file=sys.stderr,
            )
    else:
        _ds_root   = Path(args.dataset_root)
        _image_dir = _ds_root / args.split / "images"
        _mask_dir  = _ds_root / args.split / "masks"

    if not _image_dir.exists():
        print(
            f"[error] Image directory not found: {_image_dir}\n"
            "        Check --dataset-root / --split / --image-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    # If mask dir does not exist, fall back to no-GT mode
    if _mask_dir is not None and not _mask_dir.exists():
        print(
            f"[warn] Mask directory not found: {_mask_dir}\n"
            "       Evaluation metrics will be skipped.",
            file=sys.stderr,
        )
        _mask_dir = None

    # Override the module-level text prompt
    _text_prompt = args.text_prompt

    print("=" * 65)
    print("Landslide Test-Set Evaluation")
    print("=" * 65)
    print(f"Device          : {DEVICE}")
    if args.load_from_hf:
        print("Weights         : HuggingFace (facebook/sam3)")
    else:
        print(f"Checkpoint      : {args.checkpoint_path}")
    print(f"Image dir       : {_image_dir}")
    print(f"Mask dir        : {_mask_dir or '(none – metrics skipped)'}")
    print(f"Text prompt     : {_text_prompt!r}")
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

    # Build dataset (requires mask_dir; falls back gracefully if absent)
    if _mask_dir is not None:
        test_dataset = LandslideTestDataset(image_dir=_image_dir, mask_dir=_mask_dir)
    else:
        # No GT masks – create a dataset that only yields images
        # (reuse LandslideTestDataset with a dummy mask dir check removed via
        # a minimal subclass)
        class _ImageOnlyDataset(LandslideTestDataset):
            def __init__(self, image_dir):
                import re as _re
                self.image_dir = Path(image_dir)
                def _nid(p):
                    m = _re.search(r"(\d+)", p.stem)
                    return int(m.group(1)) if m else None
                imgs = {_nid(p): p for p in sorted(self.image_dir.glob("*.png")) if _nid(p) is not None}
                if not imgs:
                    raise RuntimeError(f"No PNG images found in: {image_dir}")
                self.samples = [(imgs[i], None) for i in sorted(imgs)]
            def __getitem__(self, idx):
                img_path, _ = self.samples[idx]
                from PIL import Image as _PIL
                image = _PIL.open(img_path).convert("RGB")
                import numpy as _np
                w, h = image.size
                dummy_mask = _np.zeros((h, w), dtype=_np.uint8)
                return image, dummy_mask, str(img_path)
        test_dataset = _ImageOnlyDataset(image_dir=_image_dir)

    print(f"Test set: {len(test_dataset)} image(s)\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Running evaluation (one image at a time) ...")
    test_results = evaluate_test_set(
        model, transform, postprocessor, test_dataset, DEVICE
    )

    print()
    print("Test Results")
    print(f"Test loss         : {test_results['test_loss']:.4f}")
    print(f"Test mean IoU     : {test_results['test_mean_iou']:.4f}")
    print(f"Test global IoU   : {test_results['test_global_iou']:.4f}")
    print(f"Test pixel acc.   : {test_results['test_pixel_accuracy']:.4f}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    print("\nGenerating prediction visualisations ...")
    show_predictions(
        dataset       = test_dataset,
        num_images    = args.num_vis,
        model         = model,
        transform     = transform,
        postprocessor = postprocessor,
        device        = DEVICE,
        save_path     = args.save_path,
    )
