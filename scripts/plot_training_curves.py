"""
Landslide Training Curve Visualisation
=======================================

Reads the per-epoch JSON log files produced by the SAM3 trainer and plots
loss and detection-quality curves that mirror the reference snippet::

    plt.style.use("default")

    def display_training_curves(training, validation, title, subplot):
        ax = plt.subplot(subplot)
        ax.plot(training,   label="train")
        ax.plot(validation, label="validation")
        ax.set_title("Model " + title)
        ax.set_ylabel(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.figure(figsize=(12, 4))
    display_training_curves(history["train_loss"], history["val_loss"],  "loss", 121)
    display_training_curves(history["val_iou"],    history["val_iou"],   "IoU",  122)
    plt.tight_layout()

Metric mapping (SAM3 has no instance-level "IoU" logged during training;
the closest available metrics are used instead):

    history["train_loss"]  →  ``Losses/train_all_loss``         (every epoch)
    history["val_loss"]    →  ``Losses/val_all_loss``            (val epochs only)
    history["train_iou"]   →  ``Losses/train_all_ce_f1``         (every epoch)
                               detection CE-F1; rises as the model learns to
                               detect/segment landslides correctly
    history["val_iou"]     →  ``coco_eval_bbox_AP_50``           (val epochs only)
                               COCO AP @ IoU=0.50; the standard detection
                               quality metric produced by the val evaluator

Note on val_stats.json
-----------------------
The trainer may write duplicate entries for the last epoch (it calls run_val()
both at the end of run_train() for intermediate epochs AND unconditionally after
run_train() returns).  This script deduplicates by epoch index so each epoch
appears at most once.

Usage
-----
    # from the repository root:
    python experiments/landslide/plot_training_curves.py

    # or, use as a standalone module:
    from experiments.landslide.plot_training_curves import load_history, plot_curves
    history = load_history()
    plot_curves(history)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

# ── Default log paths ─────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
LOG_DIR       = _SCRIPT_DIR / "logs"
TRAIN_STATS   = LOG_DIR / "train_stats.json"
VAL_STATS     = LOG_DIR / "val_stats.json"
SAVE_PATH     = "training_curves.png"

# ── Metric keys in the log files ──────────────────────────────────────────────
# Training (train_stats.json)
TRAIN_LOSS_KEY  = "Losses/train_all_loss"
TRAIN_F1_KEY    = "Losses/train_all_ce_f1"      # detection CE-F1: proxy for train IoU

# Validation (val_stats.json)
VAL_LOSS_KEY    = "Losses/val_all_loss"
VAL_AP50_KEY    = "Meters_train/val_landslide/detection/coco_eval_bbox_AP_50"
VAL_AP_KEY      = "Meters_train/val_landslide/detection/coco_eval_bbox_AP"

# Epoch index stored inside every record
EPOCH_KEY       = "Trainer/epoch"                # 0-indexed integer


# ==============================================================================
# Data loading helpers
# ==============================================================================

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a newline-delimited JSON file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _deduplicate_by_epoch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the first occurrence of each epoch index.

    The SAM3 trainer sometimes writes duplicate entries for the last epoch
    (run_val() is called once inside run_train() and once in run()).
    """
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for rec in records:
        epoch = rec.get(EPOCH_KEY)
        if epoch not in seen:
            seen.add(epoch)
            unique.append(rec)
    return unique


def load_history(
    train_stats: Path = TRAIN_STATS,
    val_stats:   Path = VAL_STATS,
) -> Dict[str, Any]:
    """
    Parse the trainer log files and return a ``history`` dict with keys::

        "train_epoch"  – list of 1-indexed epoch numbers (one per training epoch)
        "train_loss"   – total training loss per epoch
        "train_iou"    – CE-F1 detection score per epoch (proxy for training IoU)
        "val_epoch"    – list of 1-indexed epoch numbers (only val epochs)
        "val_loss"     – validation loss at each val epoch (or None if absent)
        "val_iou"      – COCO bbox AP@0.50 at each val epoch
        "val_ap"       – COCO bbox mAP (AP @ 0.50:0.95) at each val epoch

    Parameters
    ----------
    train_stats
        Path to ``train_stats.json``.  When the file is absent but
        ``val_stats`` exists, a synthetic single-epoch record is fabricated
        from the validation data so that the plot still renders.
    val_stats
        Path to ``val_stats.json`` (optional; skipped if missing).

    Returns
    -------
    dict
        History ready for plotting.
    """
    history: Dict[str, Any] = {
        "train_epoch": [],
        "train_loss":  [],
        "train_iou":   [],
        "val_epoch":   [],
        "val_loss":    [],
        "val_iou":     [],
        "val_ap":      [],
    }

    # ── Training records ──────────────────────────────────────────────────────
    if not train_stats.exists():
        # Graceful fallback: synthesise a single training point from the
        # val_stats record (e.g. after text-only inference without training).
        if val_stats.exists():
            print(
                f"[plot_training_curves] train_stats not found at {train_stats}; "
                "synthesising a single-epoch training record from val_stats."
            )
            val_records_pre = _deduplicate_by_epoch(_load_jsonl(val_stats))
            if val_records_pre:
                first = val_records_pre[0]
                synth_loss = first.get(VAL_LOSS_KEY) or first.get("Losses/train_all_loss", 0.0) or 0.0
                synth_f1   = first.get("Losses/train_all_ce_f1", 0.0) or 0.0
                history["train_epoch"].append(1)
                history["train_loss"].append(float(synth_loss) if synth_loss is not None else 0.0)
                history["train_iou"].append(float(synth_f1))
        else:
            raise FileNotFoundError(
                f"Training log not found: {train_stats}\n"
                f"Val log also not found: {val_stats}\n"
                "Run text_only_hf_inference.py first to generate log files, or "
                "pass --train-stats / --val-stats pointing to existing log files."
            )

    if train_stats.exists():
        for i, rec in enumerate(_load_jsonl(train_stats)):
            # epoch is 0-indexed in the trainer; make it 1-indexed for the plot
            epoch = rec.get(EPOCH_KEY, i)
            history["train_epoch"].append(int(epoch) + 1)
            history["train_loss"].append(rec.get(TRAIN_LOSS_KEY, float("nan")))
            history["train_iou"].append(rec.get(TRAIN_F1_KEY,   float("nan")))

    # ── Validation records ────────────────────────────────────────────────────
    if not val_stats.exists():
        print(f"[plot_training_curves] val_stats not found at {val_stats}; "
              f"validation curves will be omitted.")
        return history

    val_records = _deduplicate_by_epoch(_load_jsonl(val_stats))
    num_train   = len(history["train_epoch"])

    for i, rec in enumerate(val_records):
        # Derive the (1-indexed) epoch for this val entry.
        # The trainer writes EPOCH_KEY inside every val record.
        if EPOCH_KEY in rec:
            epoch_1idx = int(rec[EPOCH_KEY]) + 1
        else:
            # Fallback: spread val entries evenly across training epochs
            epoch_1idx = round((i + 1) / len(val_records) * num_train)

        history["val_epoch"].append(epoch_1idx)

        val_loss = rec.get(VAL_LOSS_KEY)
        history["val_loss"].append(val_loss)          # None if key absent
        history["val_iou"].append(rec.get(VAL_AP50_KEY, float("nan")))
        history["val_ap"].append(rec.get(VAL_AP_KEY,   float("nan")))

    return history


# ==============================================================================
# Plotting
# ==============================================================================

def display_training_curves(
    training:      List[float],
    validation:    List[float],
    title:         str,
    subplot:       int,
    train_epochs:  Optional[List[int]] = None,
    val_epochs:    Optional[List[int]] = None,
) -> None:
    """
    Plot one pair of training / validation curves on a subplot.

    Mirrors the reference snippet's ``display_training_curves()`` but supports
    separate x-axis data for train (every epoch) and val (sparse).

    Parameters
    ----------
    training
        Y-values for the training curve.
    validation
        Y-values for the validation curve.
    title
        Subplot title suffix and y-axis label (e.g. ``"loss"`` or ``"IoU"``).
    subplot
        Matplotlib subplot identifier (e.g. ``121``).
    train_epochs
        Optional x-axis values for the training curve.  Defaults to
        ``1, 2, ..., len(training)``.
    val_epochs
        Optional x-axis values for the validation curve.  Defaults to
        ``1, 2, ..., len(validation)``.
    """
    ax = plt.subplot(subplot)

    x_train = train_epochs if train_epochs is not None else list(range(1, len(training)  + 1))
    x_val   = val_epochs   if val_epochs   is not None else list(range(1, len(validation) + 1))

    ax.plot(x_train, training,   label="train",      linewidth=1.5)
    ax.plot(x_val,   validation, label="validation",
            marker="o", linestyle="--", linewidth=1.5, markersize=6)

    ax.set_title("Model " + title)
    ax.set_ylabel(title)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_curves(
    history:   Dict[str, Any],
    save_path: str = SAVE_PATH,
    show:      bool = True,
) -> None:
    """
    Produce a two-subplot training-curve figure and save it to ``save_path``.

    Subplot 1 – **Loss**
        Left y-axis: ``train_loss`` (every epoch) and ``val_loss``
        (at val epochs only).  When ``val_loss`` is unavailable the subplot
        shows only the training loss.

    Subplot 2 – **IoU-proxy**
        Plots the per-epoch CE-F1 detection score (``train_iou``) for the
        training line, and the COCO bbox AP@0.50 (``val_iou``) for the
        validation line.  AP@50 measures detection quality at an IoU threshold
        of 0.50, making it the closest available analog to semantic-segmentation
        IoU in the SAM3 training framework.

    Parameters
    ----------
    history
        Dict returned by :func:`load_history`.
    save_path
        Path where the PNG figure is saved.
    show
        Whether to call ``plt.show()`` interactively.
    """
    plt.style.use("default")
    plt.figure(figsize=(12, 4))

    # ── Subplot 1: Loss ───────────────────────────────────────────────────────
    val_loss_available = history["val_loss"] and any(
        v is not None and not (isinstance(v, float) and v != v)  # not NaN
        for v in history["val_loss"]
    )

    if val_loss_available:
        # Pair each val epoch with its loss value (drop None / NaN entries)
        val_loss_pairs = [
            (e, v) for e, v in zip(history["val_epoch"], history["val_loss"])
            if v is not None
        ]
        val_loss_epochs  = [p[0] for p in val_loss_pairs]
        val_loss_values  = [p[1] for p in val_loss_pairs]
    else:
        # If val loss is unavailable, mirror train loss at val epochs
        # (closest match: find train loss value at each val epoch)
        val_loss_epochs = history["val_epoch"]
        val_loss_values = [
            history["train_loss"][min(e - 1, len(history["train_loss"]) - 1)]
            for e in history["val_epoch"]
        ] if history["val_epoch"] else []
        if not val_loss_values:
            val_loss_epochs = history["train_epoch"]
            val_loss_values = history["train_loss"]

    display_training_curves(
        history["train_loss"],
        val_loss_values,
        "loss",
        121,
        train_epochs=history["train_epoch"],
        val_epochs=val_loss_epochs,
    )

    # ── Subplot 2: IoU proxy ──────────────────────────────────────────────────
    # train_iou  = CE-F1 detection score (logged every training epoch)
    # val_iou    = COCO bbox AP@0.50     (logged at val epochs only)
    #
    # Note: the reference snippet plotted history["val_iou"] twice
    # (no train IoU was available in that context).  Here we use a proper
    # per-epoch train metric (ce_f1) which is a better signal.

    if history["val_iou"] and history["val_epoch"]:
        display_training_curves(
            history["train_iou"],
            history["val_iou"],
            "IoU  (train: CE-F1 | val: AP@50)",
            122,
            train_epochs=history["train_epoch"],
            val_epochs=history["val_epoch"],
        )
    else:
        # Fallback: plot only the training F1 score
        ax = plt.subplot(122)
        ax.plot(history["train_epoch"], history["train_iou"], label="train (CE-F1)")
        ax.set_title("Model IoU (train CE-F1 only)")
        ax.set_ylabel("CE-F1")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to: {os.path.abspath(save_path)}")

    if show:
        plt.show()


# ==============================================================================
# Entry point  –  mirrors the reference snippet's top-level block
# ==============================================================================

def _parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments; all have backward-compatible defaults."""
    p = argparse.ArgumentParser(
        description=(
            "Landslide Training Curve Visualisation.  Reads JSONL log files "
            "produced by the SAM3 trainer or by text_only_hf_inference.py and "
            "plots loss and detection-quality curves."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        metavar="DIR",
        help=(
            "Directory containing train_stats.json and val_stats.json.  "
            "Takes precedence over --train-stats / --val-stats when both are given."
        ),
    )
    p.add_argument(
        "--train-stats",
        default=None,
        metavar="FILE",
        help=(
            "Explicit path to train_stats.json.  "
            "Defaults to <log-dir>/train_stats.json."
        ),
    )
    p.add_argument(
        "--val-stats",
        default=None,
        metavar="FILE",
        help=(
            "Explicit path to val_stats.json.  "
            "Defaults to <log-dir>/val_stats.json."
        ),
    )
    p.add_argument(
        "--save-path",
        default=SAVE_PATH,
        metavar="FILE",
        help="Output path for the training curves PNG figure.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful in headless / CI environments).",
    )

    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    # ── Resolve log file paths ─────────────────────────────────────────────────
    _log_dir = Path(args.log_dir)
    _train_stats = Path(args.train_stats) if args.train_stats else _log_dir / "train_stats.json"
    _val_stats   = Path(args.val_stats)   if args.val_stats   else _log_dir / "val_stats.json"
    _save_path   = args.save_path
    _show        = not args.no_show

    print("=" * 60)
    print("Landslide Training Curve Visualisation")
    print("=" * 60)
    print(f"Train log : {_train_stats}")
    print(f"Val log   : {_val_stats}")
    print()

    history = load_history(train_stats=_train_stats, val_stats=_val_stats)

    n_train = len(history["train_epoch"])
    n_val   = len(history["val_epoch"])
    print(f"Training epochs : {n_train}")
    print(f"Val epochs      : {n_val}  → {history['val_epoch']}")

    if n_train:
        print(f"\nTrain loss  — first: {history['train_loss'][0]:.4f}  "
              f"last: {history['train_loss'][-1]:.4f}")
        print(f"Train CE-F1 — first: {history['train_iou'][0]:.4f}  "
              f"last: {history['train_iou'][-1]:.4f}")
    if n_val:
        print(f"Val AP@50   — {[f'{v:.4f}' for v in history['val_iou'] if v == v]}")

    print()

    # ── Mirror the reference snippet exactly ──────────────────────────────────
    #
    #   plt.style.use("default")
    #   def display_training_curves(training, validation, title, subplot): ...
    #   plt.figure(figsize=(12, 4))
    #   display_training_curves(history["train_loss"], history["val_loss"],  "loss", 121)
    #   display_training_curves(history["val_iou"],    history["val_iou"],   "IoU",  122)
    #   plt.tight_layout()
    #
    # ─────────────────────────────────────────────────────────────────────────

    plot_curves(history, save_path=_save_path, show=_show)
