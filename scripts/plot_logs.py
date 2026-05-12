import json
from pathlib import Path

import matplotlib.pyplot as plt


# LOG_DIR = Path("experiments/road/logs")
# TRAIN_PATH = LOG_DIR / "train_stats.json"
# VAL_PATH = LOG_DIR / "val_stats.json"

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log-dir",
    type=Path,
    default=Path("experiments/road/logs"),
    help="Directory containing train_stats.json and val_stats.json",
)
args = parser.parse_args()

LOG_DIR = args.log_dir
TRAIN_PATH = LOG_DIR / "train_stats.json"
VAL_PATH = LOG_DIR / "val_stats.json"

print("Using log dir:", LOG_DIR)

def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_runs(rows, epoch_key="Trainer/epoch"):
    if not rows:
        return []

    runs = []
    current = [rows[0]]

    for row in rows[1:]:
        prev_epoch = current[-1][epoch_key]
        curr_epoch = row[epoch_key]

        # New run starts when epoch resets or goes backward
        if curr_epoch <= prev_epoch:
            runs.append(current)
            current = [row]
        else:
            current.append(row)

    runs.append(current)
    return runs


def plot_train_loss(train_runs):
    plt.figure(figsize=(8, 5))
    for i, run in enumerate(train_runs, start=1):
        epochs = [r["Trainer/epoch"] for r in run]
        losses = [r["Losses/train_all_loss"] for r in run]
        label = f"run {i}"
        linewidth = 2.5 if i == len(train_runs) else 1.5
        alpha = 1.0 if i == len(train_runs) else 0.6
        plt.plot(epochs, losses, marker="o", label=label, linewidth=linewidth, alpha=alpha)

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss by Epoch")
    plt.xticks(sorted(set(e for run in train_runs for e in [r["Trainer/epoch"] for r in run])))
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_loss_runs.png", bbox_inches="tight")
    plt.close()


def plot_val_ap(val_runs):
    plt.figure(figsize=(8, 5))
    for i, run in enumerate(val_runs, start=1):
        epochs = [r["Trainer/epoch"] for r in run]
        ap = [r["Meters_train/val_road/detection/coco_eval_bbox_AP"] for r in run]
        label = f"run {i}"
        linewidth = 2.5 if i == len(val_runs) else 1.5
        alpha = 1.0 if i == len(val_runs) else 0.6
        plt.plot(epochs, ap, marker="o", label=label, linewidth=linewidth, alpha=alpha)

    plt.xlabel("Epoch")
    plt.ylabel("Validation AP")
    plt.title("Validation AP by Epoch")
    plt.xticks(sorted(set(e for run in val_runs for e in [r["Trainer/epoch"] for r in run])))
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_ap_runs.png", bbox_inches="tight")
    plt.close()


def print_best_val_per_run(val_runs):
    print("\nBest validation AP per run:")
    for i, run in enumerate(val_runs, start=1):
        best = max(run, key=lambda r: r["Meters_train/val_road/detection/coco_eval_bbox_AP"])
        print(
            f"run {i}: "
            f"best epoch={best['Trainer/epoch']}, "
            f"AP={best['Meters_train/val_road/detection/coco_eval_bbox_AP']:.6f}"
        )


def main():
    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)

    train_runs = split_runs(train_rows)
    val_runs = split_runs(val_rows)

    print(f"Found {len(train_runs)} train run(s)")
    print(f"Found {len(val_runs)} val run(s)")

    for i, run in enumerate(val_runs, start=1):
        epochs = [r["Trainer/epoch"] for r in run]
        aps = [r["Meters_train/val_road/detection/coco_eval_bbox_AP"] for r in run]
        print(f"val run {i}: epochs={epochs}, APs={[round(x, 6) for x in aps]}")

    print_best_val_per_run(val_runs)

    if train_runs:
        plot_train_loss(train_runs)
    if val_runs:
        plot_val_ap(val_runs)

    print("\nSaved:")
    print("  train_loss_runs.png")
    print("  val_ap_runs.png")


if __name__ == "__main__":
    main()