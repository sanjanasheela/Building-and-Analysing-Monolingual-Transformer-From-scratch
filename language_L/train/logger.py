"""
logger.py — Training log utilities
===================================
  log_step()       — append one JSON record to the training log file.
  save_loss_curve() — read the log and render a train/val loss PNG.
"""

from __future__ import annotations

import json
from pathlib import Path


def log_step(log_path: Path, record: dict):
    """Append a single JSON-lines record to the training log."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_loss_curve(
    log_path: Path,
    out_path: Path,
    title:    str = "Training Loss Curve",
):
    """
    Read the JSONL training log and save a train + val loss PNG.
    Silently skips if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps, train_losses, val_losses = [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            steps.append(r["step"])
            train_losses.append(r.get("train_loss"))
            if "val_loss" in r:
                val_losses.append((r["step"], r["val_loss"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_losses, label="Train loss", alpha=0.7, linewidth=1)
    if val_losses:
        vx, vy = zip(*val_losses)
        ax.plot(vx, vy, "o-", label="Val loss", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Loss curve saved → {out_path}")
