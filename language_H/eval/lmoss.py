from __future__ import annotations

import sys
from pathlib import Path
import re
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path Setup ─────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _EVAL_DIR.parent
sys.path.append(str(_PROJECT_DIR / "train"))
sys.path.append(str(_EVAL_DIR))

from model import build_from_config
from data_loader import SingleBinLoader
from lm_eval import cross_entropy


def main():
    proj_dir = Path(_PROJECT_DIR)
    config_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/configs/51.yaml")
    
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Build model framework
    model = build_from_config(cfg).to(device)
    
    checkpoint_dir = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/checkpoints/checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("step_*.pt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def extract_step(path: Path) -> int:
        match = re.search(r"step_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    checkpoint_files.sort(key=extract_step)

    val_steps, val_losses = [], []
    test_steps, test_losses = [], []

    vocab_size = 7500
    test_bin_path = "language_H/token_shards_7500/test.bin"
    val_bin_path = "language_H/token_shards_7500/val.bin" # Assumes val.bin exists alongside test.bin

    class SafeSanitizedLoader:
        def __init__(self, loader, vocab_size):
            self.loader = loader
            self.vocab_size = vocab_size
        def next_batch(self):
            x, y = self.loader.next_batch()
            return torch.clamp(x, min=0, max=self.vocab_size - 1), torch.clamp(y, min=0, max=self.vocab_size - 1)

    n_batches = cfg["training"].get("eval_batches", 50)

    print(f"📦 Evaluating {len(checkpoint_files)} checkpoints from {checkpoint_dir}...")

    for ckpt_path in checkpoint_files:
        step = extract_step(ckpt_path)
        print(f"\n🔍 Processing Checkpoint Step {step} ({ckpt_path.name})...")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Check if validation loss is already saved inside the checkpoint dictionary metadata
        if isinstance(checkpoint, dict) and "val_loss" in checkpoint:
            v_loss = float(checkpoint["val_loss"])
            val_steps.append(step)
            val_losses.append(v_loss)
            print(f"  ⚡ Found saved validation loss in checkpoint: {v_loss:.4f}")
        
        # Load model weights for evaluation
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # If val_bin exists and wasn't cached in dict, compute it
        if val_bin_path.exists() and step not in val_steps:
            val_loader = SafeSanitizedLoader(SingleBinLoader(
                bin_path=val_bin_path,
                context_length=cfg["model"]["context_length"],
                batch_size=cfg["evaluation"]["eval_batch_size"],
                device=device,
            ), vocab_size)
            v_loss = cross_entropy(model, val_loader, n_batches=n_batches, device=device)
            val_steps.append(step)
            val_losses.append(v_loss)
            print(f"  📊 Computed Validation Loss: {v_loss:.4f}")

        # Compute Test Loss for the latest/all checkpoints
        if test_bin_path.exists():
            test_loader = SafeSanitizedLoader(SingleBinLoader(
                bin_path=test_bin_path,
                context_length=cfg["model"]["context_length"],
                batch_size=cfg["evaluation"]["eval_batch_size"],
                device=device,
            ), vocab_size)
            t_loss = cross_entropy(model, test_loader, n_batches=n_batches, device=device)
            test_steps.append(step)
            test_losses.append(t_loss)
            print(f"  📉 Computed Test Loss: {t_loss:.4f}")

    # Plot Validation & Test Losses
    report_loss_dir = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/bonus/bonus_logs")
    report_loss_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    if val_steps:
        plt.plot(val_steps, val_losses, marker='s', label='Validation Loss', color='darkorange', linewidth=2)
    if test_steps:
        plt.plot(test_steps, test_losses, marker='o', linestyle='--', label='Test Loss', color='crimson', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Validation and Test Loss Curves Across Checkpoints')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plot_path = report_loss_dir / "val_test_loss_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n📊 Successfully generated validation and test loss plot: {plot_path}")


if __name__ == "__main__":
    main()