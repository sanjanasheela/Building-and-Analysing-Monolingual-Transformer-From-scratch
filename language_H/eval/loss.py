from __future__ import annotations

import sys
from pathlib import Path
import json
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
from tokenizers import Tokenizer


def compute_bits_per_byte(total_loss_nats: float, total_tokens: int, test_txt_path: Path) -> float:
    """Compute Bits-Per-Byte (BPB) using the UTF-8 byte length of the test text."""
    if not test_txt_path.exists():
        return 0.0
    with open(test_txt_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    total_bytes = len(text_data.encode("utf-8"))
    if total_bytes == 0:
        return 0.0
    
    # Total bits = total_loss_nats / ln(2)
    total_bits = (total_loss_nats * total_tokens) / torch.log(torch.tensor(2.0)).item()
    bpb = total_bits / total_bytes
    return float(bpb)


def main():
    proj_dir = Path(_PROJECT_DIR)
    config_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/configs/51.yaml")
    
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Build model & load latest checkpoint weights
    model = build_from_config(cfg).to(device)
    checkpoint_dir = proj_dir / "checkpoints" / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("step_*.pt"))
    
    if checkpoint_files:
        def extract_step(path: Path) -> int:
            match = re.search(r"step_(\d+)\.pt", path.name)
            return int(match.group(1)) if match else -1

        checkpoint_files.sort(key=extract_step)
        latest_path = checkpoint_files[-1]
        print(f"📦 Loading latest checkpoint weights from: {latest_path.name}")
        checkpoint = torch.load(latest_path, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    vocab_size = cfg["tokenizer"]["vocab_size"]

    # Compute test loss on test.bin
    test_bin_path = proj_dir / "token_shards_7500" / "test.bin"
    test_txt_path = proj_dir / "data" / "final_data_set_7500" / "test.txt"
    
    raw_loader = SingleBinLoader(
        bin_path=test_bin_path,
        context_length=cfg["model"]["context_length"],
        batch_size=cfg["evaluation"]["eval_batch_size"],
        device=device,
    )

    class SafeSanitizedLoader:
        def __init__(self, loader, vocab_size):
            self.loader = loader
            self.vocab_size = vocab_size
        def next_batch(self):
            x, y = self.loader.next_batch()
            return torch.clamp(x, min=0, max=self.vocab_size - 1), torch.clamp(y, min=0, max=self.vocab_size - 1)

    loader = SafeSanitizedLoader(raw_loader, vocab_size)
    n_batches = cfg["training"].get("eval_batches", 50)
    
    print(f"🔍 Computing test cross-entropy loss over {n_batches} batches...")
    test_loss = cross_entropy(model, loader, n_batches=n_batches, device=device)
    test_ppl = torch.exp(torch.tensor(min(test_loss, 20))).item()
    
    # Calculate BPB
    total_eval_tokens = n_batches * cfg["model"]["context_length"] * cfg["evaluation"]["eval_batch_size"]
    test_bpb = compute_bits_per_byte(test_loss, total_eval_tokens, test_txt_path)

    print(f"📉 Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.4f} | Test BPB: {test_bpb:.4f}")

    # Parse train_log.jsonl for real metrics history
    train_log_path = proj_dir / "checkpoints" / "logs" / "train_log.jsonl"
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    if train_log_path.exists():
        print(f"📖 Parsing training logs from {train_log_path}...")
        with open(train_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "train_loss" in data and "step" in data:
                        train_steps.append(int(data["step"]))
                        train_losses.append(float(data["train_loss"]))
                    if "val_loss" in data and "step" in data:
                        val_steps.append(int(data["step"]))
                        val_losses.append(float(data["val_loss"]))
                except json.JSONDecodeError:
                    continue

    train_ppl = [torch.exp(torch.tensor(min(l, 20))).item() for l in train_losses]
    val_ppl = [torch.exp(torch.tensor(min(l, 20))).item() for l in val_losses] if val_losses else []

    # Generate plots and save to report/loss_curves/
    report_loss_dir = proj_dir / "report" / "loss_curves"
    report_loss_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_losses, label='Train Loss', color='royalblue', alpha=0.8)
    if val_steps:
        plt.plot(val_steps, val_losses, marker='s', label='Validation Loss', color='darkorange', linewidth=2)
    plt.axhline(y=test_loss, color='crimson', linestyle='--', label=f'Test Loss ({test_loss:.4f})')
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Train, Validation, and Test Loss Curves')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    loss_plot_path = report_loss_dir / "loss_comparison.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_ppl, label='Train PPL', color='forestgreen', alpha=0.8)
    if val_steps and val_ppl:
        plt.plot(val_steps, val_ppl, marker='s', label='Validation PPL', color='purple', linewidth=2)
    plt.axhline(y=test_ppl, color='crimson', linestyle='--', label=f'Test PPL ({test_ppl:.2f})')
    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity (PPL)')
    plt.title('Train, Validation, and Test Perplexity Curves')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    ppl_plot_path = report_loss_dir / "perplexity_comparison.png"
    plt.savefig(ppl_plot_path, dpi=150)
    plt.close()

    # Generate Final Summary TXT Report
    tables_dir = proj_dir / "report" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    final_summary_path = tables_dir / "final_evaluation_summary.txt"

    with open(final_summary_path, "w", encoding="utf-8") as f:
        f.write("========================================================\n")
        f.write("MODEL H — FINAL INTRINSIC & EVALUATION SUMMARY REPORT\n")
        f.write("========================================================\n\n")
        f.write(f"Test Cross-Entropy Loss : {test_loss:.4f}\n")
        f.write(f"Test Perplexity (PPL)   : {test_ppl:.4f}\n")
        f.write(f"Test Bits-Per-Byte (BPB): {test_bpb:.4f}\n\n")
        f.write("--------------------------------------------------------\n")
        f.write("INTRINSIC EVALUATION METRICS NOTE:\n")
        f.write("- Bits-per-byte (BPB) normalizes cross-entropy across varying tokenizers.\n")
        f.write("- Perplexity measures predictive uncertainty on held-out text shards.\n")
        f.write("========================================================\n")

    print(f"\n📊 Successfully generated plots and final summary report:")
    print(f"  - Loss plot       : {loss_plot_path}")
    print(f"  - Perplexity plot : {ppl_plot_path}")
    print(f"  - Final Summary   : {final_summary_path}")

if __name__ == "__main__":
    main()