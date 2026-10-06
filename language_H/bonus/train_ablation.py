"""
train_ablation.py — Training Script for Ablated Model (No Positional Embeddings)
================================================================================
Trains Model H (Telugu) without positional embeddings for 15,000+ steps.
Supports:
  - Sharded binary token data loading (ShardLoader / SingleBinLoader)
  - Automatic Mixed Precision (AMP)
  - AdamW optimizer with matrix weight decay decoupling
  - Cosine learning rate decay with linear warmup
  - Periodic evaluation on validation shards (val.bin)
  - Full checkpoint resumption (step_XXXXXXX.pt)
  - Live loss/perplexity logging and plotting to report/bonus/loss_curves/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Allow imports from bonus/ and language_H/
# Ensure bonus directory is first in sys.path
_BONUS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BONUS_DIR.parent

for _p in (str(_PROJECT_DIR / "model"), str(_PROJECT_DIR / "train"), str(_PROJECT_DIR), str(_BONUS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Put _BONUS_DIR strictly at index 0
if sys.path[0] != str(_BONUS_DIR):
    sys.path.insert(0, str(_BONUS_DIR))

from ablation_model import AblatedCausalLanguageModel, build_from_config

# ---------------------------------------------------------------------------
# Path Helper
# ---------------------------------------------------------------------------
def resolve_path(p: str | Path, base_dir: Path) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to base_dir
    cand1 = base_dir / p
    if cand1.exists():
        return cand1
    # Try relative to bonus dir
    cand2 = _BONUS_DIR / p
    if cand2.exists():
        return cand2
    # Try relative to parent
    cand3 = _PROJECT_DIR / p
    if cand3.exists():
        return cand3
    return cand1


# ---------------------------------------------------------------------------
# Dataset & Data Loader (Self-contained for robust Kaggle / local usage)
# ---------------------------------------------------------------------------
class ShardDataset:
    """Memory-mapped dataset for a single uint32 binary token shard."""

    def __init__(self, path: Path, context_length: int):
        self.path = Path(path)
        self.context_length = context_length
        self.n_tokens = self.path.stat().st_size // 4
        self.n_samples = max(0, self.n_tokens - context_length)
        self._mmap = np.memmap(str(self.path), dtype=np.uint32, mode="r")

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, indices: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(indices)
        T = self.context_length
        x = np.empty((B, T), dtype=np.int64)
        y = np.empty((B, T), dtype=np.int64)
        for i, idx in enumerate(indices):
            chunk = self._mmap[idx : idx + T + 1].astype(np.int64)
            x[i] = chunk[:T]
            y[i] = chunk[1 : T + 1]
        return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


class ShardStreamLoader:
    """Streams randomized batches across all train_shard_*.bin files."""

    def __init__(
        self,
        shard_dir: Path,
        context_length: int,
        batch_size: int,
        device: torch.device,
        seed: int = 42,
    ):
        self.shard_dir = Path(shard_dir)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.rng = random.Random(seed)

        self.shard_paths = sorted(self.shard_dir.glob("train_shard_*.bin"))
        if not self.shard_paths:
            # Check for any .bin files if specific pattern not found
            self.shard_paths = sorted(self.shard_dir.glob("*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No binary shards found in {shard_dir}")

        self.epoch = 0
        self.queue: List[Path] = []
        self.current_ds: Optional[ShardDataset] = None
        self.indices: List[int] = []
        self.cursor = 0
        self._refill_shards()

    def _refill_shards(self):
        self.queue = list(self.shard_paths)
        self.rng.shuffle(self.queue)
        self.epoch += 1
        self._load_next_shard()

    def _load_next_shard(self):
        if not self.queue:
            self._refill_shards()
            return
        path = self.queue.pop()
        self.current_ds = ShardDataset(path, self.context_length)
        self.indices = list(range(len(self.current_ds)))
        self.rng.shuffle(self.indices)
        self.cursor = 0

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_ds is None or self.cursor + self.batch_size > len(self.indices):
            self._load_next_shard()
        batch_indices = self.indices[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return self.current_ds.get_batch(batch_indices, self.device)


class SingleBinBatchLoader:
    """Loader for validation/testing from a single binary file (val.bin / test.bin)."""

    def __init__(self, bin_path: Path, context_length: int, batch_size: int, device: torch.device):
        self.bin_path = Path(bin_path)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.ds = ShardDataset(self.bin_path, context_length)
        self.cursor = 0

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + self.batch_size > len(self.ds):
            self.cursor = 0
        batch_indices = list(range(self.cursor, self.cursor + self.batch_size))
        self.cursor += self.batch_size
        return self.ds.get_batch(batch_indices, self.device)


# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------
def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return lr * max(step, 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Optimizer Builder
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, t_cfg: dict) -> torch.optim.Optimizer:
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": float(t_cfg.get("weight_decay", 0.1))},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    lr = float(t_cfg.get("learning_rate", 8e-4))
    b1 = float(t_cfg.get("beta1", 0.9))
    b2 = float(t_cfg.get("beta2", 0.95))
    eps = float(t_cfg.get("epsilon", 1e-8))
    return torch.optim.AdamW(groups, lr=lr, betas=(b1, b2), eps=eps)


# ---------------------------------------------------------------------------
# Validation Evaluator
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_loader: SingleBinBatchLoader,
    eval_batches: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for _ in range(eval_batches):
        x, y = val_loader.next_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    model.train()
    return {"val_loss": round(avg_loss, 6), "val_ppl": round(ppl, 4)}


# ---------------------------------------------------------------------------
# Plotting Loss Curves
# ---------------------------------------------------------------------------
def plot_loss_curves(
    train_log_file: Path,
    out_dir: Path,
    title_suffix: str = " (Ablated: No Positional Embeddings)",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps, train_losses, train_ppls = [], [], []
    val_steps, val_losses, val_ppls = [], [], []

    if not train_log_file.exists():
        return

    with open(train_log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if "train_loss" in rec:
                    steps.append(rec["step"])
                    train_losses.append(rec["train_loss"])
                    train_ppls.append(math.exp(min(rec["train_loss"], 20)))
                if "val_loss" in rec:
                    val_steps.append(rec["step"])
                    val_losses.append(rec["val_loss"])
                    val_ppls.append(rec["val_ppl"])
            except Exception:
                continue

    if not steps:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_losses, label="Train Loss", color="crimson", alpha=0.8)
    if val_steps:
        plt.plot(val_steps, val_losses, marker="o", label="Val Loss", color="navy", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss (nats)")
    plt.title(f"Ablation Training Loss{title_suffix}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_comparison.png", dpi=150)
    plt.close()

    # PPL curve
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_ppls, label="Train PPL", color="darkorange", alpha=0.8)
    if val_steps:
        plt.plot(val_steps, val_ppls, marker="s", label="Val PPL", color="purple", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.title(f"Ablation Perplexity Curve{title_suffix}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "perplexity_comparison.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Model H Ablation (No Positional Embeddings)")
    parser.add_argument("--config", type=str, default=str(_BONUS_DIR / "config.yaml"), help="Path to config YAML")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, default=None, help="Override shard directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"📖 Loading config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using compute device: {device}")

    # Set seeds
    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build Ablated Model
    model = build_from_config(cfg).to(device)
    param_counts = model.count_parameters()
    print("=" * 60)
    print(f"MODEL H ABLATION: NO POSITIONAL EMBEDDINGS")
    print(f"Total Parameters     : {param_counts['total']:,}")
    print(f"Embedding Parameters : {param_counts['embedding']:,}")
    print(f"Non-Embedding Params : {param_counts['non_embedding']:,}")
    print(f"Positional Params    : {param_counts['positional_params']}")
    print("=" * 60)

    # Resolve Directories
    base_dir = _PROJECT_DIR
    shard_dir = resolve_path(args.data_dir or cfg["data"]["shard_dir"], base_dir)
    val_bin = resolve_path(cfg["data"]["val_bin"], base_dir)

    ckpt_out_dir = Path(args.output_dir or cfg["checkpoint"]["output_dir"])
    if not ckpt_out_dir.is_absolute():
        ckpt_out_dir = _BONUS_DIR / "checkpoints"
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg["logging"].get("log_dir", "bonus/logs"))
    if not log_dir.is_absolute():
        log_dir = _BONUS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log_file = log_dir / cfg["logging"].get("train_log_filename", "train_log.jsonl")

    report_loss_dir = _PROJECT_DIR / "report" / "bonus" / "loss_curves"
    report_loss_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Shard directory      : {shard_dir}")
    print(f"📁 Validation bin        : {val_bin}")
    print(f"💾 Checkpoints directory: {ckpt_out_dir}")
    print(f"📊 Logs directory       : {log_dir}")
    print(f"📈 Report loss curves   : {report_loss_dir}")

    # Build Data Loaders
    ctx_len = cfg["model"]["context_length"]
    bs = cfg["training"]["batch_size"]
    train_loader = ShardStreamLoader(shard_dir, ctx_len, bs, device, seed=seed)
    val_loader = SingleBinBatchLoader(val_bin, ctx_len, bs, device)

    # Build Optimizer & Scaler
    t_cfg = cfg["training"]
    optimizer = build_optimizer(model, t_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and t_cfg.get("mixed_precision", True)))

    start_step = 0
    resume_path = args.resume or cfg["checkpoint"].get("resume_from")
    if resume_path and Path(resume_path).exists():
        print(f"🔄 Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"⏩ Resumed at step {start_step}")

    max_steps = t_cfg.get("max_steps", 15000)
    warmup_steps = t_cfg.get("warmup_steps", 3000)
    base_lr = t_cfg.get("learning_rate", 8e-4)
    min_lr = t_cfg.get("min_learning_rate", 8e-5)
    grad_clip = t_cfg.get("max_grad_norm", 1.0)
    log_interval = cfg["logging"].get("log_interval", 100)
    eval_interval = t_cfg.get("eval_interval", 1000)
    ckpt_interval = t_cfg.get("checkpoint_interval", 1000)
    eval_batches = t_cfg.get("eval_batches", 20)

    print(f"\n🚀 Beginning training for {max_steps - start_step} steps (Target: {max_steps})...\n")
    model.train()
    step_start_time = time.time()
    accum_tokens = 0

    for step in range(start_step + 1, max_steps + 1):
        # 1. Update Learning Rate
        lr = cosine_lr_with_warmup(step, warmup_steps, max_steps, base_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 2. Fetch Batch
        x, y = train_loader.next_batch()

        # 3. Forward Pass (AMP)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and t_cfg.get("mixed_precision", True))):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        # 4. Backward Pass & Step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        scaler.step(optimizer)
        scaler.update()

        accum_tokens += x.numel()

        # 5. Periodic Logging
        if step % log_interval == 0 or step == start_step + 1:
            elapsed = time.time() - step_start_time
            tps = int(accum_tokens / max(elapsed, 1e-4))
            train_loss = loss.item()
            train_ppl = math.exp(min(train_loss, 20))

            log_entry = {
                "step": step,
                "train_loss": round(train_loss, 6),
                "train_ppl": round(train_ppl, 4),
                "lr": round(lr, 8),
                "grad_norm": round(grad_norm, 4),
                "tokens_per_sec": tps,
                "elapsed_s": round(elapsed, 2),
            }
            with open(train_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(
                f"[Step {step:6d}/{max_steps}] Loss: {train_loss:.4f} | PPL: {train_ppl:7.2f} | "
                f"LR: {lr:.2e} | Grad: {grad_norm:.2f} | Speed: {tps:,} tok/s"
            )
            step_start_time = time.time()
            accum_tokens = 0

        # 6. Periodic Evaluation
        if step % eval_interval == 0 or step == max_steps:
            print(f"\n🔍 Evaluating on validation set at step {step}...")
            val_res = evaluate_validation(model, val_loader, eval_batches)
            val_loss, val_ppl = val_res["val_loss"], val_res["val_ppl"]
            print(f"📊 [Step {step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}\n")

            val_entry = {
                "step": step,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }
            with open(train_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_entry) + "\n")

            # Update loss curves in report
            plot_loss_curves(train_log_file, report_loss_dir)

        # 7. Periodic Checkpointing
        if step % ckpt_interval == 0 or step == max_steps:
            ckpt_path = ckpt_out_dir / f"step_{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"💾 Checkpoint saved: {ckpt_path.name}")

    print("\n🎉 Pretraining completed successfully!")
    plot_loss_curves(train_log_file, report_loss_dir)


if __name__ == "__main__":
    main()
