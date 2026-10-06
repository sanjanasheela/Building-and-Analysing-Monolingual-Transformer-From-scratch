"""
train.py — Phase 2 Pretraining Orchestrator
============================================
Thin entry point. All logic lives in sibling modules:

    model.py        — CausalLanguageModel + build_from_config()
    scheduler.py    — get_lr() / cosine_lr_with_warmup()
    data_loader.py  — ShardLoader
    checkpoint.py   — save / load / analyse / print
    logger.py       — log_step / save_loss_curve

Usage:
    python train/train.py
    python train/train.py --config configs/telugu_1.yaml
    python train/train.py --resume checkpoints/step_0001000.pt
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  pip install pyyaml")

# ---------------------------------------------------------------------------
# Path bootstrap — all sibling modules share this pattern
# ---------------------------------------------------------------------------
_TRAIN_DIR   = Path(__file__).resolve().parent
_PROJECT_DIR = _TRAIN_DIR.parent

for _p in (str(_TRAIN_DIR), str(_PROJECT_DIR / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model       import CausalLanguageModel, build_from_config
from scheduler   import get_lr
from data_loader import ShardLoader, SingleBinLoader
from checkpoint  import (save_checkpoint, load_checkpoint,
                          analyse_checkpoint, print_checkpoint_analysis)
from logger      import log_step, save_loss_curve

DEFAULT_CFG = _PROJECT_DIR / "configs" / "telugu_1.yaml"


# ===========================================================================
# Validation
# ===========================================================================

@torch.no_grad()
def evaluate(
    model:        CausalLanguageModel,
    loader:       ShardLoader,
    eval_batches: int,
) -> dict:
    """Run `eval_batches` forward passes and return val loss + perplexity."""
    model.eval()
    total_loss = 0.01
    total_tok  = 0

    for _ in range(eval_batches):
        x, y   = loader.next_batch()
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        logits    = model(x)
        loss      = F.cross_entropy(
            logits.view(-1, raw_model.vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tok  += y.numel()

    avg_loss = total_loss / total_tok
    ppl      = math.exp(min(avg_loss, 100))   # cap avoids overflow on bad init
    model.train()
    return {"val_loss": round(avg_loss, 6), "val_ppl": round(ppl, 4)}


# ===========================================================================
# Optimiser factory — fully config-driven
# ===========================================================================

def build_optimizer(model: CausalLanguageModel, t_cfg: dict) -> torch.optim.Optimizer:
    """
    Build AdamW / Adam / SGD from training config.
    Weight-decay is applied only to matrix weights (dim >= 2).
    """
    decay    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <  2]
    groups   = [
        {"params": decay,    "weight_decay": t_cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    name = t_cfg["optimizer"].lower()
    lr   = t_cfg["learning_rate"]
    b1, b2, eps = t_cfg["beta1"], t_cfg["beta2"], t_cfg["epsilon"]

    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(b1, b2), eps=eps)
    if name == "adam":
        return torch.optim.Adam(groups,  lr=lr, betas=(b1, b2), eps=eps)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=t_cfg.get("momentum", 0.9))

    sys.exit(f"Unknown optimizer '{t_cfg['optimizer']}'. Supported: AdamW, Adam, SGD")


# ===========================================================================
# Main training loop
# ===========================================================================

def train(cfg: dict, resume_from: str | None = None):
    # ── Unpack config sections ─────────────────────────────────────────────
    t_cfg  = cfg["training"]
    d_cfg  = cfg["data"]
    c_cfg  = cfg["checkpoint"]
    l_cfg  = cfg["logging"]
    hw_cfg = cfg["hardware"]

    # ── Device & reproducibility ───────────────────────────────────────────
    seed   = t_cfg["seed"]
    device = torch.device(
        "cuda" if hw_cfg["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"\n  Device : {device} (Available GPUs: {gpu_count})")

    # ── Directories ────────────────────────────────────────────────────────
    ckpt_dir = Path(c_cfg["output_dir"])
    log_dir  = Path(l_cfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path     = log_dir / l_cfg["train_log_filename"]
    analysis_log = log_dir / l_cfg["checkpoint_log_filename"]
    curve_path   = log_dir / l_cfg["loss_curve_filename"]
    curve_title  = l_cfg["loss_curve_title"]

    # ── Model ──────────────────────────────────────────────────────────────
    raw_model = build_from_config(cfg).to(device)
    pi        = raw_model.count_parameters()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model
    print(f"\n  {cfg.get('model_name', 'Model')} — parameter count")
    print(f"    Total          : {pi['total']:>12,}  ({pi['total']/1e6:.2f}M)")
    print(f"    Embedding      : {pi['embedding']:>12,}")
    print(f"    Non-embedding  : {pi['non_embedding']:>12,}")

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, t_cfg)
    print(f"  Optimizer : {t_cfg['optimizer']}  |  Scheduler : {t_cfg['scheduler']}")

    # ── AMP ────────────────────────────────────────────────────────────────
    amp_on = hw_cfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    # ── Data loaders ───────────────────────────────────────────────────────
    shard_dir    = Path(d_cfg["shard_dir"])
    val_bin      = Path(d_cfg["val_bin"])
    ctx_len      = cfg["model"]["context_length"]
    batch_size   = t_cfg["batch_size"]
    train_loader = ShardLoader(shard_dir, ctx_len, batch_size, device, seed)
    val_loader   = SingleBinLoader(val_bin, ctx_len, batch_size, device, seed + 1)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step  = 0
    sched_state = {}
    target      = resume_from or c_cfg.get("resume_from")
    if target:
        start_step, sched_state = load_checkpoint(target, model, optimizer, device)
        start_step += 1
        # Fast-forward train_loader to exact resume position
        total_batches_to_skip = (start_step - 1) * t_cfg["gradient_accumulation_steps"]
        train_loader.fast_forward(total_batches_to_skip)

    # ── Loop config ────────────────────────────────────────────────────────
    grad_accum    = t_cfg["gradient_accumulation_steps"]
    max_steps     = t_cfg["max_steps"]
    log_interval  = l_cfg["log_interval"]
    eval_interval = t_cfg["eval_interval"]
    eval_batches  = t_cfg["eval_batches"]
    ckpt_interval = t_cfg["checkpoint_interval"]
    grad_clip     = t_cfg["max_grad_norm"]
    vocab_size    = raw_model.vocab_size

    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    running_loss  = 0.0
    best_val_loss = float("inf")
    val_metrics   = {}

    import time
    step_t0 = time.time()

    print(f"\n  Training steps {start_step} → {max_steps}")
    print(f"  Batch {batch_size}  |  Grad accum {grad_accum}"
          f"  |  Effective batch {batch_size * grad_accum}")
    print(f"  Checkpoints → {ckpt_dir}")
    print(f"  Logs        → {log_dir}\n")

    for step in range(start_step, max_steps + 1):

        # ── LR ────────────────────────────────────────────────────────────
        current_lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # ── Forward + backward (gradient accumulation) ─────────────────────
        accum_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = model(x)
                loss   = F.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # ── Gradient clip + step ───────────────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        running_loss += accum_loss

        # ── Periodic logging ───────────────────────────────────────────────
        if step % log_interval == 0 and step > 0:
            elapsed  = time.time() - step_t0
            avg_loss = running_loss / log_interval
            tok_s    = batch_size * ctx_len * grad_accum * log_interval / elapsed

            record = {
                "step":           step,
                "train_loss":     round(avg_loss, 6),
                "lr":             round(current_lr, 8),
                "grad_norm":      round(grad_norm, 6),
                "tokens_per_sec": int(tok_s),
                "elapsed_s":      round(elapsed, 2),
            }
            log_step(log_path, record)
            print(f"  step {step:>7,} | loss {avg_loss:.4f} | "
                  f"lr {current_lr:.2e} | gnorm {grad_norm:.3f} | {tok_s:,.0f} tok/s")

            running_loss = 0.0
            step_t0      = time.time()

        # ── Validation ─────────────────────────────────────────────────────
        if step % eval_interval == 0 and step > 0:
            val_metrics = evaluate(model, val_loader, eval_batches)
            is_best     = val_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["val_loss"]

            log_step(log_path, {"step": step, **val_metrics, "is_best": is_best})
            print(f"\n  ── VAL  step {step:,} | "
                  f"loss {val_metrics['val_loss']:.4f} | "
                  f"PPL {val_metrics['val_ppl']:.2f}"
                  f"{'  ★ best' if is_best else ''}\n")

        # ── Checkpoint ─────────────────────────────────────────────────────
        if step % ckpt_interval == 0 and step > 0:
            train_metrics = {
                "train_loss": round(running_loss / max(log_interval, 1), 6),
                "lr":         round(current_lr, 8),
                "grad_norm":  round(grad_norm, 6),
                **val_metrics,
            }
            sched_state = {
                "step": step, "lr": current_lr,
                "warmup": t_cfg["warmup_steps"], "max_steps": max_steps,
            }

            ckpt_path = save_checkpoint(
                ckpt_dir, step, model, optimizer, sched_state, cfg, train_metrics
            )
            print(f"  ✓ Checkpoint → {ckpt_path.name}")

            analysis = analyse_checkpoint(ckpt_path)
            analysis["train_metrics"] = train_metrics
            print_checkpoint_analysis(analysis, log_file=analysis_log)

            if l_cfg.get("save_loss_curve", True):
                save_loss_curve(log_path, curve_path, title=curve_title)

    print(f"\n  Training complete.  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints : {ckpt_dir}")
    print(f"  Logs        : {log_dir}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Model H (Telugu)")
    parser.add_argument("--config", "-c", default=str(DEFAULT_CFG))
    parser.add_argument("--resume", "-r", default=None,
                        help="Path to a checkpoint .pt to resume from")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
