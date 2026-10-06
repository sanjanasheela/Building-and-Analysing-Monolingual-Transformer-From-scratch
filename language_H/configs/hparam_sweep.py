"""
hparam_sweep.py — Hyperparameter sweep for Model H (Telugu)
============================================================
Runs a grid of configs for a short number of steps each, evaluates on
val.bin, and records:
    - Validation loss (nats)
    - Validation perplexity (PPL)
    - Bits Per Byte (BPB)

Results are saved to:
    configs/sweep_results/sweep_results.jsonl   (one JSON line per run)
    configs/sweep_results/sweep_summary.txt     (formatted human-readable table)
    configs/sweep_results/sweep_loss_curve.png  (val loss vs run index)

BPB formula
-----------
    BPB = val_loss_nats / ln(2) / avg_bytes_per_token
        = val_loss_nats * 1.4427 / avg_bytes_per_token

avg_bytes_per_token for Telugu BPE (vocab 12500):
    From tokenizer report: avg chars/token ≈ 4.2
    Telugu script in UTF-8: each character = 3 bytes
    → avg_bytes_per_token ≈ 4.2 * 3 ≈ 12.6

Usage (from the project root, e.g. language_H/):
    python configs/hparam_sweep.py

    # Optionally override probe steps and results dir:
    python configs/hparam_sweep.py --probe_steps 2000 --out_dir configs/sweep_results
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as train.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent          # configs/
_PROJECT_DIR = _SCRIPT_DIR.parent                       # language_H/
_TRAIN_DIR   = _PROJECT_DIR / "train"
_MODEL_DIR   = _PROJECT_DIR / "model"

for _p in (str(_TRAIN_DIR), str(_MODEL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model       import CausalLanguageModel
from data_loader import ShardLoader, SingleBinLoader
from scheduler   import cosine_lr_with_warmup


# ===========================================================================
# BPB constant
# ===========================================================================
# Telugu BPE vocab 12500: avg chars/token ≈ 4.2 (from tokenizer report),
# each Telugu char ≈ 3 UTF-8 bytes → avg 12.6 bytes/token.
# Override with --avg_bytes_per_token if you have a more accurate number.
DEFAULT_AVG_BYTES_PER_TOKEN = 12.6


# ===========================================================================
# Hyperparameter grid
# ===========================================================================
# Add / remove values freely. Every combination will be run.
GRID = {
    "context_length": [256, 512],
    "batch_size":     [8, 16],
    "num_layers":     [4, 6],
    "num_heads":      [8],
    "d_model":        [256, 512],
    "learning_rate":  [3e-4, 1e-3],
}

# Fixed base config (non-swept params)
BASE = {
    "vocab_size":        12500,
    "ffn_dim_multiplier": 4,       # ffn_dim = d_model * ffn_dim_multiplier
    "dropout":            0.1,
    "tie_embeddings":     True,
    "init_std":           0.02,
    "rope_theta":         10000.0,

    "min_lr_fraction":    0.1,     # min_lr = lr * min_lr_fraction
    "warmup_fraction":    0.05,    # warmup_steps = probe_steps * warmup_fraction

    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "max_grad_norm": 1.0,

    "shard_dir": "token_shards",
    "val_bin":   "token_shards/val.bin",
}


# ===========================================================================
# Training / evaluation helpers
# ===========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(hp: dict, device: torch.device) -> CausalLanguageModel:
    """Build a fresh model from a hyperparameter dict."""
    model = CausalLanguageModel(
        vocab_size     = hp["vocab_size"],
        d_model        = hp["d_model"],
        num_heads      = hp["num_heads"],
        ffn_dim        = int(hp["ffn_dim"]),
        num_layers     = hp["num_layers"],
        context_length = hp["context_length"],
        dropout        = hp["dropout"],
        tie_embeddings = hp["tie_embeddings"],
        rope_theta     = hp["rope_theta"],
        activation     = hp["activation"],
    )
    model._init_weights(init_std=hp["init_std"])
    return model.to(device)


def count_params(model: CausalLanguageModel) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(
    model:      CausalLanguageModel,
    val_loader: SingleBinLoader,
    n_batches:  int,
    avg_bpt:    float,
) -> dict:
    """Evaluate on val.bin for n_batches batches.

    Returns:
        val_loss (nats), val_ppl, bpb
    """
    model.eval()
    total_loss = 0.0
    total_tok  = 0

    for _ in range(n_batches):
        x, y   = val_loader.next_batch()
        logits = model(x)
        loss   = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tok  += y.numel()

    avg_loss = total_loss / max(total_tok, 1)
    ppl      = math.exp(min(avg_loss, 100))
    bpb      = avg_loss / math.log(2) / avg_bpt   # nats → bits → bits-per-byte

    model.train()
    return {
        "val_loss": round(avg_loss, 6),
        "val_ppl":  round(ppl, 4),
        "bpb":      round(bpb, 6),
    }


def run_probe(
    hp:          dict,
    device:      torch.device,
    probe_steps: int,
    eval_every:  int,
    eval_batches: int,
    avg_bpt:     float,
    seed:        int = 42,
) -> dict:
    """
    Train a fresh model for `probe_steps` steps and return results.

    Returns a dict with all hyperparams + final metrics.
    """
    set_seed(seed)
    amp_on = device.type == "cuda"

    # ── Model ──────────────────────────────────────────────────────────────
    model     = build_model(hp, device)
    n_params  = count_params(model)

    # ── Optimiser ──────────────────────────────────────────────────────────
    lr      = hp["learning_rate"]
    min_lr  = lr * hp["min_lr_fraction"]
    warmup  = max(1, int(probe_steps * hp["warmup_fraction"]))

    decay    = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": hp["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=(hp["beta1"], hp["beta2"]), eps=hp["epsilon"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    # ── Data ───────────────────────────────────────────────────────────────
    shard_dir = _PROJECT_DIR / hp["shard_dir"]
    val_bin   = _PROJECT_DIR / hp["val_bin"]
    ctx_len   = hp["context_length"]
    batch_sz  = hp["batch_size"]

    train_loader = ShardLoader(shard_dir, ctx_len, batch_sz, device, seed)
    val_loader   = SingleBinLoader(val_bin, ctx_len, batch_sz, device, seed + 1)

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    val_history: list[dict] = []
    t0 = time.time()

    for step in range(1, probe_steps + 1):
        # LR schedule
        current_lr = cosine_lr_with_warmup(step, warmup, probe_steps, lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Forward + backward
        x, y = train_loader.next_batch()
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Periodic eval
        if step % eval_every == 0 or step == probe_steps:
            metrics = evaluate(model, val_loader, eval_batches, avg_bpt)
            val_history.append({"step": step, **metrics})
            elapsed = time.time() - t0
            print(
                f"    step {step:>5}/{probe_steps}"
                f"  val_loss={metrics['val_loss']:.4f}"
                f"  PPL={metrics['val_ppl']:.2f}"
                f"  BPB={metrics['bpb']:.4f}"
                f"  ({elapsed:.0f}s)"
            )

    # Final metrics are from the last eval
    final = val_history[-1]

    # Cleanup — free GPU memory before next run
    del model, optimizer, scaler, train_loader, val_loader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        # Hyperparams
        "context_length": hp["context_length"],
        "batch_size":     hp["batch_size"],
        "num_layers":     hp["num_layers"],
        "num_heads":      hp["num_heads"],
        "d_model":        hp["d_model"],
        "ffn_dim":        int(hp["ffn_dim"]),
        "learning_rate":  hp["learning_rate"],
        "n_params":       n_params,
        # Final metrics
        "val_loss":       final["val_loss"],
        "val_ppl":        final["val_ppl"],
        "bpb":            final["bpb"],
        # Val history (for plotting)
        "val_history":    val_history,
        "elapsed_s":      round(time.time() - t0, 1),
    }


# ===========================================================================
# Reporting helpers
# ===========================================================================

def save_jsonl(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            # val_history is too verbose for the summary line
            row = {k: v for k, v in r.items() if k != "val_history"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Results saved → {path}")


def save_summary(results: list[dict], path: Path, probe_steps: int, avg_bpt: float):
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda r: r["val_loss"])

    lines = []
    lines.append("=" * 110)
    lines.append(f"  HYPERPARAMETER SWEEP SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Probe steps : {probe_steps}  |  avg_bytes_per_token : {avg_bpt}")
    lines.append("=" * 110)

    header = (
        f"  {'Rank':>4}  {'ctx':>4}  {'bs':>4}  {'layers':>6}  {'heads':>5}  "
        f"{'d_model':>7}  {'lr':>8}  {'n_params':>10}  "
        f"{'val_loss':>9}  {'PPL':>8}  {'BPB':>8}  {'time(s)':>8}"
    )
    lines.append(header)
    lines.append("-" * 110)

    for rank, r in enumerate(sorted_results, 1):
        lines.append(
            f"  {rank:>4}  {r['context_length']:>4}  {r['batch_size']:>4}  "
            f"{r['num_layers']:>6}  {r['num_heads']:>5}  {r['d_model']:>7}  "
            f"{r['learning_rate']:>8.1e}  {r['n_params']:>10,}  "
            f"{r['val_loss']:>9.4f}  {r['val_ppl']:>8.2f}  {r['bpb']:>8.4f}  "
            f"{r['elapsed_s']:>8.1f}"
        )

    lines.append("=" * 110)
    lines.append(f"\n  BEST CONFIG (lowest val_loss): Rank 1 above")
    best = sorted_results[0]
    for k in ("context_length","batch_size","num_layers","num_heads","d_model","learning_rate"):
        lines.append(f"    {k:<20} : {best[k]}")
    lines.append(f"    {'val_loss':<20} : {best['val_loss']}")
    lines.append(f"    {'val_ppl':<20} : {best['val_ppl']}")
    lines.append(f"    {'bpb':<20} : {best['bpb']}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  Summary saved → {path}")


def save_plots(results: list[dict], out_dir: Path):
    if not MATPLOTLIB_OK:
        print("  matplotlib not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- val_loss per run ---
    sorted_r = sorted(results, key=lambda r: r["val_loss"])
    labels   = [
        f"ctx{r['context_length']}_bs{r['batch_size']}_L{r['num_layers']}"
        f"_d{r['d_model']}_lr{r['learning_rate']:.0e}"
        for r in sorted_r
    ]
    losses = [r["val_loss"] for r in sorted_r]

    plt.figure(figsize=(max(10, len(sorted_r) * 0.6), 5))
    bars = plt.bar(range(len(losses)), losses, color="steelblue")
    plt.xticks(range(len(losses)), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("Val Loss (nats)")
    plt.title("Hyperparameter Sweep — Final Val Loss per Config")
    plt.tight_layout()
    p = out_dir / "sweep_bar_val_loss.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Plot saved → {p}")

    # --- training curves for each run ---
    plt.figure(figsize=(12, 5))
    for r, label in zip(sorted_r, labels):
        steps  = [v["step"]     for v in r["val_history"]]
        losses = [v["val_loss"] for v in r["val_history"]]
        plt.plot(steps, losses, marker="o", markersize=3, label=label)

    plt.xlabel("Training step")
    plt.ylabel("Val Loss (nats)")
    plt.title("Hyperparameter Sweep — Val Loss Curves")
    plt.legend(fontsize=6, loc="upper right", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "sweep_val_loss_curves.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Plot saved → {p}")

    # --- BPB bar chart ---
    bpbs = [r["bpb"] for r in sorted_r]
    plt.figure(figsize=(max(10, len(sorted_r) * 0.6), 5))
    plt.bar(range(len(bpbs)), bpbs, color="coral")
    plt.xticks(range(len(bpbs)), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("BPB (bits per byte)")
    plt.title("Hyperparameter Sweep — Final BPB per Config")
    plt.tight_layout()
    p = out_dir / "sweep_bar_bpb.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Plot saved → {p}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Model H")
    parser.add_argument(
        "--probe_steps", type=int, default=1000,
        help="Number of training steps per config (default: 1000)"
    )
    parser.add_argument(
        "--eval_every", type=int, default=200,
        help="Evaluate on val.bin every N steps within a probe run (default: 200)"
    )
    parser.add_argument(
        "--eval_batches", type=int, default=30,
        help="Val batches per evaluation call (default: 30)"
    )
    parser.add_argument(
        "--avg_bytes_per_token", type=float, default=DEFAULT_AVG_BYTES_PER_TOKEN,
        help=f"Avg UTF-8 bytes per token for BPB calculation (default: {DEFAULT_AVG_BYTES_PER_TOKEN})"
    )
    parser.add_argument(
        "--sweep_dir", type=str,
        default="configs/sweep_yamls",
        help="Directory containing YAML configs to sweep over"
    )
    parser.add_argument(
        "--out_dir", type=str,
        default="configs/sweep_results",
        help="Directory to save sweep outputs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on (default: auto)"
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device : {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"\n  Sweep directory not found: {sweep_dir}")
        return
        
    yaml_files = sorted(sweep_dir.glob("*.yaml"))
    total = len(yaml_files)

    print(f"\n  Hyperparameter sweep (from YAMLs)")
    print(f"  Total configs    : {total}")
    print(f"  Probe steps each : {args.probe_steps}")
    print(f"  Eval every       : {args.eval_every} steps ({args.eval_batches} batches)")
    print(f"  avg_bytes/token  : {args.avg_bytes_per_token}")
    print(f"  Sweep dir        : {sweep_dir}")
    print(f"  Output dir       : {out_dir}\n")

    results = []

    for idx, yaml_path in enumerate(yaml_files, 1):
        import yaml
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        m = cfg["model"]
        t = cfg["training"]
        
        hp = {
            "vocab_size":        cfg.get("tokenizer", {}).get("vocab_size") or m["vocab_size"],
            "d_model":           m["d_model"],
            "num_layers":        m["num_layers"],
            "num_heads":         m["num_heads"],
            "ffn_dim":           m["ffn_dim"],
            "context_length":    m["context_length"],
            "dropout":           m["dropout"],
            "tie_embeddings":    m["tie_embeddings"],
            "init_std":          m["init_std"],
            "rope_theta":        m.get("rope_theta", 10000.0),
            "activation":        m.get("activation", "GELU"),
            
            "batch_size":        t["batch_size"],
            "learning_rate":     t["learning_rate"],
            "min_lr_fraction":   t["min_learning_rate"] / max(t["learning_rate"], 1e-9),
            "warmup_fraction":   t["warmup_steps"] / max(t["max_steps"], 1),
            "weight_decay":      t["weight_decay"],
            "beta1":             t["beta1"],
            "beta2":             t["beta2"],
            "epsilon":           t["epsilon"],
            "max_grad_norm":     t["max_grad_norm"],
            
            "shard_dir":         cfg["data"]["shard_dir"],
            "val_bin":           cfg["data"]["val_bin"],
        }

        run_label = cfg.get("model_name", yaml_path.stem)
        print(f"\n  [{idx}/{total}]  {run_label}")

        try:
            result = run_probe(
                hp           = hp,
                device       = device,
                probe_steps  = args.probe_steps,
                eval_every   = args.eval_every,
                eval_batches = args.eval_batches,
                avg_bpt      = args.avg_bytes_per_token,
                seed         = args.seed,
            )
            result["run_label"] = run_label
            result["run_idx"]   = idx
            results.append(result)

            print(
                f"  ✓  val_loss={result['val_loss']:.4f}"
                f"  PPL={result['val_ppl']:.2f}"
                f"  BPB={result['bpb']:.4f}"
                f"  params={result['n_params']:,}"
            )

        except Exception as e:
            print(f"  ✗  Run failed: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n  No successful runs. Exiting.")
        return

    # --- Save outputs ---
    print(f"\n  {'='*50}")
    print(f"  Sweep complete — {len(results)} runs finished")

    save_jsonl(results, out_dir / "sweep_results.jsonl")
    save_summary(results, out_dir / "sweep_summary.txt", args.probe_steps, args.avg_bytes_per_token)
    save_plots(results, out_dir)

    # Print top-5
    top5 = sorted(results, key=lambda r: r["val_loss"])[:5]
    print("\n  Top-5 configs by val loss:")
    for i, r in enumerate(top5, 1):
        print(
            f"    #{i}  {r['run_label']:<60}"
            f"  loss={r['val_loss']:.4f}  PPL={r['val_ppl']:.2f}  BPB={r['bpb']:.4f}"
        )


if __name__ == "__main__":
    main()
