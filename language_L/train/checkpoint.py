"""
checkpoint.py — Checkpoint save, load, and analysis
=====================================================
Each checkpoint contains (as required by Phase 2 spec):
    model_state, optimizer_state, scheduler_state,
    current step, config, training metrics, timestamp.

analyse_checkpoint() loads a .pt file and returns a rich analysis dict
(parameter counts, weight norms per component) without requiring the
model to be instantiated first.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    out_dir:       Path,
    step:          int,
    model,                          # CausalLanguageModel
    optimizer:     torch.optim.Optimizer,
    scheduler_state: dict,
    cfg:           dict,
    train_metrics: dict,
) -> Path:
    """
    Save a resume-capable checkpoint.
    Filename: step_XXXXXXX.pt
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:07d}.pt"

    torch.save(
        {
            # ── Required by Phase 2 spec ────────────────────────────
            "step":              step,
            "model_state":       model.state_dict(),
            "optimizer_state":   optimizer.state_dict(),
            "scheduler_state":   scheduler_state,
            "config":            cfg,
            # ── Extra analysis fields ───────────────────────────────
            "train_metrics":     train_metrics,
            "param_info":        model.count_parameters(),
            "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Load (resume)
# ---------------------------------------------------------------------------

def load_checkpoint(
    path:      str | Path,
    model,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[int, dict]:
    """
    Restore model and optimizer state from a checkpoint.
    Uses strict=False to handle architecture changes (e.g. pos_emb → RoPE).
    Returns (step, scheduler_state).
    """
    ckpt = torch.load(str(path), map_location=device)
    result = model.load_state_dict(ckpt["model_state"], strict=False)

    if result.missing_keys:
        print(f"  ⚠ Missing keys (new params, will use init): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  ⚠ Unexpected keys (old params, ignored): {result.unexpected_keys}")

    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Resumed from {Path(path).name}  (step={ckpt['step']})")
    return ckpt["step"], ckpt.get("scheduler_state", {})


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def analyse_checkpoint(ckpt_path: Path) -> dict:
    """
    Load a saved .pt file and return a structured analysis dict:
        - step, timestamp
        - parameter counts
        - training metrics recorded at save time
        - model config snapshot
        - per-tensor weight norms (aggregated per component)

    Does NOT require the model class to be instantiated.
    """
    ckpt  = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model_state"]

    # Per-tensor L2 norms (float tensors only)
    weight_norms = {
        k: round(v.float().norm().item(), 6)
        for k, v in state.items()
        if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
    }

    # Aggregate by component prefix (e.g. "wqs.0" → "wqs.0")
    component_norms: dict[str, list[float]] = {}
    for key, norm in weight_norms.items():
        prefix = key.rsplit(".", 1)[0] if "." in key else key
        component_norms.setdefault(prefix, []).append(norm)

    component_avg = {
        k: round(sum(v) / len(v), 6)
        for k, v in component_norms.items()
    }

    return {
        "step":               ckpt["step"],
        "timestamp":          ckpt.get("timestamp", "N/A"),
        "param_info":         ckpt.get("param_info", {}),
        "train_metrics":      ckpt.get("train_metrics", {}),
        "config_snapshot":    ckpt.get("config", {}).get("model", {}),
        "weight_norms":       weight_norms,
        "component_avg_norm": component_avg,
    }


# ---------------------------------------------------------------------------
# Print analysis report
# ---------------------------------------------------------------------------

def print_checkpoint_analysis(
    analysis: dict,
    log_file: Path | None = None,
):
    """Print (and optionally append to a log file) a formatted analysis report."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  Checkpoint Analysis — step {analysis['step']:,}")
    lines.append(f"  Saved at : {analysis['timestamp']}")
    lines.append("=" * 65)

    pi = analysis["param_info"]
    if pi:
        lines.append("\n  Parameters:")
        lines.append(f"    Total          : {pi.get('total', 0):>12,}")
        lines.append(f"    Embedding      : {pi.get('embedding', 0):>12,}")
        lines.append(f"    Non-embedding  : {pi.get('non_embedding', 0):>12,}")
        lines.append(f"    Tie embeddings : {pi.get('lm_head_shared', '?')}")

    tm = analysis["train_metrics"]
    if tm:
        lines.append("\n  Training metrics:")
        for k, v in tm.items():
            lines.append(f"    {k:<20} : {v}")

    lines.append("\n  Weight norms (component averages):")
    for comp, avg_norm in list(analysis["component_avg_norm"].items())[:20]:
        lines.append(f"    {comp:<35} avg_norm={avg_norm:.4f}")

    lines.append("=" * 65)

    text = "\n".join(lines)
    print(text)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")
