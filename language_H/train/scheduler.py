"""
scheduler.py — Learning-rate schedule functions
================================================
Currently implements cosine decay with linear warmup.
Add new schedules here; train.py selects by name from config.
"""

from __future__ import annotations
import math


def cosine_lr_with_warmup(
    step:         int,
    warmup_steps: int,
    max_steps:    int,
    lr:           float,
    min_lr:       float,
) -> float:
    """
    Linear warmup from 0 → lr over `warmup_steps`, then cosine decay
    from lr → min_lr over the remaining steps.
    """
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def get_lr(step: int, cfg: dict) -> float:
    """
    Dispatch to the correct scheduler by name (training.scheduler in YAML).
    Supported: "cosine"
    """
    t        = cfg["training"]
    name     = t["scheduler"]
    lr       = t["learning_rate"]
    min_lr   = t["min_learning_rate"]
    warmup   = t["warmup_steps"]
    max_step = t["max_steps"]

    if name == "cosine":
        return cosine_lr_with_warmup(step, warmup, max_step, lr, min_lr)

    raise ValueError(f"Unknown scheduler '{name}'. Supported: cosine")
