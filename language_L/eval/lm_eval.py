"""
lm_eval.py — Intrinsic language-modelling metrics
===================================================
Functions:
    cross_entropy()   — mean token-level cross-entropy on a held-out set
    perplexity()      — exp(cross_entropy)
    bpb()             — bits-per-byte (language-independent comparison metric)

All functions accept a loaded CausalLanguageModel and a ShardLoader (or any
iterator that yields (input_ids, target_ids) torch Tensors).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ── path bootstrap ──────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_PROJECT_DIR = _HERE.parent

for _p in (str(_PROJECT_DIR / "model"), str(_PROJECT_DIR / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Core: cross-entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def cross_entropy(
    model,
    loader,
    n_batches: int,
    device:    torch.device | None = None,
) -> float:
    """
    Compute mean token-level cross-entropy loss over `n_batches` batches.

    Args:
        model:     CausalLanguageModel in eval mode.
        loader:    Any object with a .next_batch() → (x, y) method.
        n_batches: Number of batches to average over.
        device:    Override device (defaults to model's device).

    Returns:
        float: Mean cross-entropy (nats per token).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tok  = 0

    for _ in range(n_batches):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)

        logits = model(x)                                    # (B, T, V)
        loss   = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tok  += y.numel()

    return total_loss / total_tok


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def perplexity(
    model,
    loader,
    n_batches: int,
    device:    torch.device | None = None,
) -> float:
    """
    Compute perplexity = exp(cross-entropy).

    Returns:
        float: Perplexity (lower is better).
    """
    ce = cross_entropy(model, loader, n_batches, device)
    return math.exp(min(ce, 20))    # cap avoids inf on a random-init model


# ---------------------------------------------------------------------------
# Bits-per-byte (BPB)
# ---------------------------------------------------------------------------

@torch.no_grad()
def bpb(
    model,
    loader,
    tokenizer,
    n_batches: int,
    device:    torch.device | None = None,
) -> float:
    """
    Compute bits-per-byte on a held-out set.

    BPB = cross_entropy_nats / log(2) / avg_bytes_per_token

    avg_bytes_per_token is estimated by encoding a sample of the raw text
    with `tokenizer` and measuring |UTF-8 bytes| / |tokens|.

    Args:
        model:     CausalLanguageModel.
        loader:    Batch iterator with .next_batch().
        tokenizer: HuggingFace-style tokenizer with .decode(ids) → str.
        n_batches: Batches for loss estimation.
        device:    Target device.

    Returns:
        float: Bits per byte (lower is better; enables cross-lingual comparison).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tok  = 0
    total_bytes = 0

    for _ in range(n_batches):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss   = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total_loss  += loss.item()
        total_tok   += y.numel()

        # Estimate bytes by decoding target token IDs
        flat_ids = y.view(-1).cpu().tolist()
        text     = tokenizer.decode(flat_ids)
        total_bytes += len(text.encode("utf-8"))

    ce_nats            = total_loss / total_tok
    avg_bytes_per_tok  = total_bytes / total_tok
    bits_per_byte      = ce_nats / math.log(2) / avg_bytes_per_tok
    return round(bits_per_byte, 6)


# ---------------------------------------------------------------------------
# Convenience: run all three and return a dict
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_all(
    model,
    loader,
    n_batches:  int,
    tokenizer = None,
    device:    torch.device | None = None,
) -> dict:
    """
    Run cross_entropy, perplexity (and optionally bpb) in one pass.

    Returns:
        dict with keys "cross_entropy", "perplexity", and optionally "bpb".
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss  = 0.0
    total_tok   = 0
    total_bytes = 0

    for _ in range(n_batches):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss   = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tok  += y.numel()

        if tokenizer is not None:
            text         = tokenizer.decode(y.view(-1).cpu().tolist())
            total_bytes += len(text.encode("utf-8"))

    ce  = total_loss / total_tok
    ppl = math.exp(min(ce, 20))
    out = {"cross_entropy": round(ce, 6), "perplexity": round(ppl, 4)}

    if tokenizer is not None and total_bytes > 0:
        avg_bpb = (ce / math.log(2)) / (total_bytes / total_tok)
        out["bpb"] = round(avg_bpb, 6)

    return out
