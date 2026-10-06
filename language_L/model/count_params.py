"""
count_params.py — Parameter counter for Model H / Model L
==========================================================
Reads a YAML config file and computes the exact trainable parameter
count for every component of the decoder-only transformer, then prints
a clean breakdown table.

Usage:
    python model/count_params.py                          # uses default config
    python model/count_params.py --config configs/telugu_1.yaml
    python model/count_params.py --config path/to/other.yaml
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

# ---------------------------------------------------------------------------
# Default config path (relative to this file)
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent        # model/
_PROJECT_DIR = _HERE.parent                           # language_H/
DEFAULT_CFG  = _PROJECT_DIR / "configs" / "telugu_1.yaml"


# ---------------------------------------------------------------------------
# Core parameter-counting logic (pure arithmetic — no PyTorch needed)
# ---------------------------------------------------------------------------

def count_parameters(
    vocab_size:     int,
    d_model:        int,
    num_layers:     int,
    num_heads:      int,
    ffn_dim:        int,
    context_length: int,
    tie_embeddings: bool = True,
    activation:     str  = "GELU",
) -> dict:
    """
    Return a dict mapping component name → parameter count.

    Architecture assumed (matches model/ implementation):
        Embeddings
            token_embedding        : vocab_size × d_model
            positional_embedding   : context_length × d_model
        Per transformer block (× num_layers):
            LayerNorm 1            : 2 × d_model  (weight + bias)
            W_Q (no bias)          : d_model × d_model
            W_K (no bias)          : d_model × d_model
            W_V (no bias)          : d_model × d_model
            W_O (no bias)          : d_model × d_model
            LayerNorm 2            : 2 × d_model
            FFN W1 (bias)          : d_model × ffn_dim  + ffn_dim
            FFN W2 (bias)          : ffn_dim × d_model  + d_model
            FFN W3 (bias)          : d_model × ffn_dim  + ffn_dim (SwiGLU only)
        Final LayerNorm            : 2 × d_model
        LM head (linear, no bias)  : vocab_size × d_model
            (skipped if tie_embeddings=True, shared with token_embedding)
    """
    components = {}
    activation = activation.upper()

    # ---- Embeddings ----
    components["Token Embedding"] = vocab_size * d_model
    components["Positional Embedding"] = 0  # 0 because we use RoPE

    # ---- Per-layer components ----
    per_block = {}
    per_block["  LayerNorm 1 (γ,β)"] = 2 * d_model
    per_block["  W_Q"]               = d_model * d_model
    per_block["  W_K"]               = d_model * d_model
    per_block["  W_V"]               = d_model * d_model
    per_block["  W_O"]               = d_model * d_model
    per_block["  LayerNorm 2 (γ,β)"] = 2 * d_model
    per_block["  FFN W1 (w + b)"]    = d_model * ffn_dim + ffn_dim
    per_block["  FFN W2 (w + b)"]    = ffn_dim * d_model + d_model
    
    if activation == "SWIGLU":
        per_block["  FFN W3 (w + b)"] = d_model * ffn_dim + ffn_dim

    block_total = sum(per_block.values())

    # Add per-block breakdown with layer count suffix
    for name, count in per_block.items():
        components[f"{name}  ×{num_layers} layers"] = count * num_layers

    # ---- Final LayerNorm ----
    components["Final LayerNorm (γ,β)"] = 2 * d_model

    # ---- LM Head ----
    if tie_embeddings:
        components["LM Head (tied — no extra params)"] = 0
    else:
        components["LM Head (unshared)"] = vocab_size * d_model

    # ---- Totals ----
    total            = sum(components.values())
    embedding_params = (components["Token Embedding"] +
                        components["Positional Embedding"])
    transformer_params = block_total * num_layers
    non_embedding    = total - embedding_params

    meta = {
        "per_block_total":    block_total,
        "embedding_params":   embedding_params,
        "transformer_params": transformer_params,
        "non_embedding":      non_embedding,
        "total":              total,
        "tie_embeddings":     tie_embeddings,
    }

    return components, meta


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(cfg: dict, components: dict, meta: dict):
    m   = cfg["model"]
    tok = cfg.get("tokenizer", {})

    print()
    print("=" * 62)
    print(f"  Parameter Count Report — {cfg.get('model_name', 'Model')}")
    print(f"  Language : {cfg.get('language', 'N/A')}")
    print("=" * 62)

    print("\n  ── Architecture ──────────────────────────────────────")
    print(f"  vocab_size      : {tok.get('vocab_size', m.get('vocab_size', '?')):>10,}")
    print(f"  d_model         : {m['d_model']:>10,}")
    print(f"  num_layers      : {m['num_layers']:>10,}")
    print(f"  num_heads       : {m['num_heads']:>10,}")
    print(f"  ffn_dim         : {m['ffn_dim']:>10,}")
    print(f"  context_length  : {m['context_length']:>10,}")
    print(f"  head_dim (d/h)  : {m['d_model'] // m['num_heads']:>10,}")
    print(f"  tie_embeddings  : {str(m.get('tie_embeddings', True)):>10}")
    print(f"  activation      : {m.get('activation', 'GELU'):>10}")
    print(f"  normalization   : {m.get('normalization', 'pre_norm'):>10}")

    print("\n  ── Parameter Breakdown ───────────────────────────────")
    col_w = 42
    for name, count in components.items():
        if count == 0:
            label = f"  {name}"
            print(f"  {label:<{col_w}}   (shared)")
        else:
            print(f"  {name:<{col_w}} {count:>12,}")

    print()
    print(f"  {'─'*58}")
    print(f"  {'Embedding params':<{col_w}} {meta['embedding_params']:>12,}")
    print(f"  {'Transformer blocks (excl. embeddings)':<{col_w}} {meta['transformer_params']:>12,}")
    print(f"  {'Final LayerNorm':<{col_w}} {2 * m['d_model']:>12,}")
    print(f"  {'─'*58}")
    print(f"  {'TOTAL parameters':<{col_w}} {meta['total']:>12,}")
    print(f"  {'TOTAL (M)':<{col_w}} {meta['total']/1e6:>11.3f}M")
    print(f"  {'Non-embedding parameters':<{col_w}} {meta['non_embedding']:>12,}")

    # Target check
    target = 25_000_000
    pct    = meta["total"] / target * 100
    status = "✓ within ±20% of 25M target" if abs(pct - 100) < 2 else "⚠ outside ±20% of 25M target"
    print(f"\n  Target ≈ 25M params — {meta['total']/1e6:.2f}M ({pct:.1f}%)  {status}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Count transformer parameters from a YAML config."
    )
    parser.add_argument(
        "--config", "-c",
        default=str(DEFAULT_CFG),
        help=f"Path to YAML config file (default: {DEFAULT_CFG})",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    m   = cfg["model"]
    tok = cfg.get("tokenizer", {})

    vocab_size     = tok.get("vocab_size") or m.get("vocab_size")
    d_model        = m["d_model"]
    num_layers     = m["num_layers"]
    num_heads      = m["num_heads"]
    ffn_dim        = m["ffn_dim"]
    context_length = m["context_length"]
    tie_embeddings = m.get("tie_embeddings", True)
    activation     = m.get("activation", "GELU")

    if vocab_size is None:
        sys.exit("vocab_size must be set under tokenizer.vocab_size or model.vocab_size in the YAML.")
    if d_model % num_heads != 0:
        print(f"  WARNING: d_model ({d_model}) is not divisible by num_heads ({num_heads})!")

    components, meta = count_parameters(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        context_length=context_length,
        tie_embeddings=tie_embeddings,
        activation=activation,
    )

    print_report(cfg, components, meta)


if __name__ == "__main__":
    main()
