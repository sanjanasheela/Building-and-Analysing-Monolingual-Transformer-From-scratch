"""
model.py — CausalLanguageModel (decoder-only GPT-style transformer)
====================================================================
Owns all learnable layers. Checkpoints saved from this module are
fully self-contained.

Imports functional building blocks from model/:
    attention.py  — multi_head_causal_self_attention, precompute_rope_freqs
    block.py      — transformer_block
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow imports from the sibling model/ directory
_TRAIN_DIR   = Path(__file__).resolve().parent
_PROJECT_DIR = _TRAIN_DIR.parent
_MODEL_DIR   = _PROJECT_DIR / "model"

for _p in (str(_MODEL_DIR), str(_TRAIN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from block import transformer_block
from attention import precompute_rope_freqs


class CausalLanguageModel(nn.Module):
    """
    Decoder-only GPT-style transformer with RoPE positional embeddings.
    All weights are owned by this module so checkpoints are self-contained.
    """

    def __init__(
        self,
        vocab_size:     int,
        d_model:        int,
        num_heads:      int,
        ffn_dim:        int,
        num_layers:     int,
        context_length: int,
        dropout:        float = 0.1,
        tie_embeddings: bool  = True,
        rope_theta:     float = 10000.0,
        activation:     str   = "GELU",
    ):
        super().__init__()
        self.vocab_size     = vocab_size
        self.d_model        = d_model
        self.num_heads      = num_heads
        self.ffn_dim        = ffn_dim
        self.num_layers     = num_layers
        self.context_length = context_length
        self.dropout        = dropout
        self.tie_embeddings = tie_embeddings
        self.rope_theta     = rope_theta
        self.activation     = activation.upper()

        # --- Token Embedding (no positional embedding — RoPE handles position) ---
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # --- Precompute RoPE frequencies and register as non-learnable buffers ---
        head_dim = d_model // num_heads
        freqs_cos, freqs_sin = precompute_rope_freqs(
            head_dim, context_length, theta=rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # --- Transformer block weights (per layer) ---
        self.lns1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.wqs  = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)])
        self.wks  = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)])
        self.wvs  = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)])
        self.wos  = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)])
        self.w1s  = nn.ModuleList([nn.Linear(d_model, ffn_dim,  bias=True) for _ in range(num_layers)])
        self.w2s  = nn.ModuleList([nn.Linear(ffn_dim,  d_model, bias=True) for _ in range(num_layers)])
        
        # SwiGLU requires an extra weight matrix per layer
        if self.activation == "SWIGLU":
            self.w3s = nn.ModuleList([nn.Linear(d_model, ffn_dim, bias=True) for _ in range(num_layers)])
        else:
            self.w3s = None

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights(init_std=0.02)   # overridden by build_from_config

    # ------------------------------------------------------------------
    def _init_weights(self, init_std: float = 0.02):
        """Weight init — std controlled by model.init_std in the YAML."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T   = input_ids.shape

        # Token embeddings only — RoPE provides positional info inside attention
        x = self.tok_emb(input_ids)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Slice precomputed RoPE frequencies to current sequence length
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]

        for i in range(self.num_layers):
            w3_layer = self.w3s[i] if self.w3s is not None else None
            x = transformer_block(
                x=x, d_model=self.d_model, num_head=self.num_heads,
                ffn_dim=self.ffn_dim, dropout_rate=self.dropout,
                ln1=self.lns1[i], ln2=self.lns2[i],
                wq=self.wqs[i], wk=self.wks[i], wv=self.wvs[i], wo=self.wos[i],
                w1=self.w1s[i],  w2=self.w2s[i], w3=w3_layer,
                training=self.training,
                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                activation=self.activation,
            )

        x = self.final_ln(x)
        return self.lm_head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        embed = sum(p.numel() for p in self.tok_emb.parameters())
        lm_head_params = (
            0 if self.tie_embeddings
            else sum(p.numel() for p in self.lm_head.parameters())
        )
        return {
            "total":          total,
            "embedding":      embed,
            "non_embedding":  total - embed,
            "lm_head_shared": self.tie_embeddings,
            "lm_head_params": lm_head_params,
        }


# ---------------------------------------------------------------------------
# Factory: build from YAML config dict
# ---------------------------------------------------------------------------

def build_from_config(cfg: dict) -> CausalLanguageModel:
    """Instantiate and initialise a CausalLanguageModel from a YAML config."""
    m   = cfg["model"]
    tok = cfg.get("tokenizer", {})
    vocab_size = tok.get("vocab_size") or m["vocab_size"]

    model = CausalLanguageModel(
        vocab_size     = vocab_size,
        d_model        = m["d_model"],
        num_heads      = m["num_heads"],
        ffn_dim        = m["ffn_dim"],
        num_layers     = m["num_layers"],
        context_length = m["context_length"],
        dropout        = m["dropout"],
        tie_embeddings = m["tie_embeddings"],
        rope_theta     = m.get("rope_theta", 10000.0),
        activation     = m.get("activation", "GELU"),
    )
    model._init_weights(init_std=m["init_std"])
    return model
