"""
model.py — Ablated Causal Language Model (No Positional Embeddings)
===================================================================
Ablation study for Model H (Telugu):
In this ablated architecture, ALL positional encodings are removed:
  - NO Rotary Position Embeddings (RoPE is completely disabled).
  - NO learned or sinusoidal absolute positional embeddings.
  - Token embeddings are mapped directly to d_model and processed through
    pre-norm Transformer blocks with causal autoregressive masking.

Within the causal receptive field [0..t], self-attention receives ZERO positional
signal and is completely permutation-invariant with respect to token order.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: (swish(x W1) * (x W3)) W2."""

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=True)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(w1(x)) * w3(x) -> projected via w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Ablated Multi-Head Causal Self-Attention (NO Positional Embeddings)
# ---------------------------------------------------------------------------
class AblatedMultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Positional Embeddings Removed.

    Computes standard scaled dot-product attention with lower-triangular causal
    masking, but WITHOUT any positional information (no RoPE, no learned pos emb).
    Attention weights are based strictly on token semantic content and causality:
        Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M_causal) V
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, T, d_model)
            return_weights: If True, returns post-softmax attention weights (B, H, T, T)
        Returns:
            out: (B, T, d_model)
            attn_weights: (B, num_heads, T, T) if return_weights else None
        """
        B, T, _ = x.shape

        # Linear projections
        q = self.wq(x)  # (B, T, d_model)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to multi-head: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores (NO RoPE / NO pos embeddings)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        # Causal autoregressive mask: upper triangle set to -inf
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + causal_mask

        # Softmax over key positions
        attn_weights = F.softmax(scores, dim=-1)

        # Post-softmax dropout (active only during training)
        out_weights = self.attn_dropout(attn_weights)

        # Output projection
        out = torch.matmul(out_weights, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.wo(out)

        if return_weights:
            return out, attn_weights
        return out, None


# ---------------------------------------------------------------------------
# Ablated Transformer Block (Pre-Norm)
# ---------------------------------------------------------------------------
class AblatedTransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block without positional encoding:
      x = x + SelfAttn(LayerNorm(x))
      x = x + SwiGLUFFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = AblatedMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm Self-Attention
        normed1 = self.ln1(x)
        attn_out, weights = self.attn(normed1, return_weights=return_weights)
        x = x + self.dropout(attn_out)

        # Pre-norm Feed-Forward
        normed2 = self.ln2(x)
        ffn_out = self.ffn(normed2)
        x = x + self.dropout(ffn_out)

        return x, weights


# ---------------------------------------------------------------------------
# Full Ablated Causal Language Model
# ---------------------------------------------------------------------------
class AblatedCausalLanguageModel(nn.Module):
    """
    Decoder-only Transformer LM with positional embeddings completely ablated.
    Matches Model H architecture (vocab 7,500, d_model 512, 6 layers, 8 heads,
    SwiGLU ffn 1,600, tied embeddings), but positional embeddings are absent.
    """

    def __init__(
        self,
        vocab_size: int = 7500,
        d_model: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 1600,
        num_layers: int = 6,
        context_length: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.context_length = context_length
        self.dropout_rate = dropout
        self.tie_embeddings = tie_embeddings
        self.positional_encoding = "none"

        # Token embeddings only — NO positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AblatedTransformerBlock(d_model, num_heads, ffn_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(d_model)

        # LM Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights(init_std=0.02)

    def _init_weights(self, init_std: float = 0.02):
        """Initialise weights with truncated normal / normal(0, init_std)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        Args:
            input_ids: (B, T) token tensor
            return_attention: If True, returns (logits, all_attention_weights)
        Returns:
            logits: (B, T, vocab_size)
            (optional) all_attention_weights: list of (B, num_heads, T, T) tensors
        """
        B, T = input_ids.shape
        if T > self.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds max context length {self.context_length}"
            )

        # Token lookup only — ZERO position information injected
        x = self.tok_emb(input_ids)
        x = self.emb_dropout(x)

        all_weights = []
        for block in self.blocks:
            x, weights = block(x, return_weights=return_attention)
            if return_attention:
                all_weights.append(weights)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        if return_attention:
            return logits, all_weights
        return logits

    @torch.no_grad()
    def extract_attention(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract post-softmax attention weights for all layers in eval mode.
        Returns:
            list of L tensors, each of shape (B, num_heads, T, T) on CPU.
        """
        self.eval()
        _, weights = self.forward(input_ids, return_attention=True)
        return [w.detach().cpu() for w in weights]

    def count_parameters(self) -> Dict[str, Any]:
        """Parameter accounting dictionary."""
        total = sum(p.numel() for p in self.parameters())
        embed = sum(p.numel() for p in self.tok_emb.parameters())
        lm_head_p = 0 if self.tie_embeddings else sum(p.numel() for p in self.lm_head.parameters())
        return {
            "total": total,
            "embedding": embed,
            "non_embedding": total - embed,
            "lm_head_shared": self.tie_embeddings,
            "lm_head_params": lm_head_p,
            "positional_params": 0,
        }


# ---------------------------------------------------------------------------
# Factory Helper
# ---------------------------------------------------------------------------
def build_from_config(cfg: dict) -> AblatedCausalLanguageModel:
    """Instantiate AblatedCausalLanguageModel from YAML configuration dictionary."""
    m = cfg.get("model", {})
    tok = cfg.get("tokenizer", {})
    vocab_size = tok.get("vocab_size", m.get("vocab_size", 7500))

    model = AblatedCausalLanguageModel(
        vocab_size=vocab_size,
        d_model=m.get("d_model", 512),
        num_heads=m.get("num_heads", 8),
        ffn_dim=m.get("ffn_dim", 1600),
        num_layers=m.get("num_layers", 6),
        context_length=m.get("context_length", 512),
        dropout=m.get("dropout", 0.1),
        tie_embeddings=m.get("tie_embeddings", True),
    )
    init_std = m.get("init_std", 0.02)
    model._init_weights(init_std=init_std)
    return model


if __name__ == "__main__":
    dummy_model = AblatedCausalLanguageModel()
    params = dummy_model.count_parameters()
    print("Ablated Model Parameters:", params)
    dummy_x = torch.randint(0, 7500, (2, 32))
    logits, attns = dummy_model(dummy_x, return_attention=True)
    print("Logits shape:", logits.shape)
    print(f"Extracted attention layers: {len(attns)}, Layer 0 shape: {attns[0].shape}")
