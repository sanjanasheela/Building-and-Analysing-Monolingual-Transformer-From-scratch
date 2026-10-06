import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import multi_head_causal_self_attention
from ffnn import feed_forward_network


def transformer_block(
    x: torch.Tensor,
    d_model: int,
    num_head: int,
    ffn_dim: int,
    dropout_rate: float = 0.1,
    ln1: nn.LayerNorm = None,
    ln2: nn.LayerNorm = None,
    wq: nn.Linear = None,
    wk: nn.Linear = None,
    wv: nn.Linear = None,
    wo: nn.Linear = None,
    w1: nn.Linear = None,
    w2: nn.Linear = None,
    w3: nn.Linear = None,
    training: bool = None,
    freqs_cos: torch.Tensor = None,
    freqs_sin: torch.Tensor = None,
    activation: str = "GELU",
) -> torch.Tensor:
    """Applies a single Pre-Norm Transformer block using the standalone attention

    and feed-forward functions.

    Args:
        training: If None (default), dropout is active only when gradient
                  computation is enabled (i.e., outside torch.no_grad()). Pass
                  True/False to override explicitly.
        freqs_cos: Optional RoPE cosine frequencies (T, head_dim).
        freqs_sin: Optional RoPE sine frequencies (T, head_dim).
        activation: Activation function to use in FFN (GELU, RELU, SWIGLU).
    """
    # Honour the caller's explicit flag; otherwise follow grad-enabled state.
    is_training = training if training is not None else torch.is_grad_enabled()

    if ln1 is None:
        ln1 = nn.LayerNorm(d_model, device=x.device)
    if ln2 is None:
        ln2 = nn.LayerNorm(d_model, device=x.device)

    # --- Sub-layer 1: Multi-Head Causal Self-Attention with Residual Connection ---
    normed_x = ln1(x)
    
    # Use the standalone attention function from Step 2
    attn_output = multi_head_causal_self_attention(
        x=normed_x,
        d_model=d_model,
        num_head=num_head,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        attn_dropout_rate=dropout_rate,
        training=is_training,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
    )

    x = x + F.dropout(attn_output, p=dropout_rate, training=is_training)

    # --- Sub-layer 2: Feed-Forward Network with Residual Connection ---
    normed_x2 = ln2(x)
    
    # Use the standalone FFN function from Step 3
    ffn_output = feed_forward_network(
        x=normed_x2,
        d_model=d_model,
        ffn_dim=ffn_dim,
        w1=w1,
        w2=w2,
        w3=w3,
        activation=activation,
    )

    out = x + F.dropout(ffn_output, p=dropout_rate, training=is_training)

    return out


# --- Example Usage ---
if __name__ == "__main__":
    from attention import precompute_rope_freqs

    B, T = 2, 16
    d_model = 768
    num_head = 12
    ffn_dim = 4 * d_model
    head_dim = d_model // num_head

    dummy_input = torch.randn(B, T, d_model)

    # Without RoPE, default GELU
    output = transformer_block(
        dummy_input, d_model=d_model, num_head=num_head, ffn_dim=ffn_dim
    )
    print("Transformer Block Output Shape (no RoPE, GELU):", output.shape)

    # With RoPE, SWIGLU
    freqs_cos, freqs_sin = precompute_rope_freqs(head_dim, T)
    output_rope = transformer_block(
        dummy_input, d_model=d_model, num_head=num_head, ffn_dim=int(d_model*8/3),
        freqs_cos=freqs_cos, freqs_sin=freqs_sin, activation="SWIGLU",
    )
    print("Transformer Block Output Shape (with RoPE, SWIGLU):", output_rope.shape)