import torch
import torch.nn as nn
from embeddings import create_token_embeddings, create_token_and_positional_embeddings
from attention import precompute_rope_freqs
from block import transformer_block
from logits import language_modeling_head


def full_transformer_model(
    input_ids: torch.Tensor,
    vocab_size: int,
    d_model: int,
    num_head: int,
    ffn_dim: int,
    num_layers: int,
    context_length: int,
    dropout_rate: float = 0.1,
    lm_head_layer: nn.Linear = None,
    use_rope: bool = True,
    rope_theta: float = 10000.0,
    activation: str = "GELU",
) -> torch.Tensor:
    """Passes input token IDs through token embeddings, stacked
    transformer blocks (with optional RoPE), a final LayerNorm,
    and the standalone language modeling head to produce vocabulary logits.

    Args:
        use_rope: If True, uses Rotary Position Embeddings instead of
                  learned absolute positional embeddings.
        rope_theta: Base frequency for RoPE (default: 10000.0).
        activation: Activation function to use in FFN (GELU, RELU, SWIGLU).
    """
    device = input_ids.device
    head_dim = d_model // num_head
    _, seq_len = input_ids.shape

    # 1. Token Embeddings (+ positional if not using RoPE)
    if use_rope:
        x = create_token_embeddings(
            input_ids=input_ids,
            vocab_size=vocab_size,
            d_model=d_model,
            dropout_rate=dropout_rate,
            device=device,
        )
        # Precompute RoPE frequencies for this sequence length
        freqs_cos, freqs_sin = precompute_rope_freqs(head_dim, seq_len, rope_theta, device)
    else:
        x = create_token_and_positional_embeddings(
            input_ids=input_ids,
            vocab_size=vocab_size,
            d_model=d_model,
            context_length=context_length,
            dropout_rate=dropout_rate,
            device=device,
        )
        freqs_cos, freqs_sin = None, None

    # 2. Stacked Transformer Blocks
    for _ in range(num_layers):
        x = transformer_block(
            x=x,
            d_model=d_model,
            num_head=num_head,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            activation=activation,
        )

    # 3. Final LayerNorm
    final_ln = nn.LayerNorm(d_model, device=device)
    x = final_ln(x)

    # 4. Language Modeling Head using the standalone function
    logits = language_modeling_head(
        x=x,
        vocab_size=vocab_size,
        d_model=d_model,
        lm_head_layer=lm_head_layer,
    )

    return logits


# --- Example Usage ---
if __name__ == "__main__":
    B, T = 2, 16
    vocab_size = 32000
    d_model = 768
    num_head = 12
    ffn_dim = int(d_model * 8/3)
    num_layers = 6
    context_length = 1024

    dummy_input_ids = torch.randint(0, vocab_size, (B, T))

    # With RoPE and SwiGLU
    logits = full_transformer_model(
        input_ids=dummy_input_ids,
        vocab_size=vocab_size,
        d_model=d_model,
        num_head=num_head,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        context_length=context_length,
        use_rope=True,
        activation="SWIGLU",
    )
    print("Full Model Logits Shape (RoPE + SwiGLU):", logits.shape)