import torch
import torch.nn as nn


def language_modeling_head(
    x: torch.Tensor,
    vocab_size: int,
    d_model: int,
    lm_head_layer: nn.Linear = None,
) -> torch.Tensor:
    """Projects hidden states of shape (B, T, d_model) to vocabulary logits of

    shape (B, T, vocab_size).
    """
    if lm_head_layer is None:
        lm_head_layer = nn.Linear(d_model, vocab_size, bias=False, device=x.device)

    logits = lm_head_layer(x)
    return logits


# --- Example Usage ---
if __name__ == "__main__":
    B, T = 2, 16
    d_model = 768
    vocab_size = 32000

    # Output from the final layer of your transformer stack
    transformer_output = torch.randn(B, T, d_model)

    logits = language_modeling_head(
        x=transformer_output,
        vocab_size=vocab_size,
        d_model=d_model,
    )

    print("LM Head Logits Shape:", logits.shape)  # Expected: (B, T, vocab_size)