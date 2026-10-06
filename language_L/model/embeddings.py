import torch
import torch.nn as nn


def create_token_embeddings(
    input_ids: torch.Tensor,
    vocab_size: int,
    d_model: int,
    dropout_rate: float = 0.1,
    training: bool = None,
    device: torch.device = None,
    tok_emb_layer: nn.Embedding = None,
) -> torch.Tensor:
    """Creates token embeddings (no positional embedding — use RoPE instead).

    Applies embedding dropout after the token embedding lookup.

    Args:
        input_ids (torch.Tensor): Tensor of token IDs with shape (B, T).
        vocab_size (int): Total size of the vocabulary.
        d_model (int): Hidden dimension size of the model.
        dropout_rate (float): Dropout probability applied after token embeddings.
        training: If None, dropout follows torch.is_grad_enabled().
        device (torch.device, optional): Device to place layers on.
        tok_emb_layer (nn.Embedding, optional): Pre-initialized embedding layer.

    Returns:
        torch.Tensor: Embedded output with shape (B, T, d_model).
    """
    if device is None:
        device = input_ids.device

    # 1. Token Embedding
    if tok_emb_layer is None:
        tok_emb_layer = nn.Embedding(vocab_size, d_model).to(device)

    out = tok_emb_layer(input_ids)

    # 2. Embedding dropout
    is_training = training if training is not None else torch.is_grad_enabled()
    import torch.nn.functional as F
    out = F.dropout(out, p=dropout_rate, training=is_training)

    return out


# Keep backward compatibility alias
def create_token_and_positional_embeddings(
    input_ids: torch.Tensor,
    vocab_size: int,
    d_model: int,
    context_length: int,
    dropout_rate: float = 0.1,
    training: bool = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Legacy function: Combines token embeddings and learnable positional embeddings,
    then applies embedding dropout.

    NOTE: This is kept for backward compatibility. For RoPE-based models,
    use create_token_embeddings() instead — positional information is
    injected via rotary embeddings inside the attention layer.

    Args:
        input_ids (torch.Tensor): Tensor of token IDs with shape (B, T).
        vocab_size (int): Total size of the vocabulary.
        d_model (int): Hidden dimension size of the model.
        context_length (int): Maximum sequence length (T).
        dropout_rate (float): Dropout probability applied after adding
            token + positional embeddings (embedding dropout).
        training: If None, dropout follows torch.is_grad_enabled().
        device (torch.device, optional): Device to place layers on.

    Returns:
        torch.Tensor: Embedded output with shape (B, T, d_model).
    """
    if device is None:
        device = input_ids.device

    batch_size, seq_len = input_ids.shape

    if seq_len > context_length:
        raise ValueError(
            f"Sequence length {seq_len} exceeds maximum context length {context_length}."
        )

    # 1. Initialize Embedding Layers
    token_embedding_layer = nn.Embedding(vocab_size, d_model).to(device)
    position_embedding_layer = nn.Embedding(context_length, d_model).to(device)

    # 2. Extract token embeddings: (B, T) -> (B, T, d_model)
    tok_emb = token_embedding_layer(input_ids)

    # 3. Create position indices: (T,) -> (1, T) -> (B, T)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

    # 4. Extract position embeddings: (B, T) -> (B, T, d_model)
    pos_emb = position_embedding_layer(positions)

    # 5. Sum token and positional embeddings
    out = tok_emb + pos_emb

    # 6. Embedding dropout
    is_training = training if training is not None else torch.is_grad_enabled()
    import torch.nn.functional as F
    out = F.dropout(out, p=dropout_rate, training=is_training)

    return out


# --- Example Usage ---
if __name__ == "__main__":
    B, T = 2, 16
    vocab_size = 32000
    d_model = 8

    dummy_input_ids = torch.randint(0, vocab_size, (B, T))

    # RoPE-style (token only)
    embeddings = create_token_embeddings(
        input_ids=dummy_input_ids,
        vocab_size=vocab_size,
        d_model=d_model,
    )
    print("Token-only embedding shape:", embeddings.shape)

    # Legacy (token + positional)
    embeddings_legacy = create_token_and_positional_embeddings(
        input_ids=dummy_input_ids,
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=1024,
    )
    print("Token+Positional embedding shape:", embeddings_legacy.shape)