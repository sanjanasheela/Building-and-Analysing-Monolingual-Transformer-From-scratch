import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) helpers
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos and sin rotation matrices for RoPE.

    Args:
        head_dim:    Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta:       Base frequency (10 000 is the standard default).
        device:      Target device.

    Returns:
        (freqs_cos, freqs_sin) each of shape (max_seq_len, head_dim).
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

    # Frequencies for each pair of dimensions: θ_i = 1 / (theta^(2i/d))
    freq_exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (freq_exponents / head_dim))  # (head_dim/2,)

    # Position indices
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (T,)

    # Outer product: (T,) x (head_dim/2,) -> (T, head_dim/2)
    freqs = torch.outer(positions, inv_freq)

    # Duplicate to full head_dim: (T, head_dim)
    freqs = torch.cat([freqs, freqs], dim=-1)

    return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to a Q or K tensor.

    Args:
        x:         (B, num_head, T, head_dim)
        freqs_cos: (T, head_dim) or broadcastable
        freqs_sin: (T, head_dim) or broadcastable

    Returns:
        Rotated tensor, same shape as x.
    """
    # Split x into pairs and rotate
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]    # first half
    x2 = x[..., head_dim // 2 :]    # second half

    # Build the "rotated" version: [-x2, x1]
    x_rotated = torch.cat([-x2, x1], dim=-1)

    # Reshape freqs for broadcasting: (T, head_dim) -> (1, 1, T, head_dim)
    cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    return x * cos + x_rotated * sin


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------

def multi_head_causal_self_attention(
    x: torch.Tensor,
    d_model: int,
    num_head: int,
    wq: nn.Linear = None,
    wk: nn.Linear = None,
    wv: nn.Linear = None,
    wo: nn.Linear = None,
    attn_dropout_rate: float = 0.1,
    training: bool = None,
    return_weights: bool = False,
    freqs_cos: torch.Tensor = None,
    freqs_sin: torch.Tensor = None,
) -> torch.Tensor:
    """Computes multi-head causal self-attention from scratch.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, d_model).
        d_model (int): Hidden dimension size.
        num_head (int): Number of attention heads.
        wq, wk, wv, wo (nn.Linear, optional): Pre-initialized linear layers.
        attn_dropout_rate (float): Dropout applied to attention weights after
            softmax (disabled automatically inside torch.no_grad()).
        training: If None, dropout follows torch.is_grad_enabled().
        return_weights (bool): If True, returns (out, attn_weights) where
            attn_weights has shape (B, num_head, T, T). Default False.
        freqs_cos (torch.Tensor, optional): Precomputed RoPE cosine freqs
            of shape (T, head_dim). If provided (along with freqs_sin),
            rotary embeddings are applied to Q and K.
        freqs_sin (torch.Tensor, optional): Precomputed RoPE sine freqs.

    Returns:
        torch.Tensor | tuple: Output (B, T, d_model), or (output, attn_weights)
            when return_weights=True.
    """
    batch_size, seq_len, _ = x.shape

    if d_model % num_head != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_head ({num_head})."
        )

    head_dim = d_model // num_head

    # Initialize weights if not provided
    if wq is None:
        wq = nn.Linear(d_model, d_model, bias=False, device=x.device)
    if wk is None:
        wk = nn.Linear(d_model, d_model, bias=False, device=x.device)
    if wv is None:
        wv = nn.Linear(d_model, d_model, bias=False, device=x.device)
    if wo is None:
        wo = nn.Linear(d_model, d_model, bias=False, device=x.device)

    # 1. Q, K, V Projections: (B, T, d_model) -> (B, T, d_model)
    q = wq(x)
    k = wk(x)
    v = wv(x)

    # 2. Reshape and Transpose for Multi-Head:
    # (B, T, num_head, head_dim) -> (B, num_head, T, head_dim)
    q = q.view(batch_size, seq_len, num_head, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_head, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_head, head_dim).transpose(1, 2)

    # 2b. Apply RoPE to Q and K (if provided)
    if freqs_cos is not None and freqs_sin is not None:
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

    # 3. Scaled Dot-Product Attention Scores: (B, num_head, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # 4. Apply Causal Mask (prevent looking into the future)
    # Creates an upper triangular matrix of 1s above the main diagonal
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
    )
    scores = scores + mask

    # Softmax to get attention probabilities
    attn_weights = F.softmax(scores, dim=-1)

    # Attention dropout (on probabilities, before multiplying by V)
    # Disabled inside torch.no_grad() via is_grad_enabled().
    is_training = training if training is not None else torch.is_grad_enabled()
    attn_weights = F.dropout(attn_weights, p=attn_dropout_rate, training=is_training)

    # 5. Multiply by Values: (B, num_head, T, head_dim)
    attn_output = torch.matmul(attn_weights, v)

    # 6. Concatenate Heads and Transpose Back:
    # (B, num_head, T, head_dim) -> (B, T, num_head * head_dim) = (B, T, d_model)
    attn_output = (
        attn_output.transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_len, d_model)
    )

    # 7. Final Output Projection
    out = wo(attn_output)

    if return_weights:
        return out, attn_weights   # attn_weights: (B, num_head, T, T)
    return out


# --- Example Usage ---
if __name__ == "__main__":
    B, T = 2, 16
    d_model = 768
    num_head = 12
    head_dim = d_model // num_head

    dummy_input = torch.randn(B, T, d_model)

    # Without RoPE (original behavior)
    output = multi_head_causal_self_attention(
        dummy_input, d_model=d_model, num_head=num_head
    )
    print("Attention Output Shape (no RoPE):", output.shape)

    # With RoPE
    freqs_cos, freqs_sin = precompute_rope_freqs(head_dim, T)
    output_rope = multi_head_causal_self_attention(
        dummy_input, d_model=d_model, num_head=num_head,
        freqs_cos=freqs_cos, freqs_sin=freqs_sin,
    )
    print("Attention Output Shape (with RoPE):", output_rope.shape)