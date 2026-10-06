import torch
import torch.nn as nn
import torch.nn.functional as F


def feed_forward_network(
    x: torch.Tensor,
    d_model: int,
    ffn_dim: int,
    w1: nn.Linear = None,
    w2: nn.Linear = None,
    w3: nn.Linear = None,
    activation: str = "GELU",
) -> torch.Tensor:
    """Applies a Position-Wise Feed-Forward Network with GELU, ReLU, or SwiGLU.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, d_model).
        d_model (int): Hidden dimension size.
        ffn_dim (int): Expanded inner dimension size.
        w1, w2, w3 (nn.Linear, optional): Pre-initialized linear layers.
            w3 is only used if activation is "SWIGLU".
        activation (str): "GELU", "RELU", or "SWIGLU".

    Returns:
        torch.Tensor: Output tensor of shape (B, T, d_model).
    """
    activation = activation.upper()

    # Initialize linear layers if not provided
    if w1 is None:
        w1 = nn.Linear(d_model, ffn_dim, bias=True, device=x.device)
    if w2 is None:
        w2 = nn.Linear(ffn_dim, d_model, bias=True, device=x.device)
    
    if activation == "SWIGLU":
        if w3 is None:
            w3 = nn.Linear(d_model, ffn_dim, bias=True, device=x.device)
        
        # SwiGLU: (SiLU(x * W1)) * (x * W3) -> W2
        gate = F.silu(w1(x))
        up   = w3(x)
        out  = w2(gate * up)
    else:
        # Standard FFN
        h = w1(x)
        if activation == "GELU":
            h = F.gelu(h)
        elif activation == "RELU":
            h = F.relu(h)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        out = w2(h)

    return out


# --- Example Usage ---
if __name__ == "__main__":
    B, T = 2, 16
    d_model = 768
    ffn_dim = 4 * d_model

    dummy_input = torch.randn(B, T, d_model)
    
    out_gelu = feed_forward_network(dummy_input, d_model, ffn_dim, activation="GELU")
    print("GELU Output:", out_gelu.shape)
    
    out_relu = feed_forward_network(dummy_input, d_model, ffn_dim, activation="RELU")
    print("ReLU Output:", out_relu.shape)
    
    # SwiGLU usually reduces ffn_dim (e.g. 8/3 instead of 4) to keep param count matched
    swiglu_ffn_dim = int(d_model * (8/3))
    out_swiglu = feed_forward_network(dummy_input, d_model, swiglu_ffn_dim, activation="SWIGLU")
    print("SwiGLU Output:", out_swiglu.shape)