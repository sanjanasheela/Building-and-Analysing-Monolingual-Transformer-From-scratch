from __future__ import annotations

import sys
from pathlib import Path
import json

import yaml
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import numpy as np

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _EVAL_DIR.parent
sys.path.append(str(_PROJECT_DIR / "train"))
sys.path.append(str(_EVAL_DIR))

from model import build_from_config
from attention import multi_head_causal_self_attention


# --- Core Attention Extraction & Diagnostics ---------------------------------

@torch.no_grad()
def extract_attention(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Run a forward pass and collect post-softmax attention weights for every layer."""
    model.eval()
    device = input_ids.device
    B, T = input_ids.shape

    x = model.tok_emb(input_ids)
    x = F.dropout(x, p=0.0, training=False)

    freqs_cos = model.freqs_cos[:T] if hasattr(model, "freqs_cos") else None
    freqs_sin = model.freqs_sin[:T] if hasattr(model, "freqs_sin") else None

    all_weights = []

    for i in range(model.num_layers):
        normed = model.lns1[i](x)
        out, weights = multi_head_causal_self_attention(
            normed,
            d_model=model.d_model,
            num_head=model.num_heads,
            wq=model.wqs[i], wk=model.wks[i],
            wv=model.wvs[i], wo=model.wos[i],
            attn_dropout_rate=0.0,
            training=False,
            return_weights=True,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )
        all_weights.append(weights.cpu())  # (1, H, T, T)

        x = x + out
        normed2 = model.lns2[i](x)

        from ffnn import feed_forward_network
        w3 = model.w3s[i] if hasattr(model, "w3s") and model.w3s is not None else None
        ffn_out = feed_forward_network(
            normed2, model.d_model, model.ffn_dim,
            w1=model.w1s[i], w2=model.w2s[i], w3=w3,
            activation=getattr(model, "activation", "GELU"),
        )
        x = x + ffn_out

    return all_weights


def plot_heatmap(
    attn_weights: list[torch.Tensor],
    layer: int,
    head: int,
    tokens: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "",
) -> None:
    """Plot and save an attention heatmap for a specific layer and head."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = attn_weights[layer][0, head].numpy()  # (T, T)
    T = w.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, T // 3), max(5, T // 3)))
    im = ax.imshow(w, cmap="Blues", vmin=0, vmax=max(w.max(), 1e-5))
    plt.colorbar(im, ax=ax)

    if tokens and len(tokens) == T:
        ax.set_xticks(range(T))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(T))
        ax.set_yticklabels(tokens, fontsize=8)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title or f"Layer {layer}  Head {head}")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150)
        plt.close()
    else:
        plt.show()


def attention_entropy(attn_weights: list[torch.Tensor]) -> torch.Tensor:
    """Compute Shannon entropy of each attention distribution: -sum(p * log2(p))."""
    entropies = []
    for w in attn_weights:
        w = w.squeeze(0)  # (H, T, T)
        p = w.clamp(min=1e-9)
        ent = -(p * p.log2()).sum(dim=-1)  # (H, T)
        entropies.append(ent)
    return torch.stack(entropies)  # (L, H, T)


def mean_attention_distance(attn_weights: list[torch.Tensor]) -> torch.Tensor:
    """Compute mean token distance each query attends along the sequence."""
    distances = []
    for w in attn_weights:
        w = w.squeeze(0)  # (H, T, T)
        _, T, _ = w.shape
        pos = torch.arange(T, dtype=torch.float)
        q_pos = pos.unsqueeze(1)  # (T, 1)
        k_pos = pos.unsqueeze(0)  # (1, T)
        dist = (q_pos - k_pos).abs()  # (T, T)
        mean_dist = (w * dist.unsqueeze(0)).sum(dim=-1)  # (H, T)
        distances.append(mean_dist)
    return torch.stack(distances)  # (L, H, T)


# --- Main Orchestration -----------------------------------------------------

def main():
    config_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/configs/51.yaml")
    checkpoint_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/checkpoints/checkpoints/step_0045000.pt")
    tokenizer_path = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_7500/tokenizer.json"
    test_bin_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/token_shards_7500/test.bin")
    attention_dir = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/report/attention")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Build model & load checkpoint
    model = build_from_config(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load a representative sample snippet from test.bin (e.g., 24 tokens for clear heatmaps)
    context_len = 24
    if test_bin_path.exists():
        with open(test_bin_path, "rb") as f:
            bin_tokens = np.frombuffer(f.read(context_len * 4), dtype=np.uint32)
        sample_tensor = torch.tensor(bin_tokens[:context_len].astype(np.int64), dtype=torch.long, device=device).unsqueeze(0)
    else:
        sample_tensor = torch.randint(0, cfg["tokenizer"]["vocab_size"], (1, context_len), dtype=torch.long, device=device)

    # Get string token representations for axis labels
    token_strings = [tokenizer.decode([tid], skip_special_tokens=True) for tid in sample_tensor[0].tolist()]

    print("🔍 Extracting attention weights and calculating diagnostics...")
    weights = extract_attention(model, sample_tensor)
    entropies = attention_entropy(weights)
    mean_distances = mean_attention_distance(weights)

    attention_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save heatmaps for early layer (Layer 0) and late layer (Last Layer) across all heads
    layers_to_plot = layers_to_plot = range(model.num_layers)
    for layer_idx in layers_to_plot:
        n_heads = weights[layer_idx].shape[1]
        for head_idx in range(n_heads):
            save_path = attention_dir / f"heatmap_layer_{layer_idx}_head_{head_idx}.png"
            plot_heatmap(
                weights,
                layer=layer_idx,
                head=head_idx,
                tokens=None,
                save_path=save_path,
                title=f"Layer {layer_idx} | Head {head_idx}"
            )
    print(f"📊 Heatmaps saved to: {attention_dir}")

    # 2. Save summary statistics (Entropy and Mean Distance averages)
    summary_stats = {
        "entropy_layer_head_mean": entropies.mean(dim=-1).tolist(),       # (L, H)
        "mean_distance_layer_head_mean": mean_distances.mean(dim=-1).tolist(),  # (L, H)
    }

    summary_json_path = attention_dir / "attention_summary_stats.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4)

    print(f"📈 Attention summary statistics saved to: {summary_json_path}")
    print("\n✅ Attention analysis completed successfully!")


if __name__ == "__main__":
    main()