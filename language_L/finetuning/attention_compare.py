"""
Post-finetune attention analysis for Model H (Nepali).

Compares pretrained vs finetuned heatmaps and attention statistics on a held-out
comparative reasoning prompt across all checkpoints in finetuned_checkpoints.
Each checkpoint output is organized into its own subfolder under report/attention_finetune/
(e.g., finetuned_checkpoints_100, finetuned_checkpoints_2k, finetuned_checkpoints_10k).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from tokenizers import Tokenizer

_FINETUNE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _FINETUNE_DIR.parent
for _p in (
    str(_FINETUNE_DIR),
    str(_PROJECT_DIR / "eval"),
    str(_PROJECT_DIR / "train"),
    str(_PROJECT_DIR / "model"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attention_analysis import (  # noqa: E402
    attention_entropy,
    extract_attention,
    mean_attention_distance,
    plot_heatmap,
)
from evaluate import _load_cfg, load_model_for_eval  # noqa: E402
from template import format_prompt  # noqa: E402
from train_finetune import _resolve  # noqa: E402

CONFIG_PATH = _PROJECT_DIR / "configs" / "finetune.yaml"
CHECKPOINTS_DIR = _PROJECT_DIR / "finetuned_checkpoints"
OUT_DIR = _PROJECT_DIR / "report" / "attention_finetune"
HEADS = (0, 3, 7)
PRETRAINED = _PROJECT_DIR / "checkpoints" / "checkpoints" / "step_0045000.pt"
TEST_FILE = _FINETUNE_DIR / "data" / "test_fixed.jsonl"
_INDIC_FONTS = [
    Path.home() / ".local/share/fonts/NotoSansDevanagari.ttf",
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
]


def extract_sample_count(p: Path) -> int:
    """Extract sample count integer from checkpoint filename for numerical sorting."""
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else float("inf")


def get_checkpoint_tag(ckpt_path: Path) -> str:
    """Derive human-readable checkpoint tag: e.g., 2000 -> '2k', 100 -> '100'."""
    m = re.search(r"samples_(\d+)", ckpt_path.stem)
    if not m:
        m = re.search(r"(\d+)", ckpt_path.stem)
    if m:
        val = int(m.group(1))
        if val >= 1000:
            return f"{val / 1000:g}k"
        return str(val)
    stem = ckpt_path.stem
    for prefix in ("finetuned_train_samples_", "finetuned_train_", "finetuned_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def get_output_folder_name(ckpt_path: Path) -> str:
    """Folder name matching pattern: finetuned_checkpoints_<tag> (e.g. finetuned_checkpoints_2k)."""
    tag = get_checkpoint_tag(ckpt_path)
    return f"finetuned_checkpoints_{tag}"


def _setup_indic_font() -> None:
    """Register Noto fonts so heatmap tick labels render Nepali (Devanagari) script."""
    names = []
    for path in _INDIC_FONTS:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            names.append(font_manager.FontProperties(fname=str(path)).get_name())
    plt.rcParams["font.family"] = names + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_jsonl_rows(path: Path) -> list[dict]:
    """Load JSON objects supporting both single-line JSONL and formatted concatenated JSON."""
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = 0
    items = []
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        obj, end = decoder.raw_decode(content, pos)
        items.append(obj)
        pos = end
    return items


def _pick_prompt() -> tuple[str, dict]:
    """Prefer a short 2-hop comparison so heatmaps stay readable."""
    rows = _load_jsonl_rows(TEST_FILE)
    chosen = None
    for row in rows:
        if chosen is None:
            chosen = row
        if row.get("hop_count") == 2 and row.get("category") == "2-hop comparisons":
            chosen = row
            break
    if chosen is None:
        raise ValueError(f"No valid sample found in {TEST_FILE}")
    user = chosen["messages"][0]["content"]
    return format_prompt(user), chosen


def _token_labels(tokenizer: Tokenizer, ids: list[int]) -> list[str]:
    labels = []
    for tid in ids:
        piece = tokenizer.decode([tid], skip_special_tokens=False).replace("\n", "↵")
        labels.append(piece[:10] if piece else str(tid))
    return labels


def _side_by_side(
    pre_w: list[torch.Tensor],
    ft_w: list[torch.Tensor],
    layer: int,
    head: int,
    tokens: list[str],
    save_path: Path,
    title: str,
    ft_label: str = "Finetuned",
) -> None:
    pre = pre_w[layer][0, head].numpy()
    ft = ft_w[layer][0, head].numpy()
    t = pre.shape[0]
    fig, axs = plt.subplots(1, 2, figsize=(max(12, t // 1.5), max(5, t // 3)))
    vmax = max(float(pre.max()), float(ft.max()), 1e-5)
    for ax, mat, name in ((axs[0], pre, "Pretrained"), (axs[1], ft, ft_label)):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"{name} — {title}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        if tokens and len(tokens) == t:
            ax.set_xticks(range(t))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(t))
            ax.set_yticklabels(tokens, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _layer_summary(entropies: torch.Tensor, distances: torch.Tensor) -> dict:
    ent = entropies.mean(dim=-1)  # (L, H)
    dist = distances.mean(dim=-1)
    return {
        "entropy_layer_head_mean": ent.tolist(),
        "mean_distance_layer_head_mean": dist.tolist(),
        "entropy_per_layer": ent.mean(dim=-1).tolist(),
        "distance_per_layer": dist.mean(dim=-1).tolist(),
    }


def process_single_checkpoint(
    ckpt_path: Path,
    cfg: dict,
    device: torch.device,
    input_ids: torch.Tensor,
    tokens: list[str],
    pre_w: list[torch.Tensor],
    pre_stats: dict,
    out_base_dir: Path,
    prompt_info: dict,
    layers_to_plot: list[int],
    heads: tuple[int, ...],
) -> dict:
    tag = get_checkpoint_tag(ckpt_path)
    folder_name = get_output_folder_name(ckpt_path)
    ckpt_out_dir = out_base_dir / folder_name
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    ft_label = f"Finetuned ({tag})"

    print(f"  -> Checkpoint: {ckpt_path.name}")
    print(f"     Output folder: {ckpt_out_dir.name}")

    # 1. Write prompt info
    (ckpt_out_dir / "prompt.json").write_text(
        json.dumps(prompt_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. Load finetuned model & extract attention
    ft_model = load_model_for_eval(ckpt_path, cfg, device)
    ft_w = extract_attention(ft_model, input_ids)
    del ft_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Heatmaps
    for layer in layers_to_plot:
        for head in heads:
            _side_by_side(
                pre_w,
                ft_w,
                layer,
                head,
                tokens,
                ckpt_out_dir / f"compare_layer_{layer}_head_{head}.png",
                f"Layer {layer} Head {head}",
                ft_label=ft_label,
            )
            plot_heatmap(
                ft_w,
                layer=layer,
                head=head,
                tokens=tokens,
                save_path=ckpt_out_dir / f"finetuned_layer_{layer}_head_{head}.png",
                title=f"{ft_label} — Layer {layer} Head {head}",
            )

    # 4. Statistics & Deltas
    ft_stats = _layer_summary(attention_entropy(ft_w), mean_attention_distance(ft_w))
    del ft_w
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "prompt_category": prompt_info["sample"]["category"],
        "checkpoint_file": ckpt_path.name,
        "checkpoint_tag": tag,
        "folder_name": folder_name,
        "sample_count": extract_sample_count(ckpt_path),
        "pretrained": pre_stats,
        f"finetuned_{tag}": ft_stats,
        "finetuned": ft_stats,
        "delta_entropy_per_layer": [
            ft - pre
            for pre, ft in zip(pre_stats["entropy_per_layer"], ft_stats["entropy_per_layer"])
        ],
        "delta_distance_per_layer": [
            ft - pre
            for pre, ft in zip(pre_stats["distance_per_layer"], ft_stats["distance_per_layer"])
        ],
    }
    (ckpt_out_dir / "pretrained_vs_finetuned_attention.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # 5. Layer-wise entropy and distance plot
    layers = list(range(len(pre_stats["entropy_per_layer"])))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].plot(layers, pre_stats["entropy_per_layer"], marker="o", label="Pretrained")
    axs[0].plot(layers, ft_stats["entropy_per_layer"], marker="s", label=ft_label)
    axs[0].set_title("Mean attention entropy by layer")
    axs[0].set_xlabel("Layer")
    axs[0].set_ylabel("Entropy (bits)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(layers, pre_stats["distance_per_layer"], marker="o", label="Pretrained")
    axs[1].plot(layers, ft_stats["distance_per_layer"], marker="s", label=ft_label)
    axs[1].set_title("Mean attention distance by layer")
    axs[1].set_xlabel("Layer")
    axs[1].set_ylabel("Mean |query − key|")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(ckpt_out_dir / "entropy_distance_pretrained_vs_finetuned.png", dpi=200)
    plt.close(fig)

    return summary


def generate_scaling_attention_plots(all_summaries: list[dict], out_base_dir: Path) -> None:
    """Generate combined plots showing how attention metrics scale with training samples."""
    valid = [s for s in all_summaries if s["sample_count"] != float("inf")]
    if len(valid) < 2:
        return

    valid.sort(key=lambda s: s["sample_count"])
    samples = [s["sample_count"] for s in valid]
    tags = [s["checkpoint_tag"] for s in valid]

    # Pretrained references
    pre = valid[0]["pretrained"]
    pre_mean_entropy = sum(pre["entropy_per_layer"]) / len(pre["entropy_per_layer"])
    pre_mean_dist = sum(pre["distance_per_layer"]) / len(pre["distance_per_layer"])
    pre_l0_ent = pre["entropy_per_layer"][0]
    pre_last_ent = pre["entropy_per_layer"][-1]

    ft_mean_entropies = [
        sum(s["finetuned"]["entropy_per_layer"]) / len(s["finetuned"]["entropy_per_layer"])
        for s in valid
    ]
    ft_mean_distances = [
        sum(s["finetuned"]["distance_per_layer"]) / len(s["finetuned"]["distance_per_layer"])
        for s in valid
    ]
    ft_l0_entropies = [s["finetuned"]["entropy_per_layer"][0] for s in valid]
    ft_last_entropies = [s["finetuned"]["entropy_per_layer"][-1] for s in valid]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Mean Entropy vs Train Samples
    axs[0].plot(samples, ft_mean_entropies, marker="o", color="tab:blue", label="Finetuned (All Layers Avg)")
    axs[0].plot(samples, ft_l0_entropies, marker="^", linestyle="--", color="tab:cyan", label="Finetuned Layer 0")
    axs[0].plot(samples, ft_last_entropies, marker="v", linestyle="--", color="tab:purple", label=f"Finetuned Layer {len(pre['entropy_per_layer'])-1}")
    axs[0].axhline(pre_mean_entropy, color="tab:blue", linestyle=":", label="Pretrained (All Layers Avg)")
    axs[0].axhline(pre_l0_ent, color="tab:cyan", linestyle=":", label="Pretrained Layer 0")
    axs[0].axhline(pre_last_ent, color="tab:purple", linestyle=":", label=f"Pretrained Layer {len(pre['entropy_per_layer'])-1}")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Finetuning Training Samples (Log Scale)")
    axs[0].set_ylabel("Attention Entropy (bits)")
    axs[0].set_title("Attention Entropy vs. Finetuning Data Size")
    axs[0].set_xticks(samples)
    axs[0].set_xticklabels(tags, rotation=45, fontsize=8)
    axs[0].legend(fontsize=8)
    axs[0].grid(True, which="both", alpha=0.3)

    # Right: Mean Distance vs Train Samples
    axs[1].plot(samples, ft_mean_distances, marker="s", color="tab:red", label="Finetuned (Mean Attention Distance)")
    axs[1].axhline(pre_mean_dist, color="tab:red", linestyle=":", label="Pretrained Mean Distance")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Finetuning Training Samples (Log Scale)")
    axs[1].set_ylabel("Mean Token Distance |query − key|")
    axs[1].set_title("Mean Attention Distance vs. Finetuning Data Size")
    axs[1].set_xticks(samples)
    axs[1].set_xticklabels(tags, rotation=45, fontsize=8)
    axs[1].legend(fontsize=8)
    axs[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plot_path = out_base_dir / "scaling_attention_entropy_distance.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"[Attention] Saved scaling summary plot -> {plot_path}")

    # Multi-checkpoint summary JSON
    summary_data = {
        "samples": samples,
        "tags": tags,
        "pretrained_overall_mean_entropy": pre_mean_entropy,
        "pretrained_overall_mean_distance": pre_mean_dist,
        "checkpoints": {
            s["checkpoint_tag"]: {
                "checkpoint": s["checkpoint_file"],
                "folder": s["folder_name"],
                "samples": s["sample_count"],
                "mean_entropy": sum(s["finetuned"]["entropy_per_layer"]) / len(s["finetuned"]["entropy_per_layer"]),
                "mean_distance": sum(s["finetuned"]["distance_per_layer"]) / len(s["finetuned"]["distance_per_layer"]),
                "delta_entropy_per_layer": s["delta_entropy_per_layer"],
                "delta_distance_per_layer": s["delta_distance_per_layer"],
            }
            for s in valid
        },
    }
    summary_file = out_base_dir / "all_checkpoints_attention_summary.json"
    summary_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"[Attention] Saved multi-checkpoint summary -> {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-finetune attention analysis for Model H (Nepali) across checkpoints"
    )
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to config file")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(CHECKPOINTS_DIR),
        help="Directory containing finetuned checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Run analysis for a single checkpoint file only",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=str(PRETRAINED),
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUT_DIR),
        help="Base output directory",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Plot heatmaps for all layers (0..L-1) instead of only first and last (0, last)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=list(HEADS),
        help="Attention heads to plot (default: 0 3 7)",
    )
    args = parser.parse_args()

    _setup_indic_font()
    config_path = Path(args.config)
    cfg = _load_cfg(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = Path(args.pretrained)
    out_dir = Path(args.output_dir)
    heads = tuple(args.heads)

    prompt, sample = _pick_prompt()
    tokenizer = Tokenizer.from_file(str(_resolve(cfg["tokenizer"]["tokenizer_path"])))
    ids = tokenizer.encode(prompt).ids[:48]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    tokens = _token_labels(tokenizer, ids)

    prompt_info = {
        "language": "Nepali",
        "prompt": prompt,
        "sample": {
            "category": sample.get("category"),
            "hop_count": sample.get("hop_count"),
            "answer_option": sample.get("answer_option"),
            "question": sample["messages"][0]["content"] if "messages" in sample else "",
        },
        "n_tokens": len(ids),
    }

    print(f"[Attention] Device: {device} | Tokens: {len(ids)}")
    print(f"[Attention] Category: {sample.get('category')} | Hop count: {sample.get('hop_count')}")
    print(f"[Attention] Question snippet: {sample['messages'][0]['content'][:120]}...")
    print(f"[Attention] Loading pretrained model once: {pretrained_path.name}")

    pre_model = load_model_for_eval(pretrained_path, cfg, device)
    pre_w = extract_attention(pre_model, input_ids)
    num_layers = pre_model.num_layers
    last = num_layers - 1
    layers_to_plot = list(range(num_layers)) if args.all_layers else [0, last]
    del pre_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pre_stats = _layer_summary(attention_entropy(pre_w), mean_attention_distance(pre_w))

    # Find checkpoints to process
    if args.checkpoint:
        ckpt_paths = [Path(args.checkpoint)]
        if not ckpt_paths[0].exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_paths[0]}")
    else:
        ckpts_dir = Path(args.checkpoints_dir)
        if not ckpts_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {ckpts_dir}")
        ckpt_paths = [p for p in ckpts_dir.glob("*.pt") if not p.name.startswith(".")]
        ckpt_paths.sort(key=extract_sample_count)

    print(f"[Attention] Found {len(ckpt_paths)} checkpoint(s) to analyze.")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for idx, ckpt_path in enumerate(ckpt_paths, 1):
        print(f"\n[{idx}/{len(ckpt_paths)}] Analyzing checkpoint: {ckpt_path.name}")
        summary = process_single_checkpoint(
            ckpt_path=ckpt_path,
            cfg=cfg,
            device=device,
            input_ids=input_ids,
            tokens=tokens,
            pre_w=pre_w,
            pre_stats=pre_stats,
            out_base_dir=out_dir,
            prompt_info=prompt_info,
            layers_to_plot=layers_to_plot,
            heads=heads,
        )
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print("\n[Attention] Generating multi-checkpoint scaling analysis...")
        generate_scaling_attention_plots(all_summaries, out_dir)

    print(f"\n[Attention] Completed all {len(all_summaries)} checkpoints!")
    print(f"[Attention] All results saved under: {out_dir}")


if __name__ == "__main__":
    main()