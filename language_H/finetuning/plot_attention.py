#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/home/sanjana/Documents/7/LMA/"
    "individual-project-sanjanasheela/language_H/"
    "report/attention_finetune"
)

OUTPUT_DIR = BASE_DIR / "json_attention_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND ALL CHECKPOINT JSON FILES
# ============================================================

json_files = list(
    BASE_DIR.glob(
        "finetuned_checkpoints_*/"
        "pretrained_vs_finetuned_attention.json"
    )
)

if not json_files:
    raise FileNotFoundError(
        "No pretrained_vs_finetuned_attention.json files found."
    )


# ============================================================
# READ JSON FILES & VALIDATE SHAPES
# ============================================================

results = []
expected_num_layers = None
reference_pretrained_entropy = None
reference_pretrained_distance = None

for json_path in json_files:

    print(f"Reading: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = int(data["sample_count"])
    folder_name = data.get("folder_name", json_path.parent.name)

    pretrained_entropy = np.array(data["pretrained"]["entropy_per_layer"], dtype=float)
    pretrained_distance = np.array(data["pretrained"]["distance_per_layer"], dtype=float)
    
    num_layers = len(pretrained_entropy)
    
    # Validate layer consistency across checkpoints
    if expected_num_layers is None:
        expected_num_layers = num_layers
        reference_pretrained_entropy = pretrained_entropy
        reference_pretrained_distance = pretrained_distance
    else:
        if num_layers != expected_num_layers:
            raise ValueError(
                f"Layer mismatch in {json_path}: found {num_layers} layers, expected {expected_num_layers}"
            )

    finetuned_entropy = np.array(data["finetuned"]["entropy_per_layer"], dtype=float)
    finetuned_distance = np.array(data["finetuned"]["distance_per_layer"], dtype=float)

    delta_entropy = np.array(data["delta_entropy_per_layer"], dtype=float)
    delta_distance = np.array(data["delta_distance_per_layer"], dtype=float)

    pretrained_entropy_heads = np.array(data["pretrained"]["entropy_layer_head_mean"], dtype=float)
    finetuned_entropy_heads = np.array(data["finetuned"]["entropy_layer_head_mean"], dtype=float)
    pretrained_distance_heads = np.array(data["pretrained"]["mean_distance_layer_head_mean"], dtype=float)
    finetuned_distance_heads = np.array(data["finetuned"]["mean_distance_layer_head_mean"], dtype=float)

    mean_pre_entropy = float(np.mean(pretrained_entropy))
    mean_ft_entropy = float(np.mean(finetuned_entropy))
    mean_pre_distance = float(np.mean(pretrained_distance))
    mean_ft_distance = float(np.mean(finetuned_distance))

    entropy_pct = (
        100.0
        * (finetuned_entropy - pretrained_entropy)
        / np.where(pretrained_entropy != 0, pretrained_entropy, np.nan)
    )

    distance_pct = (
        100.0
        * (finetuned_distance - pretrained_distance)
        / np.where(pretrained_distance != 0, pretrained_distance, np.nan)
    )

    results.append({
        "samples": samples,
        "folder": folder_name,
        "pretrained_entropy": pretrained_entropy,
        "finetuned_entropy": finetuned_entropy,
        "pretrained_distance": pretrained_distance,
        "finetuned_distance": finetuned_distance,
        "delta_entropy": delta_entropy,
        "delta_distance": delta_distance,
        "pretrained_entropy_heads": pretrained_entropy_heads,
        "finetuned_entropy_heads": finetuned_entropy_heads,
        "pretrained_distance_heads": pretrained_distance_heads,
        "finetuned_distance_heads": finetuned_distance_heads,
        "mean_pre_entropy": mean_pre_entropy,
        "mean_ft_entropy": mean_ft_entropy,
        "mean_pre_distance": mean_pre_distance,
        "mean_ft_distance": mean_ft_distance,
        "entropy_pct": entropy_pct,
        "distance_pct": distance_pct,
    })


# ============================================================
# SORT BY TRAINING SIZE
# ============================================================

results.sort(key=lambda x: x["samples"])


# ============================================================
# BASIC ARRAYS
# ============================================================

samples = np.array([r["samples"] for r in results])

labels = [
    (f"{r['samples'] / 1000:g}k" if r["samples"] >= 1000 else str(r["samples"]))
    for r in results
]

layers = np.arange(expected_num_layers)


# ============================================================
# HELPER
# ============================================================

def save_plot(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 1. MEAN ENTROPY
# ============================================================

plt.figure(figsize=(11, 7))
plt.plot(samples, [r["mean_pre_entropy"] for r in results], marker="s", linestyle="--", linewidth=2, label="Pretrained")
plt.plot(samples, [r["mean_ft_entropy"] for r in results], marker="o", linewidth=2, label="Fine-tuned")
plt.xscale("log")
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Mean attention entropy")
plt.title("Mean Attention Entropy Across All Checkpoints")
plt.legend()
plt.grid(alpha=0.3)
save_plot("01_mean_entropy_vs_samples.png")


# ============================================================
# 2. MEAN DISTANCE
# ============================================================

plt.figure(figsize=(11, 7))
plt.plot(samples, [r["mean_pre_distance"] for r in results], marker="s", linestyle="--", linewidth=2, label="Pretrained")
plt.plot(samples, [r["mean_ft_distance"] for r in results], marker="o", linewidth=2, label="Fine-tuned")
plt.xscale("log")
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Mean attention distance")
plt.title("Mean Attention Distance Across All Checkpoints")
plt.legend()
plt.grid(alpha=0.3)
save_plot("02_mean_distance_vs_samples.png")


# ============================================================
# 3. ENTROPY PER LAYER
# ============================================================

plt.figure(figsize=(12, 8))
for layer in layers:
    values = [r["finetuned_entropy"][layer] for r in results]
    plt.plot(samples, values, marker="o", linewidth=2, label=f"Layer {layer}")
plt.xscale("log")
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Mean attention entropy")
plt.title("Fine-tuned Attention Entropy by Layer")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
save_plot("03_entropy_per_layer_vs_samples.png")


# ============================================================
# 4. DISTANCE PER LAYER
# ============================================================

plt.figure(figsize=(12, 8))
for layer in layers:
    values = [r["finetuned_distance"][layer] for r in results]
    plt.plot(samples, values, marker="o", linewidth=2, label=f"Layer {layer}")
plt.xscale("log")
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Mean attention distance")
plt.title("Fine-tuned Attention Distance by Layer")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
save_plot("04_distance_per_layer_vs_samples.png")


# ============================================================
# 5. ENTROPY DELTA PER LAYER
# ============================================================

plt.figure(figsize=(12, 8))
for layer in layers:
    values = [r["delta_entropy"][layer] for r in results]
    plt.plot(samples, values, marker="o", linewidth=2, label=f"Layer {layer}")
plt.axhline(0, linestyle="--", linewidth=1.5, color="gray")
plt.xscale("log")
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Δ attention entropy")
plt.title("Change in Attention Entropy Relative to Pretrained")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
save_plot("05_entropy_delta_per_layer_vs_samples.png")


# ============================================================
# 6. DISTANCE DELTA PER LAYER
# ============================================================

plt.figure(figsize=(12, 8))
for layer in layers:
    values = [r["delta_distance"][layer] for r in results]
    plt.plot(samples, values, marker="o", linewidth=2, label=f"Layer {layer}")
plt.axhline(0, linestyle="--", linewidth=1.5, color="gray")
plt.xscale("log")
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Δ attention distance")
plt.title("Change in Attention Distance Relative to Pretrained")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
save_plot("06_distance_delta_per_layer_vs_samples.png")


# ============================================================
# 7. ENTROPY HEATMAP
# ============================================================

entropy_matrix = np.array([r["finetuned_entropy"] for r in results]).T

plt.figure(figsize=(13, 7))
plt.imshow(entropy_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="Mean attention entropy")
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(layers, [f"Layer {i}" for i in layers])
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Layer")
plt.title("Fine-tuned Attention Entropy: Layer × Checkpoint")
save_plot("07_entropy_heatmap_layers_x_checkpoints.png")


# ============================================================
# 8. DISTANCE HEATMAP
# ============================================================

distance_matrix = np.array([r["finetuned_distance"] for r in results]).T

plt.figure(figsize=(13, 7))
plt.imshow(distance_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="Mean attention distance")
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(layers, [f"Layer {i}" for i in layers])
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Layer")
plt.title("Fine-tuned Attention Distance: Layer × Checkpoint")
save_plot("08_distance_heatmap_layers_x_checkpoints.png")


# ============================================================
# 9. ENTROPY DELTA HEATMAP (Diverging Colormap)
# ============================================================

delta_entropy_matrix = np.array([r["delta_entropy"] for r in results]).T
vmax_ent = np.nanmax(np.abs(delta_entropy_matrix))

plt.figure(figsize=(13, 7))
plt.imshow(delta_entropy_matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax_ent, vmax=vmax_ent)
plt.colorbar(label="Δ attention entropy")
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(layers, [f"Layer {i}" for i in layers])
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Layer")
plt.title("Attention Entropy Change: Layer × Checkpoint")
save_plot("09_entropy_delta_heatmap_layers_x_checkpoints.png")


# ============================================================
# 10. DISTANCE DELTA HEATMAP (Diverging Colormap)
# ============================================================

delta_distance_matrix = np.array([r["delta_distance"] for r in results]).T
vmax_dist = np.nanmax(np.abs(delta_distance_matrix))

plt.figure(figsize=(13, 7))
plt.imshow(delta_distance_matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax_dist, vmax=vmax_dist)
plt.colorbar(label="Δ attention distance")
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(layers, [f"Layer {i}" for i in layers])
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Layer")
plt.title("Attention Distance Change: Layer × Checkpoint")
save_plot("10_distance_delta_heatmap_layers_x_checkpoints.png")


# ============================================================
# 11. PRETRAINED VS FINE-TUNED ENTROPY BY LAYER
# ============================================================

plt.figure(figsize=(11, 7))
plt.plot(layers, reference_pretrained_entropy, marker="s", linestyle="--", linewidth=2, label="Pretrained")
for r in results:
    plt.plot(layers, r["finetuned_entropy"], marker="o", linewidth=1.5, label=r["folder"])
plt.xlabel("Layer")
plt.ylabel("Mean attention entropy")
plt.title("Pretrained vs Fine-tuned Attention Entropy by Layer")
plt.xticks(layers)
plt.legend(fontsize=8, ncol=2)
plt.grid(alpha=0.3)
save_plot("11_entropy_pretrained_vs_finetuned_by_layer.png")


# ============================================================
# 12. PRETRAINED VS FINE-TUNED DISTANCE BY LAYER
# ============================================================

plt.figure(figsize=(11, 7))
plt.plot(layers, reference_pretrained_distance, marker="s", linestyle="--", linewidth=2, label="Pretrained")
for r in results:
    plt.plot(layers, r["finetuned_distance"], marker="o", linewidth=1.5, label=r["folder"])
plt.xlabel("Layer")
plt.ylabel("Mean attention distance")
plt.title("Pretrained vs Fine-tuned Attention Distance by Layer")
plt.xticks(layers)
plt.legend(fontsize=8, ncol=2)
plt.grid(alpha=0.3)
save_plot("12_distance_pretrained_vs_finetuned_by_layer.png")


# ============================================================
# 13. ENTROPY PERCENTAGE CHANGE (Diverging Colormap)
# ============================================================

entropy_pct_matrix = np.array([r["entropy_pct"] for r in results]).T
vmax_pct_ent = np.nanmax(np.abs(entropy_pct_matrix))

plt.figure(figsize=(13, 7))
plt.imshow(entropy_pct_matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax_pct_ent, vmax=vmax_pct_ent)
plt.colorbar(label="Entropy change (%)")
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(layers, [f"Layer {i}" for i in layers])
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Layer")
plt.title("Percentage Change in Attention Entropy")
save_plot("13_entropy_change_percentage.png")


# ============================================================
# 14. DISTANCE PERCENTAGE CHANGE (Diverging Colormap)
# ============================================================

distance_pct_matrix = np.array([r["distance_pct"] for r in results]).T
vmax_pct_dist = np.nanmax(np.abs(distance_pct_matrix))

plt.figure(figsize=(13, 7))
plt.imshow(distance_pct_matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax_pct_dist, vmax=vmax_pct_dist)
plt.colorbar(label="Distance change (%)")
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(layers, [f"Layer {i}" for i in layers])
plt.xlabel("Fine-tuning training samples")
plt.ylabel("Layer")
plt.title("Percentage Change in Attention Distance")
save_plot("14_distance_change_percentage.png")


# ============================================================
# 15. LAYER TRAJECTORY: ENTROPY VS. DISTANCE PHASE SPACE
# ============================================================

plt.figure(figsize=(10, 8))
pre_ent = reference_pretrained_entropy
pre_dist = reference_pretrained_distance
plt.plot(pre_ent, pre_dist, marker="s", linestyle="--", color="black", linewidth=2, label="Pretrained")
for i, (x, y) in enumerate(zip(pre_ent, pre_dist)):
    plt.text(x, y, str(i), fontsize=8, ha="right", va="bottom")

final_r = results[-1]
ft_ent = final_r["finetuned_entropy"]
ft_dist = final_r["finetuned_distance"]
plt.plot(ft_ent, ft_dist, marker="o", linestyle="-", color="red", linewidth=2, label=f"Fine-tuned ({final_r['folder']})")
for i, (x, y) in enumerate(zip(ft_ent, ft_dist)):
    plt.text(x, y, str(i), fontsize=8, ha="left", va="top", color="red")

plt.xlabel("Mean Attention Entropy")
plt.ylabel("Mean Attention Distance")
plt.title("Layer Information Flow: Entropy vs. Distance Trajectory")
plt.legend()
plt.grid(alpha=0.3)
save_plot("15_layer_trajectory_entropy_vs_distance.png")


# ============================================================
# 16. FINAL CHECKPOINT: ENTROPY LAYER X HEAD HEATMAP
# ============================================================

plt.figure(figsize=(12, 6))
plt.imshow(final_r["finetuned_entropy_heads"], aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="Mean attention entropy")
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.title(f"Fine-tuned Entropy by Layer & Head ({final_r['folder']})")
save_plot("16_entropy_layer_head_heatmap.png")


# ============================================================
# 17. FINAL CHECKPOINT: DISTANCE LAYER X HEAD HEATMAP
# ============================================================

plt.figure(figsize=(12, 6))
plt.imshow(final_r["finetuned_distance_heads"], aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="Mean attention distance")
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.title(f"Fine-tuned Distance by Layer & Head ({final_r['folder']})")
save_plot("17_distance_layer_head_heatmap.png")


# ============================================================
# SAVE NUMERICAL SUMMARY
# ============================================================

summary = []
for r in results:
    summary.append({
        "samples": r["samples"],
        "folder": r["folder"],
        "mean_pretrained_entropy": r["mean_pre_entropy"],
        "mean_finetuned_entropy": r["mean_ft_entropy"],
        "mean_pretrained_distance": r["mean_pre_distance"],
        "mean_finetuned_distance": r["mean_ft_distance"],
        "pretrained_entropy_per_layer": r["pretrained_entropy"].tolist(),
        "finetuned_entropy_per_layer": r["finetuned_entropy"].tolist(),
        "pretrained_distance_per_layer": r["pretrained_distance"].tolist(),
        "finetuned_distance_per_layer": r["finetuned_distance"].tolist(),
        "delta_entropy_per_layer": r["delta_entropy"].tolist(),
        "delta_distance_per_layer": r["delta_distance"].tolist(),
        "entropy_change_percent": r["entropy_pct"].tolist(),
        "distance_change_percent": r["distance_pct"].tolist(),
    })

with open(OUTPUT_DIR / "attention_analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("ATTENTION ANALYSIS COMPLETE")
print("=" * 100)
print(f"Number of checkpoints: {len(results)}")
print(f"Number of layers: {expected_num_layers}")
print()

for r in results:
    print(
        f"{r['samples']:>8} samples | "
        f"Entropy: {r['mean_ft_entropy']:.6f} | "
        f"Distance: {r['mean_ft_distance']:.6f}"
    )

print()
print(f"All plots saved to:\n{OUTPUT_DIR}")