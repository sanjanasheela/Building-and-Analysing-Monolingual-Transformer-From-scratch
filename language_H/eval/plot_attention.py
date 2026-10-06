#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION & DATA INPUT
# ============================================================

# Path to your JSON file (update this if necessary)
JSON_FILE_PATH = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/report/attention/attention_summary_stats.json")

# Output directory for the generated plots
OUTPUT_DIR = Path("./single_file_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the JSON data
if JSON_FILE_PATH.exists():
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    # If you are pasting the JSON directly for testing, you can also inject it here.
    raise FileNotFoundError(f"Could not find {JSON_FILE_PATH}. Please verify the path.")

# Extract head-level matrices
entropy_heads = np.array(data["entropy_layer_head_mean"], dtype=float)     # Shape: (num_layers, num_heads)
distance_heads = np.array(data["mean_distance_layer_head_mean"], dtype=float) # Shape: (num_layers, num_heads)

num_layers, num_heads = entropy_heads.shape
layers = np.arange(num_layers)
heads = np.arange(num_heads)

# Compute layer-level averages from the head matrices
entropy_per_layer = np.mean(entropy_heads, axis=1)
distance_per_layer = np.mean(distance_heads, axis=1)

print(f"Loaded successfully! Layers: {num_layers}, Heads per layer: {num_heads}")


# ============================================================
# PLOT HELPERS
# ============================================================

def save_plot(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 1. LAYER-WISE MEAN ENTROPY & DISTANCE
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(layers, entropy_per_layer, marker="o", color="b", linewidth=2)
ax1.set_xlabel("Layer")
ax1.set_ylabel("Mean Entropy")
ax1.set_title("Mean Attention Entropy per Layer")
ax1.set_xticks(layers)
ax1.grid(alpha=0.3)

ax2.plot(layers, distance_per_layer, marker="s", color="r", linewidth=2)
ax2.set_xlabel("Layer")
ax2.set_ylabel("Mean Distance")
ax2.set_title("Mean Attention Distance per Layer")
ax2.set_xticks(layers)
ax2.grid(alpha=0.3)

save_plot("01_layer_wise_averages.png")


# ============================================================
# 2. ENTROPY LAYER x HEAD HEATMAP
# ============================================================

plt.figure(figsize=(10, 6))
plt.imshow(entropy_heads, aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="Mean attention entropy")
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.title("Attention Entropy Heatmap (Layer × Head)")
plt.xticks(heads)
plt.yticks(layers, [f"Layer {i}" for i in layers])
save_plot("02_entropy_layer_head_heatmap.png")


# ============================================================
# 3. DISTANCE LAYER x HEAD HEATMAP
# ============================================================

plt.figure(figsize=(10, 6))
plt.imshow(distance_heads, aspect="auto", interpolation="nearest", cmap="plasma")
plt.colorbar(label="Mean attention distance")
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.title("Attention Distance Heatmap (Layer × Head)")
plt.xticks(heads)
plt.yticks(layers, [f"Layer {i}" for i in layers])
save_plot("03_distance_layer_head_heatmap.png")


# ============================================================
# 4. PHASE SPACE TRAJECTORY (ENTROPY VS. DISTANCE BY LAYER)
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(entropy_per_layer, distance_per_layer, marker="o", linestyle="-", color="purple", linewidth=2)

for i, (x, y) in enumerate(zip(entropy_per_layer, distance_per_layer)):
    plt.text(x, y, f" L{i}", fontsize=9, ha="left", va="bottom", color="black")

plt.xlabel("Mean Attention Entropy")
plt.ylabel("Mean Attention Distance")
plt.title("Layer Information Flow: Entropy vs. Distance Trajectory")
plt.grid(alpha=0.3)
save_plot("04_layer_trajectory_entropy_vs_distance.png")

print(f"\nAll plots generated successfully in: {OUTPUT_DIR.resolve()}")