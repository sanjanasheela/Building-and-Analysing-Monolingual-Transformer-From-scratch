import json
import os
import matplotlib.pyplot as plt
import pandas as pd

# Target directory path
output_dir = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/tokenizer_runs/configs"
os.makedirs(output_dir, exist_ok=True)

# 1. Load data directly from your JSON file (assuming the file is named tokenizer_stats.json)
# Alternatively, you can paste your JSON dictionary directly here.
with open("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/tokenizer_runs/all_models_report.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Extract metrics into a pandas DataFrame
rows = []
for key, val in data.items():
    vocab_size = val["vocab_size_stats"]["actual_vocab_size"]
    coverage = val["token_frequency_stats"]["vocab_coverage_pct"]
    mean_freq = val["token_frequency_stats"]["mean_frequency"]
    median_freq = val["token_frequency_stats"]["median_frequency"]
    avg_chars = val["avg_chars_per_token"]["avg_chars_per_token"]
    total_tokens = val["unk_token_stats"]["total_tokens"]
    fertility = val["unk_token_stats"]["fertility"]
    
    rows.append({
        "Model": key,
        "Vocab Size": vocab_size,
        "Coverage (%)": coverage,
        "Mean Frequency": mean_freq,
        "Median Frequency": median_freq,
        "Avg Chars per Token": avg_chars,
        "Total Tokens": total_tokens,
        "Fertility": fertility
    })

df = pd.DataFrame(rows)

# 3. Plot 1: Vocabulary Size vs. Vocabulary Coverage (%)
plt.figure(figsize=(8, 5))
plt.plot(df["Vocab Size"], df["Coverage (%)"], marker='o', color='blue', linewidth=2, markersize=8)
plt.title("Vocabulary Size vs. Vocabulary Coverage (%)", fontsize=14, fontweight='bold')
plt.xlabel("Vocabulary Size", fontsize=12)
plt.ylabel("Vocabulary Coverage (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "coverage_vs_vocab.png"), dpi=300)
plt.close()

# 4. Plot 2: Vocabulary Size vs. Average Characters per Token
plt.figure(figsize=(8, 5))
plt.plot(df["Vocab Size"], df["Avg Chars per Token"], marker='s', color='green', linewidth=2, markersize=8)
plt.title("Vocabulary Size vs. Average Characters per Token", fontsize=14, fontweight='bold')
plt.xlabel("Vocabulary Size", fontsize=12)
plt.ylabel("Average Characters per Token", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "chars_vs_vocab.png"), dpi=300)
plt.close()

# 5. Plot 3: Vocabulary Size vs. Token Fertility & Total Token Count
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel('Vocabulary Size', fontsize=12)
ax1.set_ylabel('Token Fertility', color=color, fontsize=12)
ax1.plot(df["Vocab Size"], df["Fertility"], marker='o', color=color, linewidth=2, label='Fertility')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:purple'
ax2.set_ylabel('Total Tokens', color=color, fontsize=12)
ax2.plot(df["Vocab Size"], df["Total Tokens"], marker='^', color=color, linestyle='--', linewidth=2, label='Total Tokens')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Vocabulary Size vs. Token Fertility & Total Token Count", fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, "fertility_tokens_vs_vocab.png"), dpi=300)
plt.close()

# 6. Plot 4: Token Frequencies (Mean & Median)
plt.figure(figsize=(8, 5))
plt.plot(df["Vocab Size"], df["Mean Frequency"], marker='o', color='tab:orange', label='Mean Frequency', linewidth=2)
plt.plot(df["Vocab Size"], df["Median Frequency"], marker='x', color='tab:brown', label='Median Frequency', linewidth=2)
plt.title("Vocabulary Size vs. Token Frequencies", fontsize=14, fontweight='bold')
plt.xlabel("Vocabulary Size", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "frequencies_vs_vocab.png"), dpi=300)
plt.close()

print(f"All plots have been successfully saved to: {output_dir}")