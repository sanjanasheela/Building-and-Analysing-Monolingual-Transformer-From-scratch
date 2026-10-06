# Bonus (Optional): Ablation Study — No Positional Embeddings

This directory contains the complete codebase and pipeline for the **Bonus Ablation Study** on **Model H (Telugu)** with **positional embeddings removed**.

---

## 1. Overview and Objective

In this ablation, we investigate the fundamental role of positional encodings in Transformer language modeling:
- **Standard Model H:** Employs Rotary Position Embeddings (RoPE, $\theta = 10,000$), injecting relative positional awareness directly into the Query and Key representations at every attention head.
- **Ablated Model H:** Positional encodings are completely removed. Both RoPE and learned positional embeddings are disabled. Self-attention operates strictly on semantic token representations subject only to the causal autoregressive triangular mask.

The goal is to retrain Model H from scratch without positional information for a minimum of 15,000 steps, execute the **full Phase 2 evaluation suite** on the ablated checkpoint, and compare it against the standard model to document **what breaks without position information**.

---

## 2. Directory Layout

```
language_H/
├── bonus/
│   ├── model.py              # Ablated Transformer LM with NO positional encodings
│   ├── config.yaml           # Training & model configuration for ablation
│   ├── train_ablation.py     # Training orchestrator (AMP, AdamW, cosine LR, checkpoints)
│   ├── eval_ablation.py      # Full Phase 2 evaluation suite (CE/PPL/BPB, generation, attention)
│   ├── run_bonus.sh          # One-click execution script (Kaggle or local)
│   └── README.md             # Detailed documentation and Kaggle instructions
└── report/
    └── bonus/
        ├── ablation_report.md       # Comprehensive scientific ablation report
        ├── tables/
        │   ├── final_evaluation_summary.txt  # Intrinsic metrics (CE, PPL, BPB)
        │   ├── all_temperatures_metrics.txt  # BLEU, chrF, ROUGE-L, diversity across temps
        │   └── comparison_with_baseline.txt  # Side-by-side standard vs ablated comparison
        ├── loss_curves/
        │   ├── loss_comparison.png           # Training and validation loss curves
        │   └── perplexity_comparison.png     # Perplexity curves
        ├── generated_samples/
        │   ├── samples_temp_0.0.txt          # Greedy generated text
        │   ├── samples_temp_0.2.txt          # T=0.2 continuations
        │   ├── samples_temp_0.5.txt          # T=0.5 continuations
        │   ├── samples_temp_0.8.txt          # T=0.8 continuations
        │   ├── samples_temp_1.0.txt          # T=1.0 continuations
        │   ├── samples_temp_1.2.txt          # T=1.2 continuations
        │   └── samples_temp_1.5.txt          # T=1.5 continuations
        └── attention/
            ├── attention_summary_stats.json  # Entropy and mean distance per layer/head
            ├── heatmap_layer_0_head_*.png    # Layer 0 attention heatmaps (all 8 heads)
            └── heatmap_layer_5_head_*.png    # Layer 5 attention heatmaps (all 8 heads)
```

---

## 3. How to Run on Kaggle

### Step 1: Upload Files
Upload your `token_shards_7500` dataset (or whole `language_H` repo) as a Kaggle Dataset.

### Step 2: In a Kaggle Notebook (GPU T4 or P100)
Run the following commands:

```bash
# Set up workspace
cd /kaggle/working
git clone <YOUR_REPO_URL> repo   # or copy from /kaggle/input
cd repo/language_H

# Execute the complete training and evaluation pipeline
bash bonus/run_bonus.sh 15000 /kaggle/input/your-dataset/token_shards_7500
```

Alternatively, invoke Python directly:
```bash
# 1. Train ablated model for 15,000 steps
python bonus/train_ablation.py \
    --config bonus/config.yaml \
    --max_steps 15000 \
    --data_dir /kaggle/input/your-dataset/token_shards_7500 \
    --output_dir bonus/checkpoints

# 2. Run full Phase 2 evaluation suite
python bonus/eval_ablation.py \
    --config bonus/config.yaml \
    --checkpoint bonus/checkpoints/step_0015000.pt \
    --report_dir report/bonus
```

### Step 3: Download Results
The generated artifacts in `report/bonus/` and checkpoints in `bonus/checkpoints/` can be downloaded directly from Kaggle output.

---

## 4. How to Run Locally

Activate your virtual environment and execute:
```bash
source /home/sanjana/Documents/7/LMA/.venv/bin/activate
cd /home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H

# Train and evaluate in one step
bash bonus/run_bonus.sh 15000
```
