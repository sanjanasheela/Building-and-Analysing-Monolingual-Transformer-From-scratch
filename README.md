[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Language Models and Agents (Monsoon 2026) — Individual Project
## Building and Analyzing Monolingual Transformer Language Models from Scratch

---

## 📌 Project Overview

This repository contains the complete implementation, training, evaluation, reasoning finetuning, and attention analysis pipelines for two independent decoder-only Transformer language models trained from scratch without using any pretrained models or HuggingFace Transformer abstractions:

- **Model H (Telugu — Higher-Resource):** ~24.91M parameter decoder-only Transformer trained on a curated monolingual Telugu corpus with a 7,500-token BPE vocabulary.
- **Model L (Nepali — Lower-Resource):** ~24.91M parameter decoder-only Transformer trained on a curated monolingual Nepali corpus with a 7,500-token BPE vocabulary.
- **Bonus Ablation Study (Model H):** Ablation study evaluating the role of positional encodings by training Model H with RoPE completely removed for 15,000 steps.

The two language pipelines are completely isolated: they share no data, tokenizers, vocabularies, or weights.

---

## 🔗 Data & Artifacts (Google Drive Links)

### 1. Full Corpora & Datasets
- **Telugu Full Corpus (`language_H`):** [Google Drive File Link](https://drive.google.com/file/d/1qIT_xnEBOkDdZapH7tiLZW4GE5WcQGu8/view?usp=drive_link)
- **Nepali Full Corpus (`language_L`):** [Google Drive File Link](https://drive.google.com/file/d/1VH5zZKzLQXL1KLpgn8Sy0AuJTfPfr4tp/view?usp=drive_link)
- **Master Datasets & Models Drive Folder:** [Google Drive Folder](https://drive.google.com/drive/folders/1-P_5jNOofC_cF2f5wItIaU66uUTqLd-d?usp=sharing)

### 2. Pretrained Model Checkpoints (Phase 2)
- **Language H (Telugu) Pretrained Checkpoints Folder:** [Google Drive Folder](https://drive.google.com/drive/folders/1jbtt_cHGJAI3t2bL5MidkBjtgx9cwPWI?usp=drive_link)
- **Language L (Nepali) Pretrained Checkpoints Folder:** [Google Drive Folder](https://drive.google.com/drive/folders/1PPopn6C050OliKhHqIHMjD3pm_SGJkTy?usp=drive_link)
- **Phase 2 Combined Pretrained Checkpoints & Logs:** [Google Drive Folder](https://drive.google.com/drive/folders/1CgoMXlmRY12bmKOCc0iN1pyY7-csapkD?usp=sharing)

### 3. Reasoning Finetuned & Ablation Checkpoints (Phase 3 & Bonus)
- **Language H (Telugu) Finetuned Checkpoints:** [Google Drive File Link](https://drive.google.com/file/d/1FKmrsF7LijnI_ZykgIeWhKO6NtWIhMfK/view?usp=drive_link)
- **Language L (Nepali) Finetuned Checkpoints:** [Google Drive File Link](https://drive.google.com/file/d/1VmkJHiQaGyiqdkZb5XjJY-8mC_ynM3oR/view?usp=drive_link)
- **Bonus Ablation Study Checkpoints (No Position Embeddings):** [Google Drive File Link](https://drive.google.com/file/d/1j-y0E-GfDzxXGjKAxRjaw90j_37KwQuS/view?usp=drive_link)

### Reports & Scientific Documentation
- **Consolidated Phase 2 Report:** [`finalphase_2_report.md`](./finalphase_2_report.md)
- **Telugu Tokenizer & Corpus Report (`language_H`):** [`language_H/telugu_corpus_tokenizer_report.md`](./language_H/telugu_corpus_tokenizer_report.md)
- **Nepali Tokenizer & Corpus Report (`language_L`):** [`language_L/nepali_corpus_tokenizer_report.md`](./language_L/nepali_corpus_tokenizer_report.md)
- **Bonus Ablation Study Report (No Position Embeddings):** [`language_H/report/bonus/ablation_report.md`](./language_H/report/bonus/ablation_report.md)

---

## 🏗️ Architecture Specifications

Both models share a matching compute-optimal decoder-only Transformer architecture built with raw PyTorch components (`nn.Linear`, `nn.Embedding`, manual causal self-attention):

| Hyperparameter / Feature | Model H (Telugu) | Model L (Nepali) |
| :--- | :--- | :--- |
| **Model Type** | Decoder-only Transformer | Decoder-only Transformer |
| **Trainable Parameters** | **24,912,640 (~24.91M)** | **24,912,640 (~24.91M)** |
| **Vocabulary Size** | 7,500 BPE | 7,500 BPE |
| **Model Dimension (`d_model`)** | 512 | 512 |
| **Layers (`num_layers`)** | 6 | 6 |
| **Attention Heads (`num_heads`)** | 8 | 8 |
| **Head Dimension (`d_k`)** | 64 | 64 |
| **FFN Inner Dimension (`ffn_dim`)** | 1,600 | 1,600 |
| **Activation Function** | SwiGLU | SwiGLU |
| **Positional Encoding** | RoPE ($\theta = 10,000$) | RoPE ($\theta = 10,000$) |
| **Normalization Scheme** | Pre-Norm (LayerNorm) | Pre-Norm (LayerNorm) |
| **Context Length** | 512 tokens | 512 tokens |
| **Embedding Weight Tying** | Yes (Input tied to LM Head) | Yes (Input tied to LM Head) |
| **Dropout** | 0.1 | 0.1 |

---

## 🏆 Best Finetuned Checkpoints (Empirical Model Selection)

Based on comprehensive evaluation across all 14 finetuned checkpoints per language (evaluating validation & held-out test exact-match accuracy, token NLL loss, perplexity, hop-wise breakdown, and attention localization):

| Language Model | Recommended Best Checkpoint | Training Samples | Test Exact Match | Test / Val NLL | Test / Val PPL | Selection Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Model H (Telugu)** | **`finetuned_train_samples_8000.pt`** (`finetuned_checkpoints_8k`) | 8,000 | **68.33%** | **0.7446** | **2.11** | Peak test exact match (tied with 10k), but achieves significantly better loss (0.74 vs 0.98), lower PPL (2.11 vs 2.67), and the highest multi-hop reasoning accuracy (**51.7%**; 55.0% on 2-hop, 57.5% on 3-hop) without over-memorization. |
| *Model H Efficiency Pick* | `finetuned_train_samples_2000.pt` (`finetuned_checkpoints_2k`) | 2,000 | 63.75% | 0.6809 | 1.98 | Reaches 93.3% of peak reasoning performance using only 25% of training samples. |
| **Model L (Nepali)** | **`finetuned_train_samples_2500.pt`** (`finetuned_checkpoints_2.5k`) | 2,500 | **84.17%** | **0.4917** | **1.64** | **Highest overall reasoning accuracy across both languages** (202/240 correct), with the lowest loss (0.4917) and lowest perplexity (1.64). Generalizes better than 10k (which dropped to 77.92% due to template over-fitting). |
| *Model L Runner-Up* | `finetuned_train_samples_3000.pt` (`finetuned_checkpoints_3k`) | 3,000 | 83.33% | 0.5396 | 1.72 | Peak 1-hop accuracy (**94.2%**) and 100% accuracy on direct comparisons. |

### Key Empirical Findings:
1. **Model L (Nepali) Outperforms Model H (Telugu) in Reasoning:** Model L reaches **84.17%** test EM compared to Model H's **68.33%** test EM. This stems from Devanagari's cleaner tokenization regularity on entity names and numerical strings under the 7,500 BPE vocabulary.
2. **Optimal Data Regimes (Avoiding Overfitting):** In both languages, models achieve compute-optimal reasoning convergence around **2,500 – 8,000 samples**. Scaling to 10,000 samples causes training loss to drop to near zero ($< 0.0003$) while validation/test loss begins to diverge, signaling template memorization.

---

## 📂 Repository Structure

```
individual-project-sanjanasheela/
├── README.md                           # Main documentation & Reproducibility Guide
├── finalphase_2_report.md              # Consolidated Phase 2 evaluation report
├── language_H/                         # Model H (Telugu Pipeline)
│   ├── configs/                        # Pretraining & finetuning configs (51.yaml, finetune.yaml)
│   ├── preprocessing/                  # Deduplication, filtering, and normalization scripts
│   ├── tokenizer/                      # BPE training, token counting, vocabulary sweeps
│   ├── tokenizer_runs/                 # Saved tokenizers (bpe_vocab_7500, etc.)
│   ├── train/                          # Model architecture, dataset loaders, pretrainer
│   ├── eval/                           # LM evaluation (PPL/BPB), generation metrics, attention analysis
│   ├── finetuning/                     # Phase 3 reasoning dataset generation, finetuning, scaling
│   ├── bonus/                          # Bonus ablation study (No Positional Embeddings)
│   ├── checkpoints/                    # Pretrained model checkpoints (step_0045000.pt)
│   ├── finetuned_checkpoints/          # Scaling finetuned checkpoints (100 -> 10,000 samples)
│   └── report/                         # Plots, tables, samples, and attention heatmaps
├── language_L/                         # Model L (Nepali Pipeline)
│   ├── configs/                        # Pretraining & finetuning configs (51.yaml, finetune.yaml)
│   ├── preprocessing/                  # Deduplication, normalization, and quality filters
│   ├── tokenizer/                      # BPE tokenizer training and evaluation sweeps
│   ├── tokenizer_runs/                 # Saved tokenizers (bpe_vocab_7500, etc.)
│   ├── train/                          # Model architecture, dataset loaders, pretrainer
│   ├── eval/                           # Intrinsic metrics, generation benchmarks, attention plots
│   ├── finetuning/                     # Phase 3 reasoning finetuning and evaluation
│   ├── checkpoints/                    # Pretrained model checkpoints (step_0045000.pt)
│   ├── finetuned_checkpoints/          # Scaling finetuned checkpoints
│   └── report/                         # Phase 1, 2, and 3 report artifacts
└── report/                             # Consolidated cross-model comparison plots
```

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Linux OS (Ubuntu 20.04+ or Debian recommended)
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with CUDA support (tested on RTX 2050 / T4 / A100)

### 2. Activate or Create Virtual Environment
The repository is configured to use the local virtual environment directly:

```bash
# Option A: Activate the repository's local virtual environment directly:
source .venv/bin/activate

# Option B: Create a fresh virtual environment and install dependencies:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Indic Fonts (Required for Heatmaps)
To ensure Telugu and Devanagari (Nepali) scripts render properly in attention heatmap labels:
```bash
sudo apt-get update
sudo apt-get install -y fonts-noto-core fonts-noto-extra fonts-noto-ui-core
# Or install specifically:
sudo apt-get install -y fonts-noto-telu fonts-noto-deva
```

---

## 🚀 Reproducibility Guide: Step-by-Step Pipeline

---

### Phase 1: Corpus Preprocessing & Tokenizer Construction

Each language pipeline starts with raw text cleaning, Unicode normalization, language identification filtering, deduplication, and BPE tokenizer training.

#### For Telugu (`language_H`):
```bash
cd language_H

# 1. Run corpus cleaning & deduplication
python /preprocessing/cleaning_pipeline_with_nearduplicates.py

# 2. Train the selected 7,500-vocab BPE tokenizer from scratch
python tokenizer/train_tokenizer.py \
    --train data/final_data_set_7500/train.txt \
    --val data/final_data_set_7500/val.txt \
    --test data/final_data_set_7500/test.txt

# 3. Evaluate tokenizer fertility, character-per-token, and UNK rate sweeps
python tokenizer/plot.py

# 4. Tokenize clean text into binary uint32 shards for compute-optimal training
python train/txt_to_bin.py
```

#### For Nepali (`language_L`):
```bash
cd language_L

# 1. Run corpus cleaning & deduplication
python preprocessing/cleaning_pipeline_with_neardup.py

# 2. Train the BPE tokenizer from scratch
python tokenizer/train_tokenizer.py

# 3. Evaluate tokenizer sweeps
python tokenizer/plot.py

# 4. Convert tokenized text into binary shards
python train/txt_to_bin.py
```

---

### Phase 2: Pretraining & Full LM Evaluation

Pretrain each monolingual model independently using raw PyTorch, AdamW with cosine learning rate schedule, and resume-capable checkpointing.

#### 1. Pretraining from Scratch / Resuming
```bash
# Telugu (Model H):
cd language_H
python train/train.py --config configs/51.yaml

# To resume an interrupted training run:
python train/train.py --config configs/51.yaml --resume checkpoints/checkpoints/step_0040000.pt

# Nepali (Model L):
cd language_L
python train/train.py --config configs/51.yaml
```

#### 2. Intrinsic Evaluation (Cross-Entropy, Perplexity, Bits-Per-Byte)
```bash
# Evaluates loss, PPL, and BPB on held-out test token shards
python eval/lm_eval.py --config configs/51.yaml --checkpoint checkpoints/checkpoints/step_0045000.pt
```

#### 3. Generation Quality & Diversity Benchmarks
Generates continuations across temperatures $T \in \{0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5\}$ and calculates corpus BLEU-4, chrF++, ROUGE-L, repetition rate, and Distinct-1/2:
```bash
python eval/gm.py --config configs/51.yaml --checkpoint checkpoints/checkpoints/step_0045000.pt
```

#### 4. Attention Analysis (Pretrained)
Extracts post-softmax attention weights, calculates Shannon attention entropy and mean attention distance, and plots heatmaps across all 6 layers and 8 heads:
```bash
python eval/attention_analysis.py
# Results saved to: report/attention/
```

#### 5. Empirical Causal Masking Validation
Validates that modifying token $t+1$ does not perturb logits at position $t$:
```bash
python eval/causal_masking.py
```

---

### Phase 3: Reasoning Finetuning & Multi-Checkpoint Scaling

Phase 3 evaluates how reasoning ability scales as finetuning data increases from 100 to 10,000 samples, and tracks attention pattern changes.

#### 1. Synthetic Reasoning Data Generation
Generates programmatic multi-hop transitive and numerical comparison datasets with strict entity and relation holdouts:
```bash
cd language_H
# Generates train/val/test splits and data scaling slices (100 to 10,000)
python finetuning/data_splitter.py
```

#### 2. Data-Size Scaling Finetuning Loop
Finetunes the pretrained 45k-step checkpoint across all 14 sample sizes (`100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 8000, 10000`):
```bash
python finetuning/main.py --scaling
# Checkpoints saved to: finetuned_checkpoints/
# Scaling curves saved to: report/finetuning/finetuning_scaling_analysis.png
# Metrics JSON saved to: report/finetuning/scaling_results.json
```

#### 3. Evaluating Reasoning Exact Match Accuracy
```bash
# Evaluate a single checkpoint on held-out reasoning test set
python finetuning/evaluate.py \
    --config configs/finetune.yaml \
    --checkpoint finetuned_checkpoints/finetuned_train_samples_10000.pt \
    --eval-file finetuning/data/test_fixed.jsonl
```

#### 4. Post-Finetune Attention Comparison Loop (All Checkpoints)
To analyze how attention distributions evolve across data sizes, run `attention_compare.py`. It loops over **all checkpoints** in `finetuned_checkpoints/` and outputs dedicated subfolders for each:
```bash
python finetuning/attention_compare.py
```

##### What this script generates:
- **Individual Checkpoint Subfolders:**
  - `report/attention_finetune/finetuned_checkpoints_100/`
  - `report/attention_finetune/finetuned_checkpoints_250/`
  - `report/attention_finetune/finetuned_checkpoints_500/`
  - `report/attention_finetune/finetuned_checkpoints_1k/`
  - `report/attention_finetune/finetuned_checkpoints_1.5k/`
  - `report/attention_finetune/finetuned_checkpoints_2k/`
  - `report/attention_finetune/finetuned_checkpoints_2.5k/`
  - `report/attention_finetune/finetuned_checkpoints_3k/`
  - `report/attention_finetune/finetuned_checkpoints_4k/`
  - `report/attention_finetune/finetuned_checkpoints_5k/`
  - `report/attention_finetune/finetuned_checkpoints_6k/`
  - `report/attention_finetune/finetuned_checkpoints_7.5k/`
  - `report/attention_finetune/finetuned_checkpoints_8k/`
  - `report/attention_finetune/finetuned_checkpoints_10k/`
- **Inside each folder:**
  - `compare_layer_*_head_*.png`: Side-by-side heatmaps comparing Pretrained vs Finetuned.
  - `finetuned_layer_*_head_*.png`: Standalone finetuned attention heatmap.
  - `entropy_distance_pretrained_vs_finetuned.png`: Layer-wise attention entropy and token distance curves.
  - `pretrained_vs_finetuned_attention.json`: Quantitative entropy, distance, and delta statistics.
  - `prompt.json`: Prompt text and Telugu reasoning metadata.
- **Aggregate Scaling Outputs:**
  - `report/attention_finetune/scaling_attention_entropy_distance.png`: Cross-checkpoint scaling curve of entropy and distance vs training data size.
  - `report/attention_finetune/all_checkpoints_attention_summary.json`: Multi-checkpoint consolidated metrics.

##### Optional CLI Arguments for `attention_compare.py`:
```bash
# Run for a single specific checkpoint:
python finetuning/attention_compare.py --checkpoint finetuned_checkpoints/finetuned_train_samples_2000.pt

# Plot heatmaps for all 6 layers instead of first and last:
python finetuning/attention_compare.py --all-layers

# Custom attention heads to visualize:
python finetuning/attention_compare.py --heads 0 1 2 3 4 5 6 7
```

---

### Bonus (Optional): Positional Embeddings Ablation Study

An ablation study on Model H (Telugu) training a Transformer from scratch with positional encodings (RoPE) completely removed:

```bash
cd language_H

# Option A: One-click training and full Phase 2 evaluation pipeline (15,000 steps)
bash bonus/run_bonus.sh 15000

# Option B: Run training and evaluation directly via Python:
# 1. Train ablated model
python bonus/train_ablation.py \
    --config bonus/config.yaml \
    --max_steps 15000 \
    --output_dir bonus/checkpoints

# 2. Run full evaluation suite on the ablated checkpoint
python bonus/eval_ablation.py \
    --config bonus/config.yaml \
    --checkpoint bonus/checkpoints/step_0015000.pt \
    --report_dir report/bonus
```

All ablation reports, comparative loss curves, generation samples, and attention heatmaps are available in [`language_H/report/bonus/ablation_report.md`](./language_H/report/bonus/ablation_report.md).

---

## ⚡ Quick Reproduction Cheat Sheet (TL;DR)

For graders and reviewers seeking to verify results immediately:

```bash
# 1. Activate Environment
source .venv/bin/activate

# 2. Evaluate Model H (Telugu) Pretrained LM Performance
cd language_H
python eval/lm_eval.py --config configs/51.yaml --checkpoint checkpoints/checkpoints/step_0045000.pt

# 3. Evaluate Model H Generation (BLEU, chrF, ROUGE-L)
python eval/gm.py --config configs/51.yaml --checkpoint checkpoints/checkpoints/step_0045000.pt

# 4. Run Reasoning Finetuning Evaluation (10k Checkpoint)
python finetuning/evaluate.py --config configs/finetune.yaml --checkpoint finetuned_checkpoints/finetuned_train_samples_10000.pt

# 5. Run Full Attention Comparison Across All 14 Checkpoints
python finetuning/attention_compare.py

# 6. Evaluate Model L (Nepali) Pretrained LM Performance
cd ../language_L
python eval/lm_eval.py --config configs/51.yaml --checkpoint checkpoints/checkpoints/step_0045000.pt

# 7. Evaluate Model L Generation
python eval/gm.py --config configs/51.yaml --checkpoint checkpoints/checkpoints/step_0045000.pt
```

---

## 📄 License & Course Attribution
- **Course:** Language Models and Agents (CL3-410) — Monsoon 2026
- **Assignment:** Individual Project (Phases 1, 2, 3 & Bonus)
- **Author:** Sanjana Sheela