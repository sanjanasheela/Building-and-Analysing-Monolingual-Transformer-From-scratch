# Phase 2 Report — Model H (Telugu): Implementation, Pretraining, and Evaluation

**Course:** Language Models and Agents — Monsoon 2026 · Individual Project
**Phase:** 2 of 3 — Model Implementation, Pretraining, and Evaluation [40 Marks]
**Deadline:** 5 September 2026, 11:59 P.M.
**Config used:** `configs/51.yaml` (`run51`) · **Eval checkpoint:** `step_0045000.pt`

> This report covers **Model H (Telugu)** only. Model L results will be appended once that model has been trained and evaluated.

---

## 1. Architecture

Decoder-only (GPT-style) Transformer, implemented from scratch per `configs/51.yaml`.

| Parameter | Value |
|---|---|
| Vocabulary size (BPE, own tokenizer) | 7,500 |
| Model dimension (`d_model`) | 512 |
| Number of layers | 6 |
| Number of attention heads | 8 (head dim = 64) |
| Feed-forward inner dim | 1,600 |
| Positional encoding | **RoPE** (rotary, θ = 10,000) |
| Normalization | **Pre-norm** |
| FFN activation | **SwiGLU** |
| Tied input/output embeddings | Yes |
| Context length | 512 |
| Dropout | 0.1 |
| Weight init std | 0.02 |
| **Total trainable params** | **≈24.88M** (see below) |

**Positional encoding — RoPE:** rotary position embeddings are applied directly to the query/key vectors inside attention (rotating pairs of dimensions by an angle proportional to absolute position, θ = 10,000), rather than adding a separate learned position table to the token embeddings. Because RoPE is a relative scheme computed on the fly, it introduces no additional parameters and in principle generalizes somewhat beyond the training context length, though this model was trained and evaluated at a fixed context length of 512.

**Normalization — pre-norm:** LayerNorm/RMSNorm is applied before each sublayer (attention and FFN) rather than after, so the residual stream itself stays un-normalized end-to-end. This is generally more stable for deeper stacks since gradients can flow through the residual path without repeatedly passing through a normalization non-linearity, at the cost of the final representation needing one extra normalization layer before the output head.

**FFN — SwiGLU:** each feed-forward block uses a gated SwiGLU FFN (a "gate" and an "up" projection combined via SiLU-gating, followed by a "down" projection) instead of a plain two-layer GELU MLP, which is the main reason the FFN accounts for three weight matrices rather than two in the parameter count below.

**Parameter accounting (computed from the config, tied embeddings, RoPE = 0 extra params):**

| Component | Formula | Params |
|---|---|---|

  ── Parameter Breakdown ───────────────────────────────
  Token Embedding                               3,840,000
    Positional Embedding                       (shared)
    LayerNorm 1 (γ,β)  ×6 layers                    6,144
    W_Q  ×6 layers                              1,572,864
    W_K  ×6 layers                              1,572,864
    W_V  ×6 layers                              1,572,864
    W_O  ×6 layers                              1,572,864
    LayerNorm 2 (γ,β)  ×6 layers                    6,144
    FFN W1 (w + b)  ×6 layers                   4,924,800
    FFN W2 (w + b)  ×6 layers                   4,918,272
    FFN W3 (w + b)  ×6 layers                   4,924,800
  Final LayerNorm (γ,β)                             1,024
    LM Head (tied — no extra params)           (shared)

  ──────────────────────────────────────────────────────────
  Embedding params                              3,840,000
  Transformer blocks (excl. embeddings)        21,071,616
  Final LayerNorm                                   1,024
  ──────────────────────────────────────────────────────────
  TOTAL parameters                             24,912,640
  TOTAL (M)                                       24.913M
  Non-embedding parameters                     21,072,640

  Target ≈ 25M params — 24.91M (99.7%)  ✓ within ±20% of 25M target
|

This lands squarely in the ~25M-parameter target for the assignment.
---

## 2. Training Configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1e-8, weight decay=0.1) |
| Learning rate schedule | 8e-4 → 8e-5 (cosine decay) |
| Warmup steps | 3,000 |
| Batch size | 16 (gradient accumulation ×1 → effective batch 16) |
| Gradient clipping | max norm 1.0 |
| Planned max steps | 120,000 |
| Target pretraining tokens | 600,000,000 |
| **Actual steps completed (eval checkpoint)** | **45,000** |
| Checkpoint / eval interval | every 1,000 steps |
| Mixed precision | enabled (CUDA) |
| Seed | 42 |
| Checkpoint contents | model weights, optimizer state, scheduler state, step, config (resume-capable) |


### Loss Curves

![Train/Val/Test loss curves — Model H](/language_H/report/loss_curves/loss_comparison.png)

![Train/Val/Test perplexity curves — Model H](/language_H/report/loss_curves/perplexity_comparison.png)

**Reading the curves:** loss falls sharply from ≈8.6 to ≈4 within the first ~5,000 steps, then decays slowly and flattens from roughly step 20,000 onward. Train and validation loss track closely throughout with only a small, stable gap (val sitting marginally above train), indicating **no significant overfitting** — the model appears more under-trained relative to its capacity/token budget than memorizing. The test loss (3.7361, dashed red line) sits right on the converged train/val band, confirming the held-out test set behaves consistently with validation.

---

## 3. Language Modeling Metrics

| Metric | Model H (Telugu) |
|---|---|
| Test Cross-Entropy Loss | **3.7361** |
| Test Perplexity (PPL) | **41.9356** |
| Bits-per-Byte (BPB) | **0.0152** |

A test PPL of ~42 for a from-scratch, ~24.9M-parameter model trained on ~369M tokens (of a 600M target) for 45k steps is a reasonable result for this compute/data budget — well below the vocabulary-implied random-guess ceiling (7,500) but well short of a fully-resourced pretrained LM, consistent with the training budget actually used. BPB is reported here specifically so it can be compared against Model L on a tokenizer-agnostic basis later, since PPL alone is sensitive to vocabulary size and tokens-per-character.

---

## 4. Generation Quality

Checkpoint used: `step_0045000.pt`. Continuations were generated from 250 held-out prefixes and scored against reference continuations at each decoding setting. The config's `evaluation.temperatures` lists [0.5, 1.0, 1.5], but the actual run swept a wider set: **0.2, 0.5, 0.8, 1.0, 1.2, 1.5**.


### Temperature Sweep
| Temperature | BLEU-4 |  chrF | chrF++ | ROUGE-L | Rep-rate-3 | Distinct-1 | Distinct-2 |
| ----------- | -----: | ----: | -----: | ------: | ---------: | ---------: | ---------: |
| 0.0         | 0.7501 | 17.66 |  15.24 |    7.20 |     0.5509 |     0.2000 |     0.4292 |
| 0.2         | 0.7591 | 18.43 |  15.84 |    7.34 |     0.4218 |     0.2145 |     0.4955 |
| 0.5         | 0.6284 | 19.12 |  16.38 |    7.31 |     0.1908 |     0.2646 |     0.6603 |
| 0.8         | 0.7253 | 18.82 |  15.98 |    6.78 |     0.0438 |     0.3259 |     0.8518 |
| 1.0         | 0.6494 | 18.83 |  15.84 |    6.32 |     0.0066 |     0.3609 |     0.9227 |
| 1.2         | 0.6441 | 18.52 |  15.49 |    6.17 |     0.0023 |     0.3873 |     0.9560 |
| 1.5         | 0.6183 | 18.33 |  15.07 |    5.80 |     0.0011 |     0.4198 |     0.9812 |

****Trend discussion:****

* ****BLEU-4 peaks at T=0.2, while chrF and chrF++ peak at T=0.5; ROUGE-L is also highest at T=0.2**** and generally declines at higher temperatures. This is expected because BLEU, chrF, and ROUGE-L rely on lexical/n-gram overlap with a single reference continuation. As temperature increases, the model samples more varied continuations, which may be valid and fluent but share fewer surface forms with the reference.

* ****Repetition rate decreases sharply as temperature increases****, from **0.5509 at T=0.0** to **0.0011 at T=1.5**, while ****Distinct-1 and Distinct-2 increase almost monotonically****, from **0.2000 → 0.4198** and **0.4292 → 0.9812**, respectively. This shows the classic low-temperature **“safe but repetitive”** versus high-temperature **“more diverse but potentially less reference-aligned”** trade-off.

* With only one reference continuation per prefix, ****BLEU-4, chrF/chrF++, and ROUGE-L are only weakly informative for open-ended LM generation****. A fluent continuation that uses different words or phrasing can receive a low overlap score, even though it is perfectly reasonable. Conversely, high lexical overlap does not necessarily guarantee good generation quality. ****Distinct-n and repetition rate are more informative for measuring diversity and detecting repetitive/degenerate generation****, while qualitative inspection is important for evaluating fluency and coherence.

* ****Overall, the results show a clear diversity–reference-overlap trade-off.**** At lower temperatures, the model produces outputs that are more similar to the reference and consequently obtain higher BLEU/ROUGE scores, but they also exhibit substantially more repetition. At higher temperatures, repetition almost disappears and lexical diversity increases considerably, but BLEU-4 and ROUGE-L decline, suggesting that the generated continuations move further away from the particular reference continuation. Thus, **T≈0.5 provides a reasonable middle ground**, achieving the highest chrF/chrF++ scores while substantially reducing repetition and increasing diversity compared with very low temperatures.

### Generated Samples

Full outputs (250 prefixes × reference vs. generated continuation, all six temperatures) are in [`generated_samples/`](language_H/report/generated_samples); raw console output is in [`generated_samples/evaluation.log`](language_H/report/generated_samples/evaluation.log).

---

## 5. Attention Analysis

Based on `attention/attention_summary_stats.json` (per-layer, per-head mean entropy and mean attention distance) and the 16 heatmaps (layers 0 and 5, all 8 heads each; file naming `heatmap_layer_{L}_head_{H}.png`).

### Heatmaps

![Layer 0, Head 0 — broad, recency-biased attention](/language_H/report/attention/heatmap_layer_0_head_0.png)!
![Layer 5, Head 0 — attention-sink + local diagonal](/language_H/report/attention/heatmap_layer_5_head_6.png)

The causal (lower-triangular) mask is clearly visible in every heatmap — no query attends to a future key position, verified by checking that logits at position *t* are unaffected by changing token *t+1*.

- **Early layer (Layer 0):** heads are diffuse and recency-biased — attention mass decays smoothly from each query back over its preceding tokens (`heatmap_layer_0_head_0.png`), consistent with early layers doing local, position-driven mixing before content-based routing emerges.
- **Late layer (Layer 5):** several heads develop a sharp **attention-sink** pattern, concentrating most mass on the first token (position 0) regardless of query position, with a secondary local diagonal near the query's own recent context (`heatmap_layer_5_head_6.png`) — a well-known emergent behavior where an early, always-available token position acts as a default/no-op attention target.

### Entropy per Layer

| Layer | Mean entropy across heads | Range (min–max) |
|---|---|---|
| 0 | 3.228 | 3.14 – 3.29 |
| 1 | 1.703 | 0.64 – 2.84 |
| 2 | 1.805 | 0.99 – 2.89 |
| 3 | 1.737 | 0.43 – 2.62 |
| 4 | 1.548 | 0.70 – 2.02 |
| 5 | 1.255 | 0.53 – 2.25 |

Full per-head values: [`attention/attention_summary_stats.json`](attention/attention_summary_stats.json). Entropy is **highest and most uniform across heads in Layer 0** (all 8 heads within a narrow 3.14–3.29 band — near-maximum-entropy, diffuse attention) and **drops and diverges from Layer 1 onward**, with some heads specializing into sharply peaked, content-based attention while others stay diffuse — typical of early layers doing generic local mixing and later layers differentiating into specialized heads.

### Mean Attention Distance per Layer

| Layer | Mean distance across heads | Range (min–max) |
|---|---|---|
| 0 | 5.650 | 5.39 – 5.86 |
| 1 | 3.157 | 1.39 – 7.41 |
| 2 | 4.103 | 1.72 – 9.08 |
| 3 | 5.891 | 3.15 – 10.54 |
| 4 | 5.142 | 1.87 – 7.80 |
| 5 | 7.731 | 3.70 – 10.72 |

**Most local heads:** Layer 1 / Head 6 (distance 1.39, entropy 0.78) and Layer 1 / Head 2 (distance 1.48, entropy 0.64) — sharply peaked, near-neighbor attention, functioning like a learned local window.

**Most long-range / content-based heads:** Layer 5 / Head 6 (distance 10.72, entropy 0.53) and Layer 3 / Head 4 (distance 10.54, entropy 0.43) — both combine *low entropy* (peaked, confident attention) with *high distance*, matching the attention-sink pattern from the heatmaps: these heads reach far back (often to position 0) with high confidence rather than attending broadly and diffusely.

### Discussion

Model H shows a clear depth-wise progression: **Layer 0 behaves almost like a uniform local smoothing filter** (high, uniform entropy and moderate, uniform distance across all 8 heads), while **deeper layers — 3 and 5 especially — differentiate into a mix of tight local heads and long-range, low-entropy "sink" heads**. This suggests some division of labor between position-tracking and content-routing heads has emerged even at ~24.9M parameters and 45k/120k planned steps, though with only 6 layers × 8 heads there isn't a large population of heads to draw strong general conclusions from — a richer head-specialization story would likely need more layers/heads or the remaining training budget (the model was stopped at ~61% of its token target).

---


