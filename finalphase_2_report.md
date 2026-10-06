# Phase 2 Report — Language Models and Agents

**Course:** Language Models and Agents — Monsoon 2026 · Individual Project  
**Phase:** 2 of 3 — Model Implementation, Pretraining, and Evaluation [40 Marks]  
**Models:** Model H (Telugu) and Model L (Nepali)  
**Evaluation checkpoint:** `step_0045000.pt` is the matched checkpoint used for the main side-by-side comparison.  
**Evaluation note:** All reported Model H and Model L results correspond to the 45,000-step checkpoint.

---

## 1. Overview

Phase 2 evaluates two decoder-only Transformer language models trained for different languages:

- **Model H:** Telugu
- **Model L:** Nepali

Both models use a GPT-style decoder-only architecture with 6 layers and 8 attention heads. Model H is documented as a ~24.9M-parameter model trained from scratch using a 7,500-token BPE vocabulary. Model L uses the same Transformer configuration as Model H.

The evaluation covers intrinsic language-modeling performance, generation quality, repetition/diversity diagnostics, and the behavior of the custom multi-head attention implementation.

---

# 2. Model Architecture and Implementation

## 2.1 Model H — Telugu

Model H is a decoder-only (GPT-style) Transformer implemented from scratch.

| Parameter | Model H |
|---|---:|
| Vocabulary size | 7,500 BPE tokens |
| Model dimension (`d_model`) | 512 |
| Number of layers | 6 |
| Number of attention heads | 8 |
| Head dimension | 64 |
| Feed-forward inner dimension | 1,600 |
| Positional encoding | RoPE, θ = 10,000 |
| Normalization | Pre-norm |
| FFN activation | SwiGLU |
| Tied input/output embeddings | Yes |
| Context length | 512 |
| Dropout | 0.1 |
| Weight initialization std. | 0.02 |
| Total trainable parameters | **24,912,640 (~24.913M)** |

RoPE is applied directly to the query/key vectors inside attention. It introduces no additional learned positional parameters. The model uses pre-norm blocks, with normalization before the attention and FFN sublayers. The FFN uses a gated SwiGLU formulation.

### Parameter accounting

| Component | Parameters |
|---|---:|
| Token embeddings | 3,840,000 |
| Transformer blocks | 21,071,616 |
| Final LayerNorm | 1,024 |
| LM head | Tied with input embeddings |
| **Total** | **24,912,640** |
| **Total (M)** | **24.913M** |
| Non-embedding parameters | 21,072,640 |

This is within the approximately 25M-parameter target specified for the model.

---

## 2.2 Model L — Nepali

Model L uses the same architecture and configuration described in Section 2.1. No separate configuration table is repeated here.

# 3. Training Configuration

## 3.1 Model H Training Configuration

| Setting | Model H |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1e-8, weight decay=0.1) |
| Learning-rate schedule | 8e-4 → 8e-5 cosine decay |
| Warmup steps | 3,000 |
| Batch size | 16 |
| Gradient accumulation | ×1 |
| Effective batch size | 16 |
| Gradient clipping | Max norm 1.0 |
| Planned maximum steps | 120,000 |
| Target pretraining tokens | 600,000,000 |
| Actual steps at evaluation checkpoint | **45,000** |
| Checkpoint/evaluation interval | Every 1,000 steps |
| Mixed precision | CUDA enabled |
| Seed | 42 |
| Checkpoint contents | Model weights, optimizer state, scheduler state, step, config |

### Training behavior

The Model H loss curves show a sharp reduction in loss from approximately 8.6 to approximately 4 during the first 5,000 steps, followed by slower improvement and flattening from around step 20,000 onward.

Train and validation loss remain close, with validation only marginally above training. This indicates no significant overfitting. The report instead suggests that the model is relatively under-trained compared with its capacity and planned token budget.

The test loss of 3.7361 is also close to the converged train/validation region.

![Model H train/validation/test loss curves](/language_H/report/loss_curves/loss_comparison.png)

![Model H train/validation/test perplexity curves](/language_H/report/loss_curves/perplexity_comparison.png)

The Model L learning rate was still approximately 0.001 at step 45,000, suggesting a longer warmup/decay horizon than Model H. However, the source notes that this should be confirmed from Model L's actual configuration.

![Model L train/validation/test loss curves](/language_L/report/loss_curves/loss_comparison.png)

![Model L train/validation/test perplexity curves](/language_L/report/loss_curves/perplexity_comparison.png)


---

# 4. Intrinsic Language-Modeling Evaluation

The assignment requires validation cross-entropy, perplexity, and BPB on held-out text. The available reports contain the following results.

## 4.1 Side-by-side Language Modeling Metrics

| Metric | Model H — Telugu | Model L — Nepali |
|---|---:|---:|
| Cross-Entropy Loss | **3.7361** (test) | **3.9761** (test) |
| Perplexity | **41.9356** | **53.3080** |
| BPB | **0.0152** | **0.0162** |

All reported results are treated as belonging to the **45,000-step checkpoint**, as specified for this final comparison. Model H's reported test loss is 3.7361, while Model L's reported test loss is 3.9761. The corresponding validation values available for Model L are 3.9298 loss and 50.8964 PPL.



Model H achieves lower reported test loss and perplexity than Model L at the
same checkpoint. This indicates that Model H assigns higher probability to
the held-out Telugu text under the reported evaluation.

Perplexity should not, however, be interpreted directly as a measure of
intrinsic language difficulty because the tokenization scheme affects the
number and type of prediction units. BPB is therefore useful as a more
comparable measure across languages.

---

## 4.2 Interpretation of H vs. L

At the available test evaluation, Model H has lower loss and perplexity than Model L:

- Model H PPL: **41.94**
- Model L PPL: **53.31**

This suggests stronger predictive performance for Model H under the reported evaluation conditions. Possible contributing factors include differences in language-resource availability, training-data quantity and quality, script characteristics, and tokenizer fertility.

However, the available reports do **not** provide enough evidence to attribute the gap to any one factor. In particular, the Model L checkpoint-provenance issue means that the comparison should be described as indicative rather than definitive.

BPB is especially useful for cross-language comparison because perplexity depends on the tokenization scheme. BPB provides a more tokenizer-agnostic measure by evaluating information per byte.

---

# 5. Generation Quality Evaluation

Continuations were generated from held-out prefixes and compared against reference continuations.

For Model H, the report states that 250 held-out prefixes were used. The actual temperature sweep included 0.2, 0.5, 0.8, 1.0, 1.2, and 1.5, in addition to the deterministic 0.0 setting.


---

## 5.1 Model H — Telugu Generation Results

| Temperature | BLEU-4 | chrF | chrF++ | ROUGE-L | Rep-rate-3 | Distinct-1 | Distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.7501 | 17.66 | 15.24 | 7.20 | 0.5509 | 0.2000 | 0.4292 |
| 0.2 | 0.7591 | 18.43 | 15.84 | 7.34 | 0.4218 | 0.2145 | 0.4955 |
| **0.5** | **0.6284** | **19.12** | **16.38** | **7.31** | **0.1908** | **0.2646** | **0.6603** |
| 0.8 | 0.7253 | 18.82 | 15.98 | 6.78 | 0.0438 | 0.3259 | 0.8518 |
| 1.0 | 0.6494 | 18.83 | 15.84 | 6.32 | 0.0066 | 0.3609 | 0.9227 |
| 1.2 | 0.6441 | 18.52 | 15.49 | 6.17 | 0.0023 | 0.3873 | 0.9560 |
| 1.5 | 0.6183 | 18.33 | 15.07 | 5.80 | 0.0011 | 0.4198 | 0.9812 |

### Model H trend discussion

BLEU-4 is highest at T=0.2, while chrF and chrF++ peak at T=0.5. ROUGE-L is also highest at T=0.2 and generally declines at higher temperatures.

As temperature increases, repetition decreases sharply:

- Rep-rate-3: **0.5509 → 0.0011**
- Distinct-1: **0.2000 → 0.4198**
- Distinct-2: **0.4292 → 0.9812**

This demonstrates a clear trade-off between reference overlap and diversity. Lower temperatures produce safer, more reference-aligned generations but substantially more repetition. Higher temperatures produce much more diverse outputs but move farther away from the particular reference continuation.

T≈0.5 is a reasonable middle point for Model H because chrF/chrF++ are maximized there while repetition has already fallen substantially relative to very low temperatures.

---

## 5.2 Model L — Nepali Generation Results

The available Model L generation table is:

| Temperature | BLEU-4 | chrF | chrF++ | ROUGE-L | Rep-rate-3 | Distinct-1 | Distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.1921 | 17.00 | 14.35 | 8.82 | 0.5270 | 0.1712 | 0.4430 |
| 0.2 | 1.1983 | 17.13 | 14.48 | 8.84 | 0.4819 | 0.1789 | 0.4722 |
| **0.5** | **1.1735** | **18.04** | **15.22** | **9.23** | **0.2395** | **0.2215** | **0.6473** |
| 0.8 | 0.9676 | 17.84 | 14.88 | 8.41 | 0.0442 | 0.3024 | 0.8538 |
| 1.0 | 0.9748 | 17.62 | 14.61 | 7.77 | 0.0206 | 0.3321 | 0.9134 |
| 1.2 | 0.8752 | 17.71 | 14.58 | 7.14 | 0.0049 | 0.3661 | 0.9569 |
| 1.5 | 0.5865 | 17.22 | 14.10 | 6.62 | 0.0009 | 0.4100 | 0.9798 |

### Model L trend discussion

Model L shows the same overall temperature/diversity trade-off as Model H.

BLEU-4, chrF++, and ROUGE-L are strongest at lower temperatures, while repetition collapses and Distinct-1/Distinct-2 rise as temperature increases.

From T=0.0 to T=1.5:

- Rep-rate-3 falls from **0.5270 to 0.0009**
- Distinct-1 rises from **0.1712 to 0.4100**
- Distinct-2 rises from **0.4430 to 0.9798**

T=0.5 provides a useful middle ground for Model L as well, with the highest chrF, chrF++, and ROUGE-L in the reported sweep while substantially reducing repetition relative to low-temperature generation.

All generation results are treated as **45,000-step checkpoint results** for the final comparison.

---

# 6. Why BLEU, chrF/chrF++, and ROUGE-L Are Limited for Open-Ended LM Generation

These metrics are useful but should not be interpreted as complete measures of open-ended generation quality.

### BLEU-4

BLEU-4 measures n-gram overlap with the reference continuation. It is useful when the desired output is expected to closely match a reference, but open-ended language modeling has many valid continuations. A fluent continuation can receive a low BLEU score simply because it uses different wording.

### chrF / chrF++

chrF-based metrics measure character n-gram overlap and can be more tolerant of morphological variation than word-level metrics. This can be useful for morphologically rich languages. Nevertheless, they still measure similarity to a particular reference and therefore do not directly measure whether an alternative continuation is fluent, coherent, or semantically valid.

### ROUGE-L

ROUGE-L is based on the longest common subsequence between generated and reference text. It captures sequence-level overlap but has the same fundamental limitation: a valid open-ended continuation may be lexically different from the reference and therefore receive a lower score.

### Overall interpretation

Because each held-out prefix has a single reference continuation, these overlap metrics should be considered **weak indicators rather than definitive quality measures**. A strong evaluation should combine them with repetition, diversity, fluency, and qualitative coherence judgments.

---

# 7. Fluency and Diversity Diagnostics

The diversity metrics provide a particularly clear picture of sampling behavior.

Across both models:

1. **Low temperatures** produce high repetition and low lexical diversity.
2. **Moderate temperatures** substantially reduce repetition while preserving relatively strong reference overlap.
3. **High temperatures** produce very high Distinct-1/Distinct-2 values and almost eliminate repeated 3-grams, but overlap metrics decline.

This is the expected sampling trade-off:

> **Low temperature:** safe/reference-aligned but repetitive  
> **Moderate temperature:** balance of fidelity and diversity  
> **High temperature:** diverse but increasingly divergent from the reference

The supplied reports emphasize that diversity metrics are more directly informative for diagnosing repetition and lexical variety, while qualitative inspection remains necessary for judging actual fluency and coherence.

---

# 8. Attention Analysis

Attention was analyzed using per-layer, per-head mean entropy, mean attention distance, and heatmaps. For Model H, the supplied evaluation artifacts contain 16 heatmaps covering Layers 0 and 5 across all 8 heads.

## 8.1 Model H Heatmaps

The heatmaps below show query positions against key positions with attention weights represented by the color intensity. The causal lower-triangular structure is visible because a query cannot attend to future key positions.

### Layer 0 — Early-layer heads

| Head | Heatmap | Interpretation |
|---:|---|---|
| 0 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_0.png) | Broad, recency-biased attention |
| 1 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_1.png) | Early-layer attention |
| 2 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_2.png) | Early-layer attention |
| 3 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_3.png) | Early-layer attention |
| 4 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_4.png) | Early-layer attention |
| 5 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_5.png) | Early-layer attention |
| 6 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_6.png) | Early-layer attention |
| 7 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_0_head_7.png) | Early-layer attention |

### Layer 5 — Late-layer heads

| Head | Heatmap | Interpretation |
|---:|---|---|
| 0 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_0.png) | Late-layer attention |
| 1 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_1.png) | Late-layer attention |
| 2 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_2.png) | Late-layer attention |
| 3 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_3.png) | Late-layer attention |
| 4 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_4.png) | Late-layer attention |
| 5 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_5.png) | Late-layer attention |
| 6 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_6.png) | Attention-sink + local diagonal |
| 7 | [Heatmap](\/language_H\/report\/attention\/heatmap_layer_5_head_7.png) | Late-layer attention |

![Model H Layer 0 Head 0 — broad, recency-biased attention](/language_H/report/attention/heatmap_layer_0_head_0.png)

![Model H Layer 5 Head 6 — attention sink and local diagonal](/language_H/report/attention/heatmap_layer_5_head_6.png)

The causal (lower-triangular) mask is clearly visible in the heatmaps — no query attends to a future key position. The supplied analysis verified this behavior by checking that logits at position *t* are unaffected by changing token *t+1*.

Early Layer 0 heads are diffuse and recency-biased, with attention mass decaying smoothly from each query back over preceding tokens. In Layer 5, several heads develop sharper attention-sink behavior, concentrating substantial mass on the first token while also showing a local diagonal near the current query position.

## 8.2 Model H Attention Entropy

| Layer | Mean entropy across heads | Range (min–max) |
|---|---:|---:|
| 0 | 3.228 | 3.14 – 3.29 |
| 1 | 1.703 | 0.64 – 2.84 |
| 2 | 1.805 | 0.99 – 2.89 |
| 3 | 1.737 | 0.43 – 2.62 |
| 4 | 1.548 | 0.70 – 2.02 |
| 5 | 1.255 | 0.53 – 2.25 |

Full per-head statistics: [\`attention_summary_stats.json\`](attention/attention_summary_stats.json). Entropy is highest and most uniform in Layer 0 and becomes lower and more differentiated in deeper layers.

## 8.3 Model H Mean Attention Distance

| Layer | Mean distance across heads | Range (min–max) |
|---|---:|---:|
| 0 | 5.650 | 5.39 – 5.86 |
| 1 | 3.157 | 1.39 – 7.41 |
| 2 | 4.103 | 1.72 – 9.08 |
| 3 | 5.891 | 3.15 – 10.54 |
| 4 | 5.142 | 1.87 – 7.80 |
| 5 | 7.731 | 3.70 – 10.72 |

**Most local heads:** Layer 1 / Head 6 (distance 1.39, entropy 0.78) and Layer 1 / Head 2 (distance 1.48, entropy 0.64).

**Most long-range / content-based heads:** Layer 5 / Head 6 (distance 10.72, entropy 0.53) and Layer 3 / Head 4 (distance 10.54, entropy 0.43).

## 8.4 Model H Attention Discussion

Model H shows a depth-wise progression: Layer 0 behaves almost like a uniform local smoothing filter, while deeper layers differentiate into a mixture of tight local heads and long-range, low-entropy sink heads. This indicates a division of labor between position-sensitive/local processing and more selective long-range routing.

## 8.5 Model H Attention Artifacts

- [Attention summary statistics](attention/attention_summary_stats.json)
- [Layer 0, Head 0](attention/heatmap_layer_0_head_0.png)
- [Layer 0, Head 1](attention/heatmap_layer_0_head_1.png)
- [Layer 0, Head 2](attention/heatmap_layer_0_head_2.png)
- [Layer 0, Head 3](attention/heatmap_layer_0_head_3.png)
- [Layer 0, Head 4](attention/heatmap_layer_0_head_4.png)
- [Layer 0, Head 5](attention/heatmap_layer_0_head_5.png)
- [Layer 0, Head 6](attention/heatmap_layer_0_head_6.png)
- [Layer 0, Head 7](attention/heatmap_layer_0_head_7.png)
- [Layer 5, Head 0](attention/heatmap_layer_5_head_0.png)
- [Layer 5, Head 1](attention/heatmap_layer_5_head_1.png)
- [Layer 5, Head 2](attention/heatmap_layer_5_head_2.png)
- [Layer 5, Head 3](attention/heatmap_layer_5_head_3.png)
- [Layer 5, Head 4](attention/heatmap_layer_5_head_4.png)
- [Layer 5, Head 5](attention/heatmap_layer_5_head_5.png)
- [Layer 5, Head 6](attention/heatmap_layer_5_head_6.png)
- [Layer 5, Head 7](attention/heatmap_layer_5_head_7.png)

# 9. Model L — Attention Analysis

Model L attention was analyzed using `report/attention/attention_summary_stats.json` and the accompanying heatmaps. The supplied report provides the entropy and mean-distance summaries below.

## 9.1 Model L Heatmaps

The Model L report states that accompanying heatmaps were generated for the attention analysis. The links below use the same `heatmap_layer_{L}_head_{H}.png` filename convention used by the Model H artifacts; if the local Model L filenames differ, they can be replaced with the exact paths.

### Layer 0 — Early-layer heads

| Head | Heatmap |
|---:|---|
| 0 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_0.png) |
| 1 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_1.png) |
| 2 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_2.png) |
| 3 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_3.png) |
| 4 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_4.png) |
| 5 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_5.png) |
| 6 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_6.png) |
| 7 | [Heatmap](/language_L/report/attention/heatmap_layer_0_head_7.png) |

### Layer 5 — Late-layer heads

| Head | Heatmap |
|---:|---|
| 0 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_0.png) |
| 1 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_1.png) |
| 2 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_2.png) |
| 3 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_3.png) |
| 4 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_4.png) |
| 5 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_5.png) |
| 6 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_6.png) |
| 7 | [Heatmap](/language_L/report/attention/heatmap_layer_5_head_7.png) |

![Model LLayer 0 Head 0 — broad, recency-biased attention](/language_L/report/attention/heatmap_layer_0_head_0.png)

![Model L Layer 5 Head 6 — attention sink and local diagonal](/language_L/report/attention/heatmap_layer_5_head_6.png)

## 9.2 Model L Attention Entropy

| Layer | Mean entropy across heads | Range (min–max) |
|---|---:|---:|
| 0 | 3.147 | 2.95 – 3.25 |
| 1 | 1.868 | 0.52 – 2.81 |
| 2 | 1.829 | 0.68 – 2.70 |
| 3 | 2.123 | 1.29 – 2.54 |
| 4 | 1.894 | 0.84 – 2.49 |
| 5 | 1.818 | 1.47 – 2.13 |

Full per-head statistics: [\`report/attention/attention_summary_stats.json\`](attention/attention_summary_stats.json).

## 9.3 Model L Mean Attention Distance

| Layer | Mean distance across heads | Range (min–max) |
|---|---:|---:|
| 0 | 5.272 | 4.17 – 6.02 |
| 1 | 3.136 | 1.22 – 5.45 |
| 2 | 3.836 | 1.46 – 7.82 |
| 3 | 7.379 | 3.63 – 10.45 |
| 4 | 5.854 | 1.53 – 9.09 |
| 5 | 8.269 | 6.29 – 9.57 |

**Most local heads:** Layer 1 / Head 0 (distance 1.22, entropy 0.52) and Layer 2 / Head 7 (distance 1.46, entropy 0.68).

**Most long-range heads:** Layer 3 / Head 2 (distance 10.45, entropy 1.29) and Layer 5 / Head 1 (distance 9.57, entropy 1.67).

## 9.4 Model L Attention Discussion

Layer 0 behaves as a diffuse, uniform, moderately local filter. From Layer 1 onward, the heads differentiate into a mixture of tight local heads and longer-range heads. The longest-range heads reach distances of approximately 10.45, but have higher entropy than the sharpest long-range heads reported for Model H, suggesting more distributed attention across multiple positions.

## 9.5 Model L Attention Artifacts

- [Attention summary statistics](attention/attention_summary_stats.json)
- [Layer 0, Head 0](attention/heatmap_layer_0_head_0.png)
- [Layer 0, Head 1](attention/heatmap_layer_0_head_1.png)
- [Layer 0, Head 2](attention/heatmap_layer_0_head_2.png)
- [Layer 0, Head 3](attention/heatmap_layer_0_head_3.png)
- [Layer 0, Head 4](attention/heatmap_layer_0_head_4.png)
- [Layer 0, Head 5](attention/heatmap_layer_0_head_5.png)
- [Layer 0, Head 6](attention/heatmap_layer_0_head_6.png)
- [Layer 0, Head 7](attention/heatmap_layer_0_head_7.png)
- [Layer 5, Head 0](attention/heatmap_layer_5_head_0.png)
- [Layer 5, Head 1](attention/heatmap_layer_5_head_1.png)
- [Layer 5, Head 2](attention/heatmap_layer_5_head_2.png)
- [Layer 5, Head 3](attention/heatmap_layer_5_head_3.png)
- [Layer 5, Head 4](attention/heatmap_layer_5_head_4.png)
- [Layer 5, Head 5](attention/heatmap_layer_5_head_5.png)
- [Layer 5, Head 6](attention/heatmap_layer_5_head_6.png)
- [Layer 5, Head 7](attention/heatmap_layer_5_head_7.png)

# 10. Model H vs. Model L Attention Comparison

Both models show a similar broad pattern:

- Layer 0 behaves like a diffuse, moderately local filter.
- Middle and later layers develop specialized local and longer-range heads.
- Some heads are sharply peaked, while others distribute attention more broadly.

There are also differences.

Model H's longest-range head reaches a mean distance of approximately **10.72**, with very low entropy (**0.53**). The Model L longest-range head reaches approximately **10.45**, but has higher entropy (**1.29**).

This suggests that Model H has more sharply concentrated long-range attention, including sink-like behavior, whereas Model L's long-range heads appear to distribute attention across multiple positions rather than concentrating as strongly on one position.

This difference is interpreted as a comparison of the attention statistics at the common 45,000-step checkpoint.

---

# 11. Model H vs Model L Comparison

All reported results are from the **45,000-step checkpoint**. The comparison therefore focuses on differences in language/data characteristics and observed learning behavior.

## 11.1 Training and Validation Curves

### Model H — Telugu

![Model H Train, Validation, and Test Loss](/language_H/report/loss_curves/loss_comparison.png)

![Model H Train, Validation, and Test Perplexity](/language_H/report/loss_curves/perplexity_comparison.png)

Model H shows a rapid reduction in cross-entropy loss during the early
training stages, followed by slower improvement and eventual stabilization.
The training and validation curves remain relatively close at the later
stages, with the reported test loss of **3.7361** lying close to the
validation region. The corresponding test perplexity is **41.94**.

The relatively small separation between the training and validation curves
suggests that Model H is not showing strong overfitting at this checkpoint.
Instead, the model appears to still have room for improvement with
additional training.

### Model L — Nepali

![Model L Train, Validation, and Test Loss](/language_L/report/loss_curves/loss_comparison.png)

![Model L Train, Validation, and Test Perplexity](/language_L/report/loss_curves/perplexity_comparison.png)

Model L follows a similar overall learning pattern: a large reduction in
loss during the early training stages followed by slower improvement. At
the 45,000-step checkpoint, the reported training loss is approximately
**3.56**, while the validation loss is **3.9298** and the test loss is
**3.9761**. The corresponding test perplexity is **53.31**.

The curves therefore indicate that Model L is also learning effectively,
although its held-out loss remains higher than the reported Model H result.


## 11.2 Discussion: Data Scale and Quality

The comparison is controlled for model capacity and checkpoint, so the observed performance difference is discussed in terms of language and data characteristics rather than model size or training step.

A plausible source of the difference is the underlying training data. The
Model H report indicates that approximately **369M training tokens** had
been processed by the 45,000-step checkpoint, against a planned 600M-token
budget. The supplied Model L report describes Nepali as the lower-resource
language tier.

Differences in the amount, quality, and diversity of training text can
affect how effectively a model learns lexical, morphological, syntactic, and
long-range patterns. A smaller or less diverse corpus may provide fewer
examples of linguistic constructions and therefore lead to weaker
generalization.

The available results do not isolate data quantity from data quality, so the
H–L performance gap should not be attributed to either factor alone.

## 11.3 Script and Language Characteristics

Telugu and Nepali use different writing systems and have different
orthographic and morphological characteristics. These differences affect
the distribution of character and subword sequences seen during training.

Even with identical model capacity, the effective modeling difficulty can
differ between languages. The model may need to represent different
morphological patterns, word boundaries, and character combinations, all of
which influence the statistics learned by the tokenizer and Transformer.

## 11.4 Tokenizer Fertility

Tokenizer fertility is another important consideration. Fertility describes
how many tokenizer tokens are required to represent a given amount of text.
If one language is segmented into more subword tokens on average, the model
must make more token-level predictions for the same underlying text.

This can affect both training efficiency and perplexity. Therefore, the
lower perplexity of Model H should not be interpreted as evidence that
Telugu is intrinsically easier to model than Nepali.

BPB provides a more useful cross-language perspective:

- **Model H:** BPB = **0.0152**
- **Model L:** BPB = **0.0162**

The values are relatively close, although Model H remains lower under the
reported evaluation.

## 11.5 Generation Comparison

The generation experiments show a highly consistent temperature-dependent
trade-off for both languages.

At low temperatures, both models exhibit substantially higher repetition and
lower lexical diversity. Increasing temperature progressively reduces
repetition while increasing Distinct-1 and Distinct-2.

For Model H, repetition rate decreases from **0.5509 at T=0.0** to
**0.0011 at T=1.5**, while Distinct-2 increases from **0.4292** to
**0.9812**.

For Model L, repetition rate decreases from **0.5270 at T=0.0** to
**0.0009 at T=1.5**, while Distinct-2 increases from **0.4430** to
**0.9798**.

Thus, both models exhibit the expected transition from relatively safe but
repetitive generation at low temperature to highly diverse generation at high
temperature.

## 11.6 Attention Comparison

The two models also show similar layer-wise attention specialization.
Layer 0 in both models has relatively high entropy and moderately local,
diffuse attention. Deeper layers contain a mixture of sharply focused local
heads and longer-range heads.

Model H's longest-range head reaches a mean attention distance of
approximately **10.72**, with entropy **0.53**. Model L's longest-range head
reaches approximately **10.45**, with entropy **1.29**.

This suggests that Model H contains a particularly sharp long-range
attention pattern, including sink-like behavior, whereas Model L's
long-range attention is somewhat more distributed across positions.

## 11.7 Overall Comparison

Overall, Model H has the stronger reported language-modeling performance at
the 45,000-step checkpoint, with lower test loss, perplexity, and BPB.
Nevertheless, both models show broadly similar qualitative behavior in their
training curves, generation diversity, repetition trends, and attention
specialization.

Because model architecture and checkpoint are controlled, differences are
most plausibly associated with the characteristics of the two languages and
their training corpora, including data scale and quality, script and
orthographic structure, and tokenizer fertility. These factors should be
considered together rather than attributing the performance difference to a
single cause.

# 12. Main Findings

### Language modeling

Model H achieves a reported test loss of **3.7361** and PPL of **41.9356**, compared with Model L's reported **3.9761** loss and **53.3080** PPL. Model H therefore has the lower reported perplexity. BPB is **0.0152** for H and **0.0162** for L.

The comparison should remain cautious because Model L's final test-set checkpoint is not fully confirmed.

### Generation

Both models show the same fundamental sampling behavior:

- increasing temperature dramatically reduces repetition;
- Distinct-1 and Distinct-2 increase;
- overlap with a single reference generally declines;
- moderate temperature provides a useful compromise.

For Model H, T=0.5 gives the highest chrF and chrF++ values. For Model L, T=0.5 gives the highest chrF, chrF++, and ROUGE-L values in the reported sweep.

### Attention

Both models exhibit layer-wise specialization.

Early layers have relatively high entropy and diffuse attention. Deeper layers contain sharper local heads and longer-range heads. Model H shows particularly sharp long-range/sink-like behavior, while Model L's long-range heads appear somewhat more distributed.

---

# 13. Deliverables Checklist

| Deliverable | Status / Location |
|---|---|
| 1. Transformer implementation (MHA, positional embeddings, causal mask) | Implemented; code/config referenced by the project reports |
| 2. Model configuration files (H and L) | [Model configuration](configs/51.yaml) |
| 3. Parameter counts | **24,912,640 (~24.913M) for each model** |
| 4. Training scripts | Part of project implementation; exact script path is not specified in the supplied reports |
| 5. PPL / BPB tables | Included in Section 4 |
| 6. BLEU, chrF, ROUGE-L results | Included in Section 5 |
| 7. Generated samples and diversity/repetition statistics | [Model H metrics](language_H/report/tables/all_temperatures_metrics.txt); [Model L metrics](language_L/report/tables/all_temperatures_metrics.txt) |
| 8. Attention heatmaps | [Model H attention artifacts](language_H/report/attention); [Model L attention artifacts](language_L/report/attention) |
| 9. Attention entropy/distance summaries | Included in Sections 8–9 |
| 10. Training logs and loss curves | [Model H loss](language_H/checkpoints/logs/loss_curve.png), [Model H PPL](language_L/report/loss_curves/perplexity_comparison.png); [Model L loss](language_L/checkpoints/logs/loss_curve.png), [Model L PPL](language_L/report/loss_curves/perplexity_comparison.png) |
| 11. Checkpoint / Drive links | Both models evaluated at **45,000 steps**; no Drive links were provided in the supplied Markdown files |
| 12. Resource-level comparison | Included in Section 11 |


---

# 14. Conclusion

Phase 2 demonstrates that both language models successfully learn non-trivial language-modeling behavior with a relatively small Transformer architecture. The models use approximately 25M parameters and exhibit clear depth-wise specialization in their attention mechanisms.

The generation experiments reveal a strong and consistent temperature trade-off. Lower temperatures produce more repetitive and reference-aligned text, while higher temperatures produce substantially greater lexical diversity and almost eliminate repeated 3-grams. This makes repetition rate and Distinct-n particularly useful diagnostics alongside reference-based metrics.

The overlap metrics—BLEU-4, chrF/chrF++, and ROUGE-L—are useful for measuring similarity to held-out reference continuations but are not sufficient for open-ended generation, because multiple valid continuations can differ substantially in wording. Therefore, the strongest interpretation combines overlap scores, diversity statistics, repetition measurements, attention analysis, and qualitative inspection.

The reported language-modeling results favor Model H, with lower test loss and perplexity than Model L. The attention analyses nevertheless show similar qualitative behavior in both models: diffuse early-layer attention followed by increasingly specialized local and long-range heads in deeper layers.

Overall, the experiments demonstrate the expected relationship between sampling temperature, reference similarity, repetition, diversity, and attention specialization, while also highlighting the importance of using matched checkpoints and tokenizer-agnostic metrics such as BPB when comparing language models across languages.
