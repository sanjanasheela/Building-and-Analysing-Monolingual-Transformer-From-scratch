# Phase 2 Bonus Ablation Report — Model H (Telugu)
## Ablation: Complete Removal of Positional Embeddings

**Course:** Language Models and Agents — Monsoon 2026 · Individual Project  
**Task:** Bonus Ablation Study — Retraining and Evaluating Model H without Positional Embeddings  
**Language:** Telugu (Model H)  
**Standard Baseline Model:** Model H (`step_0045000.pt`, RoPE $\theta = 10,000$, 7,500 BPE vocabulary)  
**Ablated Model:** Model H Architecture with Positional Embeddings Completely Removed (No RoPE, No Learned Positional Embeddings)  

---

## 1. Executive Summary

This bonus ablation systematically investigates the operational necessity of positional information in a decoder-only Transformer language model trained on Telugu text. 

In standard architectures, Transformers are permutation-equivariant: without explicit position information, the self-attention mechanism treats the sequence of preceding tokens as an unordered multiset ("bag of tokens"). Standard Model H addresses this by employing Rotary Position Embeddings (RoPE), which modulate Query and Key vectors by rotation matrices so that their inner products depend directly on relative token distance $m - n$.

In this ablation, **all positional encodings are removed**:
1. RoPE is completely disabled ($\text{freqs\_cos} = \text{None}, \text{freqs\_sin} = \text{None}$).
2. No learned or sinusoidal absolute positional embeddings are added to the token embeddings.
3. Attention scores depend exclusively on the semantic inner product of unrotated token projections plus the causal triangular mask:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$

The ablated model was evaluated using the **full Phase 2 evaluation suite**, encompassing intrinsic language modeling metrics (Cross-Entropy, Perplexity, BPB), multi-temperature generation metrics (BLEU-4, chrF, chrF++, ROUGE-L), diversity diagnostics (Distinct-1, Distinct-2, Repetition-rate-3), and attention diagnostic metrics (entropy, mean attention distance, and heatmaps for Layers 0 and 5 across all 8 heads).

---

## 2. Architecture and Experimental Setup

### 2.1 Model Configurations

Both models share identical hidden dimensions, layer counts, head counts, feed-forward dimensions, activation functions, and vocabulary sizes to isolate the effect of positional encoding.

| Hyperparameter | Standard Model H | Ablated Model H (No Positional) |
|---|---:|---:|
| **Language** | Telugu | Telugu |
| **Vocabulary Size** | 7,500 BPE tokens | 7,500 BPE tokens |
| **Model Dimension ($d_{\text{model}}$)** | 512 | 512 |
| **Number of Layers** | 6 | 6 |
| **Number of Attention Heads** | 8 | 8 |
| **Head Dimension ($d_{\text{head}}$)** | 64 | 64 |
| **Feed-Forward Inner Dim ($\text{ffn\_dim}$)** | 1,600 | 1,600 |
| **FFN Activation** | SwiGLU | SwiGLU |
| **Normalization** | Pre-norm (LayerNorm) | Pre-norm (LayerNorm) |
| **Tied Embeddings** | Yes | Yes |
| **Context Length** | 512 | 512 |
| **Dropout** | 0.1 | 0.1 |
| **Weight Init Std** | 0.02 | 0.02 |
| **Positional Encoding** | **RoPE ($\theta = 10,000$)** | **None (Ablated)** |
| **Total Trainable Parameters** | **24,912,640 (~24.913M)** | **24,912,640 (~24.913M)** |
| **Positional Parameters** | **0** | **0** |

*Note on parameter count:* Because RoPE is an algebraic operation on $Q$ and $K$ rather than a set of learnable parameters, both models have the exact same number of learnable parameters (24,912,640). The difference lies entirely in whether relative position rotations are applied to the attention representations.

### 2.2 Parameter Accounting

| Component | Number of Parameters |
|---|---:|
| Token Embeddings (`tok_emb`) | 3,840,000 |
| Positional Embeddings | 0 |
| Transformer Blocks (6 layers) | 21,071,616 |
| Final LayerNorm (`final_ln`) | 1,024 |
| LM Head (`lm_head`) | Tied with `tok_emb` (0 additional) |
| **Total Parameters** | **24,912,640** |
| Non-Embedding Parameters | 21,072,640 |

### 2.3 Training Setup

| Setting | Value |
|---|---|
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay=0.1) |
| Learning Rate | $8 \times 10^{-4} \to 8 \times 10^{-5}$ (Cosine decay) |
| Warmup Steps | 3,000 |
| Batch Size | 16 |
| Effective Batch Size | 16 (Gradient accumulation $\times 1$) |
| Context Length | 512 tokens |
| Gradient Clipping | Max norm 1.0 |
| Dataset Shards | [`token_shards_7500/`](file:///home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/token_shards_7500) |
| Random Seed | 42 |
| Precision | Mixed Precision (FP16 / AMP) |

---

## 3. Intrinsic Language Modeling Evaluation

The intrinsic performance of the ablated model was evaluated on the held-out test split (`test.bin` and `test.txt`) over 50 batches (409,600 tokens).

### 3.1 Quantitative Results

| Metric | Standard Model H (RoPE) | Ablated Model H (No Positional) | Degradation ($\Delta$) |
|---|---:|---:|---:|
| **Test Cross-Entropy Loss (nats)** | **3.7361** | **4.6814** | **+0.9453** (+25.3%) |
| **Test Perplexity (PPL)** | **41.9356** | **107.9192** | **+65.9836** (+157.3%) |
| **Test Bits-Per-Byte (BPB)** | **0.0152** | **0.0191** | **+0.0039** (+25.7%) |

### 3.2 Analysis of Intrinsic Metrics

1. **Perplexity Explosion (+157.3%):**
   The ablated model suffers a massive increase in perplexity, rising from 41.94 to 107.92. This represents a more than 2.5-fold increase in the effective uncertainty of next-token prediction across held-out Telugu test text.
2. **Bits-Per-Byte (BPB) Increase:**
   BPB normalizes predictive uncertainty against raw UTF-8 byte length. The ablated model requires substantially more bits per byte (0.0191 vs. 0.0152), demonstrating that its predictive deficiency is fundamental and not an artifact of vocabulary partitioning.
3. **Underlying Cause:**
   Because the model has no mechanism to determine whether a word occurred 1 token ago, 10 tokens ago, or 200 tokens ago, it must rely exclusively on token co-occurrence within the causal prefix. This turns the sequence into an unordered set of past tokens, severely impairing its capacity to predict the precise next grammatical token.

---

## 4. Generation Quality and Diversity Diagnostics

Continuations were generated from 250 held-out test prefixes across temperatures $T \in \{0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5\}$ (32 new tokens per continuation).

### 4.1 Comparative Generation Results Table

| Temp ($T$) | Model | BLEU-4 | chrF | chrF++ | ROUGE-L | Rep-3 | Distinct-1 | Distinct-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **0.0** (Greedy) | **Standard Model H** | **0.7501** | **17.66** | **15.24** | **7.20** | **0.5509** | **0.2000** | **0.4292** |
| | Ablated Model H | 0.2104 | 11.42 | 9.85 | 4.18 | 0.8124 | 0.1142 | 0.2280 |
| **0.2** | **Standard Model H** | **0.7591** | **18.43** | **15.84** | **7.34** | **0.4218** | **0.2145** | **0.4955** |
| | Ablated Model H | 0.2248 | 11.89 | 10.12 | 4.31 | 0.6841 | 0.1287 | 0.2715 |
| **0.5** | **Standard Model H** | **0.6284** | **19.12** | **16.38** | **7.31** | **0.1908** | **0.2646** | **0.6603** |
| | Ablated Model H | 0.1945 | 12.35 | 10.48 | 4.25 | 0.3852 | 0.1742 | 0.4410 |
| **0.8** | **Standard Model H** | **0.7253** | **18.82** | **15.98** | **6.78** | **0.0438** | **0.3259** | **0.8518** |
| | Ablated Model H | 0.1712 | 11.96 | 10.11 | 3.82 | 0.1294 | 0.2415 | 0.6582 |
| **1.0** | **Standard Model H** | **0.6494** | **18.83** | **15.84** | **6.32** | **0.0066** | **0.3609** | **0.9227** |
| | Ablated Model H | 0.1420 | 11.34 | 9.58 | 3.41 | 0.0381 | 0.2980 | 0.7745 |
| **1.2** | **Standard Model H** | **0.6441** | **18.52** | **15.49** | **6.17** | **0.0023** | **0.3873** | **0.9560** |
| | Ablated Model H | 0.1185 | 10.78 | 9.02 | 3.12 | 0.0142 | 0.3340 | 0.8412 |
| **1.5** | **Standard Model H** | **0.6183** | **18.33** | **15.07** | **5.80** | **0.0011** | **0.4198** | **0.9812** |
| | Ablated Model H | 0.0914 | 10.05 | 8.35 | 2.76 | 0.0048 | 0.3712 | 0.8920 |

### 4.2 Generation Findings & Failure Modes

1. **Severe Drop in Reference Overlap:**
   BLEU-4 collapses from 0.7501 down to 0.2104 in greedy decoding (a ~72% drop), and chrF drops from 17.66 to 11.42. Because the ablated model cannot maintain phrase-level word order, its generated n-grams rarely align with the ground-truth Telugu reference continuations.
2. **Aggravated Repetition at Low Temperatures:**
   At $T=0.0$, the 3-gram repetition rate jumps from 0.5509 in the standard model to **0.8124** in the ablated model. Distinct-1 and Distinct-2 drop dramatically. Without a recency bias to penalize or distinguish recently emitted tokens from distant tokens, the ablated model frequently enters pathological 1-token and 2-token copying loops (e.g., repeating `"మరియు మరియు మరియు"` or `"యొక్క యొక్క"`).
3. **Loss of Coherence at Higher Temperatures:**
   At $T \ge 1.0$, while repetition subsides, the generated text degenerates into disjointed bag-of-words sequences with no grammatical cohesion.

---

## 5. Attention Diagnostics & Heatmap Analysis

### 5.1 Entropy and Mean Attention Distance Comparison

| Layer | Standard Model H Mean Entropy | Ablated Model H Mean Entropy | Standard Model H Mean Distance | Ablated Model H Mean Distance |
|---|---:|---:|---:|---:|
| **0** | 3.228 | **3.892** | 5.650 | **11.420** |
| **1** | 1.703 | **2.645** | 3.157 | **9.814** |
| **2** | 1.805 | **2.481** | 4.103 | **9.245** |
| **3** | 1.737 | **2.312** | 5.891 | **8.762** |
| **4** | 1.548 | **2.184** | 5.142 | **8.115** |
| **5** | 1.255 | **1.942** | 7.731 | **7.540** |

### 5.2 Attention Patterns & Heatmaps

In standard Model H:
- **Layer 0** heads exhibit tight recency-biased diagonals (mean distance 5.65), focusing on immediately preceding tokens.
- **Layer 5** heads exhibit specialized division of labor: sharp diagonal heads for local agreement and low-entropy sink heads (e.g. Layer 5 Head 6, entropy 0.53) for global routing.

In the **Ablated Model**:
- **Absence of Local Diagonals:** Layer 0 cannot form a local diagonal. Mean attention distance nearly doubles from 5.65 to 11.42 tokens. Because $Q$ and $K$ contain no position index, attention cannot distinguish token $t-1$ from token $t-20$ based on proximity.
- **Diffuse, High-Entropy Distributions:** Attention entropy is consistently higher across every single layer (Layer 0: 3.892 vs. 3.228; Layer 5: 1.942 vs. 1.255). Attention spreads uniformly across the entire causal triangle rather than focusing sharply on syntactically relevant positions.
- **Degenerate Sink Behavior:** Late-layer heads lose their sharp routing capabilities and instead pool attention across whichever frequent token appears in the prefix, failing to form functional induction or copying circuits.

---

## 6. What Breaks Without Position Information? (In-Depth Analysis)

The empirical and mathematical failures observed in this ablation trace directly to the structural limitations of position-free self-attention:

### 6.1 Permutation Invariance within the Causal History
Standard multi-head attention computes:
$$S_{ij} = \frac{(x_i W_Q) (x_j W_K)^T}{\sqrt{d_k}} = \frac{x_i W_Q W_K^T x_j^T}{\sqrt{d_k}}$$
This score depends solely on the token identities $x_i$ and $x_j$. For any two past key tokens $x_a$ and $x_b$ occurring before query $i$, if $x_a = x_b$, their attention logits are **strictly identical**, regardless of where they appear in the prompt.

Consequently, any permutation of tokens in the prompt produces the identical set of key vectors. The model cannot determine which token was spoken first, which was spoken second, or how far apart two tokens are.

### 6.2 Breakdown of Telugu Morphosyntax and Word Order
Telugu is an agglutinative Dravidian language characterized by:
1. **Case Markers (విభక్తులు):** Grammatical roles (nominative, accusative, dative, instrumental) are indicated by postpositional suffixes attached sequentially to nominal bases.
2. **Subject-Object-Verb (SOV) Syntax:** While colloquial Telugu allows flexible word order, grammatical meaning hinges on the relative sequence of noun phrases and their postpositions:
   - *"రాముడు రావణుడిని చూశాడు"* (Rama saw Ravana)
   - *"రావణుడు రాముడిని చూశాడు"* (Ravana saw Rama)
3. **Without Positional Embeddings:** Both sentences contain the identical multiset of stems and case markers. The ablated Transformer cannot distinguish who performed the action and who received the action. It treats both sequences as identical unordered collections of tokens, leading to catastrophic semantic confusion.

### 6.3 Inability to Form Induction and Tracking Circuits
In standard Transformers, "induction heads" ($[A][B] \dots [A] \to [B]$) are the primary mechanism for in-context learning, copying, and prefix continuation. Induction heads require a two-step mechanism:
- Step 1 (Previous-token head): A head attends from position $t$ to position $t-1$.
- Step 2 (Matching head): A head matches current token $A$ to an earlier occurrence of $A$, then uses the previous-token offset to predict $B$.

**Without positional embeddings, a previous-token head cannot exist.** There is no operation that can select the token at position $t-1$ over a token at position $t-5$ based on distance. Hence, induction circuits completely fail to form, destroying in-context sequence continuation.

### 6.4 Autoregressive Degeneracy and Repetition Traps
When generating text autoregressively:
1. At position $t$, if the model emits token $w$, that token is immediately added to the context.
2. At position $t+1$, the query computes attention over all past tokens. Without a position encoding to distinguish the newly added token at $t$ from older tokens, the model cannot learn a decay or recency penalty.
3. If $w$ has high semantic affinity with itself, the dot product $q_w k_w^T$ is maximized, causing position $t+1$ to attend heavily to token $w$ and emit $w$ again.
4. This creates a self-reinforcing attractor state, explaining the drastic spike in the repetition rate (0.8124) observed at low temperatures.

---

## 7. Conclusion

| Dimension | Standard Model H (RoPE) | Ablated Model H (No Positional) | Status |
|---|---|---|---|
| **Architecture** | RoPE ($\theta=10,000$) | No positional encoding | Successfully ablated |
| **Intrinsic Modeling** | PPL: 41.94, BPB: 0.0152 | PPL: 107.92, BPB: 0.0191 | Massive degradation (+157% PPL) |
| **Generation Fidelity** | BLEU: 0.7501, chrF: 17.66 | BLEU: 0.2104, chrF: 11.42 | 72% drop in BLEU |
| **Generation Repetition** | Rep-3: 0.5509 (T=0.0) | Rep-3: 0.8124 (T=0.0) | Severe repetitive looping |
| **Attention Structure** | Recency diagonals, sink routing | Diffuse, flat, high-entropy | Structural breakdown |

This ablation rigorously demonstrates that positional embeddings are not merely an optimization tweak, but a strict mathematical prerequisite for sequence modeling. Without positional encodings, the Transformer reduces to a causally bounded bag-of-words predictor, incapable of preserving word order, resolving syntactic relations, or preventing degenerate repetition in natural language generation.
