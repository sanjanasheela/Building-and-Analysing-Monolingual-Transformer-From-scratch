# Phase 2 Report — Model L (Nepali): Implementation, Pretraining, and Evaluation

**Course:** Language Models and Agents — Monsoon 2026 · Individual Project
**Phase:** 2 of 3 — Model Implementation, Pretraining, and Evaluation [40 Marks]
**Eval checkpoint:** `step_0045000.pt` (matched to Model H's eval checkpoint for a fair side-by-side comparison)

> This section covers **Model L (Nepali)** only, evaluated at the same checkpoint step (45,000) used for Model H, so the two can be compared apples-to-apples. It is meant to be appended to `phase2_report.md` alongside the existing Model H section.

---


## 1. Architecture (partially inferred — confirm against actual config)
same configuration as H 

---

## 2. Training Snapshot at Step 45,000

| Setting | Value |
|---|---|
| Checkpoint saved at | 2026-09-06 16:51:56 |
| Train loss (step 45,000, single batch) | 3.5625 |
| Learning rate (step 45,000) | 0.00099993 (essentially at peak — schedule has barely decayed by this step) |
| Grad norm | 0.3026 |
| Validation loss | **3.9298** |
| Validation perplexity | **50.8964** |

**Comparison note:** at the matched step 45,000, Model L's validation loss (3.9298) is noticeably higher than Model H's (~3.7 range reported for Model H's test set) — consistent with Nepali being the lower-resource language tier, but a like-for-like comparison should use Model L's *validation* loss against Model H's *validation* loss at the same step, not against Model H's test loss, to avoid an apples-to-oranges comparison.

The learning rate still sitting at ~0.001 (its apparent peak) at step 45,000 suggests Model L's LR schedule has a longer warmup/decay horizon than Model H's — again, only confirmable from Model L's actual config.

---

## 3. Language Modeling Metrics (⚠️ checkpoint not confirmed — see note above)

| Metric | Model L (Nepali) |
|---|---|
| Test Cross-Entropy Loss | 3.9761 |
| Test Perplexity (PPL) | 53.3080 |
| Bits-per-Byte (BPB) | 0.0162 |

These numbers come from `final_evaluation_summary.txt` (mislabeled "MODEL H" in the source file — corrected here). They align closely with the step-55,000 validation loss (3.9696) rather than the step-45,000 validation loss (3.9298) reported above, reinforcing the concern in note 2: **this test-set evaluation was likely run on `step_0055000.pt`, not `step_0045000.pt`.**

---

## 4. Generation Quality (⚠️ checkpoint not confirmed)

Source: `report/tables/all_temperatures_metrics.txt`.

| Temperature | BLEU-4 |  chrF | chrF++ | ROUGE-L | Rep-rate-3 | Distinct-1 | Distinct-2 |
| ----------- | -----: | ----: | -----: | ------: | ---------: | ---------: | ---------: |
| 0.0         | 1.1921 | 17.00 |  14.35 |    8.82 |     0.5270 |     0.1712 |     0.4430 |
| 0.2         | 1.1983 | 17.13 |  14.48 |    8.84 |     0.4819 |     0.1789 |     0.4722 |
| 0.5         | 1.1735 | 18.04 |  15.22 |    9.23 |     0.2395 |     0.2215 |     0.6473 |
| 0.8         | 0.9676 | 17.84 |  14.88 |    8.41 |     0.0442 |     0.3024 |     0.8538 |
| 1.0         | 0.9748 | 17.62 |  14.61 |    7.77 |     0.0206 |     0.3321 |     0.9134 |
| 1.2         | 0.8752 | 17.71 |  14.58 |    7.14 |     0.0049 |     0.3661 |     0.9569 |
| 1.5         | 0.5865 | 17.22 |  14.10 |    6.62 |     0.0009 |     0.4100 |     0.9798 |



**Trend discussion:** the same qualitative pattern as Model H holds — BLEU-4/chrF++/ROUGE-L are highest at low temperature and decline as temperature rises, while repetition rate collapses (0.48 → 0.0005) and Distinct-1/2 rise monotonically (0.18→0.41, 0.47→0.98) as temperature increases. The magnitudes are broadly comparable to Model H's (chrF ~17–19 for both, ROUGE-L somewhat lower for Nepali at higher temperatures), consistent with Nepali generation being a bit less faithful to the single reference at a given temperature — plausibly reflecting the lower-resource tier, though this should be stated cautiously given the checkpoint-provenance uncertainty above.

---

## 5. Attention Analysis 

Source: `report/attention/attention_summary_stats.json` (6 layers × 8 heads) and the accompanying heatmaps.

### Entropy per Layer

| Layer | Mean entropy across heads | Range (min–max) |
|---|---|---|
| 0 | 3.147 | 2.95 – 3.25 |
| 1 | 1.868 | 0.52 – 2.81 |
| 2 | 1.829 | 0.68 – 2.70 |
| 3 | 2.123 | 1.29 – 2.54 |
| 4 | 1.894 | 0.84 – 2.49 |
| 5 | 1.818 | 1.47 – 2.13 |

### Mean Attention Distance per Layer

| Layer | Mean distance across heads | Range (min–max) |
|---|---|---|
| 0 | 5.272 | 4.17 – 6.02 |
| 1 | 3.136 | 1.22 – 5.45 |
| 2 | 3.836 | 1.46 – 7.82 |
| 3 | 7.379 | 3.63 – 10.45 |
| 4 | 5.854 | 1.53 – 9.09 |
| 5 | 8.269 | 6.29 – 9.57 |

**Most local heads:** Layer 1 / Head 0 (distance 1.22, entropy 0.52) and Layer 2 / Head 7 (distance 1.46, entropy 0.68) — sharply peaked, near-neighbor attention.

**Most long-range heads:** Layer 3 / Head 2 (distance 10.45, entropy 1.29) and Layer 5 / Head 1 (distance 9.57, entropy 1.67).

**Discussion:** Layer 0 again behaves as a diffuse, uniform, moderately local filter (high, narrow-range entropy: 2.95–3.25), matching Model H's Layer 0 pattern closely. From Layer 1 onward, heads differentiate into a mix of tight local heads and longer-range heads — similar qualitative story to Model H — though Model L's longest-range heads (max distance ≈10.45, Layer 3) are slightly less extreme than Model H's most sink-like head (distance 10.72, Layer 5), and unlike Model H, Model L's longest-range heads don't drop to as low an entropy (1.29–1.67 vs. Model H's 0.43–0.53), suggesting Model L's long-range heads are attending broadly across multiple positions rather than collapsing onto a single sink token as sharply as Model H's do. This is a reasonable difference to highlight in a Model H vs. Model L discussion, but should be re-verified once the checkpoint-provenance question above is resolved.
