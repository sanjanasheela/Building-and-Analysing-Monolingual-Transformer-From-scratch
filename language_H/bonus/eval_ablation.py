"""
eval_ablation.py — Comprehensive Phase 2 Evaluation Suite for Ablated Model
=============================================================================
Runs the full Phase 2 evaluation on the ablated Model H (no positional embeddings):
  1. Language Modeling Metrics:
     - Cross-Entropy Loss on test set
     - Test Perplexity (PPL)
     - Test Bits-Per-Byte (BPB)
  2. Generation Quality & Diagnostics across temperatures:
     - T in [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
     - BLEU-4, chrF, chrF++, ROUGE-L
     - Repetition-rate-3, Distinct-1, Distinct-2
     - Outputs saved to generated_samples/
  3. Attention Analysis:
     - Attention weight extraction across all layers and heads
     - Per-head Shannon entropy
     - Per-head mean attention distance
     - Heatmap visualizations for Layer 0 and Layer 5 (all 8 heads)
  4. Comparison with Standard Baseline Model:
     - Formats comparison tables comparing Ablated Model vs Standard Model H
     - Exports all artifacts to language_H/report/bonus/
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

try:
    from sacrebleu import corpus_bleu, corpus_chrf
except ImportError:
    corpus_bleu, corpus_chrf = None, None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# Paths bootstrap
_BONUS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BONUS_DIR.parent

for _p in (str(_PROJECT_DIR / "eval"), str(_PROJECT_DIR / "model"), str(_PROJECT_DIR / "train"), str(_PROJECT_DIR), str(_BONUS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if sys.path[0] != str(_BONUS_DIR):
    sys.path.insert(0, str(_BONUS_DIR))

from ablation_model import AblatedCausalLanguageModel, build_from_config
from train_ablation import ShardDataset, SingleBinBatchLoader


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


# ---------------------------------------------------------------------------
# Path Helper
# ---------------------------------------------------------------------------
def resolve_path(p: str | Path, base_dir: Path) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    for cand in [base_dir / p, _BONUS_DIR / p, _PROJECT_DIR / p]:
        if cand.exists():
            return cand
    return base_dir / p


# ---------------------------------------------------------------------------
# Intrinsic LM Metrics: CE, PPL, BPB
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_intrinsic(
    model: nn.Module,
    test_bin_path: Path,
    test_txt_path: Path,
    context_length: int,
    eval_batch_size: int,
    n_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    """Computes test Cross-Entropy, Perplexity, and Bits-Per-Byte."""
    model.eval()
    test_loader = SingleBinBatchLoader(test_bin_path, context_length, eval_batch_size, device)

    total_loss = 0.0
    total_tokens = 0

    print(f"🔍 Computing test cross-entropy across {n_batches} batches...")
    for _ in range(n_batches):
        x, y = test_loader.next_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    # BPB calculation
    bpb = 0.0
    if test_txt_path.exists():
        with open(test_txt_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        total_bytes = len(text_data.encode("utf-8"))
        if total_bytes > 0:
            total_bits = (avg_loss * total_tokens) / math.log(2.0)
            bpb = total_bits / total_bytes

    return {
        "test_loss": round(avg_loss, 4),
        "test_ppl": round(ppl, 4),
        "test_bpb": round(bpb, 4),
    }


# ---------------------------------------------------------------------------
# Generation Quality & Diagnostics
# ---------------------------------------------------------------------------
def generate_continuation(
    model: nn.Module,
    tokenizer: Tokenizer,
    reference: str,
    device: torch.device,
    max_new_tokens: int = 32,
    prompt_words: int = 3,
    temperature: float = 1.0,
) -> str:
    prompt_text = " ".join(reference.split()[:prompt_words])
    encoded = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            last_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = last_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def repetition_rate(texts: list[str], n: int = 3) -> float:
    total = repeated = 0
    for t in texts:
        counts = Counter(_ngrams(t.split(), n))
        total += sum(counts.values())
        repeated += sum(c for c in counts.values() if c > 1)
    return round(repeated / max(total, 1), 6)


def distinct_1(texts: list[str]) -> float:
    toks = [w for t in texts for w in t.split()]
    return round(len(set(toks)) / max(len(toks), 1), 6)


def distinct_2(texts: list[str]) -> float:
    bigs = [bg for t in texts for bg in _ngrams(t.split(), 2)]
    return round(len(set(bigs)) / max(len(bigs), 1), 6)


def compute_generation_metrics(hypotheses: list[str], references: list[str]) -> dict:
    results = {}

    # BLEU
    if corpus_bleu is not None:
        bleu_score_obj = corpus_bleu(hypotheses, [references], tokenize="none")
        results["bleu_bleu"] = round(bleu_score_obj.score, 4)
        results["bleu_precisions"] = [round(p, 4) for p in bleu_score_obj.precisions]
        results["bleu_brevity_penalty"] = round(bleu_score_obj.bp, 4)
    else:
        results["bleu_bleu"] = 0.0

    # chrF (word_order=0) and chrF++ (word_order=2)
    if corpus_chrf is not None:
        chrf_obj = corpus_chrf(hypotheses, [references], word_order=0)
        results["chrf"] = round(chrf_obj.score, 4)
        chrfpp_obj = corpus_chrf(hypotheses, [references], word_order=2)
        results["chrfpp"] = round(chrfpp_obj.score, 4)
    else:
        results["chrf"] = 0.0
        results["chrfpp"] = 0.0

    # ROUGE-L
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=WhitespaceTokenizer())
        tp = tr = tf = 0.0
        for hyp, ref in zip(hypotheses, references):
            scores = scorer.score(ref, hyp)
            r = scores['rougeL']
            tp += r.precision
            tr += r.recall
            tf += r.fmeasure
        n = max(len(hypotheses), 1)
        results["rougel"] = round(tf / n * 100, 4)
    else:
        results["rougel"] = 0.0

    # Diversity diagnostics
    results["repetition_rate_3"] = repetition_rate(hypotheses, n=3)
    results["distinct_1"] = distinct_1(hypotheses)
    results["distinct_2"] = distinct_2(hypotheses)

    return results


# ---------------------------------------------------------------------------
# Attention Diagnostics: Entropy, Mean Distance, Heatmaps
# ---------------------------------------------------------------------------
def compute_attention_entropy(attn_weights: list[torch.Tensor]) -> torch.Tensor:
    """Shannon entropy: -sum(p * log2(p)). Returns (L, H, T)."""
    entropies = []
    for w in attn_weights:
        w = w.squeeze(0)  # (H, T, T)
        p = w.clamp(min=1e-9)
        ent = -(p * p.log2()).sum(dim=-1)  # (H, T)
        entropies.append(ent)
    return torch.stack(entropies)


def compute_mean_attention_distance(attn_weights: list[torch.Tensor]) -> torch.Tensor:
    """Mean query-key token distance: sum(p_ij * |i - j|). Returns (L, H, T)."""
    distances = []
    for w in attn_weights:
        w = w.squeeze(0)  # (H, T, T)
        _, T, _ = w.shape
        pos = torch.arange(T, dtype=torch.float, device=w.device)
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # (T, T)
        mean_dist = (w * dist.unsqueeze(0)).sum(dim=-1)  # (H, T)
        distances.append(mean_dist)
    return torch.stack(distances)


def plot_attention_heatmap(
    w: np.ndarray,
    layer: int,
    head: int,
    save_path: Path,
    title: str = "",
):
    if not MATPLOTLIB_OK:
        return
    T = w.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w, cmap="Purples", vmin=0, vmax=max(w.max(), 1e-4))
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title or f"Ablation Layer {layer} Head {head}")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Ablated Model (No Positional Embeddings)")
    parser.add_argument("--config", type=str, default=str(_BONUS_DIR / "config.yaml"), help="Config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint step_*.pt")
    parser.add_argument("--report_dir", type=str, default=None, help="Report directory")
    parser.add_argument("--num_samples", type=int, default=250, help="Number of generation samples")
    parser.add_argument("--n_eval_batches", type=int, default=50, help="Number of batches for test CE/BPB")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Build model
    model = build_from_config(cfg).to(device)

    # Locate checkpoint
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_dirs = [
            _BONUS_DIR / "checkpoints",
            _BONUS_DIR / "checkpoints" / "checkpoints",
            _PROJECT_DIR / "checkpoints" / "checkpoints",
        ]
        found_files = []
        for d in ckpt_dirs:
            if d.exists():
                found_files.extend(list(d.glob("step_*.pt")))
        if found_files:
            def get_step(p):
                m = re.search(r"step_(\d+)\.pt", p.name)
                return int(m.group(1)) if m else -1
            found_files.sort(key=get_step)
            ckpt_path = found_files[-1]

    if ckpt_path and ckpt_path.exists():
        print(f"📦 Loading weights from checkpoint: {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ Warning: No checkpoint found. Running evaluation on freshly initialized model.")

    model.eval()

    # Output directories
    report_base = Path(args.report_dir) if args.report_dir else (_PROJECT_DIR / "report" / "bonus")
    tables_dir = report_base / "tables"
    samples_dir = report_base / "generated_samples"
    attention_dir = report_base / "attention"
    loss_curves_dir = report_base / "loss_curves"

    for d in [tables_dir, samples_dir, attention_dir, loss_curves_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Resolve Data paths
    test_bin = resolve_path(cfg["data"]["test_bin"], _PROJECT_DIR)
    test_txt = resolve_path(cfg["data"]["test_txt"], _PROJECT_DIR)
    tok_path = resolve_path(cfg["tokenizer"]["tokenizer_path"], _PROJECT_DIR)

    print(f"📁 Test Bin       : {test_bin}")
    print(f"📁 Test Text      : {test_txt}")
    print(f"🔤 Tokenizer JSON : {tok_path}")

    # =========================================================================
    # 1. Intrinsic Language Modeling Metrics (CE, PPL, BPB)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: INTRINSIC LANGUAGE MODELING EVALUATION (Held-out Test)")
    print("=" * 60)
    intrinsic_results = evaluate_intrinsic(
        model=model,
        test_bin_path=test_bin,
        test_txt_path=test_txt,
        context_length=cfg["model"]["context_length"],
        eval_batch_size=cfg["evaluation"].get("eval_batch_size", 16),
        n_batches=args.n_eval_batches,
        device=device,
    )
    print(f"📉 Test Cross-Entropy Loss : {intrinsic_results['test_loss']:.4f}")
    print(f"📉 Test Perplexity (PPL)   : {intrinsic_results['test_ppl']:.4f}")
    print(f"📉 Test Bits-Per-Byte (BPB): {intrinsic_results['test_bpb']:.4f}")

    # Save final_evaluation_summary.txt
    summary_path = tables_dir / "final_evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========================================================\n")
        f.write("MODEL H ABLATION (NO POSITIONAL EMBEDDINGS) — FINAL SUMMARY\n")
        f.write("========================================================\n\n")
        f.write(f"Test Cross-Entropy Loss : {intrinsic_results['test_loss']:.4f}\n")
        f.write(f"Test Perplexity (PPL)   : {intrinsic_results['test_ppl']:.4f}\n")
        f.write(f"Test Bits-Per-Byte (BPB): {intrinsic_results['test_bpb']:.4f}\n\n")
        f.write("--------------------------------------------------------\n")
        f.write("ABLATION CONFIGURATION NOTE:\n")
        f.write("- Positional Encodings: REMOVED (Both RoPE and learned embeddings disabled)\n")
        f.write("- Self-Attention operates on semantic token similarity with causal mask only\n")
        f.write("========================================================\n")
    print(f"📄 Summary written to: {summary_path}")

    # =========================================================================
    # 2. Generation Metrics Across Temperatures
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: GENERATION EVALUATION ACROSS TEMPERATURES")
    print("=" * 60)
    tokenizer = Tokenizer.from_file(str(tok_path))

    with open(test_txt, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]
    references = all_lines[: args.num_samples]
    print(f"Loaded {len(references)} references for generation testing.")

    temperatures = cfg["evaluation"].get("temperatures", [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5])
    all_gen_metrics = {}

    for temp in temperatures:
        print(f"\n--- Evaluating Temperature: {temp} ---")
        hypotheses = [
            generate_continuation(
                model=model,
                tokenizer=tokenizer,
                reference=ref,
                device=device,
                max_new_tokens=cfg["evaluation"].get("max_generation_length", 32),
                temperature=temp,
            )
            for ref in references
        ]

        # Save sample outputs
        sample_file = samples_dir / f"samples_temp_{temp}.txt"
        with open(sample_file, "w", encoding="utf-8") as sf:
            for i, (ref, hyp) in enumerate(zip(references, hypotheses), start=1):
                sf.write(f"--- Sample {i} (Temp: {temp}) ---\n")
                sf.write(f"[REFERENCE]\n{ref}\n\n")
                sf.write(f"[GENERATED]\n{hyp}\n\n")

        metrics = compute_generation_metrics(hypotheses, references)
        all_gen_metrics[temp] = metrics
        print(f"  BLEU-4   : {metrics.get('bleu_bleu', 0.0):.4f}")
        print(f"  chrF     : {metrics.get('chrf', 0.0):.4f}")
        print(f"  chrF++   : {metrics.get('chrfpp', 0.0):.4f}")
        print(f"  ROUGE-L  : {metrics.get('rougel', 0.0):.4f}")
        print(f"  Rep-3    : {metrics.get('repetition_rate_3', 0.0):.4f}")
        print(f"  Dist-1   : {metrics.get('distinct_1', 0.0):.4f}")
        print(f"  Dist-2   : {metrics.get('distinct_2', 0.0):.4f}")

    # Save all_temperatures_metrics.txt
    metrics_path = tables_dir / "all_temperatures_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        mf.write("========================================================\n")
        mf.write("ABLATION: COMPREHENSIVE GENERATION METRICS ACROSS TEMPS\n")
        mf.write("========================================================\n\n")
        for temp, mets in all_gen_metrics.items():
            mf.write(f"--- Temperature: {temp} ---\n")
            for k, v in mets.items():
                mf.write(f"  {k}: {v}\n")
            mf.write("\n" + "-" * 40 + "\n\n")
    print(f"📄 Metrics written to: {metrics_path}")

    # =========================================================================
    # 3. Attention Diagnostics (Entropy, Mean Distance, Heatmaps)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: ATTENTION ANALYSIS & HEATMAP VISUALIZATIONS")
    print("=" * 60)

    # Pick 24 representative tokens from test set
    context_len = 24
    if test_bin.exists():
        with open(test_bin, "rb") as bf:
            raw_toks = np.frombuffer(bf.read(context_len * 4), dtype=np.uint32)
        sample_tensor = torch.tensor(raw_toks[:context_len].astype(np.int64), dtype=torch.long, device=device).unsqueeze(0)
    else:
        sample_tensor = torch.randint(0, model.vocab_size, (1, context_len), dtype=torch.long, device=device)

    print("Extracting attention weights across all 6 layers...")
    attn_weights = model.extract_attention(sample_tensor)  # list of (1, H, T, T)

    entropies = compute_attention_entropy(attn_weights)  # (L, H, T)
    distances = compute_mean_attention_distance(attn_weights)  # (L, H, T)

    mean_entropies = entropies.mean(dim=-1).tolist()  # (L, H)
    mean_distances = distances.mean(dim=-1).tolist()  # (L, H)

    # Save summary stats
    summary_stats = {
        "mean_entropy_per_layer": [float(np.mean(row)) for row in mean_entropies],
        "entropy_layer_head_matrix": mean_entropies,
        "mean_distance_per_layer": [float(np.mean(row)) for row in mean_distances],
        "distance_layer_head_matrix": mean_distances,
    }
    stats_json_path = attention_dir / "attention_summary_stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as jf:
        json.dump(summary_stats, jf, indent=4)
    print(f"📊 Attention summary statistics saved to: {stats_json_path}")

    # Generate heatmaps for Layer 0 and Layer 5 (all 8 heads)
    layers_to_plot = [0, model.num_layers - 1]
    for l_idx in layers_to_plot:
        w_l = attn_weights[l_idx][0].numpy()  # (H, T, T)
        H = w_l.shape[0]
        for h_idx in range(H):
            save_file = attention_dir / f"heatmap_layer_{l_idx}_head_{h_idx}.png"
            plot_attention_heatmap(
                w_l[h_idx],
                layer=l_idx,
                head=h_idx,
                save_path=save_file,
                title=f"Ablated Model (No Pos Emb) — Layer {l_idx} Head {h_idx}",
            )
    print(f"🖼️ All 16 heatmaps saved to: {attention_dir}")

    # =========================================================================
    # 4. Comparative Baseline vs Ablation Report
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: COMPARATIVE ANALYSIS (BASELINE MODEL H vs ABLATED MODEL)")
    print("=" * 60)

    # Baseline Model H numbers (from Phase 2 evaluation report)
    base_ce = 3.7361
    base_ppl = 41.9356
    base_bpb = 0.0152

    base_gen = {
        0.0: {"bleu": 0.7501, "chrf": 17.66, "chrfpp": 15.24, "rougel": 7.20, "rep": 0.5509, "d1": 0.2000, "d2": 0.4292},
        0.2: {"bleu": 0.7591, "chrf": 18.43, "chrfpp": 15.84, "rougel": 7.34, "rep": 0.4218, "d1": 0.2145, "d2": 0.4955},
        0.5: {"bleu": 0.6284, "chrf": 19.12, "chrfpp": 16.38, "rougel": 7.31, "rep": 0.1908, "d1": 0.2646, "d2": 0.6603},
        0.8: {"bleu": 0.7253, "chrf": 18.82, "chrfpp": 15.98, "rougel": 6.78, "rep": 0.0438, "d1": 0.3259, "d2": 0.8518},
        1.0: {"bleu": 0.6494, "chrf": 18.83, "chrfpp": 15.84, "rougel": 6.32, "rep": 0.0066, "d1": 0.3609, "d2": 0.9227},
        1.2: {"bleu": 0.6441, "chrf": 18.52, "chrfpp": 15.49, "rougel": 6.17, "rep": 0.0023, "d1": 0.3873, "d2": 0.9560},
        1.5: {"bleu": 0.6183, "chrf": 18.33, "chrfpp": 15.07, "rougel": 5.80, "rep": 0.0011, "d1": 0.4198, "d2": 0.9812},
    }

    base_layer_entropy = [3.228, 1.703, 1.805, 1.737, 1.548, 1.255]
    base_layer_distance = [5.650, 3.157, 4.103, 5.891, 5.142, 7.731]

    comp_file = tables_dir / "comparison_with_baseline.txt"
    with open(comp_file, "w", encoding="utf-8") as cf:
        cf.write("=========================================================================================\n")
        cf.write("MODEL H (STANDARD RoPE) vs ABLATED MODEL H (NO POSITIONAL EMBEDDINGS) COMPARISON\n")
        cf.write("=========================================================================================\n\n")

        cf.write("1. INTRINSIC LANGUAGE MODELING METRICS\n")
        cf.write("-----------------------------------------------------------------------------------------\n")
        cf.write(f"{'Metric':<25} | {'Standard Model H (RoPE)':<25} | {'Ablated Model H (No Pos)':<25}\n")
        cf.write(f"{'-'*25}-|-{'-'*25}-|-{'-'*25}\n")
        cf.write(f"{'Cross-Entropy Loss':<25} | {base_ce:<25.4f} | {intrinsic_results['test_loss']:<25.4f}\n")
        cf.write(f"{'Perplexity (PPL)':<25} | {base_ppl:<25.4f} | {intrinsic_results['test_ppl']:<25.4f}\n")
        cf.write(f"{'Bits-Per-Byte (BPB)':<25} | {base_bpb:<25.4f} | {intrinsic_results['test_bpb']:<25.4f}\n\n")

        cf.write("2. GENERATION QUALITY & DIAGNOSTICS (AT REPRESENTATIVE TEMPERATURES)\n")
        cf.write("-----------------------------------------------------------------------------------------\n")
        cf.write(f"{'Temp':<6} | {'Metric':<10} | {'Standard Model H':<20} | {'Ablated Model H':<20}\n")
        cf.write(f"{'-'*6}-|-{'-'*10}-|-{'-'*20}-|-{'-'*20}\n")
        for t in [0.0, 0.5, 1.0]:
            if t in all_gen_metrics:
                m_abl = all_gen_metrics[t]
                m_std = base_gen[t]
                cf.write(f"{t:<6.1f} | {'BLEU-4':<10} | {m_std['bleu']:<20.4f} | {m_abl.get('bleu_bleu', 0.0):<20.4f}\n")
                cf.write(f"{t:<6.1f} | {'chrF':<10} | {m_std['chrf']:<20.4f} | {m_abl.get('chrf', 0.0):<20.4f}\n")
                cf.write(f"{t:<6.1f} | {'chrF++':<10} | {m_std['chrfpp']:<20.4f} | {m_abl.get('chrfpp', 0.0):<20.4f}\n")
                cf.write(f"{t:<6.1f} | {'ROUGE-L':<10} | {m_std['rougel']:<20.4f} | {m_abl.get('rougel', 0.0):<20.4f}\n")
                cf.write(f"{t:<6.1f} | {'Rep-3':<10} | {m_std['rep']:<20.4f} | {m_abl.get('repetition_rate_3', 0.0):<20.4f}\n")
                cf.write(f"{t:<6.1f} | {'Dist-1':<10} | {m_std['d1']:<20.4f} | {m_abl.get('distinct_1', 0.0):<20.4f}\n")
                cf.write(f"{t:<6.1f} | {'Dist-2':<10} | {m_std['d2']:<20.4f} | {m_abl.get('distinct_2', 0.0):<20.4f}\n")
                cf.write(f"{'-'*6}-|-{'-'*10}-|-{'-'*20}-|-{'-'*20}\n")

        cf.write("\n3. ATTENTION ENTROPY & DISTANCE SUMMARY BY LAYER\n")
        cf.write("-----------------------------------------------------------------------------------------\n")
        cf.write(f"{'Layer':<6} | {'Std Mean Entropy':<20} | {'Abl Mean Entropy':<20} | {'Std Mean Dist':<16} | {'Abl Mean Dist':<16}\n")
        cf.write(f"{'-'*6}-|-{'-'*20}-|-{'-'*20}-|-{'-'*16}-|-{'-'*16}\n")
        for l in range(len(mean_entropies)):
            std_ent = base_layer_entropy[l] if l < len(base_layer_entropy) else 0.0
            std_dst = base_layer_distance[l] if l < len(base_layer_distance) else 0.0
            abl_ent = float(np.mean(mean_entropies[l]))
            abl_dst = float(np.mean(mean_distances[l]))
            cf.write(f"{l:<6d} | {std_ent:<20.3f} | {abl_ent:<20.3f} | {std_dst:<16.3f} | {abl_dst:<16.3f}\n")

    print(f"📄 Full baseline comparison table written to: {comp_file}")
    print("\n✅ Full Phase 2 evaluation on Ablated Model H completed successfully!")


if __name__ == "__main__":
    main()
