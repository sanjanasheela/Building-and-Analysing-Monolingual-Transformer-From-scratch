"""
train_tokenizer.py — Tokenizer training, evaluation, model-selection & reporting
=================================================================================
Trains BPE tokenizers from scratch for a sweep of candidate
vocabulary sizes, using SEPARATE train / val / test splits:

    - train.txt  -> used ONLY to fit each tokenizer (vocab is never shared
                     across configs — every (algo, vocab_size) pair gets its
                     own tokenizer trained independently from scratch).
    - val.txt    -> used ONLY for model selection. Every trained config is
                     scored on val fertility + UNK-rate, and the best
                     (algo, vocab_size) combination is picked from this set.
    - test.txt   -> used ONLY ONCE, at the very end, to report final,
                     unbiased held-out metrics for the single model that
                     val-based selection chose. Never touched during
                     selection, so it gives an unbiased read of generalization.

For every trained model this script reports:
    1. vocabulary size stats
    2. token-frequency statistics on held-out text
    3. average characters per token
    4. tokenization examples (with fertility)
    5. UNK-token statistics (UNK rate, word/token fertility, OOV examples)

Outputs (under --out_dir):
    <algo>_vocab_<N>/tokenizer.json     - trained tokenizer (per config, own vocab)
    <algo>_vocab_<N>/vocab.json
    <algo>_vocab_<N>/config.json
    configs/config_<algo>_<N>.json      - copy of every config, flat
    all_models_summary.json             - machine-readable sweep comparison
    all_models_report.json              - machine-readable full metrics (val)
    report_<language>.txt               - HUMAN-READABLE report (appended to,
                                           one section per model + summary +
                                           selection rationale + final test score)
    plots/fertility_vs_vocab.png
    plots/unk_rate_vs_vocab.png
    plots/chars_per_token_vs_vocab.png
    plots/vocab_coverage_vs_vocab.png
    plots/selection_score_vs_vocab.png
"""

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, still saves PNGs fine
import matplotlib.pyplot as plt

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers as tnorm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOCAB_SIZES    = [5_000, 7_500, 10_000, 12_500]   # candidate vocab sizes to sweep
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
UNK_TOKEN      = "<unk>"

# Weight given to UNK-rate vs fertility when picking the "best" model on val.
# Both are min-max normalized across the sweep before combining, so these
# weights are directly comparable. Lower combined score = better.
UNK_WEIGHT       = 0.6
FERTILITY_WEIGHT = 0.4

EXAMPLE_SENTENCES = [
    "नेपाल विभिन्न संस्कृति र परम्पराहरूको देश हो।",
    "नेपाली भाषा नेपालको राष्ट्रभाषाको रूपमा प्रयोग हुन्छ।",
    "सगरमाथा विश्वको सबैभन्दा अग्लो हिमाल हो।",
    "प्रविधिले मानव जीवनलाई धेरै सजिलो बनाएको छ।",
    "काठमाडौं नेपालको राजधानी शहर हो।",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def jsonl_text_iterator(jsonl_path: Path, max_docs: int = None):
    """Yield text strings from a JSONL file, one document at a time."""
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    yield text
                    count += 1
                    if max_docs and count >= max_docs:
                        break
            except json.JSONDecodeError:
                continue


def txt_text_iterator(txt_path: Path, max_docs: int = None):
    """
    Yield text strings from a plain .txt file.
    Supports either "<label>\\t<text>" or raw line-per-document text.
    """
    count = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\r\n")
            if not raw.strip():
                continue
            text = raw.split("\t", 1)[1].strip() if "\t" in raw else raw.strip()
            if not text:
                continue
            yield text
            count += 1
            if max_docs and count >= max_docs:
                break


def make_iter_fn(path: Path, max_docs: int = None):
    """Return a zero-arg callable that yields text from `path`, dispatching on extension."""
    if path.suffix.lower() == ".jsonl":
        return lambda: jsonl_text_iterator(path, max_docs=max_docs)
    return lambda: txt_text_iterator(path, max_docs=max_docs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_tokenizer_from_iter(algo: str, vocab_size: int, train_iter_fn, out_dir: Path) -> Tokenizer:
    """
    Train a BPE tokenizer with given vocab_size on the TRAIN
    iterator only. Every call trains a brand-new tokenizer from scratch, so
    vocabularies are never shared across (algo, vocab_size) configs.
    Saves tokenizer.json, vocab.json, and config.json to
    out_dir/<algo>_vocab_<vocab_size>/.
    """
    subdir = out_dir / f"{algo.lower()}_vocab_{vocab_size}"
    subdir.mkdir(parents=True, exist_ok=True)

    if algo.lower() == "bpe":
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            min_frequency=2,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    tokenizer.normalizer    = tnorm.NFC()
    tokenizer.pre_tokenizer = Whitespace()

    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()}  |  vocab_size = {vocab_size:,}  (TRAIN split only)")
    print(f"{'='*60}")
    t0 = time.time()

    tokenizer.train_from_iterator(train_iter_fn(), trainer=trainer)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.2f} min")

    tokenizer.save(str(subdir / "tokenizer.json"))

    vocab = tokenizer.get_vocab()
    with open(subdir / "vocab.json", "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, ensure_ascii=False, indent=2)

    config = {
        "algorithm": algo.upper(),
        "vocab_size_target": vocab_size,
        "actual_vocab_size": len(vocab),
        "special_tokens": SPECIAL_TOKENS,
        "unk_token": UNK_TOKEN,
        "normalizer": "NFC",
        "pre_tokenizer": "Whitespace",
        "training_time_seconds": round(elapsed, 2),
    }
    with open(subdir / "config.json", "w", encoding="utf-8") as cf:
        json.dump(config, cf, indent=2)

    configs_global_dir = out_dir / "configs"
    configs_global_dir.mkdir(parents=True, exist_ok=True)
    with open(configs_global_dir / f"config_{algo.lower()}_{vocab_size}.json", "w", encoding="utf-8") as gcf:
        json.dump(config, gcf, indent=2)

    print(f"  Saved -> {subdir}/ (tokenizer.json, vocab.json, config.json)")
    return tokenizer


# ---------------------------------------------------------------------------
# Evaluation metrics (run on whichever split is passed in: val or test)
# ---------------------------------------------------------------------------

def vocab_size_stats(tokenizer: Tokenizer) -> dict:
    vocab = tokenizer.get_vocab()
    return {
        "actual_vocab_size":  len(vocab),
        "special_tokens":     SPECIAL_TOKENS,
        "num_special_tokens": len(SPECIAL_TOKENS),
    }


def token_frequency_stats(tokenizer: Tokenizer, split_iter_fn) -> dict:
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    freq   = collections.Counter()

    for text in split_iter_fn():
        ids = tokenizer.encode(text).ids
        for tid in ids:
            if tid != unk_id:
                freq[tid] += 1

    vocab        = tokenizer.get_vocab()
    id_to_token  = {v: k for k, v in vocab.items()}
    total_vocab  = len(vocab) - len(SPECIAL_TOKENS)
    seen_tokens  = len(freq)
    coverage_pct = seen_tokens / total_vocab * 100 if total_vocab else 0

    counts   = list(freq.values())
    mean_f   = statistics.mean(counts)   if counts else 0
    median_f = statistics.median(counts) if counts else 0
    std_f    = statistics.stdev(counts)  if len(counts) > 1 else 0

    top20 = [
        {"token": id_to_token.get(tid, f"<id:{tid}>"), "count": cnt}
        for tid, cnt in freq.most_common(20)
    ]
    bottom20 = [
        {"token": id_to_token.get(tid, f"<id:{tid}>"), "count": cnt}
        for tid, cnt in freq.most_common()[:-21:-1]
    ]

    return {
        "total_unique_tokens_seen": seen_tokens,
        "vocab_coverage_pct":      round(coverage_pct, 2),
        "mean_frequency":          round(mean_f, 2),
        "median_frequency":        round(median_f, 2),
        "std_frequency":           round(std_f, 2),
        "max_frequency":           max(counts) if counts else 0,
        "min_frequency":           min(counts) if counts else 0,
        "top_20_most_frequent":    top20,
        "bottom_20_least_frequent": bottom20,
    }


def avg_chars_per_token_stats(tokenizer: Tokenizer) -> dict:
    vocab         = tokenizer.get_vocab()
    special_set   = set(SPECIAL_TOKENS)
    token_lengths = [len(tok) for tok in vocab if tok not in special_set]

    mean_chars   = statistics.mean(token_lengths)   if token_lengths else 0
    median_chars = statistics.median(token_lengths) if token_lengths else 0
    std_chars    = statistics.stdev(token_lengths)  if len(token_lengths) > 1 else 0

    return {
        "avg_chars_per_token":    round(mean_chars, 4),
        "median_chars_per_token": round(median_chars, 4),
        "std_chars_per_token":    round(std_chars, 4),
        "max_chars_in_token":     max(token_lengths) if token_lengths else 0,
        "min_chars_in_token":     min(token_lengths) if token_lengths else 0,
    }


def tokenization_examples(tokenizer: Tokenizer) -> list:
    results = []
    for sent in EXAMPLE_SENTENCES:
        enc = tokenizer.encode(sent)
        results.append({
            "input":       sent,
            "tokens":      enc.tokens,
            "token_count": len(enc.tokens),
            "word_count":  len(sent.split()),
            "fertility":   round(len(enc.tokens) / max(len(sent.split()), 1), 4),
        })
    return results


def unk_token_stats(tokenizer: Tokenizer, split_iter_fn) -> dict:
    unk_id       = tokenizer.token_to_id(UNK_TOKEN)
    total_words  = 0
    total_tokens = 0
    unk_count    = 0
    docs_seen    = 0
    oov_examples = []

    for text in split_iter_fn():
        words         = text.split()
        total_words  += len(words)
        docs_seen    += 1

        enc            = tokenizer.encode(text)
        ids            = enc.ids
        total_tokens  += len(ids)
        unk_count     += ids.count(unk_id)

        if len(oov_examples) < 20:
            for word in words:
                w_enc = tokenizer.encode(word)
                if unk_id in w_enc.ids and word not in oov_examples:
                    oov_examples.append(word)
                    if len(oov_examples) >= 20:
                        break

    fertility = total_tokens / total_words if total_words else float("inf")
    unk_rate  = unk_count    / total_tokens if total_tokens else float("inf")

    return {
        "docs_evaluated":    docs_seen,
        "total_words":       total_words,
        "total_tokens":      total_tokens,
        "unk_count":         unk_count,
        "unk_rate_pct":      round(unk_rate * 100, 4),
        "fertility":         round(fertility, 4),
        "oov_word_examples": oov_examples,
    }


def full_report_from_iter(tokenizer: Tokenizer, split_iter_fn) -> dict:
    return {
        "vocab_size_stats":      vocab_size_stats(tokenizer),
        "token_frequency_stats": token_frequency_stats(tokenizer, split_iter_fn),
        "avg_chars_per_token":   avg_chars_per_token_stats(tokenizer),
        "tokenization_examples": tokenization_examples(tokenizer),
        "unk_token_stats":       unk_token_stats(tokenizer, split_iter_fn),
    }


# ---------------------------------------------------------------------------
# Model selection (val-based only — test is never used here)
# ---------------------------------------------------------------------------

def min_max_normalize(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def select_best_model(sweep_results: list) -> dict:
    """
    Pick the vocab_size config that minimizes a weighted, min-max
    normalized combination of val UNK-rate and val fertility.
    Lower UNK-rate = fewer unknown tokens (better vocabulary coverage).
    Lower fertility = fewer tokens per word (more efficient encoding).
    Both are normalized to [0, 1] across the sweep before combining, so
    the weights are directly interpretable.
    """
    unk_norm = min_max_normalize([r["unk_rate_pct"] for r in sweep_results])
    fert_norm = min_max_normalize([r["fertility"] for r in sweep_results])

    for r, u, f in zip(sweep_results, unk_norm, fert_norm):
        r["selection_score"] = round(UNK_WEIGHT * u + FERTILITY_WEIGHT * f, 6)

    best = min(sweep_results, key=lambda r: r["selection_score"])
    return best


# ---------------------------------------------------------------------------
# Human-readable text report
# ---------------------------------------------------------------------------

def format_model_section(model_key: str, split_name: str, report: dict) -> str:
    lines = []
    lines.append(f"\n{'-'*78}")
    lines.append(f"MODEL: {model_key}   (metrics computed on: {split_name})")
    lines.append(f"{'-'*78}")

    vs = report["vocab_size_stats"]
    lines.append(f"\n[1] Vocabulary size stats")
    lines.append(f"    actual_vocab_size  : {vs['actual_vocab_size']}")
    lines.append(f"    special_tokens     : {vs['special_tokens']}")

    tf = report["token_frequency_stats"]
    lines.append(f"\n[2] Token-frequency stats ({split_name})")
    lines.append(f"    unique_tokens_seen : {tf['total_unique_tokens_seen']}")
    lines.append(f"    vocab_coverage_pct : {tf['vocab_coverage_pct']}%")
    lines.append(f"    mean_frequency     : {tf['mean_frequency']}")
    lines.append(f"    median_frequency   : {tf['median_frequency']}")
    lines.append(f"    std_frequency      : {tf['std_frequency']}")
    lines.append(f"    max / min freq     : {tf['max_frequency']} / {tf['min_frequency']}")
    lines.append(f"    top_5_frequent     : {[t['token'] for t in tf['top_20_most_frequent'][:5]]}")
    lines.append(f"    bottom_5_frequent  : {[t['token'] for t in tf['bottom_20_least_frequent'][:5]]}")

    ct = report["avg_chars_per_token"]
    lines.append(f"\n[3] Average characters per token")
    lines.append(f"    avg_chars_per_token    : {ct['avg_chars_per_token']}")
    lines.append(f"    median_chars_per_token : {ct['median_chars_per_token']}")
    lines.append(f"    std_chars_per_token    : {ct['std_chars_per_token']}")
    lines.append(f"    max / min chars        : {ct['max_chars_in_token']} / {ct['min_chars_in_token']}")

    lines.append(f"\n[4] Tokenization examples")
    for ex in report["tokenization_examples"]:
        lines.append(f"    input      : {ex['input']}")
        lines.append(f"    tokens     : {ex['tokens']}")
        lines.append(f"    token_count/word_count/fertility : "
                      f"{ex['token_count']}/{ex['word_count']}/{ex['fertility']}")
        lines.append("")

    uk = report["unk_token_stats"]
    lines.append(f"[5] UNK-token statistics ({split_name})")
    lines.append(f"    docs_evaluated   : {uk['docs_evaluated']}")
    lines.append(f"    total_words      : {uk['total_words']}")
    lines.append(f"    total_tokens     : {uk['total_tokens']}")
    lines.append(f"    unk_count        : {uk['unk_count']}")
    lines.append(f"    unk_rate_pct     : {uk['unk_rate_pct']}%")
    lines.append(f"    fertility        : {uk['fertility']}")
    lines.append(f"    oov_word_examples: {uk['oov_word_examples']}")

    return "\n".join(lines)


def write_text_report(report_path: Path, language: str, sweep_results: list,
                       all_reports: dict, best: dict, test_report: dict = None):
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*78}\n")
        f.write(f"TOKENIZER SWEEP REPORT — Language: {language}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*78}\n")

        # Per-model sections (val-split metrics)
        for r in sweep_results:
            key = r["model_key"]
            f.write(format_model_section(key, "VAL split", all_reports[key]))
            f.write("\n")

        # Comparison summary table
        f.write(f"\n{'='*78}\n")
        f.write("SWEEP COMPARISON SUMMARY (val split)\n")
        f.write(f"{'='*78}\n")
        header = f"{'Model':<15} {'Algo':<8} {'Vocab':>8} {'Fertility':>10} {'UNK%':>8} {'Chars/Tok':>10} {'SelScore':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in sweep_results:
            f.write(
                f"{r['model_key']:<15} {r['algorithm']:<8} {r['vocab_size']:>8,} "
                f"{r['fertility']:>10.4f} {r['unk_rate_pct']:>8.4f} "
                f"{r['avg_chars_per_token']:>10.4f} {r['selection_score']:>10.4f}\n"
            )

        # Selection rationale
        f.write(f"\n{'='*78}\n")
        f.write("MODEL SELECTION (based on VAL fertility + UNK-rate only)\n")
        f.write(f"{'='*78}\n")
        f.write(f"Selection score = {UNK_WEIGHT} * norm(unk_rate_pct) + {FERTILITY_WEIGHT} * norm(fertility)\n")
        f.write(f"(min-max normalized across the sweep; lower score is better)\n\n")
        f.write(f"BEST MODEL: {best['model_key']}\n")
        f.write(f"  algorithm         : {best['algorithm']}\n")
        f.write(f"  vocab_size        : {best['vocab_size']}\n")
        f.write(f"  val fertility     : {best['fertility']}\n")
        f.write(f"  val unk_rate_pct  : {best['unk_rate_pct']}\n")
        f.write(f"  val selection_score: {best['selection_score']}\n")

        # Final, unbiased test-set metrics for the selected model only
        if test_report is not None:
            f.write(f"\n{'='*78}\n")
            f.write(f"FINAL HELD-OUT TEST RESULTS for selected model: {best['model_key']}\n")
            f.write("(test split was NOT used anywhere during model selection above)\n")
            f.write(f"{'='*78}\n")
            f.write(format_model_section(best["model_key"], "TEST split", test_report))
            f.write("\n")

    print(f"\nText report written/appended -> {report_path}")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def make_plots(sweep_results: list, plots_dir: Path, language: str):
    plots_dir.mkdir(parents=True, exist_ok=True)
    algos = sorted(set(r["algorithm"] for r in sweep_results))

    def series_for(algo, field):
        rows = sorted([r for r in sweep_results if r["algorithm"] == algo],
                      key=lambda r: r["vocab_size"])
        return [r["vocab_size"] for r in rows], [r[field] for r in rows]

    plot_specs = [
        ("fertility", "Fertility (tokens / word)", "fertility_vs_vocab.png",
         f"Fertility vs Vocabulary Size — {language} (val split)"),
        ("unk_rate_pct", "UNK rate (%)", "unk_rate_vs_vocab.png",
         f"UNK-token Rate vs Vocabulary Size — {language} (val split)"),
        ("avg_chars_per_token", "Avg characters / token", "chars_per_token_vs_vocab.png",
         f"Avg Characters per Token vs Vocabulary Size — {language}"),
        ("vocab_coverage_pct", "Vocab coverage (%)", "vocab_coverage_vs_vocab.png",
         f"Vocabulary Coverage vs Vocabulary Size — {language} (val split)"),
        ("selection_score", "Selection score (lower=better)", "selection_score_vs_vocab.png",
         f"Model-selection Score vs Vocabulary Size — {language}"),
    ]

    for field, ylabel, fname, title in plot_specs:
        plt.figure(figsize=(8, 5))
        for algo in algos:
            xs, ys = series_for(algo, field)
            plt.plot(xs, ys, marker="o", label=algo)
        plt.xlabel("Vocabulary size")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = plots_dir / fname
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokenizer training sweep (BPE) with proper train/val/test splits, "
                    "val-based model selection, text report, and visualizations."
    )
    parser.add_argument("--train", required=True, help="Path to TRAIN split (.txt or .jsonl)")
    parser.add_argument("--val",   required=True, help="Path to VAL split (.txt or .jsonl) — used for model selection")
    parser.add_argument("--test",  required=True, help="Path to TEST split (.txt or .jsonl) — used only for final reporting")
    parser.add_argument("--out_dir", default="./tokenizer_runs", help="Output directory for all artifacts")
    parser.add_argument("--algo", default="bpe", choices=["bpe"],
                        help="Algorithm to sweep (BPE only)")
    parser.add_argument("--language", default="Nepali", help="Language label used in report/plot titles")
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=VOCAB_SIZES,
                        help="Candidate vocab sizes to sweep")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path   = Path(args.val)
    test_path  = Path(args.test)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        assert p.exists(), f"{name} file not found: {p}"

    train_iter_fn = make_iter_fn(train_path)
    val_iter_fn   = make_iter_fn(val_path)
    test_iter_fn  = make_iter_fn(test_path)

    algos = ["bpe"]

    sweep_results = []
    all_reports   = {}
    tokenizers_by_key = {}

    for algo in algos:
        for vocab_size in args.vocab_sizes:
            # Train strictly on TRAIN split. Fresh tokenizer + fresh vocab every time.
            tokenizer = build_tokenizer_from_iter(algo, vocab_size, train_iter_fn, out_dir)
            tokenizers_by_key[f"{algo.upper()}_{vocab_size}"] = tokenizer

            # Evaluate strictly on VAL split (this is what drives model selection).
            report = full_report_from_iter(tokenizer, val_iter_fn)

            unk_stats = report["unk_token_stats"]
            key_name  = f"{algo.upper()}_{vocab_size}"

            sweep_results.append({
                "model_key":           key_name,
                "algorithm":           algo.upper(),
                "vocab_size":          vocab_size,
                "unk_rate_pct":        unk_stats["unk_rate_pct"],
                "fertility":           unk_stats["fertility"],
                "avg_chars_per_token": report["avg_chars_per_token"]["avg_chars_per_token"],
                "vocab_coverage_pct":  report["token_frequency_stats"]["vocab_coverage_pct"],
            })
            all_reports[key_name] = report

    # ---- Model selection using VAL fertility + UNK-rate only ----
    best = select_best_model(sweep_results)

    # ---- Final, unbiased evaluation of the selected model on TEST split ----
    best_tokenizer = tokenizers_by_key[best["model_key"]]
    test_report = full_report_from_iter(best_tokenizer, test_iter_fn)

    # ---- Console summary ----
    print(f"\n{'='*75}")
    print("  MULTI-MODEL TOKENIZER SWEEP COMPARISON SUMMARY (val split)")
    print(f"{'='*75}")
    print(f"  {'Model':<15}  {'Algo':<8}  {'Vocab':>8}  {'Fertility':>10}  {'UNK%':>8}  {'Chars/Tok':>10}  {'SelScore':>9}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*9}")
    for r in sweep_results:
        print(
            f"  {r['model_key']:<15}  {r['algorithm']:<8}  {r['vocab_size']:>8,}  "
            f"{r['fertility']:>10.4f}  {r['unk_rate_pct']:>8.4f}  "
            f"{r['avg_chars_per_token']:>10.4f}  {r['selection_score']:>9.4f}"
        )
    print(f"\n  BEST MODEL (val fertility + UNK-rate): {best['model_key']}")
    print(f"  Test-set fertility : {test_report['unk_token_stats']['fertility']}")
    print(f"  Test-set UNK rate  : {test_report['unk_token_stats']['unk_rate_pct']}%")

    # ---- Save machine-readable JSON outputs ----
    with open(out_dir / "all_models_summary.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "all_models_report.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    with open(out_dir / "best_model.json", "w", encoding="utf-8") as f:
        json.dump({"selection": best, "test_report": test_report}, f, indent=2, ensure_ascii=False)

    # ---- Append human-readable text report ----
    report_path = out_dir / f"report_{args.language}.txt"
    write_text_report(report_path, args.language, sweep_results, all_reports, best, test_report)

    # ---- Visualizations ----
    plots_dir = out_dir / "plots"
    make_plots(sweep_results, plots_dir, args.language)

    print("\nAll model training, evaluation, selection, reporting, and plotting complete!")
    print(f"  Report : {report_path}")
    print(f"  Plots  : {plots_dir}/")
    print(f"  Best   : {best['model_key']} (see best_model.json)")


if __name__ == "__main__":
    main()