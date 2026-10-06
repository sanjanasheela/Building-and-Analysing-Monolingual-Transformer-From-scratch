from __future__ import annotations
import sacrebleu
from rouge_score import rouge_scorer
from collections import Counter
from typing import Sequence


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def bleu(hypotheses: Sequence[str], references: Sequence[str], max_n: int = 4) -> dict:
    """Corpus BLEU using sacrebleu with proper tokenization for non-English text."""
    score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="none")
    return {
        "bleu": score.score,
        "precisions": [round(p, 6) for p in score.precisions],
        "brevity_penalty": round(score.bp, 6)
    }

def chrf(hypotheses: Sequence[str], references: Sequence[str],
         beta: float = 2.0, max_n: int = 6, word_order: int = 0) -> dict:
    """chrF / chrF++ using sacrebleu."""
    score = sacrebleu.corpus_chrf(hypotheses, [references], beta=beta, char_order=max_n, word_order=word_order)
    return {
        "chrf": round(score.score, 4),
        "precision": 0.0,
        "recall": 0.0
    }


def rouge_l(hypotheses: Sequence[str], references: Sequence[str], beta: float = 1.2) -> dict:
    """Corpus ROUGE-L using rouge-score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    tp = tr = tf = 0.0
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        r = scores['rougeL']
        tp += r.precision
        tr += r.recall
        tf += r.fmeasure
    n = max(len(hypotheses), 1)
    return {
        "rouge_l": round(tf / n * 100, 4),
        "precision": round(tp / n * 100, 4),
        "recall": round(tr / n * 100, 4)
    }


def repetition_rate(texts: Sequence[str], n: int = 3) -> float:
    """Fraction of n-grams that appear more than once."""
    total = repeated = 0
    for t in texts:
        counts = Counter(_ngrams(t.split(), n))
        total += sum(counts.values())
        repeated += sum(c for c in counts.values() if c > 1)
    return round(repeated / max(total, 1), 6)


def distinct_1(texts: Sequence[str]) -> float:
    """Unique unigrams / total unigrams."""
    toks = [w for t in texts for w in t.split()]
    return round(len(set(toks)) / max(len(toks), 1), 6)


def distinct_2(texts: Sequence[str]) -> float:
    """Unique bigrams / total bigrams."""
    bigs = [bg for t in texts for bg in _ngrams(t.split(), 2)]
    return round(len(set(bigs)) / max(len(bigs), 1), 6)


def run_all(hypotheses: Sequence[str], references: Sequence[str]) -> dict:
    """Compute all metrics and return a flat dict."""
    return {
        **{f"bleu_{k}": v for k, v in bleu(hypotheses, references).items()},
        **{f"chrf_{k}": v for k, v in chrf(hypotheses, references, word_order=0).items()},
        **{f"chrfpp_{k}": v for k, v in chrf(hypotheses, references, word_order=2).items()},
        **{f"rougel_{k}": v for k, v in rouge_l(hypotheses, references).items()},
        "repetition_rate_3": repetition_rate(hypotheses, n=3),
        "distinct_1": distinct_1(hypotheses),
        "distinct_2": distinct_2(hypotheses),
    }

    