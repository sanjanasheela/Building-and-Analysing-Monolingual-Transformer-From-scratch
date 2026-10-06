from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import math

import yaml
import torch
from tokenizers import Tokenizer
from sacrebleu import corpus_bleu, corpus_chrf
from rouge_score import rouge_scorer

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _EVAL_DIR.parent
sys.path.append(str(_PROJECT_DIR / "train"))
sys.path.append(str(_EVAL_DIR))

from model import build_from_config


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


CONFIG_PATH = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
    "language_H/configs/51.yaml"
)
CHECKPOINT_PATH = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
    "language_H/checkpoints/checkpoints/step_0045000.pt"
)
TOKENIZER_PATH = (
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
    "language_H/tokenizer_runs/bpe_vocab_7500/tokenizer.json"
)
TEST_TXT_PATH = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
    "language_H/data/final_data_set_7500/test.txt"
)
OUTPUT_DIR = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
    "language_H/report/generated_samples/test"
)
TABLES_DIR = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
    "language_H/report/tables/test"
)

NUM_SAMPLES = 250
MAX_NEW_TOKENS = 32
TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    return device


def build_model(cfg: dict, checkpoint_path: Path, device: torch.device):
    model = build_from_config(cfg).to(device)
    print(f"📦 Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get(
        "model_state", checkpoint.get("model_state_dict", checkpoint)
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def load_references_from_file(test_txt_path: Path, num_samples: int = NUM_SAMPLES) -> list[str]:
    with open(test_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:num_samples]


def generate_continuation(
    model,
    tokenizer: Tokenizer,
    reference: str,
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
    prompt_words: int = 10,
    temperature: float = 1.0,
) -> str:
    prompt_text = " ".join(reference.split()[:prompt_words])
    encoded = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs[:, -1, :] if isinstance(outputs, torch.Tensor) else outputs[0][:, -1, :]
            
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def generate_hypotheses(
    model, tokenizer: Tokenizer, references: list[str], device: torch.device, temperature: float
) -> list[str]:
    print(f"🔍 Generating continuations with Temperature={temperature}...")
    return [
        generate_continuation(model, tokenizer, ref, device, temperature=temperature)
        for ref in references
    ]


def save_generated_samples(output_dir: Path, references: list[str], hypotheses: list[str], temperature: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"samples_temp_{temperature}.txt"
    
    with open(combined_path, "w", encoding="utf-8") as combined_f:
        for i, (ref, hyp) in enumerate(zip(references, hypotheses), start=1):
            block = (
                f"--- Sample {i} (Temp: {temperature}) ---\n"
                f"[REFERENCE]\n{ref}\n\n"
                f"[GENERATED]\n{hyp}\n\n"
            )
            combined_f.write(block)
    print(f"✍️ Saved samples for Temp={temperature} to {combined_path}")


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


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


def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    # BLEU
    bleu_score_obj = corpus_bleu(hypotheses, [references], tokenize="none")
    bleu_dict = {
        "bleu_bleu": bleu_score_obj.score,
        "bleu_precisions": [round(p, 6) for p in bleu_score_obj.precisions],
        "bleu_brevity_penalty": round(bleu_score_obj.bp, 6),
    }

    # chrF (word_order=0)
    chrf_obj = corpus_chrf(hypotheses, [references], word_order=0)
    chrf_dict = {
        "chrf_chrf": round(chrf_obj.score, 4),
        "chrf_precision": 0.0,
        "chrf_recall": 0.0,
    }

    # chrF++ (word_order=2)
    chrfpp_obj = corpus_chrf(hypotheses, [references], word_order=2)
    chrfpp_dict = {
        "chrfpp_chrf": round(chrfpp_obj.score, 4),
        "chrfpp_precision": 0.0,
        "chrfpp_recall": 0.0,
    }

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=WhitespaceTokenizer())
    tp = tr = tf = 0.0
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        r = scores['rougeL']
        tp += r.precision
        tr += r.recall
        tf += r.fmeasure
    n = max(len(hypotheses), 1)
    rouge_dict = {
        "rougel_rouge_l": round(tf / n * 100, 4),
        "rougel_precision": round(tp / n * 100, 4),
        "rougel_recall": round(tr / n * 100, 4),
    }

    # Diversity & Repetition
    misc_dict = {
        "repetition_rate_3": repetition_rate(hypotheses, n=3),
        "distinct_1": distinct_1(hypotheses),
        "distinct_2": distinct_2(hypotheses),
    }

    return {
        **bleu_dict,
        **chrf_dict,
        **chrfpp_dict,
        **rouge_dict,
        **misc_dict,
    }


def save_all_metrics_txt(tables_dir: Path, all_results: dict) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "all_temperatures_metrics.txt"
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("========================================================\n")
        f.write("COMPREHENSIVE METRICS & DIAGNOSTICS ACROSS TEMPS\n")
        f.write("========================================================\n\n")
        
        for temp, metrics in all_results.items():
            f.write(f"--- Temperature: {temp} ---\n")
            for metric_name, val in metrics.items():
                f.write(f"  {metric_name}: {val}\n")
            f.write("\n" + "-" * 40 + "\n\n")
            
    print(f"📊 Saved all detailed metrics report to {metrics_path}")


def main():
    cfg = load_config(CONFIG_PATH)
    device = get_device()
    model = build_model(cfg, CHECKPOINT_PATH, device)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    references = load_references_from_file(TEST_TXT_PATH, NUM_SAMPLES)

    print(f"\n🚀 Running evaluation across temperatures: {TEMPERATURES}\n")
    
    all_results = {}
    for temp in TEMPERATURES:
        print(f"========================================")
        print(f"Evaluating Temperature: {temp}")
        print(f"========================================")
        
        hypotheses = generate_hypotheses(model, tokenizer, references, device, temperature=temp)
        save_generated_samples(OUTPUT_DIR, references, hypotheses, temperature=temp)
        
        metrics = compute_metrics(hypotheses, references)
        all_results[temp] = metrics
        
        print(f"📊 Results for Temp={temp}:")
        for k, v in metrics.items():
            print(f"  - {k} : {v}")
        print()

    save_all_metrics_txt(TABLES_DIR, all_results)

    print("\n✅ All temperature loop evaluations and metric reports completed successfully!")


if __name__ == "__main__":
    main()