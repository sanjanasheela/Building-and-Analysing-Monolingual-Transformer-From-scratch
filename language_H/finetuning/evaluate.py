"""
Evaluate pretrained vs finetuned checkpoints on the synthetic reasoning set.

Metrics: answer-token NLL / perplexity, exact-match of the gold `answer_option`
against greedy generations, plus per-category and per-hop breakdowns.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

_FINETUNE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _FINETUNE_DIR.parent
for _p in (str(_FINETUNE_DIR), str(_PROJECT_DIR / "train"), str(_PROJECT_DIR / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import build_from_config
from template import format_prompt
from train_finetune import (
    PAD_ID,
    EOS_ID,
    _resolve,
    find_jsonl,
    load_yaml,
    merge_finetune_config,
    sequence_nll,
)

STOP_STRINGS = ("\n", "ప్రశ్న:")


def _load_samples(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = 0
    items = []
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        obj, end = decoder.raw_decode(content, pos)
        items.append(obj)
        pos = end
    return items


def _load_cfg(config_path: str | Path) -> dict:
    ft_cfg = load_yaml(config_path)
    base = ft_cfg.get("base_config")
    if base:
        return merge_finetune_config(load_yaml(base), ft_cfg)
    return ft_cfg


def load_model_for_eval(ckpt_path: Path, cfg: dict, device: torch.device):
    model = build_from_config(cfg).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def greedy_answer(
    model,
    tokenizer: Tokenizer,
    user_text: str,
    device: torch.device,
    max_new_tokens: int = 64,
) -> str:
    """Greedy-decode the answer continuation after the Telugu prompt prefix."""
    prompt = format_prompt(user_text) + " "
    prompt_ids = tokenizer.encode(prompt).ids
    ctx = model.context_length
    generated: list[int] = []
    input_ids = torch.tensor([prompt_ids[-ctx:]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        if next_id in (EOS_ID, PAD_ID):
            break
        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )
        if input_ids.size(1) > ctx:
            input_ids = input_ids[:, -ctx:]
        if len(generated) >= 8 and generated[-4:] == generated[-8:-4]:
            generated = generated[:-4]
            break
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        if any(s in gen_text for s in STOP_STRINGS):
            break
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    return text


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def answer_is_correct(generated: str, gold_option: str, gold_full: str) -> bool:
    """Exact match on the gold option; also accept a full-string match."""
    gen = _normalize(generated).rstrip("।.")
    opt = _normalize(str(gold_option)).rstrip("।.")
    full = _normalize(gold_full).rstrip("।.")
    if not gen or not opt:
        return False
    if gen == opt or gen == full:
        return True
    marker = f"సమాధానం {opt}"
    if marker in gen:
        return True
    gen_c, opt_c, full_c = _compact(gen), _compact(opt), _compact(full)
    if gen_c == opt_c or gen_c == full_c:
        return True
    if f"సమాధానం{opt_c}" in gen_c:
        return True
    # Names can be split across BPE pieces with spaces; require a long option
    # so numeric answers like 44 do not match 144.
    if not opt_c.isdigit() and len(opt_c) >= 3 and opt_c in gen_c:
        return True
    if "సమాధానం" in gen:
        after = _compact(gen.split("సమాధానం", 1)[1].strip().rstrip("।."))
        return after == opt_c
    return False


def evaluate_model(
    val_file: str | Path,
    ckpt_path: str | Path,
    cfg: dict | None = None,
    config_path: str | Path | None = None,
    max_samples: int | None = None,
    dump_predictions: Path | None = None,
) -> dict:
    if cfg is None:
        if config_path is None:
            raise ValueError("Provide cfg or config_path")
        cfg = _load_cfg(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve(ckpt_path)
    val_path = find_jsonl(val_file)
    samples = _load_samples(val_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    tokenizer = Tokenizer.from_file(str(_resolve(cfg["tokenizer"]["tokenizer_path"])))
    model = load_model_for_eval(ckpt_path, cfg, device)

    print(f"[Evaluator] {ckpt_path.name} on {val_path.name} (n={len(samples)})")
    val_loss, val_ppl = sequence_nll(model, samples, tokenizer, cfg, device)

    correct = 0
    category_stats: dict[str, dict[str, int]] = {}
    hop_stats: dict[int, dict[str, int]] = {}
    predictions = []
    max_new = int(cfg.get("finetuning", {}).get("max_new_tokens", 64))

    for sample in samples:
        user = sample["messages"][0]["content"]
        gold = sample["messages"][1]["content"]
        option = sample["answer_option"]
        generated = greedy_answer(model, tokenizer, user, device, max_new_tokens=max_new)
        ok = answer_is_correct(generated, option, gold)
        correct += int(ok)
        cat = sample["category"]
        hop = sample["hop_count"]
        category_stats.setdefault(cat, {"correct": 0, "total": 0})
        category_stats[cat]["total"] += 1
        category_stats[cat]["correct"] += int(ok)
        hop_stats.setdefault(hop, {"correct": 0, "total": 0})
        hop_stats[hop]["total"] += 1
        hop_stats[hop]["correct"] += int(ok)
        predictions.append(
            {
                "question": user,
                "gold": gold,
                "answer_option": option,
                "generated": generated,
                "correct": ok,
                "category": cat,
                "hop_count": hop,
            }
        )

    n = max(len(samples), 1)
    overall = 100.0 * correct / n
    print(
        f"[Evaluator] exact_match={overall:.2f}% ({correct}/{len(samples)}) | "
        f"val_nll={val_loss:.4f} | val_ppl={val_ppl:.2f}"
    )
    for cat, st in sorted(category_stats.items()):
        acc = 100.0 * st["correct"] / max(st["total"], 1)
        print(f"    {cat}: {acc:.1f}% ({st['correct']}/{st['total']})")

    if dump_predictions is not None:
        dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        dump_predictions.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {
        "overall_accuracy": overall,
        "n_correct": correct,
        "n_total": len(samples),
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
        "category_accuracy": category_stats,
        "hop_accuracy": hop_stats,
        "predictions": predictions,
    }


def compare_pretrained_finetuned(
    eval_file: str | Path,
    pretrained_ckpt: str | Path,
    finetuned_ckpt: str | Path,
    config_path: str | Path,
    out_dir: Path | None = None,
    max_samples: int | None = None,
) -> dict:
    """Run the same test set on pretrained and finetuned checkpoints."""
    cfg = _load_cfg(config_path)
    out_dir = out_dir or (_PROJECT_DIR / "report" / "finetuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = evaluate_model(
        eval_file,
        pretrained_ckpt,
        cfg=cfg,
        max_samples=max_samples,
        dump_predictions=out_dir / "pretrained_predictions.json",
    )
    ft = evaluate_model(
        eval_file,
        finetuned_ckpt,
        cfg=cfg,
        max_samples=max_samples,
        dump_predictions=out_dir / "finetuned_predictions.json",
    )
    comparison = {
        "eval_file": str(eval_file),
        "pretrained": {
            "checkpoint": str(pretrained_ckpt),
            "exact_match": pre["overall_accuracy"],
            "val_loss": pre["val_loss"],
            "val_perplexity": pre["val_perplexity"],
            "category_accuracy": pre["category_accuracy"],
            "hop_accuracy": pre["hop_accuracy"],
        },
        "finetuned": {
            "checkpoint": str(finetuned_ckpt),
            "exact_match": ft["overall_accuracy"],
            "val_loss": ft["val_loss"],
            "val_perplexity": ft["val_perplexity"],
            "category_accuracy": ft["category_accuracy"],
            "hop_accuracy": ft["hop_accuracy"],
        },
    }
    (out_dir / "pretrained_vs_finetuned.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_qualitative(pre["predictions"], ft["predictions"], out_dir / "qualitative_examples.md")
    print(
        f"[Evaluator] pretrained EM={pre['overall_accuracy']:.2f}% → "
        f"finetuned EM={ft['overall_accuracy']:.2f}%"
    )
    return comparison


def _write_qualitative(pre_preds: list[dict], ft_preds: list[dict], path: Path) -> None:
    lines = ["# Qualitative reasoning examples", ""]
    successes = [
        (p, f)
        for p, f in zip(pre_preds, ft_preds)
        if (not p["correct"]) and f["correct"]
    ]
    failures = [
        (p, f)
        for p, f in zip(pre_preds, ft_preds)
        if not f["correct"]
    ]
    lines.append("## Finetune fixed a pretrained error")
    for pre, ft in successes[:8]:
        lines += [
            f"**Q:** {ft['question']}",
            f"- Gold: `{ft['answer_option']}`",
            f"- Pretrained: {pre['generated']}",
            f"- Finetuned: {ft['generated']}",
            "",
        ]
    lines.append("## Remaining finetuned errors")
    for pre, ft in failures[:8]:
        lines += [
            f"**Q:** {ft['question']}",
            f"- Gold: `{ft['answer_option']}`",
            f"- Pretrained: {pre['generated']}",
            f"- Finetuned: {ft['generated']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_scaling_graph(experiment_results: dict, out_path: str | Path | None = None) -> None:
    """Plot loss / PPL / exact-match against training-set size (100 → 10k)."""
    keyed = {int(k): v for k, v in experiment_results.items()}
    counts = sorted(keyed)
    train_losses = [keyed[c]["train_loss"] for c in counts]
    val_losses = [keyed[c]["val_loss"] for c in counts]
    val_perplexities = [keyed[c]["val_perplexity"] for c in counts]
    val_accuracies = [keyed[c]["val_accuracy"] for c in counts]

    hop1, hop_multi = [], []
    for c in counts:
        hops = keyed[c].get("hop_accuracy") or {}
        h1 = hops.get("1") or hops.get(1) or {}
        n1, c1 = h1.get("total", 0), h1.get("correct", 0)
        cm = tm = 0
        for hk, st in hops.items():
            if str(hk) == "1":
                continue
            cm += st.get("correct", 0)
            tm += st.get("total", 0)
        hop1.append(100.0 * c1 / n1 if n1 else 0.0)
        hop_multi.append(100.0 * cm / tm if tm else 0.0)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    def _style(ax, ylabel, title):
        ax.set_xscale("log")
        ax.set_xticks(counts)
        ax.set_xticklabels([str(c) for c in counts], rotation=45, ha="right")
        ax.set_xlabel("Training sample size (100–10,000, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.set_xlim(90, 11000)

    axs[0, 0].plot(counts, train_losses, label="Training NLL", marker="o", color="blue")
    axs[0, 0].plot(counts, val_losses, label="Validation NLL", marker="s", color="orange")
    _style(axs[0, 0], "Token NLL", "Loss vs. Sample Size (100–10k)")

    axs[0, 1].plot(counts, val_perplexities, label="Validation perplexity", marker="^", color="purple")
    _style(axs[0, 1], "Perplexity", "Validation Perplexity Trend (100–10k)")

    axs[1, 0].plot(counts, val_accuracies, label="Exact match (%)", marker="o", color="green")
    _style(axs[1, 0], "Accuracy (%)", "Validation Reasoning Accuracy (100–10k)")

    axs[1, 1].plot(counts, hop1, label="1-hop exact match", marker="o", color="teal")
    axs[1, 1].plot(counts, hop_multi, label="2–4 hop exact match", marker="s", color="crimson")
    _style(axs[1, 1], "Accuracy (%)", "Hop-wise Accuracy vs. Sample Size")

    plt.tight_layout()
    out_path = Path(out_path) if out_path else (_PROJECT_DIR / "report" / "finetuning" / "finetuning_scaling_analysis.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Evaluator] Saved scaling plot → {out_path}")
