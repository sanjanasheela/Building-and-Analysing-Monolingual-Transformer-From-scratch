"""
Reasoning finetuning trainer (Phase 3).

Starts from the language's own pretrained checkpoint, keeps the tokenizer
fixed, trains with causal LM loss on the answer tokens, and writes
resume-capable checkpoints (weights, optimizer, scheduler, step, config).
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

_FINETUNE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _FINETUNE_DIR.parent
for _p in (str(_FINETUNE_DIR), str(_PROJECT_DIR / "train"), str(_PROJECT_DIR / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from checkpoint import save_checkpoint
from model import build_from_config
from scheduler import cosine_lr_with_warmup
from template import format_example, format_prompt

PAD_ID = 0
EOS_ID = 3
IGNORE_INDEX = -100


def _resolve(path: str | Path) -> Path:
    """Resolve relative paths against the language_H directory."""
    path = Path(path)
    if path.is_absolute():
        return path
    return _PROJECT_DIR / path


def find_jsonl(path: str | Path) -> Path:
    """Locate a jsonl file in CWD, language_H, or finetuning/data."""
    given = Path(path)
    candidates = [
        given,
        _resolve(given),
        _FINETUNE_DIR / given,
        _FINETUNE_DIR / "data" / given.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"JSONL not found: {path}")


def load_yaml(path: str | Path) -> dict:
    with open(_resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_finetune_config(base_cfg: dict, ft_cfg: dict) -> dict:
    """Keep pretrained model/tokenizer; replace training with finetune YAML."""
    cfg = json.loads(json.dumps(base_cfg))
    if "tokenizer" in ft_cfg:
        cfg.setdefault("tokenizer", {}).update(ft_cfg["tokenizer"])
    if "model" in ft_cfg:
        cfg.setdefault("model", {}).update(ft_cfg["model"])
    if "training" in ft_cfg:
        cfg["training"] = ft_cfg["training"]
    if "hardware" in ft_cfg:
        cfg["hardware"] = {**cfg.get("hardware", {}), **ft_cfg["hardware"]}
    if "checkpoint" in ft_cfg:
        cfg["checkpoint"] = {**cfg.get("checkpoint", {}), **ft_cfg["checkpoint"]}
    cfg["finetuning"] = ft_cfg.get("finetuning", {})
    cfg["pretrained_checkpoint"] = ft_cfg.get(
        "pretrained_checkpoint", cfg.get("pretrained_checkpoint")
    )
    cfg["base_config"] = ft_cfg.get("base_config")
    return cfg


def build_optimizer(model, t_cfg: dict) -> torch.optim.AdamW:
    """Same 2-group AdamW split as pretraining (decay vs no-decay)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": t_cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=t_cfg["learning_rate"],
        betas=(t_cfg["beta1"], t_cfg["beta2"]),
        eps=t_cfg["epsilon"],
    )


def _torch_load(path: str | Path, device: torch.device | str):
    """Load a checkpoint that contains nested dicts (config, optimizer)."""
    return torch.load(str(path), map_location=device, weights_only=False)


def load_pretrained_weights(model, checkpoint_path: Path, device: torch.device) -> int:
    """Load model weights only; optimizer is re-created for finetuning."""
    ckpt = _torch_load(checkpoint_path, device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {result.unexpected_keys}")
    step = int(ckpt.get("step", 0))
    print(f"  Loaded pretrained weights from {checkpoint_path.name} (pretrain step={step})")
    return step


def _load_samples(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_supervised(
    sample: dict,
    tokenizer: Tokenizer,
    ctx_len: int,
) -> tuple[list[int], list[int]]:
    """
    Tokenize question+answer. Prompt token targets are IGNORE_INDEX so the
    loss is computed on the answer (and EOS) only. Offsets are used so BPE
    prefix mismatch cannot leak prompt tokens into the loss.
    """
    user = sample["messages"][0]["content"]
    prompt_text = format_prompt(user) + " "
    full_text = format_example(sample)
    enc = tokenizer.encode(full_text)
    ids = list(enc.ids)
    labels: list[int] = []
    prompt_chars = len(prompt_text)
    offsets = enc.offsets if enc.offsets else [(0, 0)] * len(ids)
    for tok, (start, _end) in zip(ids, offsets):
        if start < prompt_chars:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(tok)
    ids.append(EOS_ID)
    labels.append(EOS_ID)
    if len(ids) > ctx_len:
        ids = ids[:ctx_len]
        labels = labels[:ctx_len]
    return ids, labels


def collate_batch(
    samples: list[dict],
    tokenizer: Tokenizer,
    ctx_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_supervised(s, tokenizer, ctx_len) for s in samples]
    max_len = max(len(ids) for ids, _ in encoded)
    max_len = min(max_len, ctx_len)
    input_rows, target_rows = [], []
    for ids, labels in encoded:
        ids = ids[:max_len]
        labels = labels[:max_len]
        pad_n = max_len - len(ids)
        ids = ids + [PAD_ID] * pad_n
        labels = labels + [IGNORE_INDEX] * pad_n
        input_rows.append(ids[:-1])
        target_rows.append(labels[1:])
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(target_rows, dtype=torch.long, device=device),
    )


@torch.no_grad()
def sequence_nll(
    model,
    samples: list[dict],
    tokenizer: Tokenizer,
    cfg: dict,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Mean token NLL and perplexity on answer tokens."""
    model.eval()
    batch_size = int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"]))
    ctx_len = cfg["model"]["context_length"]
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_loss = 0.0
    total_tok = 0
    n_batches = 0
    for i in range(0, len(samples), batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = samples[i : i + batch_size]
        x, y = collate_batch(batch, tokenizer, ctx_len, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, raw.vocab_size),
            y.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_tok = int((y != IGNORE_INDEX).sum().item())
        total_loss += loss.item()
        total_tok += max(n_tok, 1)
        n_batches += 1
    avg = total_loss / max(total_tok, 1)
    ppl = math.exp(min(avg, 100.0))
    model.train()
    return avg, ppl


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def finetune_model(
    train_file: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    val_file: str | Path | None = None,
    resume_from: str | Path | None = None,
    epochs: int | None = None,
) -> tuple[float, Path]:
    """
    Finetune from `checkpoint_path` on `train_file`.

    Returns (last train loss, saved checkpoint path).
    """
    ft_cfg = load_yaml(config_path)
    base_path = ft_cfg.get("base_config")
    if base_path:
        cfg = merge_finetune_config(load_yaml(base_path), ft_cfg)
    else:
        cfg = ft_cfg

    t_cfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] device={device}")

    tokenizer_path = _resolve(cfg["tokenizer"]["tokenizer_path"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"[Trainer] tokenizer={tokenizer_path} (frozen vocabulary)")

    model = build_from_config(cfg).to(device)
    optimizer = build_optimizer(model, t_cfg)

    pretrained = _resolve(checkpoint_path)
    if not pretrained.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")

    start_step = 0
    if resume_from:
        resume_path = _resolve(resume_from)
        ckpt = _torch_load(resume_path, device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt.get("step", 0))
        pretrain_step = int(ckpt.get("train_metrics", {}).get("pretrain_step", ckpt.get("step", 0)))
        print(f"[Trainer] Resumed finetune from {resume_path.name} at step={start_step}")
    else:
        pretrain_step = load_pretrained_weights(model, pretrained, device)

    train_file = find_jsonl(train_file)
    train_samples = _load_samples(train_file)

    val_samples = []
    if val_file is not None:
        val_samples = _load_samples(find_jsonl(val_file))

    ctx_len = cfg["model"]["context_length"]
    batch_size = int(t_cfg["batch_size"])
    epochs = int(
        epochs
        if epochs is not None
        else t_cfg.get("epochs", cfg.get("finetuning", {}).get("epochs", 3))
    )
    max_grad = float(t_cfg.get("max_grad_norm", 1.0))
    warmup = int(t_cfg.get("warmup_steps", 20))
    steps_per_epoch = max(math.ceil(len(train_samples) / batch_size), 1)
    max_steps = int(t_cfg.get("max_steps", steps_per_epoch * epochs))
    lr = float(t_cfg["learning_rate"])
    min_lr = float(t_cfg.get("min_learning_rate", lr * 0.1))

    out_dir = _resolve(cfg["checkpoint"].get("output_dir", "finetuned_checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _PROJECT_DIR / "logs" / "finetune_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = start_step
    last_loss = 0.0
    last_ckpt = out_dir / f"step_{global_step:07d}.pt"

    print(
        f"[Trainer] Finetuning {train_file} | n={len(train_samples)} | "
        f"epochs={epochs} | batch={batch_size} | lr={lr}"
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_tok = 0
        n_steps = 0
        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i : i + batch_size]
            x, y = collate_batch(batch, tokenizer, ctx_len, device)
            n_tok = int((y != IGNORE_INDEX).sum().item())
            if n_tok == 0:
                continue

            global_step += 1
            cur_lr = cosine_lr_with_warmup(global_step, warmup, max_steps, lr, min_lr)
            _set_lr(optimizer, cur_lr)

            logits = model(x)
            raw = model.module if isinstance(model, torch.nn.DataParallel) else model
            loss = F.cross_entropy(
                logits.reshape(-1, raw.vocab_size),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()

            last_loss = loss.item()
            epoch_loss += last_loss * n_tok
            epoch_tok += n_tok
            n_steps += 1

            if global_step % int(t_cfg.get("log_interval", 20)) == 0:
                rec = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": round(last_loss, 6),
                    "lr": cur_lr,
                    "pretrain_step": pretrain_step,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

        mean_epoch = epoch_loss / max(epoch_tok, 1)
        msg = f"  Epoch {epoch + 1}/{epochs} | train_nll={mean_epoch:.4f} | steps={n_steps}"
        if val_samples:
            val_nll, val_ppl = sequence_nll(
                model, val_samples, tokenizer, cfg, device, max_batches=8
            )
            msg += f" | val_nll={val_nll:.4f} | val_ppl={val_ppl:.2f}"
        print(msg)

        last_ckpt = save_checkpoint(
            out_dir=out_dir,
            step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler_state={
                "name": t_cfg.get("scheduler", "cosine"),
                "warmup_steps": warmup,
                "max_steps": max_steps,
                "lr": lr,
                "min_lr": min_lr,
                "last_step": global_step,
            },
            cfg=cfg,
            train_metrics={
                "phase": "finetune",
                "epoch": epoch + 1,
                "train_loss": mean_epoch,
                "train_file": str(train_file),
                "pretrained_checkpoint": str(pretrained),
                "pretrain_step": pretrain_step,
            },
        )
        print(f"  Saved {last_ckpt}")

    alias = out_dir / f"finetuned_{Path(train_file).stem}.pt"
    if last_ckpt.exists():
        shutil.copy2(last_ckpt, alias)
        print(f"  Alias → {alias}")
    return last_loss, alias if alias.exists() else last_ckpt
