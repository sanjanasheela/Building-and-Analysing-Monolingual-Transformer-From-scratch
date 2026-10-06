"""
Phase 3 entry point: generate Nepali reasoning data, finetune Model L
from its pretrained checkpoint, and report pretrained vs finetuned
exact-match accuracy on the held-out test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_FINETUNE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _FINETUNE_DIR.parent
if str(_FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(_FINETUNE_DIR))

from data_splitter import TRAIN_SIZES, create_splits
from evaluate import compare_pretrained_finetuned, evaluate_model, generate_scaling_graph
from train_finetune import _resolve, finetune_model, load_yaml

CONFIG_PATH = _PROJECT_DIR / "configs" / "finetune.yaml"


def _find_pretrained(cfg: dict, override: str | None) -> Path:
    if override:
        path = _resolve(override)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    listed = cfg.get("pretrained_checkpoint")
    candidates = []
    if listed:
        candidates.append(_resolve(listed))
    candidates.extend(
        [
            _PROJECT_DIR / "checkpoints" / "checkpoints" / "step_0045000.pt",
            _PROJECT_DIR / "checkpoints" / "step_0045000.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Pretrained checkpoint not found. Pass --pretrained or set "
        "pretrained_checkpoint in configs/finetune.yaml. Looked in:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Nepali reasoning finetuning")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Use train_samples_{N}.jsonl instead of the full train pool.",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Finetune a model for each nested train size and plot metrics.",
    )
    parser.add_argument("--skip-data", action="store_true", help="Reuse existing jsonl splits.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--finetuned", type=str, default=None, help="Checkpoint for --eval-only.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Evaluate only the first N test/val examples.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("finetuning", {})

    if not args.skip_data and not args.eval_only:
        print("Step 1: Generating synthetic Nepali reasoning splits (seed=42)...")
        create_splits()
    else:
        print("Step 1: Skipping dataset generation.")

    pretrained = _find_pretrained(cfg, args.pretrained)
    val_file = data_cfg.get("val_file", "finetuning/data/val_fixed.jsonl")
    test_file = data_cfg.get("test_file", "finetuning/data/test_fixed.jsonl")

    if args.eval_only:
        ckpt = args.finetuned or args.pretrained
        if not ckpt:
            raise SystemExit("--eval-only requires --finetuned or --pretrained")
        evaluate_model(
            test_file, ckpt, config_path=args.config, max_samples=args.max_eval
        )
        return

    if args.scaling:
        print("\nStep 2: Data-size scaling loop...")
        results = {}
        for count in TRAIN_SIZES:
            train_file = _FINETUNE_DIR / "data" / f"train_samples_{count}.jsonl"
            if not train_file.exists():
                print(f"  skip missing {train_file.name}")
                continue
            train_loss, ckpt_path = finetune_model(
                train_file=train_file,
                checkpoint_path=pretrained,
                config_path=args.config,
                val_file=val_file,
                resume_from=args.resume,
                epochs=args.epochs,
            )
            metrics = evaluate_model(
                val_file, ckpt_path, config_path=args.config, max_samples=args.max_eval
            )
            results[count] = {
                "train_loss": train_loss,
                "val_loss": metrics["val_loss"],
                "val_perplexity": metrics["val_perplexity"],
                "val_accuracy": metrics["overall_accuracy"],
                "category_accuracy": metrics["category_accuracy"],
                "hop_accuracy": metrics["hop_accuracy"],
            }
        generate_scaling_graph(results)
        out = _PROJECT_DIR / "report" / "finetuning" / "scaling_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[SUCCESS] Scaling results → {out}")
        return

    if args.train_size is not None:
        train_file = _FINETUNE_DIR / "data" / f"train_samples_{args.train_size}.jsonl"
    else:
        train_file = _resolve(data_cfg.get("train_file", "finetuning/data/train_full.jsonl"))

    print(f"\nStep 2: Finetuning from {pretrained} on {train_file.name}...")
    _, ckpt_path = finetune_model(
        train_file=train_file,
        checkpoint_path=pretrained,
        config_path=args.config,
        val_file=val_file,
        resume_from=args.resume,
        epochs=args.epochs,
    )

    print("\nStep 3: Pretrained vs finetuned exact-match on the held-out test set...")
    compare_pretrained_finetuned(
        eval_file=test_file,
        pretrained_ckpt=pretrained,
        finetuned_ckpt=ckpt_path,
        config_path=args.config,
        max_samples=args.max_eval,
    )
    print("\n[SUCCESS] Phase 3 finetuning pipeline finished.")


if __name__ == "__main__":
    main()
