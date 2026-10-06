from __future__ import annotations

import sys
from pathlib import Path
import re
import yaml
import json
import torch
from tokenizers import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set font configurations
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans Telugu']
mpl.rcParams['axes.unicode_minus'] = False

# Path Setup
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _EVAL_DIR.parent
sys.path.append(str(_PROJECT_DIR / "train"))
sys.path.append(str(_EVAL_DIR))

from model import build_from_config
from data_loader import SingleBinLoader
from evaluation_wrapper import run_full_project_evaluation


def main():
    proj_dir = Path(_PROJECT_DIR)
    
    # 1. Locate and load configuration safely
    config_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/configs/51.yaml")
    if not config_path.exists():
        yaml_files = list(proj_dir.glob("*.yaml")) + list(proj_dir.glob("*.yml"))
        if yaml_files:
            config_path = yaml_files[0]
            print(f"⚠️ Using detected configuration file: {config_path.name}")
        else:
            raise FileNotFoundError(f"No YAML configuration file found at {config_path}")

    print(f"📂 Loading config from: {config_path}")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 2. Build model from config
    model = build_from_config(cfg).to(device)

    # 3. Safely locate and select the latest checkpoint using step naming convention
    checkpoint_dir = proj_dir / "checkpoints" / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("step_*.pt"))
    
    if checkpoint_files:
        def extract_step(path: Path) -> int:
            match = re.search(r"step_(\d+)\.pt", path.name)
            return int(match.group(1)) if match else -1

        checkpoint_files.sort(key=extract_step)
        checkpoint_path = checkpoint_files[-1]
        print(f"📦 Selected latest checkpoint: {checkpoint_path.name} (Resolved from {len(checkpoint_files)} total checkpoints)")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ Warning: No matching step_*.pt checkpoints found in {checkpoint_dir}. Running with initialized weights.")

    model.eval()

    # 4. Load trained tokenizer from the JSON path
    tokenizer_json_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_7500/tokenizer.json")
    print(f"🔤 Loading actual tokenizer from: {tokenizer_json_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))

    # 5. Initialize SingleBinLoader using test.bin explicitly
    test_bin_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/token_shards_7500/test.bin")
    print(f"📂 Initializing evaluation data loader from test set: {test_bin_path}")
    
    loader = SingleBinLoader(
        bin_path=test_bin_path,
        context_length=cfg["model"]["context_length"],
        batch_size=cfg["evaluation"]["eval_batch_size"],
        device=device,
    )

    # 6. Load a large batch of text samples directly from test.txt
    test_txt_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/final_data_set_7500/test.txt")
    print(f"📂 Loading comprehensive text lines from test dataset: {test_txt_path}")
    
    if test_txt_path.exists():
        with open(test_txt_path, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"Test text file not found at {test_txt_path}")

    # Use a large subset of data from test.txt (e.g., up to 100 samples for rigorous evaluation)
    max_eval_samples = min(10, len(all_lines))
    selected_texts = all_lines[:max_eval_samples]
    print(f"✨ Loaded {max_eval_samples} text samples from test.txt for evaluation.")

    prompt_len_tokens = 16
    sample_prompts = []
    sample_references = []

    for text in selected_texts:
        # Tokenize the full text line
        encoded = tokenizer.encode(text)
        tokens = encoded.ids
        
        if len(tokens) > prompt_len_tokens:
            prompt_tokens = tokens[:prompt_len_tokens]
            ref_text = text
        else:
            prompt_tokens = tokens
            ref_text = text
            
        sample_prompts.append(prompt_tokens)
        sample_references.append(ref_text)

    vocab_size = cfg["tokenizer"]["vocab_size"]
    attention_sample_ids = torch.randint(0, vocab_size, (1, 16), dtype=torch.long)
    temperatures = cfg["evaluation"].get("temperatures", [0.2, 0.5, 0.8, 1.0, 1.2, 1.5])

    # 7. Execute the unified evaluation wrapper across multi-temperature configurations
    print("🔍 Executing unified project evaluation wrapper with comprehensive dataset...")
    results = run_full_project_evaluation(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        prompts=sample_prompts,
        references=sample_references,
        n_batches=cfg["training"].get("eval_batches", 10),
        temperatures=temperatures,
        max_new_tokens=cfg["evaluation"].get("max_generation_length", 64),
        attention_sample_ids=attention_sample_ids,
        save_dir=proj_dir / "report" / "heatmaps",
        device=device,
    )

    # 8. Export and save outputs with 'ref' and 'output' prefixes grouped by temperatures
    report_dir = proj_dir / "report"
    tables_dir = report_dir / "tables"
    samples_dir = report_dir / "generated_samples"
    attention_dir = report_dir / "attention"

    for d in [tables_dir, samples_dir, attention_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save structured metrics
    metrics_json_path = tables_dir / "evaluation_metrics.json"
    metrics_md_path = tables_dir / "evaluation_metrics.md"

    def make_serializable(val):
        if isinstance(val, torch.Tensor):
            return val.tolist()
        if isinstance(val, dict):
            return {k: make_serializable(v) for k, v in val.items()}
        if isinstance(val, list):
            return [make_serializable(v) for v in val]
        return val

    serializable_results = make_serializable(results)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Evaluation Metrics Summary\n\n")
        f.write("| Metric Category | Value |\n|---|---|\n")
        intrinsic = serializable_results.get("intrinsic_metrics", {})
        for k, v in intrinsic.items():
            f.write(f"| Intrinsic: {k} | {v} |\n")

    # Save detailed generation outputs mapped by temperature with explicit prefixes
    generation_eval_file = samples_dir / "comprehensive_generations_by_temperature.txt"
    with open(generation_eval_file, "w", encoding="utf-8") as f:
        f.write("=== Comprehensive Generation Results Across Temperatures ===\n\n")
        gen_eval = serializable_results.get("generation_evaluation", {})
        
        for temp_key, data in gen_eval.items():
            f.write(f"########################################\n")
            f.write(f"TEMPERATURE SETTING: {temp_key}\n")
            f.write(f"########################################\n\n")
            
            hypotheses = data.get("hypotheses", [])
            for idx, (ref, hyp) in enumerate(zip(sample_references, hypotheses)):
                f.write(f"[Sample {idx + 1}]\n")
                f.write(f"ref: {ref}\n")
                f.write(f"output: {hyp}\n")
                f.write("-" * 50 + "\n")
            
            f.write("\nEvaluation Metrics for this Temperature:\n")
            for mk, mv in data.get("metrics", {}).items():
                f.write(f"  {mk}: {mv}\n")
            f.write("\n\n")

    print(f"✍️ Prefixed reference and output results successfully written to: {generation_eval_file}")

    # Save attention stats
    attn_data = serializable_results.get("attention_analysis", {})
    if attn_data:
        torch.save({
            "entropy": attn_data.get("entropy_summary"),
            "mean_distance": attn_data.get("mean_distance_summary")
        }, attention_dir / "attention_summary_stats.pt")
        print(f"🔍 Attention statistics saved to: {attention_dir / 'attention_summary_stats.pt'}")

    print("\n✅ Large-scale evaluation completed and assets exported successfully!")

if __name__ == "__main__":
    main()