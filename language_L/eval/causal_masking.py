import sys
from pathlib import Path
import yaml
import torch

# Add the train directory to Python path so we can import model.py
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _EVAL_DIR.parent
sys.path.append(str(_PROJECT_DIR / "train"))

from model import build_from_config

# 1. Load the configuration with explicit utf-8 encoding
config_path = _PROJECT_DIR / "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/configs/51.yaml"  # Update filename if your config is named differently
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 2. Build the model dynamically from config
model = build_from_config(cfg)
model.eval()

# 3. Setup sequences for verification
batch_size = 1
seq_len = 16
t = 5  # Evaluation position
vocab_size = cfg["tokenizer"]["vocab_size"]

input_seq_1 = torch.randint(0, vocab_size, (batch_size, seq_len))
input_seq_2 = input_seq_1.clone()

# Modify a future token at position t + 1
input_seq_2[0, t + 1] = (input_seq_1[0, t + 1] + 123) % vocab_size

# 4. Execute forward passes
with torch.no_grad():
    logits_1 = model(input_seq_1)
    logits_2 = model(input_seq_2)

# 5. Compare logits precisely at position t
logit_t_1 = logits_1[0, t, :]
logit_t_2 = logits_2[0, t, :]

max_diff = torch.max(torch.abs(logit_t_1 - logit_t_2)).item()

print(f"Max absolute logit difference at position {t}: {max_diff}")
assert max_diff < 1e-5, "Causal violation: Future tokens modified current logits!"
print("✅ Verification passed: The model is strictly causal.")