from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

# Setup paths
_EVAL_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
_PROJECT_DIR = _EVAL_DIR.parent
for _p in (str(_EVAL_DIR), str(_PROJECT_DIR / "train"), str(_PROJECT_DIR / "model"), str(_PROJECT_DIR / "finetuning")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import build_from_config
from template import format_prompt
from train_finetune import load_yaml

def run_single_inference():
    print("[DEBUG] Starting inference script...")
    
    # Paths and configurations
    config_path = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/configs/51.yaml"
    ckpt_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/finetuned_checkpoints/finetuned_train_samples_8000.pt")
    
    print(f"[DEBUG] Loading config from {config_path}")
    cfg = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_7500/tokenizer.json"
    print(f"[DEBUG] Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Load model and checkpoint weights
    print("[DEBUG] Building model from config...")
    model = build_from_config(cfg).to(device)
    print(f"[DEBUG] Loading checkpoint weights from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[DEBUG] Model loaded and set to eval mode.")
    
    # Hardcoded input example
    user_text = "999 మరియు 257 సంఖ్యలను పోల్చండి. ఏది పెద్దది?"
    gold_answer = "40 < 244 కాబట్టి సమాధానం దీపిక."
    gold_option = "సమానం"
    
    # Format prompt for inference
    prompt = format_prompt(user_text) + " "
    prompt_ids = tokenizer.encode(prompt).ids
    print(f"[DEBUG] Prompt tokenized. Length: {len(prompt_ids)} tokens.")
    
    ctx = model.context_length
    input_ids = torch.tensor([prompt_ids[-ctx:]], dtype=torch.long, device=device)
    print(f"[DEBUG] Input tensor shape: {input_ids.shape}")
    
    # Greedy generation loop
    generated: list[int] = []
    max_new_tokens = 64
    
    print("[DEBUG] Entering generation loop...")
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
            print(f"[DEBUG] Step {step}: generated token id {next_id}")
            if next_id in (3, 0): # EOS or PAD
                print("[DEBUG] Hit EOS or PAD token. Stopping.")
                break
            generated.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if input_ids.size(1) > ctx:
                input_ids = input_ids[:, -ctx:]
            
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)
            if "\n" in gen_text or "ప్రశ్న:" in gen_text:
                print("[DEBUG] Hit stop string in generated text. Stopping.")
                break
                
    generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    for stop in ("\n", "ప్రశ్న:"):
        if stop in generated_text:
            generated_text = generated_text.split(stop, 1)[0].strip()
            
    # Print results
    print(f"\n--- Hardcoded Inference Test ---")
    print(f"Prompt Question: {user_text}")
    print(f"Model Generation: {generated_text}")

if __name__ == "__main__":
    run_single_inference()


