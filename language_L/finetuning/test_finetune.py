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
    sys.stdout.flush()
    
    # Paths and configurations
    config_path = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/configs/51.yaml"
    ckpt_path = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/finetuned_checkpoints/finetuned_train_samples_10000.pt")
    
    print(f"[DEBUG] Loading config from {config_path}")
    sys.stdout.flush()
    cfg = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")
    sys.stdout.flush()
    
    # Load tokenizer
    tokenizer_path = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/tokenizer_runs/bpe_vocab_7500/tokenizer.json"
    print(f"[DEBUG] Loading tokenizer from {tokenizer_path}")
    sys.stdout.flush()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Load model and checkpoint weights
    print("[DEBUG] Building model from config...")
    sys.stdout.flush()
    model = build_from_config(cfg).to(device)
    print(f"[DEBUG] Loading checkpoint weights from {ckpt_path}")
    sys.stdout.flush()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[DEBUG] Model loaded and set to eval mode.")
    sys.stdout.flush()
    
    # Hardcoded input example (Fixed syntax from colon to equals)
    user_text = "सुनीता को उमेर रवि भन्दा बढी हो। को जेठो हो?"
   
    # Format prompt for inference
    prompt = format_prompt(user_text) + " "
    prompt_ids = tokenizer.encode(prompt).ids
    print(f"[DEBUG] Prompt tokenized. Length: {len(prompt_ids)} tokens.")
    sys.stdout.flush()
    
    ctx = model.context_length
    input_ids = torch.tensor([prompt_ids[-ctx:]], dtype=torch.long, device=device)
    print(f"[DEBUG] Input tensor shape: {input_ids.shape}")
    sys.stdout.flush()
    
    # Greedy generation loop
    generated: list[int] = []
    max_new_tokens = 64
    
    print("[DEBUG] Entering generation loop...")
    sys.stdout.flush()
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
            print(f"[DEBUG] Step {step}: generated token id {next_id}")
            sys.stdout.flush()
            if next_id in (3, 0): # EOS or PAD
                print("[DEBUG] Hit EOS or PAD token. Stopping.")
                sys.stdout.flush()
                break
            generated.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if input_ids.size(1) > ctx:
                input_ids = input_ids[:, -ctx:]
            
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)
            if "\n" in gen_text or "ప్రశ్న:" in gen_text:
                print("[DEBUG] Hit stop string in generated text. Stopping.")
                sys.stdout.flush()
                break
                
    generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    for stop in ("\n", "ప్రశ్న:"):
        if stop in generated_text:
            generated_text = generated_text.split(stop, 1)[0].strip()
            
    # Print results
    print(f"\n--- Hardcoded Inference Test ---")
    print(f"Prompt Question: {user_text}")
    print(f"Model Generation: {generated_text}")
    sys.stdout.flush()

if __name__ == "__main__":
    run_single_inference()