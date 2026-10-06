import os
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer_runs/bpe_vocab_7500/tokenizer.json")

TARGETS = {
    "train_manual": 490_000_000 * 0.20,  # 98M tokens
    "train_public": 490_000_000 * 0.80,  # 392M tokens
    "val": 5_000_000,                    # 5M tokens
    "test": 5_000_000                    # 5M tokens
}

def stream_and_split(input_file_path):
    print(f"Attempting to read from: {os.path.abspath(input_file_path)}")
    
    if not os.path.exists(input_file_path):
        print(f"ERROR: File not found at {input_file_path}")
        return

    os.makedirs("data/final_data_set", exist_ok=True)
    
    f_train = open("data/final_data_set/train.txt", "w", encoding="utf-8")
    f_val = open("data/final_data_set/val.txt", "w", encoding="utf-8")
    f_test = open("data/final_data_set/test.txt", "w", encoding="utf-8")
    
    counts = {"train_manual": 0, "train_public": 0, "val": 0, "test": 0}
    line_count = 0
    matched_count = 0
    
    print("Streaming and processing file line by line...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line_count += 1
            line = raw_line.strip()
            
            # Print status update every 100 lines
            if line_count % 500000 == 0:
                print(f"[STATUS] Scanned {line_count:,} lines | Matched: {matched_count:,} | TrainM: {counts['train_manual']:,} | TrainP: {counts['train_public']:,} | Val: {counts['val']:,} | Test: {counts['test']:,}")
            
            if not line:
                continue
                
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                if line_count <= 10:
                    print(f"[DEBUG] Line {line_count} could not be split into prefix and text.")
                continue
                
            prefix = parts[0].strip(":\u00a0 ").lower()
            text = parts[1].strip()
            
            if prefix == "public":
                category = "public"
                matched_count += 1
            elif prefix == "manual":
                category = "manual"
                matched_count += 1
            else:
                if line_count <= 10:
                    print(f"[DEBUG] Line {line_count} unknown prefix: {repr(prefix)}")
                continue
                
            if not text:
                continue
                
            token_count = len(tokenizer.encode(text + "\n\n"))
            
            if category == "manual":
                if counts["train_manual"] < TARGETS["train_manual"]:
                    f_train.write(text + "\n\n")
                    counts["train_manual"] += token_count
            else: 
                if counts["train_public"] < TARGETS["train_public"]:
                    f_train.write(text + "\n\n")
                    counts["train_public"] += token_count
                elif counts["val"] < TARGETS["val"]:
                    f_val.write(text + "\n\n")
                    counts["val"] += token_count
                elif counts["test"] < TARGETS["test"]:
                    f_test.write(text + "\n\n")
                    counts["test"] += token_count
                    
            if all(counts[k] >= TARGETS[k] for k in TARGETS):
                print("All target token quotas met!")
                break

    f_train.close()
    f_val.close()
    f_test.close()
    
    print(f"\n--- Debug Summary ---")
    print(f"Total lines scanned: {line_count}")
    print(f"Total lines matched prefix: {matched_count}")
    print(f"Final token counts -> Train Manual: {counts['train_manual']:,} | Train Public: {counts['train_public']:,} | Val: {counts['val']:,} | Test: {counts['test']:,}")

stream_and_split("data/processed/nepali_corpus.txt")