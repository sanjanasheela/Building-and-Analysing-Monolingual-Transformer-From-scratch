import os
import sys
from tokenizers import Tokenizer
from tqdm import tqdm

DATASET_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/final_data_set_7500/test.txt"

VOCAB_MODELS = {
    "5000": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_5000/tokenizer.json",
    "7500": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_7500/tokenizer.json",
    "10000": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_10000/tokenizer.json",
    "12500": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_12500/tokenizer.json",
}

def analyze_telugu_corpus(dataset_file: str, batch_size: int = 100_000):
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found.")
        return

    print("Loading Telugu tokenizers...")
    tokenizers = {vocab: Tokenizer.from_file(path) for vocab, path in VOCAB_MODELS.items()}

    lines_by_source = {}
    words_by_source = {}

    total_lines = 0
    total_words = 0

    print(f"Counting lines and words in '{dataset_file}'...\n")

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Telugu Corpus"):
            raw_str = line.strip()
            if not raw_str:
                continue

            parts = raw_str.split(maxsplit=1)
            source = parts[0]
            text = parts[1].strip() if len(parts) > 1 else ""

            if not text:
                continue

            word_count = len(text.split())

            lines_by_source[source] = lines_by_source.get(source, 0) + 1
            words_by_source[source] = words_by_source.get(source, 0) + word_count
            total_lines += 1
            total_words += word_count

    print("\n" + "=" * 85)
    print("DETAILED TELUGU SOURCE BREAKDOWN (LINES & WORDS)")
    print("=" * 85)
    print(f"{'Source Tag':<15} | {'Line Count':<15} | {'Line %':<10} | {'Word Count':<18} | {'Word %':<10}")
    print("-" * 85)
    for src in sorted(lines_by_source.keys()):
        l_cnt = lines_by_source[src]
        l_pct = (l_cnt / total_lines) * 100
        w_cnt = words_by_source[src]
        w_pct = (w_cnt / total_words) * 100
        print(f"{src:<15} | {l_cnt:<15,} | {l_pct:<9.2f}% | {w_cnt:<18,} | {w_pct:<9.2f}%")
    print("-" * 85)
    print(f"{'TOTAL':<15} | {total_lines:<15,} | {'100.00%':<10} | {total_words:<18,} | {'100.00%':<10}")
    print("=" * 85)

    print("\nStarting Token Sweep across Vocabulary Configurations (5000, 7500, 10000, 12500)...")

    for vocab, tok in tokenizers.items():
        print(f"\nEvaluating Vocab Size: {vocab}...")
        tot_toks = 0
        man_toks = 0
        pub_toks = 0

        batch_texts = []
        batch_labels = []

        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                raw_str = line.strip()
                if not raw_str:
                    continue
                parts = raw_str.split(maxsplit=1)
                src = parts[0]
                text = parts[1].strip() if len(parts) > 1 else ""
                if not text:
                    continue

                batch_texts.append(text)
                batch_labels.append(src)

                if len(batch_texts) >= batch_size:
                    encodings = tok.encode_batch(batch_texts)
                    for enc, lbl in zip(encodings, batch_labels):
                        n_toks = len(enc.ids)
                        tot_toks += n_toks
                        if lbl == "manual":
                            man_toks += n_toks
                        elif lbl == "public":
                            pub_toks += n_toks
                    batch_texts = []
                    batch_labels = []

            if batch_texts:
                encodings = tok.encode_batch(batch_texts)
                for enc, lbl in zip(encodings, batch_labels):
                    n_toks = len(enc.ids)
                    tot_toks += n_toks
                    if lbl == "manual":
                        man_toks += n_toks
                    elif lbl == "public":
                        pub_toks += n_toks

        fertility = tot_toks / total_words
        print(f"Vocab {vocab}: Total Tokens = {tot_toks:,} | Fertility = {fertility:.4f} | Manual Tokens = {man_toks:,} | Public Tokens = {pub_toks:,}")

if __name__ == "__main__":
    analyze_telugu_corpus(DATASET_FILE)
