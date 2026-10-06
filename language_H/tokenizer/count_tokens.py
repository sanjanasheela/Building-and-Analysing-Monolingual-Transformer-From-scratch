import os
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/processed/telugu_corpus.txt"

# Local trained tokenizer:
MODEL_PATH = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/tokenizer_runs/bpe_vocab_12500/tokenizer.json"

# HuggingFace example:
# MODEL_PATH = "sarvamai/sarvam-1"

BATCH_SIZE = 5_000

# ──────────────────────────────────────────────────────────────────────────────    


def load_tokenizer(model_path: str):
    """Load tokenizer from local JSON file or HuggingFace Hub."""

    if os.path.exists(model_path):
        print(f"Loading local trained tokenizer from:")
        print(f"  {model_path}\n")
        return Tokenizer.from_file(model_path)

    else:
        print(f"Loading pretrained HuggingFace tokenizer:")
        print(f"  {model_path}\n")
        return AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True
        )


def tokenize_batch(tok, batch_texts, is_hf):
    """Tokenize a batch and return token counts."""

    if is_hf:
        encodings = tok(
            batch_texts,
            add_special_tokens=False
        )["input_ids"]

        return [len(enc) for enc in encodings]

    else:
        encodings = tok.encode_batch(batch_texts)

        return [len(enc.ids) for enc in encodings]


def count_tokens(
    dataset_file: str,
    model_path: str,
    batch_size: int = BATCH_SIZE
):

    # ──────────────────────────────────────────────────────────────────────────
    # Check dataset
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(dataset_file):
        print(f"ERROR: Dataset file not found:")
        print(f"  {dataset_file}")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Load tokenizer
    # ──────────────────────────────────────────────────────────────────────────

    tok = load_tokenizer(model_path)

    # HuggingFace tokenizer has batch_encode_plus
    is_hf = hasattr(tok, "batch_encode_plus")

    # ──────────────────────────────────────────────────────────────────────────
    # Counters
    # ──────────────────────────────────────────────────────────────────────────

    total_tokens = 0

    manual_tokens = 0
    public_tokens = 0

    total_words = 0

    manual_words = 0
    public_words = 0

    total_lines = 0
    manual_lines = 0
    public_lines = 0

    # Useful for detecting unexpected labels
    unknown_lines = 0

    batch_texts = []
    batch_labels = []

    print(f"Counting tokens in:")
    print(f"  {dataset_file}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Function to process a batch
    # ──────────────────────────────────────────────────────────────────────────

    def process_batch():

        nonlocal total_tokens
        nonlocal manual_tokens
        nonlocal public_tokens

        token_counts = tokenize_batch(
            tok,
            batch_texts,
            is_hf
        )

        for n_toks, label, text in zip(
            token_counts,
            batch_labels,
            batch_texts
        ):

            total_tokens += n_toks

            if label == "manual":
                manual_tokens += n_toks

            elif label == "public":
                public_tokens += n_toks

    # ──────────────────────────────────────────────────────────────────────────
    # Read dataset
    # ──────────────────────────────────────────────────────────────────────────

    with open(
        dataset_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in tqdm(
            f,
            desc="Counting tokens"
        ):

            raw_str = line.rstrip("\r\n")

            # Skip empty lines
            if not raw_str.strip():
                continue

            # ──────────────────────────────────────────────────────────────
            # IMPORTANT:
            #
            # Your dataset looks like:
            #
            # public  some Telugu text...
            # manual  some Telugu text...
            #
            # NOT:
            #
            # public<TAB>some Telugu text
            #
            # So we use split(maxsplit=1).
            # ──────────────────────────────────────────────────────────────

            parts = raw_str.strip().split(
                maxsplit=1
            )

            # Invalid line
            if len(parts) != 2:
                unknown_lines += 1
                continue

            label = parts[0].strip().lower()
            text = parts[1].strip()

            # Only accept manual/public
            if label not in {"manual", "public"}:
                unknown_lines += 1
                continue

            if not text:
                continue

            # ──────────────────────────────────────────────────────────────
            # Add to batch
            # ──────────────────────────────────────────────────────────────

            batch_texts.append(text)
            batch_labels.append(label)

            # Statistics
            words = len(text.split())

            total_lines += 1
            total_words += words

            if label == "manual":
                manual_lines += 1
                manual_words += words

            elif label == "public":
                public_lines += 1
                public_words += words

            # ──────────────────────────────────────────────────────────────
            # Process batch
            # ──────────────────────────────────────────────────────────────

            if len(batch_texts) >= batch_size:

                process_batch()

                batch_texts = []
                batch_labels = []

        # ──────────────────────────────────────────────────────────────────────
        # Process final batch
        # ──────────────────────────────────────────────────────────────────────

        if batch_texts:
            process_batch()

    # ──────────────────────────────────────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────────────────────────────────────

    print("\n")
    print("=" * 70)
    print("TOKEN COUNT REPORT")
    print("=" * 70)

    print(f"  Model Used             : {model_path}")
    print(f"  Total Lines Processed  : {total_lines:,}")
    print(f"  Total Words            : {total_words:,}")
    print(f"  Total BPE Tokens       : {total_tokens:,}")

    print()
    print("-" * 70)
    print("DATASET BREAKDOWN")
    print("-" * 70)

    # ──────────────────────────────────────────────────────────────────────────
    # Manual
    # ──────────────────────────────────────────────────────────────────────────

    if total_tokens > 0:

        manual_percentage = (
            manual_tokens / total_tokens
        ) * 100

        public_percentage = (
            public_tokens / total_tokens
        ) * 100

    else:
        manual_percentage = 0
        public_percentage = 0

    print()
    print("  MANUAL DATA")
    print(f"    Lines               : {manual_lines:,}")
    print(f"    Words               : {manual_words:,}")
    print(f"    Tokens              : {manual_tokens:,}")
    print(f"    Token Percentage    : {manual_percentage:.2f}%")

    print()
    print("  PUBLIC DATA")
    print(f"    Lines               : {public_lines:,}")
    print(f"    Words               : {public_words:,}")
    print(f"    Tokens              : {public_tokens:,}")
    print(f"    Token Percentage    : {public_percentage:.2f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # Fertility
    # ──────────────────────────────────────────────────────────────────────────

    if total_words > 0:

        fertility = (
            total_tokens / total_words
        )

    else:
        fertility = 0

    print()
    print("-" * 70)
    print("TOKENIZATION STATISTICS")
    print("-" * 70)

    print(
        f"  Average Fertility     : "
        f"{fertility:.4f} tokens/word"
    )

    if manual_words > 0:

        manual_fertility = (
            manual_tokens / manual_words
        )

        print(
            f"  Manual Fertility      : "
            f"{manual_fertility:.4f} tokens/word"
        )

    if public_words > 0:

        public_fertility = (
            public_tokens / public_words
        )

        print(
            f"  Public Fertility      : "
            f"{public_fertility:.4f} tokens/word"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Sanity check
    # ──────────────────────────────────────────────────────────────────────────

    print()
    print("-" * 70)
    print("SANITY CHECK")
    print("-" * 70)

    print(
        f"  Manual + Public Tokens: "
        f"{manual_tokens + public_tokens:,}"
    )

    print(
        f"  Total Tokens          : "
        f"{total_tokens:,}"
    )

    if manual_tokens + public_tokens == total_tokens:
        print("  Status                : OK")
    else:
        print(
            "  Status                : WARNING - "
            "token counts do not match"
        )

    if unknown_lines > 0:
        print(
            f"  Unknown/Skipped Lines : "
            f"{unknown_lines:,}"
        )

    print("=" * 70)


if __name__ == "__main__":
    count_tokens(
        DATASET_FILE,
        MODEL_PATH
    )