import argparse
import hashlib
import os
import sys

# ── Make imports work whether the script is run from this dir or the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Import your custom cleaning rules
from length_filtering           import drop_short_lines
from whitespace_normalisation   import normalize_whitespace
from language_filter            import filter_document
from unicode_normalisation      import normalize_telugu_text
from remove_ra                  import fix_ravattu_brackets


# ──────────────────────────────────────────────────────────────────────────────
# Hashing & Processing Functions
# ──────────────────────────────────────────────────────────────────────────────

def xxhash_or_sha(content: str) -> int:
    """Fast hash of a string. Uses xxhash if available, else sha256."""
    try:
        import xxhash
        # FIX: Added .encode('utf-8') here
        return xxhash.xxh64(content.encode('utf-8')).intdigest()
    except ImportError:
        return int.from_bytes(hashlib.sha256(content.encode('utf-8')).digest()[:8], "big")

def process_paragraph(para: str, min_words: int) -> str:
    """Run the paragraph-level cleaning steps."""
    para = drop_short_lines(para, min_words=min_words)
    if not para.strip(): return ""
    
    para = normalize_whitespace(para, collapse_blank_lines=False)
    if not para.strip(): return ""
    
    para = filter_document(para)
    if not para.strip(): return ""
    
    para = fix_ravattu_brackets(para)
    if not para.strip(): return ""
    
    para = normalize_telugu_text(para)
    return para.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def stream_clean_and_dedup(
    input_path: str,
    output_path: str,
    min_words: int = 3,
    report_every: int = 1_000_000
):
    """
    Reads a massive file chunk-by-chunk, cleans paragraphs, deduplicates them using 
    hashes, and writes to disk immediately to prevent RAM crashes.
    """
    print(f"[pipeline] Streaming from : {input_path}")
    
    seen_hashes: set[int] = set()
    buffer = []
    
    lines_read = 0
    kept_paragraphs = 0
    dropped_duplicates = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            lines_read += 1
            
            if lines_read % report_every == 0:
                print(f"  Processed {lines_read:,} lines | Kept: {kept_paragraphs:,} | Dupes Dropped: {dropped_duplicates:,}")

            # If we hit a blank line, process the accumulated paragraph buffer
            if not line.strip():
                if buffer:
                    raw_para = "\n".join(buffer)
                    buffer = [] # Clear buffer immediately
                    
                    # 1. Clean the paragraph
                    cleaned_para = process_paragraph(raw_para, min_words)
                    
                    if cleaned_para:
                        # 2. Hash for exact deduplication
                        h = xxhash_or_sha(cleaned_para)
                        
                        if h in seen_hashes:
                            dropped_duplicates += 1
                        else:
                            seen_hashes.add(h)
                            fout.write(cleaned_para + "\n\n")
                            kept_paragraphs += 1
            else:
                # Accumulate non-empty lines into the current paragraph buffer
                buffer.append(line.rstrip("\n"))

        # Process the very last buffer if the file doesn't end in a newline
        if buffer:
            raw_para = "\n".join(buffer)
            cleaned_para = process_paragraph(raw_para, min_words)
            if cleaned_para:
                h = xxhash_or_sha(cleaned_para)
                if h not in seen_hashes:
                    fout.write(cleaned_para + "\n\n")
                    kept_paragraphs += 1
                else:
                    dropped_duplicates += 1

    print(f"\n{'='*60}")
    print(f"Total lines read  : {lines_read:,}")
    print(f"Unique Paras Kept : {kept_paragraphs:,}")
    print(f"Duplicates Dropped: {dropped_duplicates:,}")
    print(f"[pipeline] Written: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Argument Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming Telugu text cleaning and deduplication pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Path to the raw input .txt file")
    parser.add_argument("output", help="Path for the cleaned output .txt file")
    parser.add_argument(
        "--min-words", type=int, default=3,
        help="Minimum words per line to keep.",
    )
    parser.add_argument(
        "--report-every", type=int, default=1000000,
        help="How often to print progress updates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    stream_clean_and_dedup(
        input_path=args.input,
        output_path=args.output,
        min_words=args.min_words,
        report_every=args.report_every
    )