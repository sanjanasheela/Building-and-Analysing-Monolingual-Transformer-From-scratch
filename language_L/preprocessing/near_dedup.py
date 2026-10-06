"""
Near-duplicate deduplication using MinHash LSH.

Memory-efficient: processes file in chunks, so only CHUNK_SIZE lines
are in RAM at a time. The LSH index grows as unique lines are seen.

Requires: pip install datasketch

Usage:
    python near_dedup.py
"""

from datasketch import MinHash, MinHashLSH
import unicodedata
import os

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_merged_deduped.txt"
OUTPUT_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_near_deduped.txt"

SIMILARITY_THRESHOLD = 0.8   # lines with Jaccard ≥ this are considered duplicates
NUM_PERM             = 128   # MinHash permutations (128 = tighter estimate, ~2x RAM vs 64)
NGRAM_SIZE           = 3     # character n-gram size
CHUNK_SIZE           = 50_000  # lines loaded into RAM at once — tune to your RAM
LOG_EVERY            = 10_000
# ─────────────────────────────────────────────────────────────────────────────


def make_minhash(text: str) -> MinHash:
    """Compute MinHash for a line using character n-grams."""
    mh = MinHash(num_perm=NUM_PERM)
    text = text.replace(" ", "")
    n = NGRAM_SIZE
    if len(text) < n:
        mh.update(text.encode("utf-8"))
    else:
        for i in range(len(text) - n + 1):
            mh.update(text[i:i + n].encode("utf-8"))
    return mh


def main():
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Threshold: {SIMILARITY_THRESHOLD} | num_perm: {NUM_PERM} | chunk: {CHUNK_SIZE:,}\n")

    # Global LSH index — persists across all chunks
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERM)

    total   = 0
    kept    = 0
    skipped = 0
    uid     = 0  # unique key counter for LSH

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        chunk = []

        for raw_line in fin:
            stripped = raw_line.strip()
            if not stripped:
                continue
            chunk.append(stripped)

            if len(chunk) >= CHUNK_SIZE:
                uid, kept, skipped = process_chunk(
                    chunk, lsh, fout, uid, kept, skipped
                )
                total += len(chunk)
                print(f"  Total {total:,} | kept {kept:,} | skipped {skipped:,}")
                chunk = []

        # Flush remaining lines
        if chunk:
            uid, kept, skipped = process_chunk(
                chunk, lsh, fout, uid, kept, skipped
            )
            total += len(chunk)

    dupe_pct = (skipped / total * 100) if total else 0
    print(f"\nDone.")
    print(f"  Total lines read   : {total:,}")
    print(f"  Unique lines kept  : {kept:,}")
    print(f"  Near-dupes removed : {skipped:,} ({dupe_pct:.1f}%)")
    print(f"  Saved to: {OUTPUT_FILE}")


def process_chunk(
    lines: list[str],
    lsh: MinHashLSH,
    fout,
    uid: int,
    kept: int,
    skipped: int,
) -> tuple[int, int, int]:
    """Dedup a chunk of lines against the global LSH index."""
    for i, line in enumerate(lines):
        # NFC normalization: ensures Telugu combining characters are
        # in canonical form so visually identical strings hash identically
        line = unicodedata.normalize("NFC", line)
        mh = make_minhash(line)
        matches = lsh.query(mh)

        if matches:
            skipped += 1
        else:
            key = f"l{uid}"
            lsh.insert(key, mh)
            fout.write(line + "\n")
            kept += 1
            uid += 1

        if (kept + skipped) % LOG_EVERY == 0:
            print(f"    chunk progress: {i+1:,}/{len(lines):,}", end="\r")

    return uid, kept, skipped


if __name__ == "__main__":
    main()
