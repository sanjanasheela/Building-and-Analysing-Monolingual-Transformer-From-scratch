"""
near_dedup_resume.py — Resume near-duplicate deduplication from a partial output.

After a crash, the LSH index (in RAM) is lost but the partial output file is intact.
This script:
  1. Rebuilds the LSH index from the partial output (fast, ~1.9 GB).
  2. Scans the full input from the beginning.
     - Lines that already appear in the partial output are skipped (they're "seen").
     - New lines are checked against the LSH; unique ones are APPENDED to the output.

Result: the output file is completed without re-doing work or losing progress.

Requires: pip install datasketch

Usage:
    python near_dedup_resume.py
"""

from datasketch import MinHash, MinHashLSH
import unicodedata
import os

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_merged_deduped.txt"
PARTIAL_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_near_deduped.txt"  # partial output (safe to read)
OUTPUT_FILE  = PARTIAL_FILE  # we will APPEND to this file

SIMILARITY_THRESHOLD = 0.8
NUM_PERM             = 128
NGRAM_SIZE           = 3
CHUNK_SIZE           = 50_000
LOG_EVERY            = 10_000
# ─────────────────────────────────────────────────────────────────────────────


def make_minhash(text: str) -> MinHash:
    """Compute MinHash for a line using character n-grams."""
    mh = MinHash(num_perm=NUM_PERM)
    text_no_space = text.replace(" ", "")
    n = NGRAM_SIZE
    if len(text_no_space) < n:
        mh.update(text_no_space.encode("utf-8"))
    else:
        for i in range(len(text_no_space) - n + 1):
            mh.update(text_no_space[i:i + n].encode("utf-8"))
    return mh


def rebuild_lsh_from_partial(partial_file: str) -> tuple[MinHashLSH, set[str], int]:
    """
    Phase 1: Read the partial output and rebuild the LSH index.
    Also stores a set of raw line strings for fast membership checks
    (to detect which input lines were already written).

    Returns (lsh, seen_lines_set, uid_counter).
    """
    print(f"[Phase 1] Rebuilding LSH index from partial output: {partial_file}")
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERM)
    seen_set: set[str] = set()  # exact text of already-output lines
    uid = 0

    with open(partial_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = unicodedata.normalize("NFC", raw_line.strip())
            if not stripped:
                continue
            mh = make_minhash(stripped)
            key = f"l{uid}"
            lsh.insert(key, mh)
            seen_set.add(stripped)
            uid += 1
            if uid % 500_000 == 0:
                print(f"  Loaded {uid:,} lines into LSH index...", flush=True)

    print(f"  Done. LSH index has {uid:,} entries. Starting uid = {uid}\n")
    return lsh, seen_set, uid


def process_chunk(
    lines: list[str],
    lsh: MinHashLSH,
    seen_set: set[str],
    fout,
    uid: int,
    kept: int,
    skipped: int,
    already_written: int,
) -> tuple[int, int, int, int]:
    """Dedup a chunk of lines against the global LSH index, skipping already-written lines."""
    for line in lines:
        line = unicodedata.normalize("NFC", line)

        # Fast exact check: was this line already in the partial output?
        if line in seen_set:
            already_written += 1
            continue

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

    return uid, kept, skipped, already_written


def main():
    print("=" * 60)
    print("Near-Dedup Resume Script")
    print("=" * 60)
    print(f"Input  : {INPUT_FILE}")
    print(f"Partial: {PARTIAL_FILE}")
    print(f"Output : {OUTPUT_FILE}  (will APPEND)")
    print(f"Threshold: {SIMILARITY_THRESHOLD} | num_perm: {NUM_PERM} | chunk: {CHUNK_SIZE:,}\n")

    # Safety check
    if not os.path.exists(PARTIAL_FILE):
        raise FileNotFoundError(f"Partial output not found: {PARTIAL_FILE}")

    # Phase 1: Rebuild LSH from partial output
    lsh, seen_set, uid = rebuild_lsh_from_partial(PARTIAL_FILE)
    partial_lines = uid  # how many lines were already written

    # Phase 2: Stream the input and append new unique lines
    print("[Phase 2] Streaming input and appending new unique lines...")
    total_read   = 0
    kept         = 0   # new lines appended in this run
    skipped      = 0   # near-dupes skipped
    already_written = 0  # input lines already present in partial output

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as fout:  # 'a' = append

        chunk: list[str] = []

        for raw_line in fin:
            stripped = raw_line.strip()
            if not stripped:
                continue
            chunk.append(stripped)
            total_read += 1

            if len(chunk) >= CHUNK_SIZE:
                uid, kept, skipped, already_written = process_chunk(
                    chunk, lsh, seen_set, fout, uid, kept, skipped, already_written
                )
                print(
                    f"  Read {total_read:,} | already_written {already_written:,} | "
                    f"new_kept {kept:,} | near_dupes {skipped:,}",
                    flush=True,
                )
                chunk = []

        # Flush remaining
        if chunk:
            uid, kept, skipped, already_written = process_chunk(
                chunk, lsh, seen_set, fout, uid, kept, skipped, already_written
            )
            total_read += len(chunk)

    total_output = partial_lines + kept
    total_removed = skipped + (total_read - already_written - kept - skipped)
    dupe_pct = (skipped / (total_read - already_written) * 100) if (total_read - already_written) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Total input lines read          : {total_read:,}")
    print(f"  Lines already in partial output : {already_written:,}")
    print(f"  New unique lines appended       : {kept:,}")
    print(f"  Near-dupes removed (this run)   : {skipped:,}  ({dupe_pct:.1f}% of new lines)")
    print(f"  Total lines in final output     : {total_output:,}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
