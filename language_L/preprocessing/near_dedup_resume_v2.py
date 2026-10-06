"""
near_dedup_resume_v2.py — Low-RAM resume of near-duplicate deduplication.

Strategy:
  1. Build a compact MD5 hash set of all lines already in the partial output
     (~16 bytes per line × 3.85M lines ≈ 60 MB). This lets us cheaply skip
     input lines that were already written.

  2. Rebuild the LSH index with REDUCED num_perm (64 instead of 128) from the
     partial output. 64 perms ≈ half the RAM vs 128; accuracy drops slightly
     but is still very good for Jaccard threshold=0.8.
     ~3.85M lines × 64 perms × 8 bytes ≈ ~2 GB for the index.

  3. Stream the full input from scratch:
     - Lines whose MD5 is in the "already seen" set → skip (already written)
     - New lines → check LSH → if unique, append to output and add to LSH

Result: output file is completed correctly, appending only new unique lines.

NOTE: Since we reduce num_perm from 128→64 for the warmup only, there may be
a tiny inconsistency at the boundary, but for practical deduplication at 0.8
threshold this is negligible.

Requires: pip install datasketch

Usage:
    python near_dedup_resume_v2.py
"""

from datasketch import MinHash, MinHashLSH
import unicodedata
import hashlib
import os
import gc

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_merged_deduped.txt"
PARTIAL_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_near_deduped.txt"
OUTPUT_FILE  = PARTIAL_FILE  # APPEND to this file

SIMILARITY_THRESHOLD = 0.8
NUM_PERM_WARMUP      = 64    # reduced for RAM-efficient warmup (was 128)
NUM_PERM_NEW         = 64    # must match warmup value
NGRAM_SIZE           = 3
CHUNK_SIZE           = 20_000  # smaller chunks to reduce peak RAM
LOG_EVERY_CHUNKS     = 5      # print every N chunks
# ─────────────────────────────────────────────────────────────────────────────


def md5_hash(text: str) -> bytes:
    """Compact 16-byte MD5 of a line."""
    return hashlib.md5(text.encode("utf-8")).digest()


def make_minhash(text: str, num_perm: int) -> MinHash:
    """Compute MinHash for a line using character n-grams."""
    mh = MinHash(num_perm=num_perm)
    text_no_space = text.replace(" ", "")
    n = NGRAM_SIZE
    if len(text_no_space) < n:
        mh.update(text_no_space.encode("utf-8"))
    else:
        for i in range(len(text_no_space) - n + 1):
            mh.update(text_no_space[i:i + n].encode("utf-8"))
    return mh


def phase1_build_seen_set(partial_file: str) -> set[bytes]:
    """
    Phase 1a: Build a compact MD5 set of already-written lines.
    16 bytes × 3.85M ≈ ~60 MB. Very fast.
    """
    print("[Phase 1a] Building MD5 seen-set from partial output...")
    seen: set[bytes] = set()
    count = 0
    with open(partial_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = unicodedata.normalize("NFC", raw_line.strip())
            if not stripped:
                continue
            seen.add(md5_hash(stripped))
            count += 1
            if count % 1_000_000 == 0:
                print(f"  MD5 set: {count:,} lines loaded ({len(seen)*16/1024/1024:.0f} MB)", flush=True)
    print(f"  Done. {count:,} lines in seen-set ({len(seen)*16/1024/1024:.0f} MB)\n")
    return seen


def phase1_build_lsh(partial_file: str, num_perm: int) -> tuple[MinHashLSH, int]:
    """
    Phase 1b: Build LSH from partial output (with reduced num_perm).
    """
    print(f"[Phase 1b] Building LSH index (num_perm={num_perm}) from partial output...")
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=num_perm)
    uid = 0
    with open(partial_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = unicodedata.normalize("NFC", raw_line.strip())
            if not stripped:
                continue
            mh = make_minhash(stripped, num_perm)
            lsh.insert(f"l{uid}", mh)
            uid += 1
            if uid % 500_000 == 0:
                print(f"  LSH: {uid:,} entries loaded...", flush=True)
    print(f"  Done. LSH index has {uid:,} entries. Starting uid = {uid}\n")
    return lsh, uid


def main():
    print("=" * 60)
    print("Near-Dedup Resume Script v2  (Low-RAM)")
    print("=" * 60)
    print(f"Input  : {INPUT_FILE}")
    print(f"Partial: {PARTIAL_FILE}")
    print(f"Output : {OUTPUT_FILE}  (APPEND mode)")
    print(f"Threshold: {SIMILARITY_THRESHOLD} | num_perm: {NUM_PERM_NEW} | chunk: {CHUNK_SIZE:,}\n")

    if not os.path.exists(PARTIAL_FILE):
        raise FileNotFoundError(f"Partial output not found: {PARTIAL_FILE}")

    # ── Phase 1a: Build MD5 seen-set (tiny RAM) ──────────────────────────────
    seen_md5 = phase1_build_seen_set(PARTIAL_FILE)
    partial_count = len(seen_md5)

    # ── Phase 1b: Build LSH (reduced num_perm) ───────────────────────────────
    lsh, uid = phase1_build_lsh(PARTIAL_FILE, NUM_PERM_WARMUP)
    gc.collect()

    print(f"[Phase 2] Streaming input and appending new unique lines...")
    total_read      = 0
    already_written = 0
    new_kept        = 0
    new_skipped     = 0
    chunks_done     = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as fout:

        chunk: list[str] = []

        for raw_line in fin:
            stripped = unicodedata.normalize("NFC", raw_line.strip())
            if not stripped:
                continue

            total_read += 1

            # Fast path: already in partial output (by MD5)
            if md5_hash(stripped) in seen_md5:
                already_written += 1
                continue

            # New line: check LSH for near-duplicates
            mh = make_minhash(stripped, NUM_PERM_NEW)
            if lsh.query(mh):
                new_skipped += 1
            else:
                lsh.insert(f"l{uid}", mh)
                fout.write(stripped + "\n")
                new_kept += 1
                uid += 1

            if total_read % (CHUNK_SIZE * LOG_EVERY_CHUNKS) == 0:
                print(
                    f"  Read {total_read:,} | already_written {already_written:,} | "
                    f"new_kept {new_kept:,} | near_dupes {new_skipped:,}",
                    flush=True,
                )
                gc.collect()

    total_output = partial_count + new_kept
    new_lines    = total_read - already_written
    dupe_pct     = (new_skipped / new_lines * 100) if new_lines > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Total input lines read           : {total_read:,}")
    print(f"  Lines already in partial output  : {already_written:,}")
    print(f"  New unique lines appended        : {new_kept:,}")
    print(f"  Near-dupes removed (this run)    : {new_skipped:,}  ({dupe_pct:.1f}%)")
    print(f"  Total lines in final output      : {total_output:,}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
