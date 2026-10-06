"""
Deduplication script: memory-efficient streaming exact deduplication.

Processes the file line-by-line — never loads the full file into RAM.
Only MD5 hashes of seen lines are stored in memory (~few GB for billions of lines).

Usage:
    python deduplicate.py
"""
import hashlib
import os

INPUT_FILE  = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_merged.txt"
OUTPUT_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_merged_deduped.txt"

LOG_EVERY = 1_000_000   # print progress every N lines


def line_hash(text: str) -> bytes:
    """Return a 16-byte MD5 digest of the line (collision-safe, compact)."""
    return hashlib.md5(text.encode("utf-8")).digest()


def main():
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("Streaming deduplication started...\n")

    seen: set[bytes] = set()
    kept = 0
    skipped = 0
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            stripped = raw_line.strip()
            if not stripped:
                continue

            total += 1
            h = line_hash(stripped)

            if h not in seen:
                seen.add(h)
                fout.write(stripped + "\n")
                kept += 1
            else:
                skipped += 1

            if total % LOG_EVERY == 0:
                print(f"  Processed {total:,} lines | kept {kept:,} | skipped {skipped:,}")

    dupe_pct = (skipped / total * 100) if total else 0
    print(f"\nDone.")
    print(f"  Total lines read : {total:,}")
    print(f"  Unique lines kept: {kept:,}")
    print(f"  Duplicates removed: {skipped:,} ({dupe_pct:.1f}%)")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()