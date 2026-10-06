"""
run_cleaning_folder.py
----------------------
Iterate over every .txt file in INPUT_DIR (recursively, excluding EXCLUDE_DIRS),
run the full Nepali cleaning pipeline on each one (chunked to manage memory),
and write cleaned output files to OUTPUT_DIR mirroring the input structure.

Usage:
    python run_cleaning_folder.py
    python run_cleaning_folder.py --input_dir /path/to/raw --output_dir /path/to/processed
"""

import os
import gc
import sys
import time
import argparse
from pathlib import Path

# Ensure preprocessing directory is on sys.path
PREPROC_DIR = Path(__file__).resolve().parent
if str(PREPROC_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROC_DIR))

from clean_pipeline import run_pipeline


# ──────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────
DEFAULT_INPUT_DIR = (
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data/onlinekhabar"
)

DEFAULT_OUTPUT_DIR = (
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/processed/onlinekhabar"
)

DEFAULT_EXCLUDE_DIRS = {"ocr_books", "nepali-corpus", "_parquet_cache", ".cache", "raygx-corpus"}
DEFAULT_CHUNK_SIZE_MB = 50
DEFAULT_MIN_WORDS = 3
# ──────────────────────────────────────────────


def clean_file_chunked(
    input_path: str,
    output_path: str,
    chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
    min_words: int = DEFAULT_MIN_WORDS,
    skip_existing: bool = False
) -> None:
    """
    Clean a single .txt file in fixed-size chunks and write to output_path.

    Chunking avoids loading the entire file into RAM at once, which is
    important for large files (hundreds of MB / multi-GB).
    """
    if skip_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"  Skipping (already processed): {output_path}")
        return

    chunk_size = chunk_size_mb * 1024 * 1024

    file_size       = os.path.getsize(input_path)
    bytes_processed = 0
    lines_processed = 0
    chunks          = 0
    start_time      = time.time()

    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Size  : {file_size / (1024**2):.2f} MB  |  chunk: {chunk_size_mb} MB")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, "r", encoding="utf-8", errors="ignore") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        buffer      = []
        buffer_size = 0

        for line in infile:
            buffer.append(line)
            line_size        = len(line.encode("utf-8"))
            buffer_size      += line_size
            bytes_processed  += line_size
            lines_processed  += 1

            # ── flush chunk ──────────────────────────────
            if buffer_size >= chunk_size:
                chunks += 1
                text    = "".join(buffer)

                cleaned = run_pipeline(text, min_words=min_words)

                if cleaned:
                    outfile.write(cleaned)
                    outfile.write("\n\n")
                    outfile.flush()

                del text, cleaned, buffer
                buffer      = []
                buffer_size = 0
                gc.collect()

                percent = bytes_processed / file_size * 100 if file_size else 0
                elapsed = time.time() - start_time
                print(
                    f"    chunk {chunks:4d} | {percent:6.2f}% | "
                    f"{elapsed / 60:.1f} min"
                )

        # ── final remaining buffer ───────────────────────
        if buffer:
            chunks += 1
            text    = "".join(buffer)
            cleaned = run_pipeline(text, min_words=min_words)
            if cleaned:
                outfile.write(cleaned)
                outfile.write("\n\n")
                outfile.flush()
            del text, cleaned, buffer
            gc.collect()

    elapsed = time.time() - start_time
    out_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    print(
        f"  Done  : {chunks} chunk(s), {lines_processed:,} lines, "
        f"output: {out_size / (1024**2):.2f} MB in {elapsed:.1f} s\n"
    )


def clean_folder(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    exclude_dirs: set = DEFAULT_EXCLUDE_DIRS,
    chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
    min_words: int = DEFAULT_MIN_WORDS,
    skip_existing: bool = False
) -> None:
    """Iterate over every .txt file in input_dir (recursively, excluding exclude_dirs) and clean it."""

    os.makedirs(output_dir, exist_ok=True)

    # Collect all txt files recursively while excluding specified directories
    txt_files = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f.lower().endswith(".txt"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir)
                txt_files.append(rel_path)

    txt_files.sort()

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    total_start = time.time()

    print("=" * 60)
    print(f"Input  folder : {input_dir}")
    print(f"Output folder : {output_dir}")
    print(f"Excluded dirs : {exclude_dirs}")
    print(f"Files found   : {len(txt_files)}")
    print("=" * 60)
    print()

    for idx, rel_path in enumerate(txt_files, 1):
        in_path  = os.path.join(input_dir,  rel_path)
        out_path = os.path.join(output_dir, rel_path)

        print(f"[{idx}/{len(txt_files)}]  {rel_path}")
        clean_file_chunked(
            in_path,
            out_path,
            chunk_size_mb=chunk_size_mb,
            min_words=min_words,
            skip_existing=skip_existing
        )

    total_elapsed = time.time() - total_start

    print("=" * 60)
    print("ALL FILES DONE")
    print(f"Total time : {total_elapsed / 60:.1f} minutes")
    print(f"Output dir : {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Clean raw Nepali text files using the preprocessing pipeline.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Path to input directory containing .txt files")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to output directory for cleaned files")
    parser.add_argument("--exclude_dirs", type=str, default=",".join(DEFAULT_EXCLUDE_DIRS), help="Comma-separated folder names to exclude")
    parser.add_argument("--chunk_size_mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB for file processing")
    parser.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS, help="Minimum words per line filter")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already exist in output directory")

    args = parser.parse_args()
    excludes = set(d.strip() for d in args.exclude_dirs.split(",") if d.strip())

    clean_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        exclude_dirs=excludes,
        chunk_size_mb=args.chunk_size_mb,
        min_words=args.min_words,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
