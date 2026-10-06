import os
import gc
import time

from clean_pipeline import run_pipeline


# ──────────────────────────────────────────────
# CONFIG — edit these two paths
# ──────────────────────────────────────────────
INPUT_DIR = (
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/1"
)

OUTPUT_DIR = (
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/processed/manual/llm/1"
)

# Chunk size for large files (MB)
CHUNK_SIZE_MB = 50

MIN_WORDS = 3
# ──────────────────────────────────────────────


def clean_file(input_path: str, output_path: str) -> None:
    """Clean a single file in chunks and write to output_path."""

    chunk_size = CHUNK_SIZE_MB * 1024 * 1024

    file_size = os.path.getsize(input_path)

    bytes_processed = 0
    lines_processed = 0
    chunks = 0

    start_time = time.time()

    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Size  : {file_size / (1024**2):.2f} MB  |  chunk: {CHUNK_SIZE_MB} MB")

    with open(
        input_path, "r", encoding="utf-8", errors="ignore"
    ) as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:

        buffer = []
        buffer_size = 0

        for line in infile:

            buffer.append(line)
            line_size = len(line.encode("utf-8"))
            buffer_size += line_size
            bytes_processed += line_size
            lines_processed += 1

            # ── flush chunk ──────────────────────────────
            if buffer_size >= chunk_size:

                chunks += 1
                text = "".join(buffer)

                cleaned = run_pipeline(text, min_words=MIN_WORDS)

                if cleaned:
                    outfile.write(cleaned)
                    outfile.write("\n\n")
                    outfile.flush()

                del text, cleaned, buffer
                buffer = []
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
            text = "".join(buffer)
            cleaned = run_pipeline(text, min_words=MIN_WORDS)
            if cleaned:
                outfile.write(cleaned)
                outfile.write("\n\n")
                outfile.flush()
            del text, cleaned, buffer
            gc.collect()

    elapsed = time.time() - start_time
    print(
        f"  Done  : {chunks} chunk(s), {lines_processed:,} lines, "
        f"{elapsed:.1f} s\n"
    )


def clean_folder() -> None:
    """Iterate over every .txt file in INPUT_DIR and clean it."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".txt")
    )

    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    total_start = time.time()

    print("=" * 60)
    print(f"Input  folder : {INPUT_DIR}")
    print(f"Output folder : {OUTPUT_DIR}")
    print(f"Files found   : {len(txt_files)}")
    print("=" * 60)
    print()

    for idx, fname in enumerate(txt_files, 1):
        input_path  = os.path.join(INPUT_DIR,  fname)
        output_path = os.path.join(OUTPUT_DIR, fname)

        print(f"[{idx}/{len(txt_files)}]  {fname}")
        clean_file(input_path, output_path)

    total_elapsed = time.time() - total_start

    print("=" * 60)
    print("ALL FILES DONE")
    print(f"Total time : {total_elapsed / 60:.1f} minutes")
    print(f"Output dir : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    clean_folder()