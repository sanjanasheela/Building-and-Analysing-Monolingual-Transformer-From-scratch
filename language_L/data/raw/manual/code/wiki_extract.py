"""
Step 1: Process Nepali Wikipedia XML dump using wikiextractor.
Step 2: Consolidate the already-extracted wiki chunk files (AA/wiki_00 etc.)
Outputs clean plain text to public/wikipedia/
"""
import json
import os
import subprocess
from pathlib import Path

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
XML_DUMP = BASE / "data/raw/manual/sources/newiki-latest-pages-articles.xml"
CHUNKS_DIR = BASE / "data/raw/manual/sources/text"
OUTPUT_DIR = BASE / "data/raw/public/wikipedia"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTPUT_DIR / "extracted_json"
WIKI_TEXT_OUT = OUTPUT_DIR / "nepali_wiki_from_dump.txt"
CHUNKS_TEXT_OUT = OUTPUT_DIR / "nepali_wiki_from_chunks.txt"


def run_wikiextractor():
    """Run wikiextractor on the XML dump to get JSON output."""
    if WIKI_TEXT_OUT.exists():
        print(f"Wiki dump text already extracted: {WIKI_TEXT_OUT}")
        return

    print(f"Running wikiextractor on {XML_DUMP.name} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/home/sanjana/.local/bin/wikiextractor",
        str(XML_DUMP),
        "--output", str(EXTRACT_DIR),
        "--bytes", "100M",
        "--json",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("wikiextractor failed!")
        return

    print("Wikiextractor done. Flattening JSON to plain text...")
    flatten_json_to_text(EXTRACT_DIR, WIKI_TEXT_OUT)


def flatten_json_to_text(json_dir: Path, out_file: Path):
    """Convert wikiextractor JSON files to plain text."""
    count = 0
    with open(out_file, "w", encoding="utf-8") as fout:
        for fpath in sorted(json_dir.rglob("wiki_*")):
            if fpath.is_file():
                with open(fpath, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "").strip()
                            if text:
                                fout.write(text)
                                fout.write("\n\n")
                                count += 1
                        except json.JSONDecodeError:
                            continue
    size_mb = out_file.stat().st_size / 1e6
    print(f"  ✓ Wrote {count} articles → {out_file.name} ({size_mb:.1f} MB)")


def consolidate_chunks():
    """Merge already-extracted wiki chunk files (AA/wiki_00 etc.) into one file."""
    if CHUNKS_TEXT_OUT.exists():
        print(f"Chunks already consolidated: {CHUNKS_TEXT_OUT}")
        return

    chunk_files = sorted(CHUNKS_DIR.rglob("wiki_*"))
    print(f"Found {len(chunk_files)} chunk files to consolidate...")

    total_bytes = 0
    with open(CHUNKS_TEXT_OUT, "w", encoding="utf-8") as fout:
        for cf in chunk_files:
            if cf.is_file():
                try:
                    text = cf.read_text(encoding="utf-8", errors="ignore")
                    fout.write(text)
                    fout.write("\n\n")
                    total_bytes += len(text)
                except Exception as e:
                    print(f"  Warning: could not read {cf}: {e}")

    size_mb = CHUNKS_TEXT_OUT.stat().st_size / 1e6
    print(f"  ✓ Consolidated chunks → {CHUNKS_TEXT_OUT.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("=== Step 1: Consolidate existing wiki chunks ===")
    consolidate_chunks()

    print("\n=== Step 2: Extract from XML dump ===")
    run_wikiextractor()

    print("\n=== Wikipedia extraction complete ===")
    for f in OUTPUT_DIR.glob("*.txt"):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
