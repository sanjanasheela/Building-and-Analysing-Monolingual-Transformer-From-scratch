#!/usr/bin/env python3
"""
Master runner: executes all Nepali data collection steps in sequence.
Run this script and go to sleep — it handles everything.

Steps:
  1. Wiki extraction (existing XML dump + chunk consolidation)
  2. OCR of existing PDFs in sources/books/
  3. Internet Archive download + OCR
  4. News website scraping
  5. Literature + Wikisource scraping
  6. Government + Educational scraping

Logs are written to data/raw/collection_log.txt
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
CODE_DIR = BASE / "data/raw/manual/code"
LOG_FILE = BASE / "data/raw/collection_log.txt"

STEPS = [
    ("Wiki Extraction",          CODE_DIR / "wiki_extract.py"),
    ("OCR Nepali Books",         CODE_DIR / "ocr_nepali.py"),
    ("Internet Archive",         CODE_DIR / "download_ia_pdfs.py"),
    ("News Scraping",            CODE_DIR / "scrape_news.py"),
    ("Literature Scraping",      CODE_DIR / "scrape_literature.py"),
    ("Govt/Educational Scraping",CODE_DIR / "scrape_govt.py"),
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, script_path):
    log(f"=== START: {name} ===")
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE),
        timeout=3600 * 8,  # 8 hour max per step
    )
    elapsed = (time.time() - start) / 60
    if result.returncode == 0:
        log(f"=== DONE: {name} ({elapsed:.1f} min) ===")
    else:
        log(f"=== FAILED: {name} (exit {result.returncode}, {elapsed:.1f} min) ===")


def print_summary():
    log("\n=== COLLECTION SUMMARY ===")
    dirs_to_check = [
        ("Wikipedia", BASE / "data/raw/public/wikipedia"),
        ("Books OCR", BASE / "data/raw/public/books_ocr"),
        ("Internet Archive", BASE / "data/raw/public/internet_archive"),
        ("News", BASE / "data/raw/public/news"),
        ("Literature", BASE / "data/raw/public/literature"),
        ("Government", BASE / "data/raw/public/government"),
        ("Educational", BASE / "data/raw/public/educational"),
    ]
    total = 0
    for label, d in dirs_to_check:
        if d.exists():
            size = sum(f.stat().st_size for f in d.glob("*.txt"))
            total += size
            log(f"  {label}: {size/1e6:.1f} MB")
        else:
            log(f"  {label}: (not created)")
    log(f"  GRAND TOTAL: {total/1e6:.1f} MB of text")


if __name__ == "__main__":
    log("====== Nepali Data Collection Started ======")
    for name, script in STEPS:
        try:
            run_step(name, script)
        except subprocess.TimeoutExpired:
            log(f"=== TIMEOUT: {name} — moving to next step ===")
        except Exception as e:
            log(f"=== ERROR in {name}: {e} — moving to next step ===")
        time.sleep(5)

    print_summary()
    log("====== All collection complete ======")
