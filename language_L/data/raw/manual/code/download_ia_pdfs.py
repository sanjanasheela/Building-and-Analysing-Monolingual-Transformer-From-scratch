"""
Internet Archive scraper for Nepali language books.
Strategy:
  1. Search IA for Nepali-language texts via the search API
  2. For each item, prefer downloading _djvu.txt (pre-extracted text, no OCR needed)
  3. If no djvu.txt, download the PDF and OCR it with tesseract nep
"""
import os
import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

os.environ["TESSDATA_PREFIX"] = "/home/sanjana/.local/share/tessdata"

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
OUTPUT_DIR = BASE / "data/raw/public/internet_archive"
PDF_CACHE = OUTPUT_DIR / "_pdf_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_CACHE.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research bot - Nepali NLP corpus collection)"
}

# IA search queries targeting Nepali content
IA_SEARCHES = [
    "subject:Nepali+language+mediatype:texts",
    "subject:Nepali+literature+mediatype:texts",
    "language:nep+mediatype:texts",
    "language:Nepali+mediatype:texts",
    "subject:Nepal+mediatype:texts+language:nep",
    "subject:Nepali+grammar+mediatype:texts",
    "subject:Nepali+poetry+mediatype:texts",
    "collection:digitallibraryindia+language:nep",
    "subject:Nepali+history+mediatype:texts",
]

# Known good IA identifiers with plain text already available
KNOWN_ITEMS = [
    "garud-puran-nepali-bhasha-tika",
    "mahabharat-sampoorna-18-parva-nepali",
    "ramayan-in-nepali-bhanubhakta-acharya",
    "swasthani-brata-katha",
    # Additional known items
    "NepaliShabdakosha",
    "nepali-bhashaको-vyakaran",
]

MAX_ITEMS_PER_SEARCH = 50
DELAY = 1.5  # seconds between requests


def search_ia(query: str, rows: int = 50) -> list:
    """Search Internet Archive and return list of identifiers."""
    url = (
        f"https://archive.org/advancedsearch.php"
        f"?q={query}&fl[]=identifier&fl[]=title&fl[]=language"
        f"&rows={rows}&page=1&output=json"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        data = r.json()
        docs = data.get("response", {}).get("docs", [])
        return docs
    except Exception as e:
        print(f"  Search failed for '{query}': {e}")
        return []


def get_item_files(identifier: str) -> dict:
    """Get file listing for an IA item. Returns dict of filename -> download URL."""
    url = f"https://archive.org/metadata/{identifier}/files"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        data = r.json()
        files = {}
        for f in data.get("result", []):
            name = f.get("name", "")
            files[name] = f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}"
        return files
    except Exception as e:
        print(f"  Could not get files for {identifier}: {e}")
        return {}


def download_file(url: str, dest: Path) -> bool:
    """Download a file from url to dest."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        if r.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Download error: {e}")
        return False


def ocr_pdf_to_text(pdf_path: Path, out_path: Path):
    """OCR a PDF using pytesseract nep lang."""
    if out_path.exists() and out_path.stat().st_size > 500:
        return
    try:
        import pytesseract
        from pdf2image import convert_from_path, pdfinfo_from_path

        total_pages = pdfinfo_from_path(str(pdf_path))["Pages"]
        print(f"    OCR: {pdf_path.name} ({total_pages} pages)")
        with open(out_path, "w", encoding="utf-8") as fout:
            for i in range(1, total_pages + 1):
                imgs = convert_from_path(str(pdf_path), dpi=250, first_page=i, last_page=i)
                text = pytesseract.image_to_string(imgs[0], lang="nep")
                fout.write(text + "\n\n")
                if i % 30 == 0:
                    print(f"      page {i}/{total_pages}")
        size_kb = out_path.stat().st_size / 1000
        print(f"    ✓ OCR done: {out_path.name} ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"    ✗ OCR failed: {e}")


def process_item(identifier: str, title: str = ""):
    """Download text from one IA item."""
    safe_id = re.sub(r"[^\w\-]", "_", identifier)
    out_path = OUTPUT_DIR / f"{safe_id}.txt"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  SKIP (exists): {identifier}")
        return

    print(f"  Processing: {identifier}  ({title})")
    files = get_item_files(identifier)
    if not files:
        return

    time.sleep(DELAY)

    # Priority 1: _djvu.txt — already plain text
    djvu_txt = {k: v for k, v in files.items() if k.endswith("_djvu.txt")}
    if djvu_txt:
        fname, url = next(iter(djvu_txt.items()))
        print(f"    Found djvu.txt: {fname}")
        tmp = PDF_CACHE / f"{safe_id}_djvu.txt"
        if download_file(url, tmp):
            text = tmp.read_text(encoding="utf-8", errors="ignore")
            # djvu.txt has page separators — clean up
            text = re.sub(r"\x0c", "\n", text)
            out_path.write_text(text, encoding="utf-8")
            size_kb = out_path.stat().st_size / 1000
            print(f"    ✓ Saved djvu text: {out_path.name} ({size_kb:.0f} KB)")
        return

    # Priority 2: _text.pdf — searchable PDF (try pdftotext first)
    text_pdfs = {k: v for k, v in files.items() if k.endswith("_text.pdf")}
    if text_pdfs:
        fname, url = next(iter(text_pdfs.items()))
        print(f"    Found text PDF: {fname}")
        pdf_dest = PDF_CACHE / f"{safe_id}_text.pdf"
        if download_file(url, pdf_dest):
            ocr_pdf_to_text(pdf_dest, out_path)
        return

    # Priority 3: main PDF — OCR it
    main_pdfs = {k: v for k, v in files.items()
                 if k.lower().endswith(".pdf") and "_text" not in k.lower()}
    if main_pdfs:
        # Pick smallest PDF if multiple
        fname = sorted(main_pdfs.keys())[0]
        url = main_pdfs[fname]
        print(f"    Found main PDF: {fname}")
        pdf_dest = PDF_CACHE / f"{safe_id}.pdf"
        if download_file(url, pdf_dest):
            # Only OCR if < 100 MB to avoid hour-long waits
            if pdf_dest.stat().st_size < 100 * 1e6:
                ocr_pdf_to_text(pdf_dest, out_path)
            else:
                print(f"    PDF too large for quick OCR ({pdf_dest.stat().st_size/1e6:.0f} MB), skipping")
        return

    print(f"    No usable text format found for {identifier}")


def main():
    seen_ids = set()

    # Add known items first
    for item_id in KNOWN_ITEMS:
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            process_item(item_id)
            time.sleep(DELAY)

    # Search IA for more items
    for query in IA_SEARCHES:
        print(f"\n--- Searching IA: {query} ---")
        docs = search_ia(query, rows=MAX_ITEMS_PER_SEARCH)
        print(f"  Found {len(docs)} items")
        for doc in docs:
            identifier = doc.get("identifier", "")
            title = doc.get("title", "")
            if identifier and identifier not in seen_ids:
                seen_ids.add(identifier)
                process_item(identifier, title)
                time.sleep(DELAY)

    print("\n=== Internet Archive download complete ===")
    txt_files = list(OUTPUT_DIR.glob("*.txt"))
    total = sum(f.stat().st_size for f in txt_files)
    print(f"Total: {len(txt_files)} files, {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
