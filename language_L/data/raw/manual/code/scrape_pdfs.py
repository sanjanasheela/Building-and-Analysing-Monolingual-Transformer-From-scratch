"""
PDF Downloader & Extractor for Nepali Books/Textbooks
======================================================
Downloads pure Nepali PDFs (books, textbooks, legal docs) and extracts text.
- Uses `pdftotext` (poppler-utils) for blazing-fast extraction.
- Keeps ALL downloaded PDFs in the `books_ocr` folder as requested.
- If a PDF is a scanned image (pdftotext yields no text), it stays in the folder
  so your existing `ocr_nepali.py` script can process it later!
"""

import os
import re
import time
import subprocess
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
BOOKS_DIR = BASE_DIR / "data/raw/manual/data/books_ocr"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Research Bot (Nepali NLP)"}
MAX_WORKERS = 6

# We search Internet Archive for Nepali language items that have PDFs
IA_API = "https://archive.org/advancedsearch.php"
QUERIES = [
    'language:"Nepali" AND format:"Text PDF"',
    'subject:"Nepal" AND format:"Text PDF" AND language:"Nepali"',
    'subject:"Education" AND language:"Nepali"'
]

def is_nepali(text, thresh=0.15):
    if not text: return False
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return (deva / max(len(text.strip()), 1)) >= thresh

def download_and_extract(identifier, title):
    safe_title = re.sub(r'[^\w\-]', '_', title)[:60]
    pdf_path = BOOKS_DIR / f"{safe_title}_{identifier}.pdf"
    txt_path = BOOKS_DIR / f"{safe_title}_{identifier}.txt"
    
    if pdf_path.exists() or txt_path.exists():
        return f"[SKIP] {safe_title} (already exists)"
        
    try:
        # Fetch file list to find the actual PDF URL
        meta_url = f"https://archive.org/metadata/{identifier}/files"
        r = requests.get(meta_url, headers=HEADERS, timeout=20)
        files = {f["name"]: f for f in r.json().get("result", [])}
        
        pdf_file = next((k for k in files if k.endswith(".pdf")), None)
        if not pdf_file:
            return f"[SKIP] No PDF found for {identifier}"
            
        download_url = f"https://archive.org/download/{identifier}/{requests.utils.quote(pdf_file)}"
        
        # Download the PDF
        print(f"  Downloading PDF: {safe_title}...", flush=True)
        resp = requests.get(download_url, headers=HEADERS, stream=True, timeout=60)
        if resp.status_code == 200:
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=2048*1024):
                    f.write(chunk)
                    
            # Run pdftotext (leaves the PDF intact!)
            subprocess.run(["pdftotext", str(pdf_path), str(txt_path)], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                           
            # Check if extraction was successful and contains Nepali
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
                if is_nepali(text[:3000]):
                    # Clean the text file
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    txt_path.write_text(text, encoding="utf-8")
                    size = txt_path.stat().st_size // 1024
                    return f"[DONE] {safe_title} -> PDF saved & {size} KB text extracted."
                else:
                    # Not enough Devanagari text, it's likely a scanned image PDF!
                    txt_path.unlink() # Remove useless empty text file
                    return f"[SCANNED PDF] {safe_title} -> PDF saved for OCR later!"
            else:
                return f"[SCANNED PDF] {safe_title} -> PDF saved for OCR later!"
    except Exception as e:
        # If it fails, clean up partial downloads
        if pdf_path.exists(): pdf_path.unlink()
        return f"[ERROR] Failed {identifier}: {e}"
        
    return f"[FAIL] Could not download {identifier}"

def run():
    print(f"=== Starting PDF Book Downloader (Keeping PDFs in {BOOKS_DIR.name}) ===", flush=True)
    identifiers = set()
    
    # Collect IDs from Archive.org
    for q in QUERIES:
        for page in range(1, 5):
            try:
                params = {"q": q, "fl[]": ["identifier", "title"], "rows": 50, "page": page, "output": "json"}
                r = requests.get(IA_API, params=params, headers=HEADERS, timeout=20)
                docs = r.json().get("response", {}).get("docs", [])
                if not docs: break
                for d in docs:
                    identifiers.add((d["identifier"], d.get("title", d["identifier"])))
                time.sleep(1)
            except Exception as e:
                print(f"Search error: {e}")
                
    print(f"Found {len(identifiers)} potential Nepali PDFs. Downloading...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_and_extract, iid, title) for iid, title in identifiers]
        for f in as_completed(futures):
            print(f.result(), flush=True)
            
    print("\n=== PDF Download Complete ===")
    print("Check the books_ocr folder! Digital PDFs have text files, scanned PDFs are ready for your OCR script.")

if __name__ == "__main__":
    run()
