"""
Scraper for Nepali government, legal, and educational websites.
Downloads PDFs where available and OCRs them; otherwise scrapes HTML text.
"""
import os, re, time, unicodedata, warnings
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
import trafilatura

os.environ["TESSDATA_PREFIX"] = "/home/sanjana/.local/share/tessdata"

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
OUTPUT_DIR = BASE / "data/raw/public/government"
EDU_DIR = BASE / "data/raw/public/educational"
PDF_CACHE = BASE / "data/raw/public/_pdf_cache_govt"
for d in [OUTPUT_DIR, EDU_DIR, PDF_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research; Nepali NLP)"}
DELAY = 1.5


def is_devanagari(text, thresh=0.2):
    count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return count / max(len(text), 1) >= thresh


def clean(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize("NFC", text or "")).strip()


def fetch(url):
    try:
        dl = trafilatura.fetch_url(url)
        return clean(trafilatura.extract(dl) or "") if dl else ""
    except Exception:
        return ""


def get_links(url, pdf_only=False):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")
        domain = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if pdf_only and not href.lower().endswith(".pdf"):
                continue
            if urlparse(href).netloc == domain or pdf_only:
                links.append(href)
        return list(set(links))
    except Exception:
        return []


def download_file(url, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        if r.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        return True
    except Exception:
        return False


def ocr_pdf(pdf_path, out_path):
    if out_path.exists() and out_path.stat().st_size > 500:
        return
    try:
        import pytesseract
        from pdf2image import convert_from_path, pdfinfo_from_path
        pages = pdfinfo_from_path(str(pdf_path))["Pages"]
        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(1, pages + 1):
                imgs = convert_from_path(str(pdf_path), dpi=250, first_page=i, last_page=i)
                f.write(pytesseract.image_to_string(imgs[0], lang="nep") + "\n\n")
        print(f"    OCR done: {out_path.name} ({out_path.stat().st_size//1000} KB)")
    except Exception as e:
        print(f"    OCR failed: {e}")


def scrape_html_site(name, seeds, out_dir, max_pages=300):
    out = out_dir / f"{name}.txt"
    if out.exists() and out.stat().st_size > 20_000:
        print(f"  SKIP {name}"); return
    print(f"\n=== {name} ===")
    visited, queue, texts = set(), list(seeds), []
    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited: continue
        visited.add(url)
        t = fetch(url)
        if t and len(t) >= 80 and is_devanagari(t):
            texts.append(t)
        for link in get_links(url):
            if link not in visited: queue.append(link)
        time.sleep(DELAY)
    if texts:
        out.write_text("\n\n---\n\n".join(texts), encoding="utf-8")
        print(f"  ✓ {name}: {len(texts)} pages → {out.stat().st_size//1000} KB")
    else:
        print(f"  ✗ {name}: no content")


def scrape_pdfs_from_site(name, seed_urls, out_dir, max_pdfs=30):
    """Find all PDF links on a site, download, OCR, save text."""
    print(f"\n=== {name} (PDF harvest) ===")
    out_combined = out_dir / f"{name}_pdfs.txt"
    if out_combined.exists() and out_combined.stat().st_size > 20_000:
        print(f"  SKIP {name} PDFs"); return

    pdf_links = set()
    for seed in seed_urls:
        links = get_links(seed, pdf_only=True)
        pdf_links.update(links)
        time.sleep(DELAY)

    print(f"  Found {len(pdf_links)} PDFs")
    all_texts = []
    for i, url in enumerate(list(pdf_links)[:max_pdfs]):
        fname = re.sub(r'[^\w\-.]', '_', url.split("/")[-1])[:80]
        pdf_dest = PDF_CACHE / f"{name}_{fname}"
        txt_dest = PDF_CACHE / f"{name}_{fname}.txt"
        if download_file(url, pdf_dest):
            ocr_pdf(pdf_dest, txt_dest)
            if txt_dest.exists():
                t = txt_dest.read_text(encoding="utf-8", errors="ignore")
                if is_devanagari(t):
                    all_texts.append(t)
        time.sleep(DELAY)

    if all_texts:
        out_combined.write_text("\n\n---\n\n".join(all_texts), encoding="utf-8")
        print(f"  ✓ {name} PDFs: {len(all_texts)} docs → {out_combined.stat().st_size//1000} KB")


# Government HTML sites
GOVT_SITES = [
    ("parliament_np", ["https://www.parliament.gov.np/ne/", "https://www.parliament.gov.np/ne/news"], OUTPUT_DIR, 300),
    ("moe_gov", ["https://moe.gov.np/"], OUTPUT_DIR, 200),
    ("ocmcm_gov", ["https://ocmcm.gov.np/ne/"], OUTPUT_DIR, 200),
    ("cec_gov", ["https://election.gov.np/ne/"], OUTPUT_DIR, 200),
    ("nhrc_org", ["https://nhrc.org.np/"], OUTPUT_DIR, 200),
    ("nasc_org", ["https://nasc.org.np/"], OUTPUT_DIR, 200),
    ("moha_gov", ["https://moha.gov.np/"], OUTPUT_DIR, 200),
]

# Educational PDF sites
EDU_PDF_SITES = [
    ("elibrary_gov", ["https://elibrary.moe.gov.np/"], EDU_DIR, 40),
    ("curriculum_gov", ["https://www.curriculum.gov.np/"], EDU_DIR, 30),
    ("trc_gov", ["https://trc.gov.np/"], EDU_DIR, 30),
]

# Educational HTML sites
EDU_HTML_SITES = [
    ("ku_edu", ["https://ku.edu.np/", "https://ku.edu.np/news"], EDU_DIR, 200),
    ("tu_edu", ["https://tribhuvan-university.edu.np/"], EDU_DIR, 200),
    ("purbanchal_edu", ["https://purbanchaluniversity.edu.np/"], EDU_DIR, 150),
]


if __name__ == "__main__":
    print("=== Government & Educational Scraper ===")

    for name, seeds, out_dir, mp in GOVT_SITES:
        scrape_html_site(name, seeds, out_dir, mp)
        time.sleep(2)

    for name, seeds, out_dir, max_p in EDU_PDF_SITES:
        scrape_pdfs_from_site(name, seeds, out_dir, max_p)
        time.sleep(2)

    for name, seeds, out_dir, mp in EDU_HTML_SITES:
        scrape_html_site(name, seeds, out_dir, mp)
        time.sleep(2)

    total_g = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.txt"))
    total_e = sum(f.stat().st_size for f in EDU_DIR.glob("*.txt"))
    print(f"\nGovernment: {total_g/1e6:.1f} MB | Educational: {total_e/1e6:.1f} MB")
