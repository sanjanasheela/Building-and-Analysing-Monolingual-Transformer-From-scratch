"""
Scraper for Nepali news websites NOT already in the existing dataset.
Sites already scraped (skip): gorkhapatra, kantipur, ratopati, setopati,
  onlinekhabar, deshsanchar, nepalpress, nayapatrika, etc.

New targets: ekantipur, nagariknews, annapurnapost, dcnepal,
  shilapatra, nepallivetoday, swasthyakhabar, nepalekhabar, sajha forums

Resumable: already-scraped sites are skipped automatically (file exists check).
"""
import re
import socket
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse

import warnings
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import trafilatura

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Global socket timeout — prevents any network call from hanging forever
socket.setdefaulttimeout(20)

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
OUTPUT_DIR = BASE / "data/raw/public/news"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research bot Nepali NLP)"
}
DELAY = 1.2
MIN_TEXT_LEN = 100

NEPALI_UNICODE_RANGE = ('\u0900', '\u097F')  # Devanagari block


def is_mostly_nepali(text: str, threshold: float = 0.3) -> bool:
    """Check if text contains enough Devanagari characters."""
    if not text:
        return False
    deva_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return deva_count / max(len(text), 1) >= threshold


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch_text_trafilatura(url: str) -> str:
    """Use trafilatura to extract main article text from URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False,
                                   include_tables=False, no_fallback=False)
        return text or ""
    except Exception:
        return ""


SITE_MAX_SECONDS = 45 * 60  # 45 minutes max per site


def fetch_links(url: str, same_domain: bool = True) -> list:
    """Fetch all <a href> links from a page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        base_domain = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if same_domain and urlparse(href).netloc != base_domain:
                continue
            if href.startswith("http"):
                links.append(href)
        return list(set(links))
    except Exception:
        return []


def save_texts(out_file: Path, texts: list, name: str):
    """Write texts to file, print summary."""
    if texts:
        out_file.write_text("\n\n---\n\n".join(texts), encoding="utf-8")
        print(f"  ✓ {name}: {len(texts)} articles → {out_file.stat().st_size//1000} KB")
    else:
        print(f"  ✗ {name}: no usable Nepali text found")


def scrape_site(name: str, start_urls: list, max_pages: int = 500):
    """Generic site scraper: BFS from start_urls, extract article text.
    Automatically skips if output already exists.
    Hard time limit per site: SITE_MAX_SECONDS.
    """
    out_file = OUTPUT_DIR / f"{name}.txt"
    if out_file.exists() and out_file.stat().st_size > 50_000:
        print(f"  SKIP {name} (already exists, {out_file.stat().st_size//1000} KB)")
        return

    print(f"\n=== Scraping: {name} ===", flush=True)
    visited = set()
    queue = list(start_urls)
    texts = []
    site_start = time.time()

    while queue and len(visited) < max_pages:
        # Hard time limit per site
        if time.time() - site_start > SITE_MAX_SECONDS:
            print(f"  ⏰ {name}: time limit reached, saving partial results")
            break

        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        text = fetch_text_trafilatura(url)
        text = clean_text(text)

        if text and len(text) >= MIN_TEXT_LEN and is_mostly_nepali(text):
            texts.append(text)
            if len(texts) % 50 == 0:
                print(f"  {name}: {len(texts)} articles, {len(visited)}/{max_pages} visited", flush=True)

        # Get more links from this page
        if len(visited) < max_pages:
            new_links = fetch_links(url)
            for link in new_links:
                if link not in visited:
                    queue.append(link)

        time.sleep(DELAY)

    save_texts(out_file, texts, name)


def scrape_ekantipur():
    """Ekantipur has a large archive — scrape category pages."""
    name = "ekantipur"
    out_file = OUTPUT_DIR / f"{name}.txt"
    if out_file.exists() and out_file.stat().st_size > 100_000:
        print(f"  SKIP {name}")
        return

    # Ekantipur category pages
    categories = [
        "https://ekantipur.com/news",
        "https://ekantipur.com/opinion",
        "https://ekantipur.com/literature",
        "https://ekantipur.com/sports",
        "https://ekantipur.com/science-technology",
        "https://ekantipur.com/business",
        "https://ekantipur.com/entertainment",
    ]
    start_urls = []
    # Get article links from category pages (paginate up to 20 pages each)
    for cat in categories:
        for page in range(1, 21):
            start_urls.append(f"{cat}?page={page}" if page > 1 else cat)

    scrape_site(name, start_urls, max_pages=1000)


# Site configurations: (name, start_urls, max_pages)
SITES = [
    (
        "nagariknews",
        ["https://nagariknews.nagariknetwork.com/",
         "https://nagariknews.nagariknetwork.com/category/news",
         "https://nagariknews.nagariknetwork.com/category/politics",
         "https://nagariknews.nagariknetwork.com/category/society"],
        600,
    ),
    (
        "annapurnapost",
        ["https://annapurnapost.com/",
         "https://annapurnapost.com/news",
         "https://annapurnapost.com/category/politics",
         "https://annapurnapost.com/category/economics",
         "https://annapurnapost.com/category/society"],
        600,
    ),
    (
        "dcnepal",
        ["https://www.dcnepal.com/",
         "https://www.dcnepal.com/news",
         "https://www.dcnepal.com/category/arts-and-literature"],
        400,
    ),
    (
        "shilapatra",
        ["https://shilapatra.com/",
         "https://shilapatra.com/category/news",
         "https://shilapatra.com/category/politics"],
        400,
    ),
    (
        "nepallivetoday",
        ["https://www.nepallivetoday.com/"],
        300,
    ),
    (
        "swasthyakhabar",
        ["https://swasthyakhabar.com/"],
        300,
    ),
    (
        "nepalekhabar",
        ["https://www.nepalekhabar.com/"],
        300,
    ),
    (
        "arthasarokar",
        ["https://arthasarokar.com/"],
        300,
    ),
    (
        "bbc_nepali",
        ["https://www.bbc.com/nepali",
         "https://www.bbc.com/nepali/topics",
         "https://www.bbc.com/nepali/news"],
        500,
    ),
    (
        "rfa_nepali",
        ["https://www.rfa.org/nepali/"],
        300,
    ),
    (
        "voa_nepali",
        ["https://www.voanepal.com/"],
        300,
    ),
    (
        "abhiyan",
        ["https://abhiyandaily.com/"],
        300,
    ),
    (
        "nepal_magazine",
        ["https://nepalmagazine.com.np/"],
        200,
    ),
    (
        "sajha_forum",
        ["https://www.sajha.com/sajha/html/forums.cfm"],
        400,
    ),
]


if __name__ == "__main__":
    print("=== Nepali News Scraper ===\n")

    # Ekantipur (special handling due to category structure)
    scrape_ekantipur()

    # All other sites
    for name, start_urls, max_pages in SITES:
        scrape_site(name, start_urls, max_pages)
        time.sleep(2)

    print("\n=== News scraping complete ===")
    total = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.txt"))
    print(f"Total: {total/1e6:.1f} MB in {OUTPUT_DIR}")
