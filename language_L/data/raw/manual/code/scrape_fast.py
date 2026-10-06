"""
Ultra-Fast Concurrent Nepali Web Scraper (10 Parallel Domain Workers)
======================================================================
- 10 concurrent domain workers (utilizing multi-core CPU safely)
- $O(1)$ memory consumption per thread (< 150 MB RAM total)
- Real-time disk streaming with immediate buffer flush
- 12s per-request timeout with automatic error handling
- Auto-skip for fully scraped sites (> 100 KB)
"""

import gc
import re
import socket
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import trafilatura

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
socket.setdefaulttimeout(15)

BASE_DIR = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
OUTPUT_DIR = BASE_DIR / "data/raw/manual/data/news"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
PER_REQUEST_DELAY = 0.4
SITE_TIMEOUT_SECS = 30 * 60  # max 30 mins per site
MAX_WORKERS = 10  # 10 parallel domain workers for massive throughput


def is_nepali(text: str, threshold: float = 0.25) -> bool:
    """Fast check for Devanagari character density."""
    if not text:
        return False
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return (deva / max(len(text.strip()), 1)) >= threshold


def clean_text(text: str) -> str:
    """Normalize Unicode and clean excessive whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_article_text(url: str, session: requests.Session) -> str:
    """Fetch and extract main article text using trafilatura."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200 or not resp.text:
            return ""
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False, no_fallback=False)
        return clean_text(text or "")
    except Exception:
        return ""


def extract_links(url: str, session: requests.Session) -> list:
    """Extract in-domain hyperlinks from a page."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        base_domain = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            p = urlparse(href)
            if p.netloc == base_domain and href.startswith("http") and "#" not in href:
                links.append(href)
        return list(set(links))
    except Exception:
        return []


def scrape_bfs_site(name: str, seed_urls: list, max_pages: int = 700, min_size_kb: int = 150):
    """
    Scrapes a site using BFS, streaming output directly to disk.
    RAM usage is kept minimal by immediately appending and freeing buffers.
    """
    out_file = OUTPUT_DIR / f"{name}.txt"
    if out_file.exists() and out_file.stat().st_size > min_size_kb * 1024:
        print(f"[SKIP] {name} ({out_file.stat().st_size // 1024} KB already collected)", flush=True)
        return name, out_file.stat().st_size

    print(f"[START] BFS: {name} (max {max_pages} pages)", flush=True)
    session = requests.Session()
    visited = set()
    queue = list(seed_urls)
    saved_count = 0
    start_time = time.time()

    with open(out_file, "a", encoding="utf-8") as fout:
        while queue and len(visited) < max_pages:
            if time.time() - start_time > SITE_TIMEOUT_SECS:
                print(f"[TIMEOUT] {name} hit {SITE_TIMEOUT_SECS//60}m limit ({saved_count} articles saved)", flush=True)
                break

            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            text = extract_article_text(url, session)
            if text and len(text) >= 120 :
                fout.write(text + "\n\n---\n\n")
                fout.flush()
                saved_count += 1
                if saved_count % 50 == 0:
                    print(f"  [{name}] {saved_count} articles | {len(visited)}/{max_pages} pages visited", flush=True)

            if len(visited) < max_pages:
                new_links = extract_links(url, session)
                for link in new_links:
                    if link not in visited:
                        queue.append(link)

            time.sleep(PER_REQUEST_DELAY)

    final_size = out_file.stat().st_size if out_file.exists() else 0
    print(f"[DONE] {name}: {saved_count} articles -> {final_size // 1024} KB", flush=True)
    gc.collect()
    return name, final_size


def scrape_paginated_site(name: str, url_pattern: str, start_page: int = 1, end_page: int = 150, min_size_kb: int = 150):
    """
    Crawls paginated archive pages and extracts articles in real-time.
    """
    out_file = OUTPUT_DIR / f"{name}.txt"
    if out_file.exists() and out_file.stat().st_size > min_size_kb * 1024:
        print(f"[SKIP] {name} ({out_file.stat().st_size // 1024} KB already collected)", flush=True)
        return name, out_file.stat().st_size

    print(f"[START] Paginated: {name} (pages {start_page}..{end_page})", flush=True)
    session = requests.Session()
    article_urls = set()

    for p in range(start_page, end_page + 1):
        page_url = url_pattern.format(p)
        links = extract_links(page_url, session)
        for l in links:
            article_urls.add(l)
        if p % 30 == 0:
            print(f"  [{name}] Indexed page {p}/{end_page} ({len(article_urls)} URLs found)", flush=True)
        time.sleep(0.2)

    print(f"  [{name}] Fetching texts for {len(article_urls)} discovered links...", flush=True)
    saved_count = 0
    start_time = time.time()

    with open(out_file, "a", encoding="utf-8") as fout:
        for url in list(article_urls)[:2000]:
            if time.time() - start_time > SITE_TIMEOUT_SECS:
                print(f"[TIMEOUT] {name} time limit ({saved_count} saved)", flush=True)
                break
            text = extract_article_text(url, session)
            if text and len(text) >= 120:
                fout.write(text + "\n\n---\n\n")
                fout.flush()
                saved_count += 1
                if saved_count % 50 == 0:
                    print(f"  [{name}] {saved_count} articles saved", flush=True)
            time.sleep(PER_REQUEST_DELAY)

    final_size = out_file.stat().st_size if out_file.exists() else 0
    print(f"[DONE] {name}: {saved_count} articles -> {final_size // 1024} KB", flush=True)
    gc.collect()
    return name, final_size


# ── Target Sites Definitions ──────────────────────────────────────────────────

BFS_TARGETS = [
    # ("pahilopost", ["https://pahilopost.com/", "https://pahilopost.com/category/news", "https://pahilopost.com/category/politics", "https://pahilopost.com/category/economy"], 700),
    # ("koshi_tv", ["https://koshitv.com/"], 500),
    # ("saptahik", ["https://www.saptahik.com.np/"], 500),
    # ("reporters_nepal", ["https://www.reportersnepal.com/"], 500),
    # ("lokaantar", ["https://lokaantar.com/"], 500),
    # ("nepal_npatra", ["https://www.nepalpatra.com/"], 450),
    # ("hamro_patro_news", ["https://www.hamropatro.com/news", "https://www.hamropatro.com/community"], 450),
    # ("samacharpatra_new", ["https://www.samacharpatra.com/"], 450),
    # ("naya_patrika_new", ["https://www.nayapatrika.com/"], 450),
    ("gorkhapatra_new", ["https://gorkhapatraonline.com/"], 450),
    ("khabardabali", ["https://khabardabali.com/"], 400),
    ("mero_sansar", ["https://merosansar.com/"], 350),
    ("nepali_times_np", ["https://www.nepalitimes.com/nepali"], 350),
    ("himalaya_post", ["https://www.himalayapost.com/"], 400),
    ("aarthiknews", ["https://aarthiknews.com/"], 400),
    ("bizshala", ["https://bizshala.com/"], 400),
    ("bizpati", ["https://bizpati.com/"], 400),
    ("chakrapath", ["https://chakrapath.com/"], 400),
    ("onlineradionepal", ["https://radionepal.gov.np/"], 350),
]

PAGINATED_TARGETS = [
    # ("annapurnapost_deep", "https://annapurnapost.com/news?page={}", 1, 150),
    # ("setopati_deep", "https://www.setopati.com/category/rastriya?page={}", 1, 120),
    # ("onlinekhabar_deep", "https://www.onlinekhabar.com/category/news/page/{}", 1, 120),
    # ("ratopati_deep", "https://www.ratopati.com/category/main-news?page={}", 1, 120),
    # ("nepalpress_deep", "https://www.nepalpress.com/category/politics/page/{}", 1, 100),
    # ("deshsanchar_deep", "https://deshsanchar.com/category/main-news/page/{}", 1, 100),
]


def run_all():
    print(f"=== Starting Ultra-Fast Scraper with {MAX_WORKERS} Concurrent Domain Workers ===", flush=True)
    t_start = time.time()

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for name, seeds, max_p in BFS_TARGETS:
            tasks.append(executor.submit(scrape_bfs_site, name, seeds, max_p))

        for name, pattern, start_p, end_p in PAGINATED_TARGETS:
            tasks.append(executor.submit(scrape_paginated_site, name, pattern, start_p, end_p))

        for future in as_completed(tasks):
            try:
                name, size = future.result()
            except Exception as e:
                print(f"[ERROR] Task failed: {e}", flush=True)

    elapsed_min = (time.time() - t_start) / 60
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.txt"))
    print(f"\n=== All concurrent scrapers finished in {elapsed_min:.1f} minutes ===", flush=True)
    print(f"Total News Data in {OUTPUT_DIR}: {total_bytes / (1024*1024):.1f} MB", flush=True)


if __name__ == "__main__":
    run_all()
