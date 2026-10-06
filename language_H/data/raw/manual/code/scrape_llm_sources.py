"""
Telugu LLM Corpus Scraper v2 - Resilient multi-source scraper
- Short timeouts (10s) to skip stuck sites quickly
- Focuses on high-yield pages with direct article links
- Output: data/raw/manual/data/llm/
"""

import os, time, re, requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse

OUTPUT_DIR = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')

def is_telugu(text, threshold=0.12):
    if not text or len(text) < 50: return False
    return len(TELUGU_RE.findall(text)) / max(len(text), 1) >= threshold

def save(output_file, text):
    path = os.path.join(OUTPUT_DIR, output_file)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n\n---\n\n")

def fetch_text(url, timeout=10):
    try:
        downloaded = trafilatura.fetch_url(url, config=trafilatura.settings.use_config())
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text
    except Exception:
        pass
    # Fallback: requests + BS4
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        soup = BeautifulSoup(r.content, "html.parser", from_encoding="utf-8")
        for tag in soup(["script", "style", "nav", "header", "footer"]): tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        pass
    return None

def get_links(url, timeout=8):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        soup = BeautifulSoup(r.content, "html.parser")
        base = urlparse(url).netloc
        links = set()
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if urlparse(href).netloc == base and href.startswith("http"):
                links.add(href)
        return links
    except Exception:
        return set()

def scrape_site(name, seeds, output_file, max_pages=200, delay=0.8):
    print(f"\n{'='*50}\n{name}")
    visited, queue = set(), list(seeds)
    count = 0
    while queue and count < max_pages:
        url = queue.pop(0)
        if url in visited: continue
        visited.add(url)
        text = fetch_text(url)
        if text and is_telugu(text) and len(text) > 300:
            save(output_file, text)
            count += 1
            kb = os.path.getsize(os.path.join(OUTPUT_DIR, output_file)) / 1024
            print(f"  [{count}] {kb:.0f}KB | {url[:70]}")
            for link in get_links(url):
                if link not in visited: queue.append(link)
        time.sleep(delay)
    total = os.path.getsize(os.path.join(OUTPUT_DIR, output_file)) if os.path.exists(os.path.join(OUTPUT_DIR, output_file)) else 0
    print(f"  Done: {count} pages, {total/1024:.0f} KB")

# ── 1. Andhra Bhoomi ────────────────────────────────────────────

# ── 2. Nava Telangana ────────────────────────────────────────────

# ── 3. Sakshi Education ──────────────────────────────────────────

# ── 4. Kinige ────────────────────────────────────────────────────

# ── 7. Prabha News ───────────────────────────────────────────────
scrape_site("Prabha News", [
    "https://www.prabhanews.com/",
    "https://www.prabhanews.com/andhra-pradesh",
    "https://www.prabhanews.com/telangana",
    "https://www.prabhanews.com/national",
    "https://www.prabhanews.com/sports",
], "prabhanews.txt", max_pages=300)

# ── 10. Telugu Quotes & Poetry ───────────────────────────────────
scrape_site("Telugu Quotes/Poetry", [
    "https://www.teluguquotes.in/",
    "https://telugupadyalu.com/",
    "https://www.telugupoems.com/",
    "https://padyam.net/",
    "https://www.ekoumudi.net/",
], "telugu_poetry.txt", max_pages=200)

# ── 11. AP Govt Press Releases ───────────────────────────────────
scrape_site("AP Government", [
    "https://www.ap.gov.in/te/press-releases/",
    "https://www.ap.gov.in/te/programmes/",
    "https://www.ap.gov.in/te/",
    "https://goir.ap.gov.in/",
    "https://apsa.ap.gov.in/",
], "ap_govt.txt", max_pages=200)

# ── 12. Telangana Govt ──────────────────────────────────────────
scrape_site("Telangana Government", [
    "https://www.telangana.gov.in/",
    "https://www.telangana.gov.in/news/",
    "https://www.telangana.gov.in/press-releases/",
    "https://tgnns.telangana.gov.in/",
], "telangana_govt.txt", max_pages=200)

# ── 13. SCERT Telangana ─────────────────────────────────────────
scrape_site("SCERT Telangana", [
    "https://scert.telangana.gov.in/",
    "https://scert.telangana.gov.in/textbooks",
    "https://scert.telangana.gov.in/news",
], "scert_telangana.txt", max_pages=150)

# ── 14. Eemaata Magazine Extra ──────────────────────────────────
scrape_site("Eemaata Extra", [
    "https://www.eemaata.com/",
    "https://www.eemaata.com/em/issues/",
    "https://www.eemaata.com/em/columns/",
    "https://www.eemaata.com/em/kavitalu/",
], "eemaata_extra.txt", max_pages=300)

# ── 15. EMESCO Books ────────────────────────────────────────────
scrape_site("EMESCO", [
    "https://www.emescobooks.com/",
    "https://www.emescobooks.com/telugu-books/",
    "https://www.emescobooks.com/category/fiction/",
    "https://www.emescobooks.com/category/non-fiction/",
], "emesco.txt", max_pages=200)

# ── 16. Janam Sakshi ────────────────────────────────────────────
scrape_site("Janam Sakshi", [
    "https://www.janamsakshi.com/",
    "https://www.janamsakshi.com/telangana/",
    "https://www.janamsakshi.com/andhra-pradesh/",
    "https://www.janamsakshi.com/national/",
], "janamsakshi.txt", max_pages=300)

# ── 17. Mangalam Telugu ─────────────────────────────────────────
scrape_site("Mangalam Telugu", [
    "https://te.mangalam.com/",
    "https://te.mangalam.com/entertainment/",
    "https://te.mangalam.com/news/",
    "https://te.mangalam.com/spirituality/",
    "https://te.mangalam.com/health/",
], "mangalam_telugu.txt", max_pages=300)

# ── 18. Silicon Andhra ──────────────────────────────────────────
scrape_site("Silicon Andhra Sujanaranjani", [
    "http://sujanaranjani.siliconandhra.org/",
    "https://www.siliconandhra.org/",
], "siliconandhra.txt", max_pages=200)

# ── 19. Telugu Wikipedia article dump (deep portal links) ────────

# ── 20. Vaartha Extra ────────────────────────────────────────────

# ── Final summary ─────────────────────────────────────────────────
print("\n" + "="*60)
total = 0
for f in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f)
    s = os.path.getsize(fp)
    total += s
    print(f"  {f:40s}  {s/1024/1024:.2f} MB")
print(f"\n  TOTAL: {total/1024/1024:.2f} MB")
