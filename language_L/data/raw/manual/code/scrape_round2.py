"""
Round 2 scraper — targets high-yield Nepali sources not yet scraped.
Focus: large-volume sites with deep archives.

Targets:
- Annapurna Post (paginated archive — deep crawl)
- Nagarik News deeper crawl
- Nepal Samacharpatra deeper crawl
- Hamro Patro (large Nepali content portal)
- Nepali Times
- Khabarhub deeper crawl
- Pahilopost
- Koshi TV
- Nepal Banda / Loktantra
- Ujyalo Online
- Global IME blog
- Nepal Rastriya Samachar Samiti (RSS news agency)
- Mero Lagani (finance but Nepali)
- Nepal Patra
- Saptahik (weekly magazine)
- Nepal Live
- Reporters Nepal
"""
import json
import re
import socket
import time
import unicodedata
import warnings
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import trafilatura

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
socket.setdefaulttimeout(20)

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
OUTPUT_DIR = BASE / "data/raw/manual/data/news"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
DELAY = 1.0
SITE_MAX_SECS = 50 * 60  # 50 min per site


def is_nepali(text, thresh=0.25):
    if not text:
        return False
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return deva / max(len(text.strip()), 1) >= thresh


def clean(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch(url):
    try:
        dl = trafilatura.fetch_url(url)
        if not dl:
            return ""
        text = trafilatura.extract(dl, include_comments=False,
                                   include_tables=False, no_fallback=False)
        return clean(text or "")
    except Exception:
        return ""


def get_links(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        domain = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            p = urlparse(href)
            if p.netloc == domain and href.startswith("http") and "#" not in href:
                links.append(href)
        return list(set(links))
    except Exception:
        return []


def scrape_site(name, seeds, max_pages=500, min_size_kb=30):
    out = OUTPUT_DIR / f"{name}.txt"
    if out.exists() and out.stat().st_size > min_size_kb * 1000:
        print(f"  SKIP {name} ({out.stat().st_size//1000} KB already)")
        return 0

    print(f"\n>>> {name}", flush=True)
    visited, queue, texts = set(), list(seeds), []
    t0 = time.time()

    while queue and len(visited) < max_pages:
        if time.time() - t0 > SITE_MAX_SECS:
            print(f"  ⏰ time limit hit, saving {len(texts)} articles")
            break

        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        text = fetch(url)
        if text and len(text) >= 100 and is_nepali(text):
            texts.append(text)
            if len(texts) % 100 == 0:
                print(f"  {name}: {len(texts)} articles / {len(visited)} visited", flush=True)

        for link in get_links(url):
            if link not in visited:
                queue.append(link)
        time.sleep(DELAY)

    if texts:
        out.write_text("\n\n---\n\n".join(texts), encoding="utf-8")
        kb = out.stat().st_size // 1000
        print(f"  ✓ {name}: {len(texts)} articles → {kb} KB")
        return out.stat().st_size
    else:
        print(f"  ✗ {name}: nothing")
        return 0


def scrape_paginated(name, url_pattern, start=1, end=200, min_size_kb=50):
    """Scrape a site with numbered pages like /archive?page=N"""
    out = OUTPUT_DIR / f"{name}.txt"
    if out.exists() and out.stat().st_size > min_size_kb * 1000:
        print(f"  SKIP {name} ({out.stat().st_size//1000} KB)")
        return 0

    print(f"\n>>> {name} (paginated)", flush=True)
    article_urls = set()

    for page in range(start, end + 1):
        url = url_pattern.format(page)
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                print(f"  404 at page {page}, stopping")
                break
            soup = BeautifulSoup(r.text, "lxml")
            domain = urlparse(url).netloc
            found = 0
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                if urlparse(href).netloc == domain and href not in article_urls:
                    article_urls.add(href)
                    found += 1
            if found == 0 and page > 5:
                print(f"  No links at page {page}, stopping")
                break
        except Exception as e:
            print(f"  Error page {page}: {e}")
            break
        time.sleep(DELAY * 0.5)

    print(f"  Found {len(article_urls)} article URLs, fetching text...")
    texts = []
    t0 = time.time()
    for url in list(article_urls)[:2000]:
        if time.time() - t0 > SITE_MAX_SECS:
            print(f"  ⏰ time limit, saving {len(texts)}")
            break
        text = fetch(url)
        if text and len(text) >= 100 and is_nepali(text):
            texts.append(text)
        time.sleep(DELAY)

    if texts:
        out.write_text("\n\n---\n\n".join(texts), encoding="utf-8")
        kb = out.stat().st_size // 1000
        print(f"  ✓ {name}: {len(texts)} articles → {kb} KB")
        return out.stat().st_size
    print(f"  ✗ {name}: nothing")
    return 0


# ── Wikisource full crawl (API — most reliable Nepali source) ────────────────

WIKISOURCE_OUT = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data/literature/nepali_wikisource_api.txt")
WIKISOURCE_API = "https://ne.wikisource.org/w/api.php"


def scrape_wikisource():
    if WIKISOURCE_OUT.exists() and WIKISOURCE_OUT.stat().st_size > 100_000:
        print(f"  SKIP Wikisource ({WIKISOURCE_OUT.stat().st_size//1000} KB)")
        return
    print("\n>>> Nepali Wikisource (MediaWiki API)")
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    pages, params = [], {
        "action": "query", "list": "allpages", "aplimit": "500",
        "apnamespace": "0", "format": "json"
    }
    retries = 0
    while retries < 5:
        try:
            r = session.get(WIKISOURCE_API, params=params, timeout=30)
            if not r.text.strip():
                time.sleep(3)
                retries += 1
                continue
            data = r.json()
            pages.extend(p["title"] for p in data["query"]["allpages"])
            retries = 0
            if "continue" in data:
                params["apcontinue"] = data["continue"]["apcontinue"]
                time.sleep(0.5)
            else:
                break
        except Exception as e:
            print(f"  Retry {retries}: {e}")
            retries += 1
            time.sleep(3)
    print(f"  {len(pages)} pages found")
    if not pages:
        return
    texts = []
    for i, title in enumerate(pages):
        for attempt in range(3):
            try:
                r = session.get(WIKISOURCE_API, timeout=30, params={
                    "action": "parse", "page": title, "prop": "wikitext", "format": "json"
                })
                if not r.text.strip():
                    time.sleep(2)
                    continue
                wt = r.json()["parse"]["wikitext"]["*"]
                wt = re.sub(r'\[\[File:[^\]]+\]\]|\[\[Image:[^\]]+\]\]', '', wt)
                wt = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', wt)
                wt = re.sub(r'\{\{[^}]*\}\}', '', wt)
                wt = re.sub(r"'{2,}|={2,}[^=]+=+", '', wt)
                wt = clean(wt)
                if wt and len(wt) >= 100 and is_nepali(wt):
                    texts.append(f"## {title}\n\n{wt}")
                break
            except Exception:
                time.sleep(2)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pages)}, {len(texts)} with content", flush=True)
        time.sleep(0.3)
    if texts:
        WIKISOURCE_OUT.parent.mkdir(parents=True, exist_ok=True)
        WIKISOURCE_OUT.write_text("\n\n---\n\n".join(texts), encoding="utf-8")
        print(f"  ✓ Wikisource: {len(texts)} pages → {WIKISOURCE_OUT.stat().st_size/1e6:.1f} MB")
    else:
        print("  ✗ Wikisource: no content")


# ── More Internet Archive targeted crawl ─────────────────────────────────────

IA_OUT = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data/internet_archive")
IA_API = "https://archive.org/advancedsearch.php"
IA_HEADERS = {"User-Agent": "Mozilla/5.0 (research; Nepali NLP)"}

# Very specific searches targeting only Nepali-script content
IA_QUERIES = [
    'subject:"Nepali" language:"Nepali" mediatype:texts',
    'language:"Nepali" mediatype:texts',
    'subject:"Nepal" language:"Nepali" mediatype:texts',
    'language:"nep" mediatype:texts',
    'subject:"Devanagari" language:"Nepali" mediatype:texts',
]


def get_ia_djvu_text(identifier):
    safe = re.sub(r'[^\w\-]', '_', identifier)
    out = IA_OUT / f"{safe}.txt"
    if out.exists() and out.stat().st_size > 500:
        return True

    try:
        r = requests.get(f"https://archive.org/metadata/{identifier}/files",
                         headers=IA_HEADERS, timeout=30)
        files = {f["name"]: f for f in r.json().get("result", [])}

        # Prefer _djvu.txt
        djvu_files = [k for k in files if k.endswith("_djvu.txt")]
        if djvu_files:
            url = f"https://archive.org/download/{identifier}/{requests.utils.quote(djvu_files[0])}"
            resp = requests.get(url, headers=IA_HEADERS, timeout=120, stream=True)
            if resp.status_code == 200:
                content = resp.content.decode("utf-8", errors="ignore")
                content = re.sub(r'\x0c', '\n', content)
                if is_nepali(content[:3000]):
                    out.write_text(content, encoding="utf-8")
                    return True
    except Exception:
        pass
    return False


def scrape_ia_targeted():
    print("\n>>> Internet Archive (targeted Nepali-only search)")
    seen = set(f.stem for f in IA_OUT.glob("*.txt"))
    new_count = 0

    for query in IA_QUERIES:
        print(f"  Query: {query[:60]}")
        for page in range(1, 11):  # up to 10 pages = 500 results
            try:
                params = {
                    "q": query,
                    "fl[]": ["identifier", "title", "language"],
                    "rows": 50, "page": page, "output": "json"
                }
                r = requests.get(IA_API, params=params, headers=IA_HEADERS, timeout=30)
                docs = r.json().get("response", {}).get("docs", [])
                if not docs:
                    break
                for doc in docs:
                    iid = doc.get("identifier", "")
                    if iid and iid not in seen:
                        seen.add(iid)
                        if get_ia_djvu_text(iid):
                            new_count += 1
                            if new_count % 20 == 0:
                                print(f"    {new_count} new Nepali items downloaded", flush=True)
                        time.sleep(DELAY)
                time.sleep(1)
            except Exception as e:
                print(f"    Error: {e}"); break

    total = sum(f.stat().st_size for f in IA_OUT.glob("*.txt"))
    print(f"  IA total: {new_count} new items, {total/1e6:.1f} MB total")


# ── Site list ────────────────────────────────────────────────────────────────

SITES = [
    # High-yield news sites
    ("pahilopost", ["https://pahilopost.com/", "https://pahilopost.com/category/news",
                    "https://pahilopost.com/category/politics",
                    "https://pahilopost.com/category/economy"], 800),
    ("ujyaalonews", ["https://www.ujyaalonews.com/",
                     "https://www.ujyaalonews.com/category/news"], 600),
    ("nepal_live", ["https://www.nepallive.com/",
                    "https://www.nepallive.com/category/news"], 500),
    ("reporters_nepal", ["https://www.reportersnepal.com/"], 400),
    ("loktantra", ["https://www.loktantra.com.np/"], 400),
    ("nepalkhabar_daily", ["https://nepalkhabar.com/"], 400),
    ("koshi_tv", ["https://koshitv.com/"], 300),
    ("saptahik", ["https://www.saptahik.com.np/"], 400),
    ("nepal_patra", ["https://www.nepalpatra.com/"], 400),
    ("hamro_patro_news", ["https://www.hamropatro.com/news",
                          "https://www.hamropatro.com/community"], 400),
    ("samacharpatra_new", ["https://www.samacharpatra.com/"], 400),
    ("naya_patrika_new", ["https://www.nayapatrika.com/"], 400),
    ("gorkhapatra_new", ["https://gorkhapatraonline.com/"], 400),
    ("eklo_nepali", ["https://www.eklolagyo.com/"], 300),
    ("khabardabali", ["https://khabardabali.com/"], 300),
    # Literature / culture
    ("nepali_times", ["https://www.nepalitimes.com/",  # has Nepali section
                      "https://www.nepalitimes.com/nepali"], 300),
    ("mero_sansar", ["https://merosansar.com/"], 300),
    ("nepali_unicode_tools", ["https://www.ashesh.com.np/"], 200),
]

PAGINATED_SITES = [
    # (name, url_pattern_with_{}, start, end)
    ("annapurnapost_deep", "https://annapurnapost.com/news?page={}", 1, 150),
    ("setopati_deep", "https://www.setopati.com/category/rastriya?page={}", 1, 100),
    ("onlinekhabar_deep", "https://www.onlinekhabar.com/category/news/page/{}", 1, 100),
]


if __name__ == "__main__":
    print("=== Round 2: Extended Nepali Scraping ===\n")

    # 1. Wikisource via API
    scrape_wikisource()

    # 2. Skip IA (archive.org is rate-limiting; already have 189MB from there)
    print("\n>>> Internet Archive: SKIPPED (already have 189MB, site rate-limiting)")

    # 3. BFS site scraping
    for name, seeds, max_p in SITES:
        scrape_site(name, seeds, max_p)
        time.sleep(2)

    # 4. Paginated scraping for high-volume sites
    for name, pattern, start, end in PAGINATED_SITES:
        scrape_paginated(name, pattern, start, end)
        time.sleep(2)

    print("\n=== Round 2 complete ===")
    DATA = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data")
    total = sum(f.stat().st_size for f in DATA.rglob("*.txt"))
    print(f"Grand total in manual/data/: {total/1e6:.0f} MB")
