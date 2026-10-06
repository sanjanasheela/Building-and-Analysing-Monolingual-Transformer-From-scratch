"""
Scraper for Nepali literature and poetry websites + Nepali Wikisource via API.
"""
import re, time, unicodedata, warnings
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import trafilatura

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
OUTPUT_DIR = BASE / "data/raw/public/literature"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research; Nepali NLP corpus)"}
DELAY = 1.5


def is_devanagari(text, thresh=0.25):
    count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return count / max(len(text), 1) >= thresh


def clean(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize("NFC", text or "")).strip()


def fetch(url):
    try:
        dl = trafilatura.fetch_url(url)
        return clean(trafilatura.extract(dl, include_comments=False) or "") if dl else ""
    except Exception:
        return ""


def get_links(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")
        d = urlparse(url).netloc
        return list(set(urljoin(url, a["href"]) for a in soup.find_all("a", href=True)
                        if urlparse(urljoin(url, a["href"])).netloc == d))
    except Exception:
        return []


def scrape_site(name, seeds, max_pages=400):
    out = OUTPUT_DIR / f"{name}.txt"
    if out.exists() and out.stat().st_size > 30_000:
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
        print(f"  ✓ {name}: {len(texts)} texts → {out.stat().st_size//1000} KB")
    else:
        print(f"  ✗ {name}: no content")


# Wikisource via MediaWiki API
WIKISOURCE_API = "https://ne.wikisource.org/w/api.php"


def scrape_wikisource():
    out = OUTPUT_DIR / "nepali_wikisource_api.txt"
    if out.exists() and out.stat().st_size > 100_000:
        print(f"  SKIP Wikisource"); return
    print("\n=== Nepali Wikisource (MediaWiki API) ===")

    # List all pages
    pages, params = [], {"action":"query","list":"allpages","aplimit":"500","apnamespace":"0","format":"json"}
    while True:
        try:
            r = requests.get(WIKISOURCE_API, params=params, headers=HEADERS, timeout=30)
            data = r.json()
            pages.extend(p["title"] for p in data["query"]["allpages"])
            if "continue" in data:
                params["apcontinue"] = data["continue"]["apcontinue"]
                time.sleep(0.5)
            else:
                break
        except Exception as e:
            print(f"  API error: {e}"); break

    print(f"  {len(pages)} pages found")
    texts = []
    for i, title in enumerate(pages):
        try:
            r = requests.get(WIKISOURCE_API, headers=HEADERS, timeout=30,
                params={"action":"parse","page":title,"prop":"wikitext","format":"json"})
            wt = r.json()["parse"]["wikitext"]["*"]
            wt = re.sub(r'\[\[File:[^\]]+\]\]', '', wt)
            wt = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', wt)
            wt = re.sub(r'\{\{[^}]+\}\}', '', wt)
            wt = re.sub(r"'{2,}|==+[^=]+=+", '', wt)
            wt = clean(wt)
            if wt and len(wt) >= 80 and is_devanagari(wt):
                texts.append(f"## {title}\n\n{wt}")
        except Exception:
            pass
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(pages)} fetched, {len(texts)} with content")
        time.sleep(0.3)

    if texts:
        out.write_text("\n\n---\n\n".join(texts), encoding="utf-8")
        print(f"  ✓ Wikisource: {len(texts)} pages → {out.stat().st_size/1e6:.1f} MB")


LIT_SITES = [
    ("nepalikavita", ["https://www.nepalikavita.com/","https://www.nepalikavita.com/kavita","https://www.nepalikavita.com/gajal"], 600),
    ("hamrosahitya", ["https://hamrosahitya.com/","https://hamrosahitya.com/category/katha","https://hamrosahitya.com/category/kavita"], 500),
    ("sahityapost", ["https://www.sahityapost.com/"], 400),
    ("nepalisahitya", ["https://nepalisahitya.com/"], 400),
    ("rachana", ["https://rachana.com.np/"], 300),
    ("srijansil", ["https://srijansil.com/"], 300),
]

if __name__ == "__main__":
    print("=== Nepali Literature Scraper ===")
    scrape_wikisource()
    for name, seeds, mp in LIT_SITES:
        scrape_site(name, seeds, mp)
        time.sleep(2)
    total = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.txt"))
    print(f"\nTotal: {total/1e6:.1f} MB")
