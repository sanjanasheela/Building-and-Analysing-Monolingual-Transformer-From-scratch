"""
Telugu LLM Corpus Scraper - Brand new sources only
"""

import os, time, re, requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse

OUTPUT_DIR = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
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

def scrape_site(name, seeds, output_file, max_pages=800, delay=0.5):
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
            print(f"  [{count}/{max_pages}] {kb:.0f}KB | {url[:70]}")
            for link in get_links(url):
                if link not in visited: queue.append(link)
        time.sleep(delay)
    
    fp = os.path.join(OUTPUT_DIR, output_file)
    total = os.path.getsize(fp) if os.path.exists(fp) else 0
    print(f"  Done: {count} pages, {total/1024:.0f} KB")


# NEW SOURCES

scrape_site("Eenadu", [
    "https://www.eenadu.net/",
    "https://www.eenadu.net/ap",
    "https://www.eenadu.net/telangana",
    "https://www.eenadu.net/politics",
    "https://www.eenadu.net/business",
], "eenadu_bulk.txt", max_pages=800)

scrape_site("Sakshi", [
    "https://www.sakshi.com/",
    "https://www.sakshi.com/telugu-news/andhra-pradesh",
    "https://www.sakshi.com/telugu-news/telangana",
    "https://www.sakshi.com/telugu-news/movies",
    "https://www.sakshi.com/telugu-news/sports",
], "sakshi_bulk.txt", max_pages=800)

scrape_site("Samayam Telugu", [
    "https://telugu.samayam.com/",
    "https://telugu.samayam.com/telangana/articlelist/71067524.cms",
    "https://telugu.samayam.com/andhra-pradesh/articlelist/71067426.cms",
], "samayam_bulk.txt", max_pages=800)

scrape_site("TV9 Telugu", [
    "https://tv9telugu.com/",
    "https://tv9telugu.com/andhra-pradesh",
    "https://tv9telugu.com/telangana",
    "https://tv9telugu.com/politics",
], "tv9_bulk.txt", max_pages=800)

scrape_site("Great Andhra", [
    "https://telugu.greatandhra.com/",
    "https://telugu.greatandhra.com/politics/ap",
    "https://telugu.greatandhra.com/politics/telangana",
    "https://telugu.greatandhra.com/movies",
], "greatandhra_bulk.txt", max_pages=800)

scrape_site("Hindustan Times Telugu", [
    "https://telugu.hindustantimes.com/",
    "https://telugu.hindustantimes.com/andhra-pradesh",
    "https://telugu.hindustantimes.com/telangana",
], "ht_telugu_bulk.txt", max_pages=800)

scrape_site("NT News", [
    "https://www.ntnews.com/",
    "https://www.ntnews.com/telangana",
    "https://www.ntnews.com/national",
    "https://www.ntnews.com/sports",
], "ntnews_bulk.txt", max_pages=800)

print("\n" + "="*60)
print("Finished scraping new sources.")
