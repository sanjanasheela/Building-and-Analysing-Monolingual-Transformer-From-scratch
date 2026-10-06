"""
Phase 1: Deep crawler to discover and save unique article links from Andhrajyothy.
This strictly grabs URLs and appends them to a text file to avoid redundancy.
"""

import re
import time
import queue
import threading
from urllib.parse import urljoin, urlparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup

LINKS_FILE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/andhrajyothy/andhrajyothy_links.txt")
LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)

NUM_THREADS = 16
DELAY = 0.05

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Andhrajyothy category slugs (top-level sections seen on the site)
CATEGORY_SLUGS = [
    "andhra-pradesh", "telangana", "national", "sports", "health",
    "lifestyle", "technology", "navya", "prathyekam", "photogallery",
    "web-stories", "latest-news", "business", "crime", "international",
    "videos", "entertainment", "spiritual", "politics",
]

BASE_DOMAIN = "www.andhrajyothy.com"

SEED_URLS = ["https://www.andhrajyothy.com/"]
for slug in CATEGORY_SLUGS:
    cat_url = f"https://www.andhrajyothy.com/{slug}"
    SEED_URLS.append(cat_url)
    for page in range(1, 501):
        SEED_URLS.append(f"https://www.andhrajyothy.com/{slug}/page/{page}")

# Article URLs look like:
# https://www.andhrajyothy.com/2026/telangana/kalyana-lakshmi-shaadi-mubarak-...-bvr-1551002.html
# https://www.andhrajyothy.com/2026/andhra-pradesh/guntur/cm-chandrababu-...-vvnp-1551019.html
# (category can be followed by an extra district/subcategory segment, so allow 1+ segments)
ARTICLE_PATTERN = re.compile(r"/\d{4}/[a-z0-9\-]+(?:/[a-z0-9\-]+)*-\d+\.html$", re.IGNORECASE)


class LinkSpider:
    def __init__(self):
        self.visited = set()
        self.article_links = set()
        self.queue = queue.Queue()
        self.lock = threading.Lock()

        self.running = True

        # Load previously discovered links
        if LINKS_FILE.exists():
            with open(LINKS_FILE, "r") as f:
                for line in f:
                    u = line.strip()
                    if u:
                        self.article_links.add(u)
                        self.visited.add(u)
            print(f"Loaded {len(self.article_links)} existing article links.")

        # Seed the queue
        for u in SEED_URLS:
            if u not in self.visited:
                self.queue.put(u)
                self.visited.add(u)

    def fetch_links(self, url):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                return

            soup = BeautifulSoup(resp.text, "lxml")
            new_articles = []

            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                parsed = urlparse(href)

                if parsed.netloc == BASE_DOMAIN or parsed.netloc == "andhrajyothy.com":
                    clean_u = href.split("#")[0].split("?")[0]

                    with self.lock:
                        if clean_u in self.visited:
                            continue
                        self.visited.add(clean_u)

                        # Is it an article?
                        if ARTICLE_PATTERN.search(clean_u):
                            if clean_u not in self.article_links:
                                self.article_links.add(clean_u)
                                new_articles.append(clean_u)
                        # Is it a category/pagination page to crawl further?
                        elif "/page/" in clean_u or clean_u in SEED_URLS or any(
                            f"/{slug}" in clean_u for slug in CATEGORY_SLUGS
                        ):
                            self.queue.put(clean_u)

            # Save new articles immediately to avoid redundancy on crash
            if new_articles:
                with self.lock:
                    with open(LINKS_FILE, "a") as f:
                        for article_url in new_articles:
                            f.write(article_url + "\n")

        except Exception:
            pass

    def worker(self):
        while self.running:
            try:
                url = self.queue.get(timeout=3)
            except queue.Empty:
                continue

            self.fetch_links(url)
            self.queue.task_done()
            time.sleep(DELAY)

    def run(self):
        print("=== Starting Link Spider ===")
        threads = []
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)

        try:
            while True:
                time.sleep(5)
                with self.lock:
                    print(f"Discovered {len(self.article_links):,} unique articles | Queue: {self.queue.qsize():,} URLs to explore")
        except KeyboardInterrupt:
            print("Stopping...")
            self.running = False
            for t in threads:
                t.join()


if __name__ == "__main__":
    spider = LinkSpider()
    spider.run()