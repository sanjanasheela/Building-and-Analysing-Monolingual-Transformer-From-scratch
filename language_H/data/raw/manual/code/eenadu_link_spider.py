"""
Phase 1: Deep crawler to discover and save unique article links from Eenadu.
This strictly grabs URLs and appends them to a text file to avoid redundancy.

NOTE ON EENADU'S STRUCTURE (different from OnlineKhabar / Andhrajyothy):
- Article URLs are 5 segments: /telugu-news/<category>/<slug>/<code>/<id>
  or /telugu-article/<category>/<slug>/<code>/<id>, plus
  /videos/playvideo/<slug>/<code>/<id> for video pages.
- Category listing pages do NOT expose plain /page/N pagination links.
  The "మరిన్ని" (More) control on eenadu.net is JS/AJAX-driven infinite
  scroll, so there is no discoverable static URL for "page 2" of a
  category. Faking /page/2, /page/3... URLs (like the OnlineKhabar/
  Andhrajyothy spiders do) would just 404 here.
- To compensate for the lack of deep pagination, this spider seeds a much
  WIDER set of listing/index pages (every section, sub-section, district
  hub, and secondary listing like top-news/editorial/trending/most-read)
  and crawls generically: any same-domain, non-asset link it finds gets
  queued, not just links matching a fixed category-slug list. This lets
  it naturally discover district pages, topic pages, and other index
  pages the site links to, without needing to guess their URLs in advance.
- Because of the missing pagination, a single run will mostly capture
  what's currently on each listing page (recent + a few "most read"/
  "top news" tie-ins). Re-running this spider periodically (e.g. daily)
  will accumulate a much larger set over time, since newly published
  articles rotate through, and everything already saved is deduplicated
  and never lost.
"""

import re
import time
import queue
import threading
from urllib.parse import urljoin, urlparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup

LINKS_FILE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/eenadu/eenadu_links.txt")
LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)

NUM_THREADS = 16
DELAY = 0.05

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

BASE_DOMAIN = "www.eenadu.net"

# Broad set of listing/index pages to seed the crawl. District hub pages
# (andhra-pradesh/districts, telangana/districts) will themselves surface
# links to each individual district's news page, so we don't need to
# guess/hardcode every district slug up front.
SEED_PATHS = [
    "/",
    "/andhra-pradesh", "/andhra-pradesh/districts", "/andhra-pradesh/top-news",
    "/telangana", "/telangana/districts", "/telangana/top-news",
    "/india", "/world", "/business", "/sports", "/movies", "/crime", "/women",
    "/education", "/health", "/technology", "/devotional", "/youth", "/recipes",
    "/kids-stories", "/real-estate", "/astrology", "/web-stories", "/nri",
    "/photos", "/videos", "/explained", "/sunday-magazine",
    "/latest-news-list", "/breaking-news-list", "/trending-news",
    "/editorial", "/antaryami", "/vyakyanam", "/telugulo-varthalu",
]
SEED_URLS = [f"https://{BASE_DOMAIN}{p}" for p in SEED_PATHS]

# Article URLs look like:
# https://www.eenadu.net/telugu-news/andhra-pradesh/women-police-officer-injured-during-ysrcp-protest/1799/126142772
# https://www.eenadu.net/telugu-news/districts/prakasam-news/8/126142711
# https://www.eenadu.net/telugu-article/health/the-importance-of-gut-health/0813/126142516
# https://www.eenadu.net/videos/playvideo/bro-bridge-swept-away-near-india-china-border-at-chamoli/6/80831
ARTICLE_PATTERN = re.compile(
    r"(?:/telugu-(?:news|article)/[a-z0-9\-]+/[a-z0-9\-]+/\d+/\d+"
    r"|/videos/playvideo/[a-z0-9\-]+/\d+/\d+)$",
    re.IGNORECASE
)

# Non-content paths / infinite or irrelevant sections to avoid wasting
# crawl budget on (search boxes, calendars, legal pages, other subdomains
# handled automatically since domain check already excludes them).
SKIP_SUBSTRINGS = [
    "/search", "/feedback", "/calendar", "/terms-conditions",
    "/privacy-policy", "javascript:", "/rashi-phalalu/",  # daily horoscope detail pages (low value, high volume)
]


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

                if parsed.netloc != BASE_DOMAIN:
                    continue

                clean_u = href.split("#")[0].split("?")[0]
                if not clean_u.startswith("https://"):
                    continue
                if any(s in clean_u for s in SKIP_SUBSTRINGS):
                    continue

                with self.lock:
                    if clean_u in self.visited:
                        continue
                    self.visited.add(clean_u)

                    if ARTICLE_PATTERN.search(clean_u):
                        if clean_u not in self.article_links:
                            self.article_links.add(clean_u)
                            new_articles.append(clean_u)
                    else:
                        # Generic crawl: queue any other same-domain page
                        # (district pages, topic pages, sub-sections, etc.)
                        # since Eenadu doesn't expose fixed pagination URLs
                        # to filter on the way OnlineKhabar/Andhrajyothy do.
                        self.queue.put(clean_u)

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
        print("=== Starting Eenadu Link Spider ===")
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