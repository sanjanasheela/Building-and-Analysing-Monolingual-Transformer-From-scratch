# """
# Phase 2: Eenadu Data Extractor

# - Reads URLs from eenadu_links.txt
# - Downloads pages using requests()
# - Extracts article text using Trafilatura
# - Falls back to BeautifulSoup if Trafilatura fails
# - NO language filter
# - Saves extracted text into 100 MB chunks
# - Stops after 1 GB
# - Never re-fetches URLs recorded in processed_links.txt
# - Safe to restart
# - Shows useful HTTP/extraction errors instead of silently hiding them
# """

# import re
# import time
# import queue
# import threading
# import unicodedata
# import warnings
# from pathlib import Path

# import requests
# import trafilatura

# from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


# # ============================================================
# # CONFIGURATION
# # ============================================================

# LINKS_FILE = Path(
#     "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
#     "language_H/data/raw/manual/data/eenadu/"
#     "eenadu_links.txt"
# )

# OUTPUT_DIR = Path(
#     "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/"
#     "language_H/data/raw/manual/data/eenadu"
# )

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )


# # ------------------------------------------------------------
# # TARGET / CHUNK SIZE
# # ------------------------------------------------------------

# TARGET_BYTES = 1 * 1024 * 1024 * 1024
# CHUNK_BYTES = 100 * 1024 * 1024


# # ------------------------------------------------------------
# # THREADING
# # ------------------------------------------------------------

# NUM_THREADS = 12
# DELAY = 0.1


# # ------------------------------------------------------------
# # EXTRACTION
# # ------------------------------------------------------------

# MIN_TEXT_LEN = 100


# # ------------------------------------------------------------
# # NETWORK
# # ------------------------------------------------------------

# REQUEST_TIMEOUT = 20


# HEADERS = {
#     "User-Agent": (
#         "Mozilla/5.0 (X11; Linux x86_64) "
#         "AppleWebKit/537.36 "
#         "(KHTML, like Gecko) "
#         "Chrome/120.0.0.0 Safari/537.36"
#     ),

#     "Accept": (
#         "text/html,application/xhtml+xml,"
#         "application/xml;q=0.9,image/avif,image/webp,"
#         "*/*;q=0.8"
#     ),

#     "Accept-Language": (
#         "te-IN,te;q=0.9,en-US;q=0.8,en;q=0.7"
#     ),

#     "Connection": "keep-alive",
# }


# # ============================================================
# # WARNINGS
# # ============================================================

# warnings.filterwarnings(
#     "ignore",
#     category=XMLParsedAsHTMLWarning
# )


# # ============================================================
# # TEXT CLEANING
# # ============================================================

# def clean_text(text: str) -> str:

#     if not text:
#         return ""

#     # Unicode normalization
#     text = unicodedata.normalize(
#         "NFC",
#         text
#     )

#     # Normalize spaces
#     text = re.sub(
#         r"[ \t]+",
#         " ",
#         text
#     )

#     # Normalize excessive newlines
#     text = re.sub(
#         r"\n{3,}",
#         "\n\n",
#         text
#     )

#     return text.strip()


# # ============================================================
# # BEAUTIFULSOUP EXTRACTION
# # ============================================================

# def extract_with_bs4(html: str) -> str:
#     """
#     Fallback extractor.

#     Tries multiple possible article containers because
#     website HTML structures can change.
#     """

#     soup = BeautifulSoup(
#         html,
#         "lxml"
#     )


#     # --------------------------------------------------------
#     # REMOVE NON-CONTENT ELEMENTS
#     # --------------------------------------------------------

#     for tag in soup([
#         "script",
#         "style",
#         "noscript",
#         "svg",
#         "iframe",
#         "header",
#         "footer",
#         "nav",
#         "form"
#     ]):

#         tag.decompose()


#     # --------------------------------------------------------
#     # FIRST: TRY ARTICLE TAG
#     # --------------------------------------------------------

#     article = soup.find(
#         "article"
#     )

#     if article:

#         paragraphs = article.find_all(
#             "p"
#         )

#         text = "\n".join(
#             p.get_text(
#                 " ",
#                 strip=True
#             )
#             for p in paragraphs
#             if len(
#                 p.get_text(
#                     " ",
#                     strip=True
#                 )
#             ) > 20
#         )

#         text = clean_text(text)

#         if len(text) >= MIN_TEXT_LEN:

#             return text


#     # --------------------------------------------------------
#     # TRY COMMON CONTENT CLASS NAMES
#     # --------------------------------------------------------

#     selectors = [

#         # General article/content containers
#         "[class*='article-content']",
#         "[class*='articleContent']",

#         "[class*='story-content']",
#         "[class*='storyContent']",

#         "[class*='content-body']",
#         "[class*='contentBody']",

#         "[class*='fullstory']",
#         "[class*='full-story']",

#         "[class*='main-content']",
#         "[class*='mainContent']",

#         "[class*='news-content']",
#         "[class*='newsContent']",

#         # Telugu news sites frequently use these kinds
#         # of wrappers
#         "[class*='story']",
#         "[class*='article']",
#         "[class*='content']",

#         # Last fallback
#         "main",
#     ]


#     candidates = []


#     for selector in selectors:

#         try:

#             elements = soup.select(
#                 selector
#             )

#         except Exception:

#             continue


#         for element in elements:

#             paragraphs = element.find_all(
#                 "p"
#             )


#             if not paragraphs:
#                 continue


#             text = "\n".join(
#                 p.get_text(
#                     " ",
#                     strip=True
#                 )
#                 for p in paragraphs
#                 if len(
#                     p.get_text(
#                         " ",
#                         strip=True
#                     )
#                 ) > 20
#             )


#             text = clean_text(
#                 text
#             )


#             if len(text) >= MIN_TEXT_LEN:

#                 candidates.append(
#                     text
#                 )


#     # --------------------------------------------------------
#     # RETURN LONGEST CANDIDATE
#     # --------------------------------------------------------

#     if candidates:

#         return max(
#             candidates,
#             key=len
#         )


#     return ""


# # ============================================================
# # EXTRACT ARTICLE TEXT
# # ============================================================

# def extract_article_text(html: str) -> str:
#     """
#     Extraction pipeline:

#         HTML
#           ↓
#         Trafilatura
#           ↓
#         BeautifulSoup fallback
#     """

#     # --------------------------------------------------------
#     # TRAFILATURA
#     # --------------------------------------------------------

#     try:

#         text = trafilatura.extract(
#             html,
#             include_comments=False,
#             include_tables=False,
#             no_fallback=False
#         )

#         text = clean_text(
#             text
#         )

#         if len(text) >= MIN_TEXT_LEN:

#             return text

#     except Exception:

#         pass


#     # --------------------------------------------------------
#     # BEAUTIFULSOUP FALLBACK
#     # --------------------------------------------------------

#     try:

#         text = extract_with_bs4(
#             html
#         )

#         text = clean_text(
#             text
#         )

#         if len(text) >= MIN_TEXT_LEN:

#             return text

#     except Exception:

#         pass


#     return ""


# # ============================================================
# # EXTRACTOR CLASS
# # ============================================================

# class Extractor:

#     def __init__(self):

#         # ----------------------------------------------------
#         # QUEUE
#         # ----------------------------------------------------

#         self.url_queue = queue.Queue()


#         # ----------------------------------------------------
#         # LOCK
#         # ----------------------------------------------------

#         self.lock = threading.Lock()


#         # ----------------------------------------------------
#         # STATE
#         # ----------------------------------------------------

#         self.running = True


#         # ----------------------------------------------------
#         # STATISTICS
#         # ----------------------------------------------------

#         self.total_bytes = 0

#         self.total_articles = 0

#         self.attempted = 0

#         self.successful = 0

#         self.failed = 0

#         self.http_errors = 0

#         self.empty_extractions = 0

#         self.request_errors = 0


#         self.start_time = time.time()


#         # ----------------------------------------------------
#         # OUTPUT CHUNK DETECTION
#         # ----------------------------------------------------

#         self.file_index = 1


#         existing_chunks = list(
#             OUTPUT_DIR.glob(
#                 "eenadu_articles_*.txt"
#             )
#         )


#         if existing_chunks:

#             for path in existing_chunks:

#                 try:

#                     index = int(
#                         path.stem.split("_")[-1]
#                     )

#                     if index > self.file_index:

#                         self.file_index = index

#                 except Exception:

#                     pass


#                 self.total_bytes += (
#                     path.stat().st_size
#                 )


#         if self.total_bytes:

#             print(
#                 f"Resuming: "
#                 f"{self.total_bytes / (1024 * 1024):.1f} MB "
#                 f"already saved"
#             )


#         # ----------------------------------------------------
#         # CURRENT OUTPUT FILE
#         # ----------------------------------------------------

#         self.current_path = (
#             OUTPUT_DIR
#             / f"eenadu_articles_{self.file_index}.txt"
#         )


#         self.out_file = open(
#             self.current_path,
#             "a",
#             encoding="utf-8"
#         )


#         self.file_bytes = (
#             self.current_path.stat().st_size
#             if self.current_path.exists()
#             else 0
#         )


#         # ----------------------------------------------------
#         # PROCESSED LINKS FILE
#         # ----------------------------------------------------

#         self.proc_path = (
#             OUTPUT_DIR
#             / "processed_links.txt"
#         )


#         self.processed = set()


#         if self.proc_path.exists():

#             with open(
#                 self.proc_path,
#                 encoding="utf-8"
#             ) as f:

#                 for line in f:

#                     url = line.strip()

#                     if url:

#                         self.processed.add(
#                             url
#                         )


#         self.proc_file = open(
#             self.proc_path,
#             "a",
#             encoding="utf-8"
#         )


#         # ----------------------------------------------------
#         # LOAD INPUT LINKS
#         # ----------------------------------------------------

#         links = []


#         if LINKS_FILE.exists():

#             with open(
#                 LINKS_FILE,
#                 encoding="utf-8"
#             ) as f:

#                 links = [
#                     line.strip()
#                     for line in f
#                     if line.strip()
#                 ]


#         print(
#             f"Total links     : {len(links):,}"
#         )

#         print(
#             f"Already done    : {len(self.processed):,}"
#         )


#         # ----------------------------------------------------
#         # QUEUE UNIQUE LINKS
#         # ----------------------------------------------------

#         queued = set()


#         for url in links:

#             if not url:
#                 continue


#             if url in self.processed:
#                 continue


#             if url in queued:
#                 continue


#             queued.add(
#                 url
#             )


#             self.url_queue.put(
#                 url
#             )


#         print(
#             f"Initial queue   : "
#             f"{self.url_queue.qsize():,}"
#         )


#     # ========================================================
#     # ROTATE OUTPUT
#     # ========================================================

#     def rotate_file(self):

#         self.out_file.close()


#         self.file_index += 1


#         self.current_path = (
#             OUTPUT_DIR
#             / f"eenadu_articles_{self.file_index}.txt"
#         )


#         self.out_file = open(
#             self.current_path,
#             "a",
#             encoding="utf-8"
#         )


#         self.file_bytes = 0


#     # ========================================================
#     # SAVE
#     # ========================================================

#     def save(
#         self,
#         url: str,
#         text: str
#     ):

#         content = (
#             text
#             + "\n\n"
#         )


#         size = len(
#             content.encode(
#                 "utf-8"
#             )
#         )


#         with self.lock:

#             # Double-check
#             if url in self.processed:

#                 return False


#             # ------------------------------------------------
#             # WRITE ARTICLE
#             # ------------------------------------------------

#             self.out_file.write(
#                 content
#             )

#             self.out_file.flush()


#             # ------------------------------------------------
#             # RECORD URL
#             # ------------------------------------------------

#             self.proc_file.write(
#                 url + "\n"
#             )

#             self.proc_file.flush()


#             # ------------------------------------------------
#             # UPDATE STATE
#             # ------------------------------------------------

#             self.processed.add(
#                 url
#             )

#             self.file_bytes += size

#             self.total_bytes += size

#             self.total_articles += 1


#             # ------------------------------------------------
#             # ROTATE
#             # ------------------------------------------------

#             if (
#                 self.file_bytes
#                 >= CHUNK_BYTES
#             ):

#                 self.rotate_file()


#         return True


#     # ========================================================
#     # PROCESS ONE URL
#     # ========================================================

#     def process(
#         self,
#         url: str
#     ):

#         # ----------------------------------------------------
#         # CHECK PROCESSED
#         # ----------------------------------------------------

#         with self.lock:

#             if url in self.processed:

#                 return


#             self.attempted += 1


#         try:

#             # ------------------------------------------------
#             # REQUEST
#             # ------------------------------------------------

#             response = requests.get(
#                 url,
#                 headers=HEADERS,
#                 timeout=REQUEST_TIMEOUT
#             )


#             status = response.status_code


#             # ------------------------------------------------
#             # HTTP STATUS
#             # ------------------------------------------------

#             if status != 200:

#                 with self.lock:

#                     self.http_errors += 1
#                     self.failed += 1


#                 print(
#                     f"[HTTP {status}] "
#                     f"{url}",
#                     flush=True
#                 )


#                 # Important:
#                 # We DON'T add failed URLs to
#                 # processed_links.txt.
#                 #
#                 # Therefore they can be retried
#                 # on a future run.

#                 return


#             # ------------------------------------------------
#             # HTML
#             # ------------------------------------------------

#             html = response.text


#             if not html:

#                 with self.lock:

#                     self.empty_extractions += 1
#                     self.failed += 1


#                 print(
#                     f"[EMPTY HTML] "
#                     f"{url}",
#                     flush=True
#                 )


#                 return


#             # ------------------------------------------------
#             # EXTRACT
#             # ------------------------------------------------

#             text = extract_article_text(
#                 html
#             )


#             # ------------------------------------------------
#             # EXTRACTION FAILED
#             # ------------------------------------------------

#             if not text:

#                 with self.lock:

#                     self.empty_extractions += 1
#                     self.failed += 1


#                 print(
#                     f"[NO TEXT] "
#                     f"{url}",
#                     flush=True
#                 )


#                 return


#             # ------------------------------------------------
#             # SAVE
#             # ------------------------------------------------

#             if len(text) < MIN_TEXT_LEN:

#                 with self.lock:

#                     self.empty_extractions += 1
#                     self.failed += 1


#                 print(
#                     f"[SHORT TEXT "
#                     f"{len(text)} chars] "
#                     f"{url}",
#                     flush=True
#                 )


#                 return


#             saved = self.save(
#                 url,
#                 text
#             )


#             if saved:

#                 with self.lock:

#                     self.successful += 1


#                 print(
#                     f"[SAVED] "
#                     f"{len(text):,} chars | "
#                     f"{url}",
#                     flush=True
#                 )


#         except requests.RequestException as e:

#             with self.lock:

#                 self.request_errors += 1
#                 self.failed += 1


#             print(
#                 f"[REQUEST ERROR] "
#                 f"{url} | {e}",
#                 flush=True
#             )


#         except Exception as e:

#             with self.lock:

#                 self.failed += 1


#             print(
#                 f"[ERROR] "
#                 f"{url} | "
#                 f"{type(e).__name__}: {e}",
#                 flush=True
#             )


#     # ========================================================
#     # WORKER
#     # ========================================================

#     def worker(self):

#         while self.running:

#             # ------------------------------------------------
#             # TARGET REACHED
#             # ------------------------------------------------

#             with self.lock:

#                 if self.total_bytes >= TARGET_BYTES:

#                     break


#             # ------------------------------------------------
#             # GET URL
#             # ------------------------------------------------

#             try:

#                 url = self.url_queue.get(
#                     timeout=5
#                 )


#             except queue.Empty:

#                 # ------------------------------------------------
#                 # QUEUE EMPTY
#                 # ------------------------------------------------

#                 if self.url_queue.empty():

#                     time.sleep(
#                         2
#                     )

#                     # Check once more
#                     if self.url_queue.empty():

#                         break


#                 continue


#             # ------------------------------------------------
#             # PROCESS
#             # ------------------------------------------------

#             try:

#                 self.process(
#                     url
#                 )

#             except Exception as e:

#                 print(
#                     f"[WORKER ERROR] "
#                     f"{type(e).__name__}: {e}",
#                     flush=True
#                 )


#             finally:

#                 self.url_queue.task_done()


#             # ------------------------------------------------
#             # DELAY
#             # ------------------------------------------------

#             time.sleep(
#                 DELAY
#             )


#     # ========================================================
#     # MONITOR
#     # ========================================================

#     def monitor(self):

#         while self.running:

#             time.sleep(
#                 15
#             )


#             with self.lock:

#                 current_bytes = (
#                     self.total_bytes
#                 )

#                 articles = (
#                     self.total_articles
#                 )

#                 attempted = (
#                     self.attempted
#                 )

#                 successful = (
#                     self.successful
#                 )

#                 failed = (
#                     self.failed
#                 )

#                 http_errors = (
#                     self.http_errors
#                 )

#                 empty = (
#                     self.empty_extractions
#                 )

#                 request_errors = (
#                     self.request_errors
#                 )

#                 queue_size = (
#                     self.url_queue.qsize()
#                 )


#             # ------------------------------------------------
#             # RATE
#             # ------------------------------------------------

#             elapsed = max(
#                 time.time()
#                 - self.start_time,
#                 1
#             )


#             mb = (
#                 current_bytes
#                 / (1024 * 1024)
#             )


#             target_mb = (
#                 TARGET_BYTES
#                 / (1024 * 1024)
#             )


#             rate = (
#                 mb
#                 / (elapsed / 60)
#             )


#             if rate > 0:

#                 eta_min = (
#                     target_mb - mb
#                 ) / rate

#             else:

#                 eta_min = 0


#             # ------------------------------------------------
#             # STATUS
#             # ------------------------------------------------

#             print(
#                 f"[STATUS] "
#                 f"{mb:.1f} MB / "
#                 f"{target_mb:.0f} MB | "
#                 f"Articles: {articles:,} | "
#                 f"Attempted: {attempted:,} | "
#                 f"Saved: {successful:,} | "
#                 f"Failed: {failed:,} | "
#                 f"HTTP errors: {http_errors:,} | "
#                 f"No text: {empty:,} | "
#                 f"Request errors: {request_errors:,} | "
#                 f"Queue: {queue_size:,} | "
#                 f"Rate: {rate:.1f} MB/min | "
#                 f"ETA: ~{eta_min:.0f} min",
#                 flush=True
#             )


#     # ========================================================
#     # RUN
#     # ========================================================

#     def run(self):

#         print()
#         print(
#             "=" * 75
#         )

#         print(
#             "EENADU EXTRACTOR"
#         )

#         print(
#             "=" * 75
#         )

#         print()

#         print(
#             f"Input : {LINKS_FILE}"
#         )

#         print(
#             f"Output: {OUTPUT_DIR}"
#         )

#         print(
#             f"Threads: {NUM_THREADS}"
#         )

#         print(
#             "Language filter: NONE"
#         )

#         print()


#         # ----------------------------------------------------
#         # MONITOR
#         # ----------------------------------------------------

#         monitor_thread = threading.Thread(
#             target=self.monitor,
#             daemon=True
#         )

#         monitor_thread.start()


#         # ----------------------------------------------------
#         # WORKERS
#         # ----------------------------------------------------

#         threads = []


#         for i in range(
#             NUM_THREADS
#         ):

#             thread = threading.Thread(
#                 target=self.worker,
#                 name=f"eenadu-worker-{i}"
#             )

#             thread.start()

#             threads.append(
#                 thread
#             )


#         # ----------------------------------------------------
#         # WAIT
#         # ----------------------------------------------------

#         for thread in threads:

#             thread.join()


#         # ----------------------------------------------------
#         # STOP
#         # ----------------------------------------------------

#         self.running = False


#         # ----------------------------------------------------
#         # CLOSE FILES
#         # ----------------------------------------------------

#         with self.lock:

#             self.out_file.close()

#             self.proc_file.close()


#         # ----------------------------------------------------
#         # FINAL REPORT
#         # ----------------------------------------------------

#         print()
#         print(
#             "=" * 75
#         )

#         print(
#             "DONE"
#         )

#         print(
#             "=" * 75
#         )

#         print(
#             f"Total MB saved : "
#             f"{self.total_bytes / (1024 * 1024):.1f}"
#         )

#         print(
#             f"Articles saved : "
#             f"{self.total_articles:,}"
#         )

#         print(
#             f"Attempted      : "
#             f"{self.attempted:,}"
#         )

#         print(
#             f"Successful     : "
#             f"{self.successful:,}"
#         )

#         print(
#             f"Failed         : "
#             f"{self.failed:,}"
#         )

#         print(
#             f"HTTP errors    : "
#             f"{self.http_errors:,}"
#         )

#         print(
#             f"No text        : "
#             f"{self.empty_extractions:,}"
#         )

#         print(
#             f"Request errors : "
#             f"{self.request_errors:,}"
#         )

#         print()

#         print(
#             f"Output directory:"
#         )

#         print(
#             OUTPUT_DIR
#         )


# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":

#     extractor = Extractor()

#     extractor.run()


"""
Phase 2: Eenadu Data Extractor

- Reads URLs from eenadu_links.txt
- Downloads pages using requests()
- Extracts article text using Trafilatura
- Falls back to BeautifulSoup if Trafilatura fails
- NO language filter
- Saves extracted text into 100 MB chunks
- Stops after 1 GB
- Never re-fetches URLs recorded in processed_links.txt
- Safe to restart
- Shows useful HTTP/extraction errors instead of silently hiding them
"""

import re
import time
import queue
import threading
import unicodedata
import warnings
from pathlib import Path

import requests
import trafilatura

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


# ============================================================
# CONFIGURATION
# ============================================================

LINKS_FILE = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/sakshi/sakshi_links.txt"
)

OUTPUT_DIR = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/sakshi/"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# TARGET / CHUNK SIZE
# ------------------------------------------------------------

TARGET_BYTES = 1 * 1024 * 1024 * 1024
CHUNK_BYTES = 100 * 1024 * 1024


# ------------------------------------------------------------
# THREADING
# ------------------------------------------------------------

NUM_THREADS = 12
DELAY = 0.1


# ------------------------------------------------------------
# EXTRACTION
# ------------------------------------------------------------

MIN_TEXT_LEN = 100


# ------------------------------------------------------------
# NETWORK
# ------------------------------------------------------------

REQUEST_TIMEOUT = 20


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),

    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,"
        "*/*;q=0.8"
    ),

    "Accept-Language": (
        "te-IN,te;q=0.9,en-US;q=0.8,en;q=0.7"
    ),

    "Connection": "keep-alive",
}


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize(
        "NFC",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize excessive newlines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# BEAUTIFULSOUP EXTRACTION
# ============================================================

def extract_with_bs4(html: str) -> str:
    """
    Fallback extractor.

    Tries multiple possible article containers because
    website HTML structures can change.
    """

    soup = BeautifulSoup(
        html,
        "lxml"
    )


    # --------------------------------------------------------
    # REMOVE NON-CONTENT ELEMENTS
    # --------------------------------------------------------

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "header",
        "footer",
        "nav",
        "form"
    ]):

        tag.decompose()


    # --------------------------------------------------------
    # FIRST: TRY ARTICLE TAG
    # --------------------------------------------------------

    article = soup.find(
        "article"
    )

    if article:

        paragraphs = article.find_all(
            "p"
        )

        text = "\n".join(
            p.get_text(
                " ",
                strip=True
            )
            for p in paragraphs
            if len(
                p.get_text(
                    " ",
                    strip=True
                )
            ) > 20
        )

        text = clean_text(text)

        if len(text) >= MIN_TEXT_LEN:

            return text


    # --------------------------------------------------------
    # TRY COMMON CONTENT CLASS NAMES
    # --------------------------------------------------------

    selectors = [

        # General article/content containers
        "[class*='article-content']",
        "[class*='articleContent']",

        "[class*='story-content']",
        "[class*='storyContent']",

        "[class*='content-body']",
        "[class*='contentBody']",

        "[class*='fullstory']",
        "[class*='full-story']",

        "[class*='main-content']",
        "[class*='mainContent']",

        "[class*='news-content']",
        "[class*='newsContent']",

        # Telugu news sites frequently use these kinds
        # of wrappers
        "[class*='story']",
        "[class*='article']",
        "[class*='content']",

        # Last fallback
        "main",
    ]


    candidates = []


    for selector in selectors:

        try:

            elements = soup.select(
                selector
            )

        except Exception:

            continue


        for element in elements:

            paragraphs = element.find_all(
                "p"
            )


            if not paragraphs:
                continue


            text = "\n".join(
                p.get_text(
                    " ",
                    strip=True
                )
                for p in paragraphs
                if len(
                    p.get_text(
                        " ",
                        strip=True
                    )
                ) > 20
            )


            text = clean_text(
                text
            )


            if len(text) >= MIN_TEXT_LEN:

                candidates.append(
                    text
                )


    # --------------------------------------------------------
    # RETURN LONGEST CANDIDATE
    # --------------------------------------------------------

    if candidates:

        return max(
            candidates,
            key=len
        )


    return ""


# ============================================================
# EXTRACT ARTICLE TEXT
# ============================================================

def extract_article_text(html: str) -> str:
    """
    Extraction pipeline:

        HTML
          ↓
        Trafilatura
          ↓
        BeautifulSoup fallback
    """

    # --------------------------------------------------------
    # TRAFILATURA
    # --------------------------------------------------------

    try:

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )

        text = clean_text(
            text
        )

        if len(text) >= MIN_TEXT_LEN:

            return text

    except Exception:

        pass


    # --------------------------------------------------------
    # BEAUTIFULSOUP FALLBACK
    # --------------------------------------------------------

    try:

        text = extract_with_bs4(
            html
        )

        text = clean_text(
            text
        )

        if len(text) >= MIN_TEXT_LEN:

            return text

    except Exception:

        pass


    return ""


# ============================================================
# EXTRACTOR CLASS
# ============================================================

class Extractor:

    def __init__(self):

        # ----------------------------------------------------
        # QUEUE
        # ----------------------------------------------------

        self.url_queue = queue.Queue()


        # ----------------------------------------------------
        # LOCK
        # ----------------------------------------------------

        self.lock = threading.Lock()


        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.running = True


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        self.total_bytes = 0

        self.total_articles = 0

        self.attempted = 0

        self.successful = 0

        self.failed = 0

        self.http_errors = 0

        self.empty_extractions = 0

        self.request_errors = 0


        self.start_time = time.time()


        # ----------------------------------------------------
        # OUTPUT CHUNK DETECTION
        # ----------------------------------------------------

        self.file_index = 1


        existing_chunks = list(
            OUTPUT_DIR.glob(
                "eenadu_articles_*.txt"
            )
        )


        if existing_chunks:

            for path in existing_chunks:

                try:

                    index = int(
                        path.stem.split("_")[-1]
                    )

                    if index > self.file_index:

                        self.file_index = index

                except Exception:

                    pass


                self.total_bytes += (
                    path.stat().st_size
                )


        if self.total_bytes:

            print(
                f"Resuming: "
                f"{self.total_bytes / (1024 * 1024):.1f} MB "
                f"already saved"
            )


        # ----------------------------------------------------
        # CURRENT OUTPUT FILE
        # ----------------------------------------------------

        self.current_path = (
            OUTPUT_DIR
            / f"eenadu_articles_{self.file_index}.txt"
        )


        self.out_file = open(
            self.current_path,
            "a",
            encoding="utf-8"
        )


        self.file_bytes = (
            self.current_path.stat().st_size
            if self.current_path.exists()
            else 0
        )


        # ----------------------------------------------------
        # PROCESSED LINKS FILE
        # ----------------------------------------------------

        self.proc_path = (
            OUTPUT_DIR
            / "processed_links.txt"
        )


        self.processed = set()


        if self.proc_path.exists():

            with open(
                self.proc_path,
                encoding="utf-8"
            ) as f:

                for line in f:

                    url = line.strip()

                    if url:

                        self.processed.add(
                            url
                        )


        self.proc_file = open(
            self.proc_path,
            "a",
            encoding="utf-8"
        )


        # ----------------------------------------------------
        # LOAD INPUT LINKS
        # ----------------------------------------------------

        links = []


        if LINKS_FILE.exists():

            with open(
                LINKS_FILE,
                encoding="utf-8"
            ) as f:

                links = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]


        print(
            f"Total links     : {len(links):,}"
        )

        print(
            f"Already done    : {len(self.processed):,}"
        )


        # ----------------------------------------------------
        # QUEUE UNIQUE LINKS
        # ----------------------------------------------------

        queued = set()


        for url in links:

            if not url:
                continue


            if url in self.processed:
                continue


            if url in queued:
                continue


            queued.add(
                url
            )


            self.url_queue.put(
                url
            )


        print(
            f"Initial queue   : "
            f"{self.url_queue.qsize():,}"
        )


    # ========================================================
    # ROTATE OUTPUT
    # ========================================================

    def rotate_file(self):

        self.out_file.close()


        self.file_index += 1


        self.current_path = (
            OUTPUT_DIR
            / f"eenadu_articles_{self.file_index}.txt"
        )


        self.out_file = open(
            self.current_path,
            "a",
            encoding="utf-8"
        )


        self.file_bytes = 0


    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        url: str,
        text: str
    ):

        content = (
            text
            + "\n\n"
        )


        size = len(
            content.encode(
                "utf-8"
            )
        )


        with self.lock:

            # Double-check
            if url in self.processed:

                return False


            # ------------------------------------------------
            # WRITE ARTICLE
            # ------------------------------------------------

            self.out_file.write(
                content
            )

            self.out_file.flush()


            # ------------------------------------------------
            # RECORD URL
            # ------------------------------------------------

            self.proc_file.write(
                url + "\n"
            )

            self.proc_file.flush()


            # ------------------------------------------------
            # UPDATE STATE
            # ------------------------------------------------

            self.processed.add(
                url
            )

            self.file_bytes += size

            self.total_bytes += size

            self.total_articles += 1


            # ------------------------------------------------
            # ROTATE
            # ------------------------------------------------

            if (
                self.file_bytes
                >= CHUNK_BYTES
            ):

                self.rotate_file()


        return True


    # ========================================================
    # PROCESS ONE URL
    # ========================================================

    def process(
        self,
        url: str
    ):

        # ----------------------------------------------------
        # CHECK PROCESSED
        # ----------------------------------------------------

        with self.lock:

            if url in self.processed:

                return


            self.attempted += 1


        try:

            # ------------------------------------------------
            # REQUEST
            # ------------------------------------------------

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )


            status = response.status_code


            # ------------------------------------------------
            # HTTP STATUS
            # ------------------------------------------------

            if status != 200:

                with self.lock:

                    self.http_errors += 1
                    self.failed += 1


                print(
                    f"[HTTP {status}] "
                    f"{url}",
                    flush=True
                )


                # Important:
                # We DON'T add failed URLs to
                # processed_links.txt.
                #
                # Therefore they can be retried
                # on a future run.

                return


            # ------------------------------------------------
            # HTML
            # ------------------------------------------------

            html = response.text


            if not html:

                with self.lock:

                    self.empty_extractions += 1
                    self.failed += 1


                print(
                    f"[EMPTY HTML] "
                    f"{url}",
                    flush=True
                )


                return


            # ------------------------------------------------
            # EXTRACT
            # ------------------------------------------------

            text = extract_article_text(
                html
            )


            # ------------------------------------------------
            # EXTRACTION FAILED
            # ------------------------------------------------

            if not text:

                with self.lock:

                    self.empty_extractions += 1
                    self.failed += 1


                print(
                    f"[NO TEXT] "
                    f"{url}",
                    flush=True
                )


                return


            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            if len(text) < MIN_TEXT_LEN:

                with self.lock:

                    self.empty_extractions += 1
                    self.failed += 1


                print(
                    f"[SHORT TEXT "
                    f"{len(text)} chars] "
                    f"{url}",
                    flush=True
                )


                return


            saved = self.save(
                url,
                text
            )


            if saved:

                with self.lock:

                    self.successful += 1


                print(
                    f"[SAVED] "
                    f"{len(text):,} chars | "
                    f"{url}",
                    flush=True
                )


        except requests.RequestException as e:

            with self.lock:

                self.request_errors += 1
                self.failed += 1


            print(
                f"[REQUEST ERROR] "
                f"{url} | {e}",
                flush=True
            )


        except Exception as e:

            with self.lock:

                self.failed += 1


            print(
                f"[ERROR] "
                f"{url} | "
                f"{type(e).__name__}: {e}",
                flush=True
            )


    # ========================================================
    # WORKER
    # ========================================================

    def worker(self):

        while self.running:

            # ------------------------------------------------
            # TARGET REACHED
            # ------------------------------------------------

            with self.lock:

                if self.total_bytes >= TARGET_BYTES:

                    break


            # ------------------------------------------------
            # GET URL
            # ------------------------------------------------

            try:

                url = self.url_queue.get(
                    timeout=5
                )


            except queue.Empty:

                # ------------------------------------------------
                # QUEUE EMPTY
                # ------------------------------------------------

                if self.url_queue.empty():

                    time.sleep(
                        2
                    )

                    # Check once more
                    if self.url_queue.empty():

                        break


                continue


            # ------------------------------------------------
            # PROCESS
            # ------------------------------------------------

            try:

                self.process(
                    url
                )

            except Exception as e:

                print(
                    f"[WORKER ERROR] "
                    f"{type(e).__name__}: {e}",
                    flush=True
                )


            finally:

                self.url_queue.task_done()


            # ------------------------------------------------
            # DELAY
            # ------------------------------------------------

            time.sleep(
                DELAY
            )


    # ========================================================
    # MONITOR
    # ========================================================

    def monitor(self):

        while self.running:

            time.sleep(
                15
            )


            with self.lock:

                current_bytes = (
                    self.total_bytes
                )

                articles = (
                    self.total_articles
                )

                attempted = (
                    self.attempted
                )

                successful = (
                    self.successful
                )

                failed = (
                    self.failed
                )

                http_errors = (
                    self.http_errors
                )

                empty = (
                    self.empty_extractions
                )

                request_errors = (
                    self.request_errors
                )

                queue_size = (
                    self.url_queue.qsize()
                )


            # ------------------------------------------------
            # RATE
            # ------------------------------------------------

            elapsed = max(
                time.time()
                - self.start_time,
                1
            )


            mb = (
                current_bytes
                / (1024 * 1024)
            )


            target_mb = (
                TARGET_BYTES
                / (1024 * 1024)
            )


            rate = (
                mb
                / (elapsed / 60)
            )


            if rate > 0:

                eta_min = (
                    target_mb - mb
                ) / rate

            else:

                eta_min = 0


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            print(
                f"[STATUS] "
                f"{mb:.1f} MB / "
                f"{target_mb:.0f} MB | "
                f"Articles: {articles:,} | "
                f"Attempted: {attempted:,} | "
                f"Saved: {successful:,} | "
                f"Failed: {failed:,} | "
                f"HTTP errors: {http_errors:,} | "
                f"No text: {empty:,} | "
                f"Request errors: {request_errors:,} | "
                f"Queue: {queue_size:,} | "
                f"Rate: {rate:.1f} MB/min | "
                f"ETA: ~{eta_min:.0f} min",
                flush=True
            )


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        print()
        print(
            "=" * 75
        )

        print(
            "EENADU EXTRACTOR"
        )

        print(
            "=" * 75
        )

        print()

        print(
            f"Input : {LINKS_FILE}"
        )

        print(
            f"Output: {OUTPUT_DIR}"
        )

        print(
            f"Threads: {NUM_THREADS}"
        )

        print(
            "Language filter: NONE"
        )

        print()


        # ----------------------------------------------------
        # MONITOR
        # ----------------------------------------------------

        monitor_thread = threading.Thread(
            target=self.monitor,
            daemon=True
        )

        monitor_thread.start()


        # ----------------------------------------------------
        # WORKERS
        # ----------------------------------------------------

        threads = []


        for i in range(
            NUM_THREADS
        ):

            thread = threading.Thread(
                target=self.worker,
                name=f"eenadu-worker-{i}"
            )

            thread.start()

            threads.append(
                thread
            )


        # ----------------------------------------------------
        # WAIT
        # ----------------------------------------------------

        for thread in threads:

            thread.join()


        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        self.running = False


        # ----------------------------------------------------
        # CLOSE FILES
        # ----------------------------------------------------

        with self.lock:

            self.out_file.close()

            self.proc_file.close()


        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        print()
        print(
            "=" * 75
        )

        print(
            "DONE"
        )

        print(
            "=" * 75
        )

        print(
            f"Total MB saved : "
            f"{self.total_bytes / (1024 * 1024):.1f}"
        )

        print(
            f"Articles saved : "
            f"{self.total_articles:,}"
        )

        print(
            f"Attempted      : "
            f"{self.attempted:,}"
        )

        print(
            f"Successful     : "
            f"{self.successful:,}"
        )

        print(
            f"Failed         : "
            f"{self.failed:,}"
        )

        print(
            f"HTTP errors    : "
            f"{self.http_errors:,}"
        )

        print(
            f"No text        : "
            f"{self.empty_extractions:,}"
        )

        print(
            f"Request errors : "
            f"{self.request_errors:,}"
        )

        print()

        print(
            f"Output directory:"
        )

        print(
            OUTPUT_DIR
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    extractor = Extractor()

    extractor.run()