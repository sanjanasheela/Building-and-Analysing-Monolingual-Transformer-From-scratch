"""
Internet Archive Telugu TXT Downloader
Downloads pre-OCR'd .txt files from IA (much faster than PDF+OCR).
Targets items tagged language:Telugu that have .txt files already.
"""
import os, time, requests

OUTPUT_DIR = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEARCH_URL = "https://archive.org/advancedsearch.php"
META_URL   = "https://archive.org/metadata/{id}"
DOWN_URL   = "https://archive.org/download/{id}/{fn}"
DELAY = 1.5
MAX_ITEMS = 1000

def search_ia(query, max_results):
    items, page = [], 1
    while len(items) < max_results:
        params = {"q": query, "fl[]": ["identifier","title"], "rows": 100,
                  "page": page, "output": "json"}
        r = requests.get(SEARCH_URL, params=params, timeout=30)
        docs = r.json()["response"]["docs"]
        if not docs:
            break
        items.extend(docs)
        page += 1
        time.sleep(DELAY)
    return items[:max_results]

def get_txt_files(identifier):
    r = requests.get(META_URL.format(id=identifier), timeout=30)
    meta = r.json()
    # Skip CDL-restricted items
    if str(meta.get("access-restricted-item","false")).lower() == "true":
        return []
    colls = meta.get("metadata",{}).get("collection","")
    if isinstance(colls, str): colls = [colls]
    if any(c in {"inlibrary","lendinglibrary","printdisabled"} for c in colls):
        return []
    files = meta.get("files", [])
    return [f["name"] for f in files if f.get("name","").endswith("_djvu.txt") or
            (f.get("name","").endswith(".txt") and "_meta" not in f.get("name",""))]

def append_to_corpus(text, label):
    path = os.path.join(OUTPUT_DIR, "ia_telugu_texts.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== {label} ===\n")
        f.write(text)

import re
TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')
def is_telugu(text, threshold=0.10):
    if not text or len(text) < 100: return False
    return len(TELUGU_RE.findall(text)) / len(text) >= threshold

# Run two queries: one for books, one for texts with djvu
for query in [
    'language:(Telugu) AND mediatype:texts AND format:"DjVuTXT"',
    'language:(Telugu) AND mediatype:texts AND -collection:lendinglibrary',
]:
    print(f"\nSearching: {query}")
    items = search_ia(query, MAX_ITEMS)
    print(f"Found {len(items)} items")

    for i, item in enumerate(items):
        identifier = item["identifier"]
        title = item.get("title", identifier)
        print(f"  [{i+1}/{len(items)}] {title[:60]}")
        try:
            txt_files = get_txt_files(identifier)
        except Exception as e:
            print(f"    meta error: {e}")
            time.sleep(DELAY)
            continue

        if not txt_files:
            print("    no txt files / restricted")
            time.sleep(DELAY)
            continue

        for fn in txt_files[:2]:  # max 2 files per item
            url = DOWN_URL.format(id=identifier, fn=fn)
            try:
                r = requests.get(url, timeout=60, stream=True)
                chunks = []
                for chunk in r.iter_content(chunk_size=65536):
                    chunks.append(chunk)
                text = b"".join(chunks).decode("utf-8", errors="replace")
                if is_telugu(text):
                    append_to_corpus(text, f"{title} | {fn}")
                    sz = len(text) / 1024
                    print(f"    SAVED {fn} ({sz:.0f} KB)")
                else:
                    print(f"    not Telugu: {fn}")
            except Exception as e:
                print(f"    download error: {e}")
            time.sleep(DELAY)

# Final size
path = os.path.join(OUTPUT_DIR, "ia_telugu_texts.txt")
if os.path.exists(path):
    print(f"\nia_telugu_texts.txt: {os.path.getsize(path)/1024/1024:.2f} MB")
print("Done.")
