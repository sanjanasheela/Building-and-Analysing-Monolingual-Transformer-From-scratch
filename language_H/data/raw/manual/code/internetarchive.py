"""
Download Telugu-language texts from Internet Archive into a local directory.

USAGE:
    pip install requests
    python download_telugu_ia.py

WHAT THIS DOES:
    1. Queries the IA advancedsearch API for Telugu-language text items.
    2. Filters to items that look public-domain / openly licensed
       (skips items flagged as "lending library only" / CDL-restricted).
    3. Downloads a preferred file type (PDF or OCR'd TXT) for each item
       into your target directory.

IMPORTANT:
    - Internet Archive hosts both public-domain works AND modern
      in-copyright books available only via Controlled Digital Lending
      (CDL) -- those are meant to be read online with a borrow limit,
      not bulk-downloaded. This script tries to skip CDL/"lending only"
      items automatically (see is_probably_public_domain), but you
      should spot-check results before using them, especially if this
      feeds into a dataset you'll redistribute.
    - Be polite to IA's servers: this script rate-limits requests.
      Don't remove the sleep() calls or crank up concurrency.
"""

import os
import time
import requests

# ── CONFIG ──────────────────────────────────────────────────────────
OUTPUT_DIR = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/sources/internetarchive"
MAX_ITEMS = 500          # how many items to attempt (raise/lower as needed)
# Any of these extensions will be downloaded -- one file per item, first match wins.
# Reorder if you want to prioritize a particular format.
ACCEPTED_EXTENSIONS = (
    ".pdf", ".txt", ".epub", ".djvu", ".djvu.txt", ".mobi", ".azw3", ".doc", ".docx"
)
DOWNLOAD_ALL_MATCHING = False  # True = grab every matching file per item, not just the first
REQUEST_DELAY = 1.0      # seconds between API/download requests
SEARCH_QUERY = 'language:(Telugu) AND mediatype:texts'

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_BASE = "https://archive.org/download/{identifier}/{filename}"


def search_items(query, max_results):
    """Page through IA search results."""
    items = []
    rows = 100
    page = 1
    while len(items) < max_results:
        params = {
            "q": query,
            "fl[]": ["identifier", "title", "year", "licenseurl", "collection"],
            "rows": rows,
            "page": page,
            "output": "json",
        }
        resp = requests.get(SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        docs = resp.json()["response"]["docs"]
        if not docs:
            break
        items.extend(docs)
        page += 1
        time.sleep(REQUEST_DELAY)
    return items[:max_results]


def is_probably_public_domain(metadata):
    """
    Heuristic check: skip items that look like CDL/lending-only books.
    Not a legal determination -- just a filter to reduce obviously
    restricted items. Always spot-check before redistributing data.
    """
    access = metadata.get("access-restricted-item", "false")
    if str(access).lower() == "true":
        return False
    collections = metadata.get("metadata", {}).get("collection", "")
    if isinstance(collections, str):
        collections = [collections]
    restricted_markers = {"inlibrary", "printdisabled", "lendinglibrary"}
    if any(c in restricted_markers for c in collections):
        return False
    return True


def pick_files(files):
    """
    Return filenames matching any accepted extension.
    If DOWNLOAD_ALL_MATCHING is False, returns at most one (first match).
    """
    matches = [
        f["name"] for f in files
        if f.get("name", "").lower().endswith(ACCEPTED_EXTENSIONS)
    ]
    if not matches:
        return []
    return matches if DOWNLOAD_ALL_MATCHING else matches[:1]


def download_file(identifier, filename, dest_dir):
    url = DOWNLOAD_BASE.format(identifier=identifier, filename=filename)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        print(f"  [skip] already exists: {filename}")
        return
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"  [ok] downloaded: {filename}")
    except requests.RequestException as e:
        print(f"  [fail] {filename}: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Searching Internet Archive for: {SEARCH_QUERY}")
    items = search_items(SEARCH_QUERY, MAX_ITEMS)
    print(f"Found {len(items)} candidate items.\n")

    for i, item in enumerate(items, 1):
        identifier = item["identifier"]
        title = item.get("title", identifier)
        print(f"[{i}/{len(items)}] {title} ({identifier})")

        try:
            meta_resp = requests.get(METADATA_URL.format(identifier=identifier), timeout=30)
            meta_resp.raise_for_status()
            metadata = meta_resp.json()
        except requests.RequestException as e:
            print(f"  [fail] couldn't fetch metadata: {e}")
            time.sleep(REQUEST_DELAY)
            continue

        if not is_probably_public_domain(metadata):
            print("  [skip] looks CDL/lending-restricted")
            time.sleep(REQUEST_DELAY)
            continue

        files = metadata.get("files", [])
        filenames = pick_files(files)
        if not filenames:
            print("  [skip] no file matching accepted extensions found")
            time.sleep(REQUEST_DELAY)
            continue

        for filename in filenames:
            download_file(identifier, filename, OUTPUT_DIR)
            time.sleep(REQUEST_DELAY)

    print("\nDone.")


if __name__ == "__main__":
    main()