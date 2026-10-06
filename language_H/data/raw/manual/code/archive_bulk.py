import os
import requests
import time

OUTPUT_DIR = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "archive_org_telugu_bulk.txt")

SEARCH_URL = "https://archive.org/advancedsearch.php"

def fetch_archive_texts():
    params = {
        "q": "language:telugu AND mediatype:texts",
        "fl[]": "identifier",
        "rows": 2000,
        "page": 1,
        "output": "json"
    }
    
    print("Fetching list of identifiers...")
    try:
        r = requests.get(SEARCH_URL, params=params, timeout=60)
        items = r.json()["response"]["docs"]
    except Exception as e:
        print(f"Search failed: {e}")
        return

    print(f"Found {len(items)} items. Downloading texts...")
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for idx, item in enumerate(items):
            identifier = item["identifier"]
            # To get .txt, we typically look for {identifier}_djvu.txt
            txt_url = f"https://archive.org/download/{identifier}/{identifier}_djvu.txt"
            try:
                txt_r = requests.get(txt_url, timeout=15)
                if txt_r.status_code == 200:
                    text = txt_r.text
                    if len(text) > 1000:
                        f.write(f"=== {identifier} ===\n{text}\n\n")
                        print(f"[{idx}] Downloaded {identifier}: {len(text)/1024:.2f} KB")
                else:
                    # Sometimes the txt file is just {identifier}.txt
                    txt_url2 = f"https://archive.org/download/{identifier}/{identifier}.txt"
                    txt_r2 = requests.get(txt_url2, timeout=15)
                    if txt_r2.status_code == 200:
                        text = txt_r2.text
                        if len(text) > 1000:
                            f.write(f"=== {identifier} ===\n{text}\n\n")
                            print(f"[{idx}] Downloaded {identifier} (.txt): {len(text)/1024:.2f} KB")
            except Exception as e:
                pass
            
            time.sleep(1)

if __name__ == "__main__":
    fetch_archive_texts()
