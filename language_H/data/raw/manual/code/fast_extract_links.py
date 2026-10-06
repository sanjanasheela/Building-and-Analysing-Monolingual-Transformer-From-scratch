import os
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import trafilatura

INPUT_FILES = [
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/asianet_bulk.txt",
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/asianet_full_bulk.txt"
]
OUTPUT_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/asianet_extracted_texts.txt"

async def fetch_and_extract(session, url, sem, out_f):
    async with sem:
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    text = trafilatura.extract(html, include_links=False, include_images=False, include_comments=False)
                    if text and len(text) > 200 and re.search(r'[\u0C00-\u0C7F]', text):
                        out_f.write(text + "\n\n---\n\n")
                        return True
        except Exception:
            pass
    return False

async def main():
    urls = set()
    print("Reading URLs from input files...")
    for fpath in INPUT_FILES:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("http"):
                        url = line.split()[0]
                        if '.jpg' not in url and '.png' not in url and 'gallery' not in url:
                            urls.add(url)
    
    urls = list(urls)
    print(f"Found {len(urls)} unique URLs.")
    
    # Process up to 100k URLs. 100k URLs will yield tons of text.
    urls_to_process = urls[:100000]
    
    print(f"Processing {len(urls_to_process)} URLs with high concurrency...")
    
    success_count = 0
    sem = asyncio.Semaphore(150)
    
    out_f = open(OUTPUT_FILE, 'a', encoding='utf-8')
    
    connector = aiohttp.TCPConnector(limit=150)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for url in urls_to_process:
            tasks.append(asyncio.create_task(fetch_and_extract(session, url, sem, out_f)))
        
        # Gather with progress
        completed = 0
        for f in asyncio.as_completed(tasks):
            res = await f
            completed += 1
            if res:
                success_count += 1
            if completed % 1000 == 0:
                print(f"Processed {completed}/{len(tasks)} URLs. Successful: {success_count}")
                out_f.flush()
                
    out_f.close()
    print(f"Done. Successfully extracted {success_count} articles.")
    
    if success_count > 1000:
        for fpath in INPUT_FILES:
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"Removed old file {fpath}")

if __name__ == '__main__':
    asyncio.run(main())
