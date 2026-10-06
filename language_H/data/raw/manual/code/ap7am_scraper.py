import os
import asyncio
import aiohttp
import trafilatura
from bs4 import BeautifulSoup
import re

OUTPUT_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/ap7am_bulk.txt"
START_ID = 700000
END_ID = 877000
CONCURRENCY = 200

TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')

def extract_text(html):
    try:
        text = trafilatura.extract(html, include_links=False, include_images=False, include_comments=False)
        if text and len(text) > 150 and len(TELUGU_RE.findall(text)) > 50:
            return text
    except:
        pass
    return None

async def fetch_and_save(session, uid, sem, f):
    url = f"https://www.ap7am.com/tn/{uid}"
    async with sem:
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    text = extract_text(html)
                    if text:
                        f.write(text + "\n\n---\n\n")
                        return len(text)
        except Exception:
            pass
    return 0

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    
    total_bytes = 0
    count = 0
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        async with aiohttp.ClientSession() as session:
            # We iterate in chunks so we don't hold 150k tasks in memory all at once
            chunk_size = 5000
            for start in range(END_ID, START_ID, -chunk_size):
                end = max(start - chunk_size, START_ID)
                tasks = [fetch_and_save(session, uid, sem, f) for uid in range(start, end, -1)]
                results = await asyncio.gather(*tasks)
                
                for r in results:
                    if r > 0:
                        total_bytes += r
                        count += 1
                        
                print(f"Processed down to ID {end}. Saved {count} articles. Total size so far: {total_bytes / (1024*1024):.2f} MB")
                
if __name__ == '__main__':
    asyncio.run(main())
