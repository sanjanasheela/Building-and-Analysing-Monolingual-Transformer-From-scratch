import os
import requests
import trafilatura
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

OUTPUT_DIR = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SITEMAPS = {
    "filmibeat": "https://telugu.filmibeat.com/sitemap.xml",
    "news18": "https://telugu.news18.com/commonfeeds/v1/tel/sitemap/today",
    "gulte": "https://www.gulte.com/telugu-sitemap.xml",
    "asianet": "https://telugu.asianetnews.com/sitemap.xml",
    "manatelangana": "https://www.manatelangana.news/sitemap_index.xml",
}

def get_urls_from_sitemap(sitemap_url):
    urls = []
    try:
        r = requests.get(sitemap_url, timeout=10)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            # Find all <loc> tags
            for child in root.iter():
                if child.tag.endswith('loc') and child.text:
                    url = child.text.strip()
                    if url.endswith('.xml'):
                        urls.extend(get_urls_from_sitemap(url))
                    elif '.jpg' not in url and '.png' not in url:
                        urls.append(url)
    except Exception as e:
        pass
    return list(set(urls))

def scrape_url(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text) > 200:
                return text
    except:
        pass
    return None

def main():
    for name, sitemap in SITEMAPS.items():
        print(f"Fetching URLs for {name} from {sitemap}...")
        urls = get_urls_from_sitemap(sitemap)
        # Limit to 5000 URLs per site to be fast
        urls = urls[:5000]
        print(f"Found {len(urls)} URLs for {name}. Starting scrape...")
        
        output_file = os.path.join(OUTPUT_DIR, f"{name}_bulk.txt")
        saved = 0
        with open(output_file, 'a', encoding='utf-8') as f:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(scrape_url, url): url for url in urls}
                for future in as_completed(future_to_url):
                    text = future.result()
                    if text:
                        f.write(text + "\n\n---\n\n")
                        saved += 1
                        if saved % 100 == 0:
                            print(f"{name}: Saved {saved} articles...")
        print(f"{name}: Done. Saved {saved} total articles.")

if __name__ == '__main__':
    main()
