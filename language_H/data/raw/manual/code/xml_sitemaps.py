import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import time

# --- CONFIGURATION ---
DATA_DIR = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data")
DATA_DIR.mkdir(exist_ok=True)

# Define the root sitemaps for the sites. 
# (Most sites use /sitemap.xml, /sitemap_index.xml, or /news-sitemap.xml)
SITEMAPS = {
    "onlinekhabar": "https://www.onlinekhabar.com/sitemap_index.xml"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def parse_sitemap(url, site_name, outfile, visited_sitemaps):
    """Recursively parse sitemaps and extract article URLs."""
    if url in visited_sitemaps:
        return
    visited_sitemaps.add(url)
    
    print(f"[{site_name}] Fetching sitemap: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"  -> Failed to fetch (Status {response.status_code})")
            return

        # Parse the XML
        root = ET.fromstring(response.content)
        
        # XML namespaces are usually present in sitemaps (e.g., xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
        # We can strip the namespace for easier searching, or use a wildcard match.
        namespace = ''
        if '}' in root.tag:
            namespace = root.tag.split('}')[0] + '}'

        # 1. If this is a Sitemap Index, it contains links to OTHER sitemaps
        if root.tag.endswith('sitemapindex'):
            sitemap_tags = root.findall(f".//{namespace}sitemap")
            print(f"  -> Found {len(sitemap_tags)} sub-sitemaps. Processing...")
            for sitemap in sitemap_tags:
                loc = sitemap.find(f"{namespace}loc")
                if loc is not None and loc.text:
                    parse_sitemap(loc.text, site_name, outfile, visited_sitemaps)
                    time.sleep(0.5) # Be polite to the server

        # 2. If this is a standard URL set, it contains actual page links
        elif root.tag.endswith('urlset'):
            url_tags = root.findall(f".//{namespace}url")
            article_count = 0
            
            for url_node in url_tags:
                loc = url_node.find(f"{namespace}loc")
                if loc is not None and loc.text:
                    link = loc.text
                    
                    # Optional: Add basic filtering here if you want to skip category pages
                    # Usually, if it ends in a number or .html, it's an article.
                    if "/tags/" not in link and "/author/" not in link:
                        outfile.write(link + "\n")
                        article_count += 1
            
            print(f"  -> Extracted {article_count} URLs.")

    except Exception as e:
        print(f"  -> Error parsing {url}: {e}")

def run_sitemap_extractor():
    for site_name, root_sitemap in SITEMAPS.items():
        print(f"\n========== Starting extraction for {site_name} ==========")
        output_file = DATA_DIR / f"{site_name}_all_links.txt"
        
        # Use a set to prevent infinite loops if sitemaps reference each other
        visited_sitemaps = set() 
        
        with open(output_file, "a", encoding="utf-8") as outfile:
            parse_sitemap(root_sitemap, site_name, outfile, visited_sitemaps)
            
        print(f"========== Finished {site_name}. Links saved to {output_file} ==========")

if __name__ == "__main__":
    run_sitemap_extractor()