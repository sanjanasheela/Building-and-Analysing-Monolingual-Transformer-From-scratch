import requests
from bs4 import BeautifulSoup
import trafilatura
import os
import time
import argparse

# Fetch article links from a homepage
def get_article_links(homepage_url):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0)'}
    response = requests.get(homepage_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('http') and len(href) > 40: 
            links.add(href)
    return links

# Extract text and append to single target file
def scrape_and_save(links, output_filepath):
    for i, link in enumerate(links):
        print(f"Scraping {link}...")
        
        downloaded = trafilatura.fetch_url(link)
        
        if downloaded:
            text = trafilatura.extract(downloaded)
            
            if text:
                with open(output_filepath, 'a', encoding='utf-8') as f:
                    f.write(text + "\n\n")
                print(f"  -> Appended to: {output_filepath}")
            else:
                print("  -> Skipped: No text found.")
                
        time.sleep(1)

if __name__ == "__main__":
    # 1. Setup argparse to accept command-line inputs
    parser = argparse.ArgumentParser(description="Scrape articles from a website and save to a file.")
    parser.add_argument("url", help="The target URL to scrape links from")
    parser.add_argument("filename", help="The output file name (e.g., dataset.txt)")
    
    # 2. Parse the arguments provided by the user
    args = parser.parse_args()
    
    # 3. Setup file paths based on user input
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGET_FILE = os.path.join(SCRIPT_DIR, args.filename)
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
    
    # 4. Run the pipeline using the provided arguments
    links_to_scrape = get_article_links(args.url)
    
    print(f"Found {len(links_to_scrape)} potential articles. Starting extraction...")
    scrape_and_save(links_to_scrape, TARGET_FILE)