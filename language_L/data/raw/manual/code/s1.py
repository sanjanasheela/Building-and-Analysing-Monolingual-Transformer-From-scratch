"""
Generic single-page scraper.

Usage:
    python3 scrape.py <url> <output_path>

Fetches the given URL, strips scripts/styles/nav/footer boilerplate,
and writes the extracted visible text to output_path.
"""

import sys
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (research data collection; contact: you@example.com)"}


def scrape(url, output_path):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip elements that aren't main content
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Saved {len(cleaned)} characters to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scrape.py <url> <output_path>")
        sys.exit(1)

    url_arg = sys.argv[1]
    output_arg = sys.argv[2]
    scrape(url_arg, output_arg)