import requests

# The API endpoint for English Wikisource
url = "https://en.wikisource.org/w/api.php"

# Parameters to get the plain text of a specific page
params = {
    "action": "query",
    "format": "json",
    "titles": "The Raven", # Replace with your target page title
    "prop": "extracts",
    "explaintext": True,   # Returns clean text instead of HTML tags
}

# Wikimedia strictly requires a descriptive User-Agent
headers = {
    "User-Agent": "MyWikisourceScraper/1.0 (your_email@example.com)"
}

response = requests.get(url, params=params, headers=headers)
data = response.json()

# Parse the JSON response to extract the text
pages = data["query"]["pages"]
for page_id, page_info in pages.items():
    print(f"Title: {page_info['title']}")
    print("-" * 20)
    
    # Print the first 500 characters of the text
    if "extract" in page_info:
        print(page_info["extract"][:500])
    else:
        print("No text found for this page.")