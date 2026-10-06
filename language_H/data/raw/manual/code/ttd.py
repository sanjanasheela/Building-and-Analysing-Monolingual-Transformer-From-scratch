import os
import requests
import time

# Create a dedicated directory to save the downloaded PDFs
output_dir = "ttd_ebooks"
os.makedirs(output_dir, exist_ok=True)

# Define the range of Book IDs you want to attempt to download.
# Note: The portal has thousands of books; test with a small range first.
start_id = 1
end_id = 50 

# Using a standard User-Agent helps prevent the server from blocking the script
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

print("Starting TTD E-Books download process...")

for book_id in range(start_id, end_id + 1):
    # Construct the download URL for the specific book ID
    download_url = f"https://ebooks.tirumala.org/download?id={book_id}"
    
    try:
        # Stream the request to avoid loading large PDFs entirely into memory
        response = requests.get(download_url, headers=headers, stream=True)
        
        # Verify the request was successful and the returned content is actually a PDF
        content_type = response.headers.get('Content-Type', '')
        if response.status_code == 200 and 'application/pdf' in content_type.lower():
            file_path = os.path.join(output_dir, f"ttd_book_{book_id}.pdf")
            
            # Write the file in chunks
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    
            print(f"✓ Successfully downloaded: ttd_book_{book_id}.pdf")
        else:
            # The ID might be invalid, deleted, or point to an HTML page instead
            print(f"✗ Book ID {book_id} not found or is not a valid PDF.")
            
        # IMPORTANT: Polite crawling. Pause between requests so you don't overload their servers.
        time.sleep(2)
        
    except requests.exceptions.RequestException as e:
        print(f"! Error attempting to download Book ID {book_id}: {e}")

print("========================================")
print("Finished downloading range.")