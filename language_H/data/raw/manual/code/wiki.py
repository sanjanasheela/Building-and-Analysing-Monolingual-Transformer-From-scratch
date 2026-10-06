import os
import json
import glob

# Paths
EXTRACTED_FOLDER = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/extracted_wiki"
OUTPUT_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/wiki.txt"

# Find all the extracted chunk files
json_files = glob.glob(f"{EXTRACTED_FOLDER}/*/*")

print(f"Found {len(json_files)} extracted files. Merging into {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    # Parse the JSON string
                    article = json.loads(line)
                    
                    # Write the text to your dataset file
                    if article.get('text'):
                        outfile.write(article['text'].strip() + "\n\n")
                        
                except json.JSONDecodeError:
                    continue

print("Merge complete! Your .txt dataset is ready.")