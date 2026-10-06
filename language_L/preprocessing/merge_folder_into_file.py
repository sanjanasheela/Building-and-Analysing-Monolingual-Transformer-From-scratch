"""
merge_folder_into_file.py
--------------------------
Merge all cleaned .txt files from a processed folder into a single corpus file.

Usage:
    python merge_folder_into_file.py
"""

from pathlib import Path

INPUT_FOLDER = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data"
)
OUTPUT_FILE = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/nepali_manual.txt"
)

# Find .txt files in the folder AND all subdirectories
txt_files = sorted(INPUT_FOLDER.rglob("*.txt"))

# Don't accidentally include the output file itself
txt_files = [f for f in txt_files if f != OUTPUT_FILE]

print(f"Found {len(txt_files)} text files")

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for txt_file in txt_files:
        print(f"Merging: {txt_file.relative_to(INPUT_FOLDER)}")

        with open(txt_file, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())

        # Separate documents
        outfile.write("\n\n")

print(f"\nDone! Merged file: {OUTPUT_FILE}")