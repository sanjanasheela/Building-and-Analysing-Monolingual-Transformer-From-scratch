import os
import sys

def merge_onlinekhabar_folder(input_folder, output_file):
    """
    Merges all .txt files from input_folder into output_file,
    prepending 'manual ' to every non-empty line.
    """
    print(f"Reading files from: {input_folder}")
    files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.txt')])
    print(f"Found {len(files)} files to merge.")

    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fpath in files:
            print(f"Processing: {os.path.basename(fpath)}")
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as infile:
                for line in infile:
                    if line.strip():
                        outfile.write('manual ' + line)
                    else:
                        outfile.write(line)
                    total_lines += 1

    print(f"Created '{output_file}' with {total_lines:,} lines.")

if __name__ == '__main__':
    raw_folder = '/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data/onlinekhabar'
    out_file = '/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_corpus_onlinekhabar_manual.txt'
    merge_onlinekhabar_folder(raw_folder, out_file)
