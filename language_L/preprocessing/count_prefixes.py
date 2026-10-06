import sys
from tqdm import tqdm

def analyze_prefixes(file_path):
    manual_count = 0
    public_count = 0
    no_prefix_count = 0
    empty_lines = 0
    total_lines = 0

    print(f"Analyzing line prefixes in: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
        for line in tqdm(infile, desc="Processing lines", unit="lines"):
            total_lines += 1
            sline = line.strip()
            if not sline:
                empty_lines += 1
                continue
            
            if sline.startswith("manual"):
                manual_count += 1
            elif sline.startswith("public"):
                public_count += 1
            else:
                no_prefix_count += 1

    valid_lines = total_lines - empty_lines
    print("\n" + "="*50)
    print("LINE PREFIX STATISTICS")
    print("="*50)
    print(f"Total lines      : {total_lines:,}")
    print(f"Non-empty lines  : {valid_lines:,}")
    print(f"Empty lines      : {empty_lines:,}")
    print("-" * 50)
    print(f"Lines with 'manual' prefix   : {manual_count:,} ({manual_count / (valid_lines or 1) * 100:.2f}%)")
    print(f"Lines with 'public' prefix   : {public_count:,} ({public_count / (valid_lines or 1) * 100:.2f}%)")
    print(f"Lines without any prefix     : {no_prefix_count:,} ({no_prefix_count / (valid_lines or 1) * 100:.2f}%)")
    print("="*50)

if __name__ == '__main__':
    target_file = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_corpus.txt"
    analyze_prefixes(target_file)
