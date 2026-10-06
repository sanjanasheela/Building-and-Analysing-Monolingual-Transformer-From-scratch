import os

def extract_top_half(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Scanning {input_file} to count total lines...")
    
    # Pass 1: Count total lines efficiently without loading the file into memory
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        total_lines = sum(1 for _ in f_in)
    
    if total_lines == 0:
        print("The file is empty.")
        return

    halfway_point = total_lines // 2
    print(f"Total lines: {total_lines:,}. Extracting the first {halfway_point:,} lines...")

    # Pass 2: Write only the top 50% to the new file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for current_line_num, line in enumerate(f_in):
            if current_line_num < halfway_point:
                f_out.write(line)
            else:
                break # Stop reading once we hit the halfway mark

    print(f"Success! Top half saved to {output_file}")

# --- How to use it ---
# Replace these with your actual file names
INPUT_TEXT_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/telugu-corpus/data/te.txt" 
OUTPUT_TEXT_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/public/IndicCorp"

extract_top_half(INPUT_TEXT_FILE, OUTPUT_TEXT_FILE)