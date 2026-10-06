import os

def concatenate_corpus_files(file1, file2, output_file):
    print(f"Merging:\n  1) {file1}\n  2) {file2}\nInto:\n  {output_file}\n")
    
    chunk_size = 64 * 1024 * 1024  # 64 MB chunk read
    
    with open(output_file, 'wb') as outfile:
        for fpath in [file1, file2]:
            print(f"Appending {fpath}...")
            with open(fpath, 'rb') as infile:
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
            # Ensure line break between files
            outfile.write(b'\n')

    print(f"\nDone! Combined output saved to '{output_file}'.")
    print(f"Total size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

if __name__ == '__main__':
    f1 = '/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_merged_deduped.txt'
    f2 = '/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_corpus_onlinekhabar_manual.txt'
    out = '/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_corpus.txt'
    
    concatenate_corpus_files(f1, f2, out)
