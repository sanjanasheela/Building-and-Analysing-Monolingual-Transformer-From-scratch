import os
import bz2
import xml.etree.ElementTree as ET
import re

dump_dir = '/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/wikimedia_dumps'
output_file = '/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/llm/wikimedia_extra_dumps.txt'

def clean_wikitext(text):
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:File|Image|Category|వర్గం|దస్త్రం):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[([^\]|]*\|)?([^\]]*)\]\]', r'\2', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"'{2,}", '', text)
    text = re.sub(r'={2,}[^=]+={2,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

total_bytes = 0

with open(output_file, 'w', encoding='utf-8') as out_f:
    for filename in os.listdir(dump_dir):
        if not filename.endswith('.xml.bz2'):
            continue
        print(f"Extracting {filename}...")
        file_path = os.path.join(dump_dir, filename)
        
        try:
            with bz2.open(file_path, 'rt', encoding='utf-8') as f:
                context = ET.iterparse(f, events=('end',))
                for event, elem in context:
                    if elem.tag.endswith('text'):
                        text = elem.text
                        if text:
                            cleaned = clean_wikitext(text)
                            if len(cleaned) > 100:
                                out_f.write(cleaned + '\n\n')
                                total_bytes += len(cleaned.encode('utf-8'))
                        elem.clear()
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

print(f"Done. Total extracted size: {total_bytes / (1024*1024):.2f} MB")
