# """
# public.py
# ---------
# Download Nepali public datasets (raygx/Nepali-Text-Corpus & ai4bharat/sangraha),
# extract raw text, and run the Nepali cleaning pipeline.

# Usage:
#     python public.py
# """

# import os
# import sys
# import time
# import gc
# import pandas as pd
# from pathlib import Path
# from huggingface_hub import hf_hub_download

# # Make preprocessing pipeline importable
# LANGUAGE_L_DIR = Path(__file__).resolve().parent.parent.parent.parent
# PREPROC_DIR = LANGUAGE_L_DIR / "preprocessing"
# if str(PREPROC_DIR) not in sys.path:
#     sys.path.insert(0, str(PREPROC_DIR))

# from clean_pipeline import run_pipeline

# BASE_DIR = Path(__file__).resolve().parent
# RAW_DATA_DIR = BASE_DIR / "data"
# PROCESSED_DIR = LANGUAGE_L_DIR / "data" / "processed" / "public"

# RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
# PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# # Datasets to fetch and process
# PUBLIC_DATASETS = [
#     {
#         "repo_id": "raygx/Nepali-Text-Corpus",
#         "files": [
#             "data/train-00000-of-00004-60d0fdefa46b7bf8.parquet",
#             "data/train-00001-of-00004-977dca81cd013026.parquet",
#             "data/train-00002-of-00004-ba35d0b900a7de7d.parquet",
#             "data/train-00003-of-00004-54a0a8b501e17261.parquet",
#         ],
#         "prefix": "raygx_corpus",
#     },
#     {
#         "repo_id": "ai4bharat/sangraha",
#         "files": [
#             "verified/nep/data-1.parquet",
#             "verified/nep/data-2.parquet",
#         ],
#         "prefix": "ai4bharat_sangraha",
#     }
# ]

# CHUNK_SIZE_MB = 50
# MIN_WORDS = 3


# def clean_text_chunked(raw_txt_path: Path, processed_txt_path: Path) -> None:
#     """Clean a raw text file in chunks and write to processed destination."""
#     chunk_size = CHUNK_SIZE_MB * 1024 * 1024
#     file_size = raw_txt_path.stat().st_size
#     bytes_processed = 0
#     lines_processed = 0
#     chunks = 0
#     t0 = time.time()

#     print(f"  Cleaning raw -> {processed_txt_path.name} ({file_size / (1024**2):.2f} MB)")

#     with open(raw_txt_path, "r", encoding="utf-8", errors="ignore") as infile, \
#          open(processed_txt_path, "w", encoding="utf-8") as outfile:

#         buffer = []
#         buffer_size = 0

#         for line in infile:
#             buffer.append(line)
#             line_size = len(line.encode("utf-8"))
#             buffer_size += line_size
#             bytes_processed += line_size
#             lines_processed += 1

#             if buffer_size >= chunk_size:
#                 chunks += 1
#                 text = "".join(buffer)
#                 cleaned = run_pipeline(text, min_words=MIN_WORDS)
#                 if cleaned:
#                     outfile.write(cleaned)
#                     outfile.write("\n\n")
#                     outfile.flush()

#                 del text, cleaned, buffer
#                 buffer = []
#                 buffer_size = 0
#                 gc.collect()

#         if buffer:
#             chunks += 1
#             text = "".join(buffer)
#             cleaned = run_pipeline(text, min_words=MIN_WORDS)
#             if cleaned:
#                 outfile.write(cleaned)
#                 outfile.write("\n\n")
#                 outfile.flush()
#             del text, cleaned, buffer
#             gc.collect()

#     elapsed = time.time() - t0
#     out_size = processed_txt_path.stat().st_size if processed_txt_path.exists() else 0
#     print(f"  Done: {chunks} chunk(s), {lines_processed:,} lines, cleaned size: {out_size / (1024**2):.2f} MB in {elapsed:.1f}s\n")


# def process_public_dataset():
#     total_start = time.time()
#     print("=" * 70)
#     print("STARTING PUBLIC DATASET DOWNLOAD & CLEANING PIPELINE")
#     print(f"Raw Output Folder      : {RAW_DATA_DIR}")
#     print(f"Processed Output Folder: {PROCESSED_DIR}")
#     print("=" * 70)

#     for ds_info in PUBLIC_DATASETS:
#         repo_id = ds_info["repo_id"]
#         files = ds_info["files"]
#         prefix = ds_info["prefix"]

#         print(f"\n>>> Processing Repository: {repo_id} ({len(files)} files)")

#         for idx, filename in enumerate(files, 1):
#             base_name = Path(filename).stem
#             raw_txt_path = RAW_DATA_DIR / f"{prefix}_{base_name}.txt"
#             proc_txt_path = PROCESSED_DIR / f"{prefix}_{base_name}.txt"

#             # Step 1: Download parquet file
#             print(f"\n[{idx}/{len(files)}] Downloading {filename} from {repo_id}...")
#             dl_start = time.time()
#             try:
#                 parquet_path = hf_hub_download(
#                     repo_id=repo_id,
#                     filename=filename,
#                     repo_type="dataset",
#                     local_dir=str(RAW_DATA_DIR / "_parquet_cache"),
#                 )
#                 print(f"  Downloaded to: {parquet_path} ({time.time() - dl_start:.1f}s)")
#             except Exception as e:
#                 print(f"  Error downloading {filename}: {e}")
#                 continue

#             # Step 2: Extract text to raw .txt file if not already extracted
#             if not raw_txt_path.exists() or raw_txt_path.stat().st_size == 0:
#                 print(f"  Extracting text from parquet to {raw_txt_path.name}...")
#                 try:
#                     df = pd.read_parquet(parquet_path)
#                     text_col = "text" if "text" in df.columns else df.columns[0]
#                     with open(raw_txt_path, "w", encoding="utf-8") as f:
#                         for text in df[text_col]:
#                             if isinstance(text, str) and text.strip():
#                                 f.write(text.strip())
#                                 f.write("\n\n")
#                     del df
#                     gc.collect()
#                     print(f"  Extracted {raw_txt_path.stat().st_size / (1024**2):.2f} MB raw text.")
#                 except Exception as e:
#                     print(f"  Error extracting text from {parquet_path}: {e}")
#                     continue
#             else:
#                 print(f"  Raw text already exists: {raw_txt_path.name} ({raw_txt_path.stat().st_size / (1024**2):.2f} MB)")

#             # Step 3: Run cleaning pipeline
#             clean_text_chunked(raw_txt_path, proc_txt_path)

#     total_elapsed = time.time() - total_start
#     print("=" * 70)
#     print(f"ALL PUBLIC DATASETS PROCESSED SUCCESSFULLY in {total_elapsed / 60:.1f} minutes")
#     print(f"Processed files saved to: {PROCESSED_DIR}")
#     print("=" * 70)


# if __name__ == "__main__":
#     process_public_dataset()


import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashokpant/nepali-news-dataset-large")

print("Path to dataset files:", path)