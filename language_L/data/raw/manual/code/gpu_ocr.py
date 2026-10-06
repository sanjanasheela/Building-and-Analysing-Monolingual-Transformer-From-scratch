import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
# If False: Runtime -> Change runtime type -> Hardware accelerator -> GPU

# ============================================================
# CELL 4 — OCR script
# ============================================================
from pathlib import Path
from pdf2image import convert_from_path, pdfinfo_from_path
import easyocr
import numpy as np

PDF_FOLDER = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/sources/books_ocr")
OUTPUT_FOLDER = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data/ocr_books")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

reader = easyocr.Reader(["te"], gpu=True)

pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF file(s) for OCR processing.\n")

for pdf_path in pdf_files:
    output_path = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

    if output_path.exists():
        print(f"Skipping {pdf_path.name}, already processed.")
        continue

    try:
        print(f"========================================")
        print(f"Starting OCR: {pdf_path.name}")
        print(f"========================================")

        info = pdfinfo_from_path(str(pdf_path))
        total_pages = info["Pages"]
        print(f"Total Pages: {total_pages}")

        with open(output_path, "w", encoding="utf-8") as f:
            for page_num in range(1, total_pages + 1):
                try:
                    images = convert_from_path(
                        str(pdf_path),
                        dpi=300,
                        first_page=page_num,
                        last_page=page_num
                    )

                    if images:
                        img_array = np.array(images[0])
                        result = reader.readtext(
                            img_array,
                            detail=0,
                            paragraph=True
                        )
                        text = "\n".join(result)
                        f.write(text)
                        f.write("\n\n")

                except Exception as page_err:
                    print(f"  \u2717 page {page_num} failed: {page_err}")
                    f.write(f"[OCR FAILED: page {page_num}]\n\n")
                    continue

                if page_num % 10 == 0 or page_num == total_pages:
                    print(f"  {pdf_path.name}: processed page {page_num}/{total_pages}")

        print(f"\n\u2713 Finished: {output_path.name} -> Saved to {output_path}\n")

    except Exception as e:
        print(f"\u2717 FAILED: {pdf_path.name}")
        print(f"  Error: {e}\n")
        continue

print("All OCR tasks finished.")