from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path


# Folder containing your PDFs
PDF_FOLDER = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/sources/internetarchive"
)

# Folder where the extracted TXT files will go
OUTPUT_FOLDER = Path(
    "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/ocr3"
)

# Create output folder if it doesn't exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# Find all PDFs
pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files")
for pdf_path in pdf_files:

    try:
        output_path = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

        print(f"\nStarting: {pdf_path.name}")

        total_pages = pdfinfo_from_path(str(pdf_path))["Pages"]

        with open(output_path, "w", encoding="utf-8") as f:

            for i in range(1, total_pages + 1):

                img = convert_from_path(
                    str(pdf_path),
                    dpi=300,
                    first_page=i,
                    last_page=i
                )[0]

                text = pytesseract.image_to_string(
                    img,
                    lang="tel"
                )

                f.write(text)
                f.write("\n\n")

                print(
                    f"  {pdf_path.name}: "
                    f"page {i}/{total_pages}"
                )

        print(f"✓ Finished: {output_path}")

    except Exception as e:
        print(f"✗ FAILED: {pdf_path.name}")
        print(f"  Error: {e}")
        continue