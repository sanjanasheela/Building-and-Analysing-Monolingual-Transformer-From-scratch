"""
OCR pipeline for Nepali PDFs.
Uses tesseract with 'nep' language pack (Devanagari).
Processes all PDFs in sources/books/ and outputs .txt files.
"""
import os
import zipfile
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path

# Point tesseract to our custom tessdata directory
os.environ["TESSDATA_PREFIX"] = "/home/sanjana/.local/share/tessdata"

BASE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L")
PDF_FOLDER = BASE / "data/raw/manual/sources/books_ocr"
OUTPUT_FOLDER = BASE / "data/raw/manual/data/ocr_books"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SWASTHANI_EXTRACT = PDF_FOLDER / "swasthani_extracted"


def unzip_swasthani():
    """Unzip Swasthani zip and return list of PDFs inside."""
    zip_path = PDF_FOLDER / "Swasthani_Brata_Katha.zip"
    if not zip_path.exists():
        print("Swasthani zip not found, skipping")
        return []
    SWASTHANI_EXTRACT.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(SWASTHANI_EXTRACT)
    pdfs = list(SWASTHANI_EXTRACT.rglob("*.pdf"))
    print(f"  Found {len(pdfs)} PDFs in Swasthani zip")
    return pdfs


def ocr_pdf(pdf_path: Path, output_path: Path, lang: str = "nep"):
    """OCR a single PDF, write text to output_path."""
    if output_path.exists():
        print(f"  SKIP (already done): {output_path.name}")
        return

    try:
        info = pdfinfo_from_path(str(pdf_path))
        total_pages = info["Pages"]
        print(f"\n  Starting: {pdf_path.name}  ({total_pages} pages)")

        with open(output_path, "w", encoding="utf-8") as fout:
            for i in range(1, total_pages + 1):
                imgs = convert_from_path(
                    str(pdf_path),
                    dpi=300,
                    first_page=i,
                    last_page=i,
                )
                text = pytesseract.image_to_string(imgs[0], lang=lang)
                fout.write(text)
                fout.write("\n\n")
                if i % 50 == 0 or i == total_pages:
                    print(f"    {pdf_path.name}: page {i}/{total_pages}")

        size_mb = output_path.stat().st_size / 1e6
        print(f"  ✓ Done: {output_path.name}  ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"  ✗ FAILED: {pdf_path.name} — {e}")
        if output_path.exists():
            output_path.unlink()  # remove incomplete file


def main():
    # Collect all PDFs to process
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    # Also add Swasthani PDFs after unzip
    swasthani_pdfs = unzip_swasthani()
    pdf_files += swasthani_pdfs

    print(f"\nFound {len(pdf_files)} PDF files to OCR")
    print(f"Output dir: {OUTPUT_FOLDER}\n")

    for pdf_path in pdf_files:
        out_name = pdf_path.stem.replace(" ", "_") + ".txt"
        output_path = OUTPUT_FOLDER / out_name
        ocr_pdf(pdf_path, output_path, lang="nep")

    print("\n=== All OCR complete ===")
    total = sum(f.stat().st_size for f in OUTPUT_FOLDER.glob("*.txt"))
    print(f"Total output: {total / 1e6:.1f} MB in {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
