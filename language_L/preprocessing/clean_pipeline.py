"""
clean_pipeline.py
-----------------
Full Nepali text cleaning pipeline.

Steps (in order):
  1. header / page-number stripping
  2. length filtering        (drop short lines)
  3. whitespace normalisation
  4. language filtering      (Nepali majority-rule, gibberish removal)
  5. unicode normalisation   (NFC)

Usage
-----
As a function:
    from clean_pipeline import run_pipeline
    clean_text = run_pipeline(raw_text)

From the command line:
    python clean_pipeline.py <input_file> <output_file> [options]

    Options:
      --page-sep STR        String used to split the raw text into pages
                            (default: form-feed \\f). Use "NONE" to skip
                            header/footer stripping.
      --min-words INT       Minimum words per line to keep  (default: 3)
      --min-repeat-ratio F  Fraction of pages a line must appear in to be
                            treated as a header/footer  (default: 0.6)
      --edge-lines INT      Lines from top/bottom per page to inspect for
                            headers/footers  (default: 3)
"""

import argparse
import re
import sys
import os

# ── make imports work whether the script is run from this dir or the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from header_pager_numberstripping import strip_headers_footers, strip_scan_artifacts
from length_filtering              import drop_short_lines
from whitespace_normalisation      import normalize_whitespace
from language_filter               import filter_document
from unicode_normalisation         import normalize_nepali_text


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline  (all steps in one function)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    text: str,
    page_sep: str = "\f",
    min_words: int = 3,
    min_repeat_ratio: float = 0.6,
    edge_lines: int = 3,
) -> str:
    """
    Run all five Nepali cleaning steps on *text* and return the cleaned string.

    Parameters
    ----------
    text             : raw input text (full file contents as a string).
    page_sep         : character / string that separates pages in the raw
                       text.  Pass None to skip header/footer stripping.
    min_words        : minimum words a line must have to survive length
                       filtering (step 2).
    min_repeat_ratio : header/footer detection threshold (step 1).
    edge_lines       : how many lines from the top/bottom of each page are
                       inspected for repeated headers/footers (step 1).

    Returns
    -------
    Cleaned text as a single string.
    """

    # ── Step 1: header / page-number stripping (document-level) ──────────────
    # Must run on the whole document so it can detect repeating patterns
    # across pages before we break into paragraphs.
    if page_sep is not None and page_sep in text:
        pages = text.split(page_sep)
        pages = strip_headers_footers(
            pages,
            min_repeat_ratio=min_repeat_ratio,
            edge_lines=edge_lines,
        )
        text = "\n".join(pages)
    text = strip_scan_artifacts(text)

    # ── Split into paragraphs NOW, before any line-level step ────────────────
    # Every subsequent step runs per-paragraph so blank lines are never lost.
    paragraphs = re.split(r'\n{2,}', text)

    cleaned = []
    for para in paragraphs:
        if not para.strip():
            continue

        # ── Step 2: length filtering ──────────────────────────────────────────
        para = drop_short_lines(para, min_words=min_words)
        if not para.strip():
            continue

        # ── Step 3: whitespace normalisation ─────────────────────────────────
        # collapse_blank_lines=False — we handle paragraph breaks ourselves
        para = normalize_whitespace(para, collapse_blank_lines=False)
        if not para.strip():
            continue

        # ── Step 4: language filtering (Nepali majority-rule + gibberish) ─────
        para = filter_document(para)
        if not para.strip():
            continue

        # ── Step 5: unicode normalisation (NFC) ──────────────────────────────
        para = normalize_nepali_text(para)
        if para.strip():
            cleaned.append(para)

    return '\n\n'.join(cleaned)


# ──────────────────────────────────────────────────────────────────────────────
# File-level helper
# ──────────────────────────────────────────────────────────────────────────────

def clean_file(
    input_path: str,
    output_path: str,
    page_sep: str = "\f",
    min_words: int = 3,
    min_repeat_ratio: float = 0.6,
    edge_lines: int = 3,
) -> str:
    """
    Read *input_path*, run the full cleaning pipeline, write to *output_path*.

    Returns the cleaned text string (also written to output_path).
    """
    print(f"[pipeline] Reading  : {input_path}")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    lines_before = raw.count("\n") + 1
    print(f"[pipeline] Lines in : {lines_before:,}")

    cleaned = run_pipeline(
        raw,
        page_sep=page_sep,
        min_words=min_words,
        min_repeat_ratio=min_repeat_ratio,
        edge_lines=edge_lines,
    )

    lines_after = cleaned.count("\n") + 1 if cleaned else 0
    print(f"[pipeline] Lines out: {lines_after:,}  "
          f"({100 * lines_after / max(lines_before, 1):.1f}% retained)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"[pipeline] Written  : {output_path}")
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Nepali text cleaning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Path to the raw input .txt file")
    parser.add_argument("output", help="Path for the cleaned output .txt file")
    parser.add_argument(
        "--page-sep", default="\f",
        help="String separating pages in the raw file. Use NONE to skip "
             "header/footer stripping.",
    )
    parser.add_argument(
        "--min-words", type=int, default=3,
        help="Minimum words per line to keep.",
    )
    parser.add_argument(
        "--min-repeat-ratio", type=float, default=0.6,
        help="Fraction of pages a line must appear in to count as a header/footer.",
    )
    parser.add_argument(
        "--edge-lines", type=int, default=3,
        help="Lines from top/bottom of each page to inspect for headers/footers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    page_sep = None if args.page_sep.upper() == "NONE" else args.page_sep
    clean_file(
        input_path=args.input,
        output_path=args.output,
        page_sep=page_sep,
        min_words=args.min_words,
        min_repeat_ratio=args.min_repeat_ratio,
        edge_lines=args.edge_lines,
    )
