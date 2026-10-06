import re
from collections import Counter


# ─────────────────────────────────────────────────────────────
# 1. Header/footer/page-number stripping
# ─────────────────────────────────────────────────────────────
def strip_headers_footers(pages, min_repeat_ratio=0.6, edge_lines=3):
    """
    Remove running headers/footers/page numbers that repeat at consistent
    positions (top/bottom) across many pages.

    Args:
        pages: list[str], raw text of each page (in reading order).
        min_repeat_ratio: fraction of pages a line must appear in
            (at the same edge position) to be considered a header/footer.
        edge_lines: how many lines from the top/bottom of each page to
            consider as "edge" candidates.

    Returns:
        list[str]: cleaned page texts with header/footer lines removed.
    """
    page_lines = [p.splitlines() for p in pages]
    n_pages = len(page_lines)
    if n_pages == 0:
        return []

    def normalize_for_match(line):
        # Collapse digits so "Page 3 of 42" ~ "Page 4 of 42" match as pattern
        s = re.sub(r'\d+', '#', line.strip())
        return s

    top_counter = Counter()
    bottom_counter = Counter()

    for lines in page_lines:
        top = lines[:edge_lines]
        bottom = lines[-edge_lines:] if len(lines) >= edge_lines else lines
        top_counter.update({normalize_for_match(l) for l in top if l.strip()})
        bottom_counter.update({normalize_for_match(l) for l in bottom if l.strip()})

    threshold = max(2, int(n_pages * min_repeat_ratio))
    header_patterns = {l for l, c in top_counter.items() if c >= threshold}
    footer_patterns = {l for l, c in bottom_counter.items() if c >= threshold}

    # Also treat pure page-number lines as footer/header noise
    page_num_re = re.compile(r'^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$', re.I)

    cleaned_pages = []
    for lines in page_lines:
        kept = []
        n = len(lines)
        for i, line in enumerate(lines):
            norm = normalize_for_match(line)
            is_edge = i < edge_lines or i >= n - edge_lines
            if is_edge and (norm in header_patterns or norm in footer_patterns):
                continue
            if is_edge and page_num_re.match(line):
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept))

    return cleaned_pages


def strip_scan_artifacts(text, artifact_patterns=None):
    """
    Remove common OCR/scan artifacts: stray form-feed chars, repeated
    dashed rule lines, watermark-style single-char noise lines, etc.
    """
    if artifact_patterns is None:
        artifact_patterns = [
            r'^\s*[-_=~]{3,}\s*$',        # rule lines: ---- ____ ====
            r'^\s*\f\s*$',                 # form feed only
            r'^\s*[|.]{3,}\s*$',           # dot/pipe leaders (....... or |||)
            r'^\s*©.*\d{4}.*$',            # copyright footer lines
        ]
    combined = re.compile('|'.join(f'(?:{p})' for p in artifact_patterns), re.M | re.I)
    lines = text.splitlines()
    cleaned = [l for l in lines if not combined.match(l)]
    return "\n".join(cleaned)


