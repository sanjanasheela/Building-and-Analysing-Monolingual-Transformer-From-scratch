import re


def normalize_whitespace(text, collapse_blank_lines=True):
    """
    Normalize all whitespace variants (spaces, tabs, non-breaking spaces,
    zero-width spaces, etc.) to single regular spaces, and optionally
    collapse multiple blank lines into one.
    """
    # Normalize exotic whitespace/invisible chars to a plain space
    weird_ws = [
        '\u00a0',  # non-breaking space
        '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005',
        '\u2006', '\u2007', '\u2008', '\u2009', '\u200a',  # various unicode spaces
        '\u200b',  # zero-width space
        '\u202f',  # narrow no-break space
        '\ufeff',  # BOM / zero-width no-break space
        '\t',
    ]
    for ch in weird_ws:
        text = text.replace(ch, ' ')

    # Collapse runs of horizontal whitespace (but keep newlines intact)
    text = re.sub(r'[ \u00a0]{2,}', ' ', text)
    text = re.sub(r'[ \t]+\n', '\n', text)   # trailing spaces before newline
    text = re.sub(r'\n[ \t]+', '\n', text)   # leading spaces after newline

    if collapse_blank_lines:
        # 3+ newlines -> 2 (i.e., at most one blank line between paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

