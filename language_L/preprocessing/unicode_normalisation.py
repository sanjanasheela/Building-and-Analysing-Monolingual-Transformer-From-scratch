import unicodedata


def normalize_nepali_text(text: str) -> str:
    """Apply NFC Unicode normalisation to Nepali (Devanagari) text."""
    return unicodedata.normalize('NFC', text)
