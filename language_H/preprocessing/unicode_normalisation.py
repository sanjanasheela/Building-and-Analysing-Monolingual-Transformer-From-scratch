import unicodedata

def normalize_telugu_text(text):
    normalized_text = unicodedata.normalize('NFC', text)
    return normalized_text

