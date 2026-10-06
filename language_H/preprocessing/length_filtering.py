
# ─────────────────────────────────────────────────────────────
# . Drop lines/documents too short to be meaningful
# ─────────────────────────────────────────────────────────────
def drop_short_lines(text, min_words=3, min_chars=None):
    """
    Remove lines that don't meet a minimum word (or character) count.
    Useful for stripping stray single-word lines, orphaned bullets, etc.
    """
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        word_count = len(stripped.split())
        if word_count < min_words:
            continue
        if min_chars is not None and len(stripped) < min_chars:
            continue
        kept.append(line)
    return "\n".join(kept)


def is_document_too_short(text, min_words=20, min_chars=None):
    """
    Check whether an entire document is too short to be meaningful
    (e.g. empty scans, near-blank pages, junk files).

    Returns True if the document should be dropped.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if min_chars is not None and len(stripped) < min_chars:
        return True
    word_count = len(stripped.split())
    return word_count < min_words


def filter_short_documents(documents, min_words=20, min_chars=None):
    """
    Filter a collection of documents, dropping any that are too short.

    Args:
        documents: list[str] or list[dict] with a 'text' key.
    Returns:
        Filtered list (same type of elements as input).
    """
    def get_text(doc):
        return doc['text'] if isinstance(doc, dict) else doc

    return [
        doc for doc in documents
        if not is_document_too_short(get_text(doc), min_words=min_words, min_chars=min_chars)
    ]