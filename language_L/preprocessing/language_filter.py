"""
nepali_language_filter.py

Full pipeline for cleaning Nepali OCR / web-scraped text:
  1. Detects and removes OCR gibberish tokens (broken conjuncts, junk scripts,
     repeated-character noise, isolated marks, fused mixed-script tokens).
  2. Applies a Nepali-vs-other-language majority rule per sentence:
       - other-language word count <= nepali word count  -> strip just those
         other-language words, keep the sentence
       - other-language word count >  nepali word count  -> drop the whole
         sentence
  3. Digits (Latin 0-9 and Devanagari ०-९) are always preserved and never
     count as "other language".

Nepali uses the Devanagari script (U+0900 – U+097F).

Usage:
    from language_filter import filter_document
    clean = filter_document(raw_text)
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Character-class regexes  (Devanagari / Nepali)
# ---------------------------------------------------------------------------

# Full Devanagari block
NEPALI_RE = re.compile(r'[\u0900-\u097F]')

# Valid Devanagari base letters (independent vowels + consonants)
NEPALI_BASE_RE = re.compile(
    r'[\u0905-\u0914'   # independent vowels: अ–औ
    r'\u0915-\u0939'   # consonants: क–ह
    r'\u0958-\u095F'   # additional consonants: क़–य़
    r'\u0960\u0961'    # ॠ ॡ
    r']'
)

# Dependent vowel signs (matras) — these cannot legitimately start a word
NEPALI_DEPENDENT_MARKS = re.compile(
    r'^[\u093E-\u094C'  # ा–ौ
    r'\u094E\u094F'     # ॎ ॏ
    r'\u0955-\u0957'    # ॕ–ॗ
    r'\u0962\u0963'     # ॢ ॣ
    r']'
)

# Devanagari virama (halant) — trailing virama = broken conjunct
VIRAMA = '\u094D'

# Devanagari digits U+0966–U+096F plus Latin digits
DIGIT_ONLY_RE = re.compile(
    r'^[0-9\u0966-\u096F]+([.,:/-][0-9\u0966-\u096F]+)*$'
)

# 4+ repeated identical characters = scan / copy-paste noise
REPEAT_RUN_RE = re.compile(r'(.)\1{3,}')

# Scripts that have no business appearing in a Nepali corpus
# (keeps Devanagari U+0900-U+097F as the ALLOWED script)
JUNK_SCRIPT_RE = re.compile(
    r'[\u0980-\u09FF'   # Bengali
    r'\u0A00-\u0A7F'   # Gurmukhi
    r'\u0A80-\u0AFF'   # Gujarati
    r'\u0B00-\u0B7F'   # Odia
    r'\u0B80-\u0BFF'   # Tamil
    r'\u0C00-\u0C7F'   # Telugu
    r'\u0C80-\u0CFF'   # Kannada
    r'\u0D00-\u0D7F'   # Malayalam
    r'\uE000-\uF8FF'   # Private Use Area
    r'\uFFF0-\uFFFF'   # Specials
    r']'
)

# Sentence boundary: danda (।), double danda (॥), . ! ?
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?\u0964\u0965])\s+')

# Digit mixed with symbols (OCR digit-soup)
HAS_DIGIT_RE  = re.compile(r'[0-9\u0966-\u096F]')
HAS_SYMBOL_RE = re.compile(r'[^\w\u0900-\u097F]')  # not word-char / not Devanagari

# Leading/trailing sentence punctuation (strip before classification)
EDGE_PUNCT_RE = re.compile(r'^[.,!?;:"\'\(\)\[\]।॥]+|[.,!?;:"\'\(\)\[\]।॥]+$')

# Emoji / pictograph ranges
EMOJI_RE = re.compile(
    '['
    '\U0001F300-\U0001FAFF'
    '\U0001F1E6-\U0001F1FF'
    '\U00002600-\U000027BF'
    '\U0001F000-\U0001F0FF'
    '\U00002190-\U000021FF'
    '\U0000FE0F'
    '\U0000200D'
    ']+'
)


# ---------------------------------------------------------------------------
# Emoji removal
# ---------------------------------------------------------------------------

def remove_emojis(text: str) -> str:
    """Strip emoji/pictograph characters from text."""
    return EMOJI_RE.sub('', text)


# ---------------------------------------------------------------------------
# Gibberish (OCR noise) detection
# ---------------------------------------------------------------------------

def is_gibberish_word(word: str) -> bool:
    """Heuristic OCR-noise detector. Returns True if the token is scan garbage."""
    if not word:
        return True

    # Contains a script that should never appear in Nepali text
    if JUNK_SCRIPT_RE.search(word):
        return True

    # Starts with a dependent vowel sign (can't begin a valid Devanagari word)
    if NEPALI_DEPENDENT_MARKS.match(word):
        return True

    # 4+ repeated characters = noise
    if REPEAT_RUN_RE.search(word):
        return True

    # Strip combining marks / punctuation; if nothing left, it's noise
    stripped = ''.join(
        ch for ch in word
        if unicodedata.category(ch) not in ('Mn', 'Mc', 'Po', 'Pc')
    )
    if not stripped:
        return True

    # Trailing virama = broken conjunct (incomplete OCR word)
    if word.endswith(VIRAMA):
        return True

    # Devanagari + Latin mixed in the same token = OCR fusion artefact
    has_devanagari = bool(NEPALI_RE.search(word))
    has_latin      = bool(re.search(r'[A-Za-z]', word))
    if has_devanagari and has_latin:
        return True

    # Too few base letters relative to total length = likely garbled conjunct
    if has_devanagari:
        base_chars = len(NEPALI_BASE_RE.findall(word))
        if len(word) > 2 and base_chars / len(word) < 0.3:
            return True

    return False


def clean_gibberish(sentence: str) -> str:
    """Remove gibberish tokens from a sentence; keep everything else."""
    words = sentence.split()
    kept  = [w for w in words if not is_gibberish_word(w)]
    return ' '.join(kept)


# ---------------------------------------------------------------------------
# Language classification (Nepali / other / digit / noise)
# ---------------------------------------------------------------------------

def classify_word(word: str) -> str:
    """Return 'nepali', 'digit', 'other', or 'noise' for a single token."""
    core = EDGE_PUNCT_RE.sub('', word)

    if DIGIT_ONLY_RE.match(core if core else word):
        return 'digit'

    devanagari_chars = len(NEPALI_RE.findall(word))
    alpha_chars      = sum(1 for ch in word if ch.isalpha())

    if alpha_chars == 0:
        # No letters: digit-symbol soup → noise; pure punctuation → neutral
        if HAS_DIGIT_RE.search(word) and HAS_SYMBOL_RE.search(word):
            return 'noise'
        return 'digit'

    return 'nepali' if devanagari_chars / alpha_chars > 0.5 else 'other'


def filter_sentence(sentence: str):
    """
    Majority-rule filter for one sentence.
    Returns cleaned sentence string, or None if it should be dropped.
    """
    words = sentence.split()
    if not words:
        return None

    tags         = [classify_word(w) for w in words]
    nepali_count = tags.count('nepali')
    other_count  = tags.count('other')

    if other_count > nepali_count:
        return None   # majority non-Nepali → drop whole sentence

    kept = [w for w, t in zip(words, tags) if t in ('nepali', 'digit')]
    return ' '.join(kept) if kept else None


def is_junk_line(line: str, min_real_word_ratio: float = 0.4,
                 min_real_words: int = 1) -> bool:
    """
    Returns True if the line is mostly OCR garbage (digit-symbol soup,
    broken conjuncts, junk scripts) with very little real Nepali content.
    """
    words = line.split()
    if not words:
        return True

    real_word_count = sum(
        1 for w in words
        if not is_gibberish_word(w) and classify_word(w) in ('nepali', 'other')
    )

    if real_word_count < min_real_words:
        return True
    return (real_word_count / len(words)) < min_real_word_ratio


# ---------------------------------------------------------------------------
# Full pipeline helpers
# ---------------------------------------------------------------------------

def filter_sentence_full(sentence: str):
    """Gibberish cleanup → language majority-rule filter."""
    cleaned = clean_gibberish(sentence)
    return filter_sentence(cleaned) if cleaned else None


def filter_nepali_text(text: str) -> str:
    """Run full sentence-level pipeline over a multi-sentence text block."""
    text      = remove_emojis(text)
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    result    = []
    for sent in sentences:
        if not sent.strip():
            continue
        filtered = filter_sentence_full(sent)
        if filtered:
            result.append(filtered)
    return ' '.join(result)


def filter_document(text: str, min_real_word_ratio: float = 0.4) -> str:
    """
    Full document-level pipeline, line by line:
      1. Drop lines that are mostly junk (is_junk_line).
      2. For surviving lines, run sentence-level gibberish + language filter.
    """
    kept_lines = []
    for line in text.splitlines():
        line = remove_emojis(line)
        if not line.strip():
            continue
        if is_junk_line(line, min_real_word_ratio=min_real_word_ratio):
            continue
        cleaned = filter_nepali_text(line)
        if cleaned:
            kept_lines.append(cleaned)
    return '\n'.join(kept_lines)
