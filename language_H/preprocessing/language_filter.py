"""
telugu_language_filter.py
 
Full pipeline for cleaning Telugu OCR text:
  1. Detects and removes OCR gibberish tokens (broken conjuncts, junk scripts,
     repeated-character noise, isolated marks, fused mixed-script tokens).
  2. Applies a Telugu-vs-other-language majority rule per sentence:
       - other-language word count <= telugu word count -> strip just those
         other-language words, keep the sentence
       - other-language word count >  telugu word count -> drop the whole
         sentence
  3. Digits are always preserved and never count as "other language".
 
Usage:
    from telugu_language_filter import filter_telugu_text
    clean_text = filter_telugu_text(raw_ocr_text)
 
Or run directly:
    python telugu_language_filter.py input.txt output.txt
"""
 
import re
import sys
import unicodedata
 
 
# --------------------------------------------------------------------------
# Character-class regexes
# --------------------------------------------------------------------------
 
TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')                     # full Telugu block
TELUGU_BASE_RE = re.compile(r'[\u0C05-\u0C39\u0C58-\u0C5A\u0C60\u0C61]')  # valid base letters
TELUGU_DEPENDENT_MARKS = re.compile(r'^[\u0C3E-\u0C56\u0C62\u0C63]')      # matras (can't start a word)
 
DIGIT_ONLY_RE = re.compile(r'^[0-9\u0966-\u096F]+([.,:/-][0-9\u0966-\u096F]+)*$')  # digits, decimals, dates
 
REPEAT_RUN_RE = re.compile(r'(.)\1{3,}')  # 4+ repeated chars = scan noise
 
# Scripts/blocks that have no business appearing in Telugu OCR output
JUNK_SCRIPT_RE = re.compile(
    r'[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F'
    r'\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F\u0E00-\u0E7F\u0F00-\u0FFF'
    r'\u1000-\u109F\u10A0-\u10FF\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF'
    r'\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0370-\u03FF\u0400-\u04FF'
    r'\u0530-\u058F\u0590-\u05FF\u0600-\u06FF\uE000-\uF8FF\uFFF0-\uFFFF\U000E0000-\U000E07FF]'
)

 
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?\u0964\u0965])\s+')  # . ! ? । ॥
 
# A token that has ANY digit (Latin or Telugu) mixed with symbol/punctuation
# characters, but is NOT a clean number (already excluded by DIGIT_ONLY_RE).
# This catches junk like "6920%2040/2%06", "౦౦౧౧౫6౧10౧/", "[12౧౧౭/&".
HAS_DIGIT_RE = re.compile(r'[0-9\u0C66-\u0C6F]')
HAS_SYMBOL_RE = re.compile(r'[^\w\u0C00-\u0C7F]')  # anything not word-char/Telugu (i.e. punctuation/symbols)
 
# Ordinary leading/trailing sentence punctuation to strip before classifying,
# so "2024." or "(5)" at a sentence boundary isn't mistaken for symbol-soup.
EDGE_PUNCT_RE = re.compile(r'^[.,!?;:"\'()\[\]]+|[.,!?;:"\'()\[\]]+$')
 
 
# Strict Emoji / Pictograph / Geometric Symbol ranges to strip from corpus
EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # Pictographs, emojis, supplemental symbols
    "\U0001F1E6-\U0001F1FF"  # Regional indicators (flags)
    "\U00002600-\U000027BF"  # Misc Symbols, Dingbats
    "\U00002100-\U0000214F"  # Letterlike Symbols (℃, ℅, №, ℠, ™)
    "\U00002190-\U000021FF"  # Arrows
    "\U00002300-\U000023FF"  # Technical Symbols (e.g. ⏰, ⏩, ⏬, ⍟)
    "\U00002460-\U000024FF"  # Enclosed Alphanumerics (e.g. ⓘ)
    "\U00002500-\U0000257F"  # Box Drawings (e.g. ─)
    "\U00002580-\U0000259F"  # Block Elements
    "\U000025A0-\U000025FF"  # Geometric Shapes (e.g. ■, □, ▲, ▼, ◆, ●, ◕, ◤)
    "\U00002800-\U000028FF"  # Braille Patterns
    "\U00002B00-\U00002BFF"  # Misc Symbols/Arrows (e.g. ⭐, ⬛, ⭄)
    "\U00003000-\U0000303F"  # CJK Symbols & Punctuation
    "\U00003200-\U000032FF"  # Enclosed CJK Letters and Months
    "\U00003300-\U000033FF"  # CJK Compatibility (e.g. ㎡)
    "\U0000FE00-\U0000FE0F"  # Variation selectors
    "\U0000FF00-\U0000FFEE"  # Halfwidth/Fullwidth Forms (e.g. ￮)
    "\U0000200D"             # Zero-width joiner
    "]+"
)

 
 
def remove_emojis(text):
    """Strip emoji/pictograph characters from text. Safe to call on a whole
    line/sentence before any other processing."""
    return EMOJI_RE.sub('', text)
 
 
# --------------------------------------------------------------------------
# Gibberish (OCR noise) detection
# --------------------------------------------------------------------------
 
def is_gibberish_word(word):
    """Heuristic OCR-noise detector. True if word looks like scan garbage."""
    if not word:
        return True
 
    if JUNK_SCRIPT_RE.search(word):
        return True
 
    if TELUGU_DEPENDENT_MARKS.match(word):
        return True
 
    if REPEAT_RUN_RE.search(word):
        return True
 
    stripped = ''.join(
        ch for ch in word
        if unicodedata.category(ch) not in ('Mn', 'Mc', 'Po', 'Pc')
    )
    if not stripped:
        return True
 
    if word.endswith('\u0C4D'):  # trailing virama, broken conjunct
        return True
 
    has_telugu = bool(TELUGU_RE.search(word))
    has_latin = bool(re.search(r'[A-Za-z]', word))
    if has_telugu and has_latin:
        return True
 
    if has_telugu:
        base_chars = len(TELUGU_BASE_RE.findall(word))
        total_len = len(word)
        if total_len > 2 and base_chars / total_len < 0.3:
            return True
 
    return False
 
 
def clean_gibberish(sentence):
    """Remove gibberish tokens from a sentence, keep everything else."""
    words = sentence.split()
    kept = [w for w in words if not is_gibberish_word(w)]
    return ' '.join(kept)
 
 
# --------------------------------------------------------------------------
# Language classification (Telugu / other / digit)
# --------------------------------------------------------------------------
 
def classify_word(word):
    """Return 'telugu', 'digit', 'other', or 'noise' for a single token."""
    core = EDGE_PUNCT_RE.sub('', word)  # strip ordinary sentence punctuation from edges
 
    if DIGIT_ONLY_RE.match(core if core else word):
        return 'digit'  # a clean number/date, e.g. 2024, 12,000, 5/12 (with optional . , ! etc. around it)
 
    telugu_chars = len(TELUGU_RE.findall(word))
    alpha_chars = sum(1 for ch in word if ch.isalpha())
 
    if alpha_chars == 0:
        # No letters at all. If it also has digits mixed with symbols
        # (%, |, [, ], /, &, : ...) it's OCR digit-symbol soup, not a
        # real number -> noise. Pure punctuation with no digits is
        # harmless and treated as neutral 'digit' (kept, doesn't affect ratio).
        if HAS_DIGIT_RE.search(word) and HAS_SYMBOL_RE.search(word):
            return 'noise'
        return 'digit'
 
    return 'telugu' if telugu_chars / alpha_chars > 0.5 else 'other'
 
 
def filter_sentence(sentence):
    """
    Apply the majority-rule filter to one sentence.
    Returns the cleaned sentence, or None if it should be dropped entirely.
    Noise tokens (digit-symbol soup) are always dropped and never counted.
    """
    words = sentence.split()
    if not words:
        return None
 
    tags = [classify_word(w) for w in words]
    telugu_count = tags.count('telugu')
    other_count = tags.count('other')
 
    if other_count > telugu_count:
        return None  # majority other-language -> drop whole sentence
 
    kept = [w for w, t in zip(words, tags) if t in ('telugu', 'digit')]
    if not kept:
        return None
 
    return ' '.join(kept)
 
 
def is_junk_line(line, min_real_word_ratio=0.4, min_real_words=1):
    """
    Line/paragraph-level check for whole lines that are mostly OCR garbage
    (digit-symbol soup, broken conjuncts, junk scripts) with little or no
    real content -- e.g. entire lines like:
        "౦౦౧౧౫6౧10౧/ 5920%2040/2%06 0 [12౧౧౭/& 9026220660"
    Returns True if the line should be dropped entirely.
    """
    words = line.split()
    if not words:
        return True
 
    real_word_count = 0
    for w in words:
        if is_gibberish_word(w):
            continue
        tag = classify_word(w)
        if tag in ('telugu', 'other'):
            real_word_count += 1
 
    if real_word_count < min_real_words:
        return True
 
    return (real_word_count / len(words)) < min_real_word_ratio
 
 
# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
 
def filter_sentence_full(sentence):
    """Gibberish cleanup, then language majority-rule filter."""
    cleaned = clean_gibberish(sentence)
    if not cleaned:
        return None
    return filter_sentence(cleaned)
 
 
def filter_telugu_text(text):
    """Run the full pipeline over a multi-sentence text block (no line breaks assumed)."""
    text = remove_emojis(text)
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
 
    result = []
    for sent in sentences:
        if not sent.strip():
            continue
        filtered = filter_sentence_full(sent)
        if filtered:
            result.append(filtered)
 
    return ' '.join(result)
 
 
def filter_document(text, min_real_word_ratio=0.4):
    """
    Full document-level pipeline, line by line. Best for raw OCR dumps where
    garbage tends to cluster into whole lines (like scanned tables, headers,
    barcodes-as-text, misdetected non-Telugu pages, etc.):
 
      1. Drop lines that are mostly junk (is_junk_line).
      2. For surviving lines, run sentence-level gibberish + language filtering.
    """
    lines = text.splitlines()
    kept_lines = []
 
    for line in lines:
        line = remove_emojis(line)
        if not line.strip():
            continue
        if is_junk_line(line, min_real_word_ratio=min_real_word_ratio):
            continue
        cleaned = filter_telugu_text(line)
        if cleaned:
            kept_lines.append(cleaned)
 
    return '\n'.join(kept_lines)
 
 


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

if __name__ == '__main__':
    
        junk_doc = (
            """Here is a multilingual message celebrating positivity, learning, and connection, featuring plenty of Telugu alongside other languages and emojis:

---

✨ **నమస్కారం! అందరికీ స్వాగతం!** 🙏💐

జీవితంలో ప్రతి క్షణం కొత్త విషయాలు నేర్చుకోవడానికి ఒక అద్భుతమైన అవకాశం! 🌟📚💡

🌏 **Global Greetings:**

* **English:** Welcome! Wishing you a day full of joy, creativity, and success! 🚀🌈✨
* **Hindi (हिंदी):** आपका स्वागत है! आपका हर दिन नई खुशियों और सफलताओं से भरा रहे। 🌸☀️🕊️
* **Tamil (தமிழ்):** உங்கள் வாழ்க்கை எப்போதுமே மகிழ்ச்சியும் அமைதியும் நிறைந்ததாக இருக்கட்டும்! 🌺🌿💫
* **Spanish (Español):** ¡Que tengas un día maravilloso lleno de energía positiva y grandes logros! ☀️🎉🎈
* **Japanese (日本語):** いつも笑顔で、素晴らしい一日をお過ごしください！ 🌸🍵🍙

---

📖 **తెలుగులో కొన్ని మంచి మాటలు (Inspiring Thoughts in Telugu):**

1. 🎯 **లక్ష్యం మరియు కృషి:**
కష్టపడి పనిచేస్తే సాధించలేనిది ఏదీ లేదు! 💪🔥 మీ కలలను నిజం చేసుకునే ప్రయాణంలో ఎప్పుడూ ఆత్మవిశ్వాసం కోల్పోకండి. 🧗‍♂️🏆
2. 🌿 **శాంతి మరియు ఆనందం:**
మనసు ప్రశాంతంగా ఉంటే, ప్రపంచమంతా అందంగా కనిపిస్తుంది. 🧘‍♀️🕊️ ప్రకృతితో కాసేపు సమయం గడపడం ఎంతో సాంత్వననిస్తుంది. 🌳🌻🍃
3. 🤝 **స్నేహం మరియు అనుబంధం:**
మంచి స్నేహితులు మరియు కుటుంబం మన జీవితానికి అసలైన బలం. 💖👨‍👩‍👧‍👦 ఎల్లప్పుడూ ఇతరులకు సహాయం చేస్తూ, చిరునవ్వులు పంచుదాం! 😊🤝❤️
4. ☕ **చిన్న చిన్న సంతోషాలు:**
ఉదయాన్నే వేడి వేడి కాఫీ ☕, వర్షపు జల్లులు 🌧️, మంచి పుస్తకం 📖... ఇవే జీవితంలో మధురమైన క్షణాలు! 🌈✨

---

🎉 **ముగింపు (Conclusion):**

ఎల్లప్పుడూ ఆనందంగా, ఉత్సాహంగా ఉండండి! 🌟🥳

మీ ప్రయాణం విజయవంతం కావాలని మనస్ఫూర్తిగా కోరుకుంటున్నాము! 🚀⭐🎊

**All the best! అందరికీ శుభం కలగాలి!** 💐🙏✨
"""
        )
        print("\n=== Document/line-level demo (junk-line dropping) ===")
        print(filter_document(junk_doc))

