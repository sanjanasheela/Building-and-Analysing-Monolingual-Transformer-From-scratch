import argparse
import hashlib
import os
import sqlite3
import sys
from datasketch import MinHash

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from header_pager_numberstripping import strip_headers_footers, strip_scan_artifacts
from length_filtering import drop_short_lines
from whitespace_normalisation import normalize_whitespace
from language_filter import filter_document
from unicode_normalisation import normalize_nepali_text

# Safety valve: force-flush a paragraph buffer even without a
# prefix-only separator if malformed input creates a huge paragraph.
MAX_BUFFER_LINES = 5_000

# Additional RAM safety valve based on total characters.
MAX_BUFFER_CHARS = 500_000

# Commit SQLite periodically based on total lines processed.
COMMIT_EVERY_N_PROCESSED = 20_000


def xxhash_or_sha(content: str) -> str:
    try:
        import xxhash
        return xxhash.xxh64(content.encode("utf-8")).hexdigest()
    except ImportError:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def process_paragraph(para: str, min_words: int) -> str:
    para = drop_short_lines(para, min_words=min_words)

    if not para.strip():
        return ""

    para = normalize_whitespace(
        para,
        collapse_blank_lines=False
    )

    if not para.strip():
        return ""

    para = filter_document(para)

    if not para.strip():
        return ""

    para = normalize_nepali_text(para)

    return para.strip()


def get_word_ngrams(text: str, n=3):
    """
    Generate word n-grams one at a time to reduce RAM usage.
    """
    tokens = text.split()

    if len(tokens) < n:
        yield text
        return

    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i + n])


def compute_minhash(text: str, num_perm=64) -> MinHash:
    """
    Computes a compact MinHash signature for near-duplicate matching.
    """
    m = MinHash(num_perm=num_perm)

    for ngram in get_word_ngrams(text, n=3):
        m.update(ngram.encode("utf-8"))

    return m


def split_prefixed_line(line: str):
    """
    Extract the prefix and content from a line.

    The prefix is assumed to be the first whitespace-delimited token.

    Examples:

        'public  తెలుగు text'
            -> ('public', 'తెలుగు text')

        'manual  తెలుగు text'
            -> ('manual', 'తెలుగు text')

        'public'
            -> ('public', '')

        'manual'
            -> ('manual', '')

        ''
            -> ('', '')
    """

    stripped = line.strip()

    if not stripped:
        return "", ""

    parts = stripped.split(None, 1)

    prefix = parts[0]

    # Prefix-only line.
    if len(parts) == 1:
        return prefix, ""

    content = parts[1]

    return prefix, content


def stream_clean_with_sqlite_and_near_dedup(
    input_path: str,
    output_path: str,
    min_words: int = 3,
    report_every: int = 500_000
):

    print(
        "[pipeline] Streaming with Disk-Based SQLite Cache "
        "(Exact + Near-Duplicates)..."
    )

    db_path = "dedup_cache.db"

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ---------------------------------------------------------
    # SQLite RAM / performance settings
    # ---------------------------------------------------------

    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.execute("PRAGMA cache_size = -8000")
    cursor.execute("PRAGMA temp_store = FILE")
    cursor.execute("PRAGMA mmap_size = 0")

    # ---------------------------------------------------------
    # Exact duplicate table
    # ---------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE exact_seen (
            hash TEXT PRIMARY KEY
        ) WITHOUT ROWID
        """
    )

    # ---------------------------------------------------------
    # LSH band table
    # ---------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE lsh_bands (
            band_id INTEGER,
            band_hash TEXT,
            PRIMARY KEY (band_id, band_hash)
        ) WITHOUT ROWID
        """
    )

    conn.commit()

    # ---------------------------------------------------------
    # Runtime state
    # ---------------------------------------------------------

    buffer = []
    buffer_chars = 0

    # Prefix belonging to the current paragraph.
    current_prefix = ""

    lines_read = 0
    lines_since_commit = 0

    stats = {
        "kept": 0,
        "exact": 0,
        "near": 0
    }

    # ---------------------------------------------------------
    # MinHash / LSH configuration
    # ---------------------------------------------------------

    num_perm = 64
    num_bands = 8
    rows_per_band = num_perm // num_bands

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)) or ".",
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Periodic SQLite commit
    # ---------------------------------------------------------

    def maybe_commit():
        nonlocal lines_since_commit

        if lines_since_commit >= COMMIT_EVERY_N_PROCESSED:
            conn.commit()
            lines_since_commit = 0

    # ---------------------------------------------------------
    # Process one paragraph
    # ---------------------------------------------------------

    def handle_paragraph(buf, prefix, fout):

        if not buf:
            return

        # Prefixes have already been removed.
        raw_para = "\n".join(buf)

        # -----------------------------------------------------
        # Normal cleaning
        # -----------------------------------------------------

        cleaned_para = process_paragraph(
            raw_para,
            min_words
        )

        if not cleaned_para:
            return

        # -----------------------------------------------------
        # 1. Exact deduplication
        # -----------------------------------------------------

        h = xxhash_or_sha(cleaned_para)

        cursor.execute(
            "SELECT 1 FROM exact_seen WHERE hash = ?",
            (h,)
        )

        if cursor.fetchone():

            stats["exact"] += 1

            return

        # -----------------------------------------------------
        # 2. Near deduplication
        # -----------------------------------------------------

        m = compute_minhash(
            cleaned_para,
            num_perm=num_perm
        )

        signature = m.digest()

        band_hashes = []

        for band_id in range(num_bands):

            start = band_id * rows_per_band
            end = start + rows_per_band

            band_bytes = signature[
                start:end
            ].tobytes()

            band_hash = hashlib.md5(
                band_bytes
            ).hexdigest()

            band_hashes.append(
                (band_id, band_hash)
            )

            cursor.execute(
                """
                SELECT 1
                FROM lsh_bands
                WHERE band_id = ? AND band_hash = ?
                """,
                (band_id, band_hash)
            )

            if cursor.fetchone():

                stats["near"] += 1

                return

        # -----------------------------------------------------
        # Not a duplicate -> persist
        # -----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO exact_seen (hash)
            VALUES (?)
            """,
            (h,)
        )

        cursor.executemany(
            """
            INSERT OR IGNORE INTO lsh_bands
            (band_id, band_hash)
            VALUES (?, ?)
            """,
            band_hashes
        )

        # -----------------------------------------------------
        # Restore the ORIGINAL prefix to every output line.
        #
        # Example:
        #
        # manual  line one
        # manual  line two
        #
        # or:
        #
        # public  line one
        # public  line two
        # -----------------------------------------------------

        for output_line in cleaned_para.splitlines():

            fout.write(
                prefix
                + "  "
                + output_line
                + "\n"
            )

        # Paragraph separator.
        fout.write("\n")

        stats["kept"] += 1

    # ---------------------------------------------------------
    # Read input
    # ---------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as fin, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        for line in fin:

            lines_read += 1
            lines_since_commit += 1

            # -------------------------------------------------
            # Progress report
            # -------------------------------------------------

            if lines_read % report_every == 0:

                print(
                    f"  Processed {lines_read:,} lines | "
                    f"Kept: {stats['kept']:,} || "
                    f"Exact Dupes: {stats['exact']:,} | "
                    f"Near Dupes: {stats['near']:,}"
                )

            text = line.rstrip("\n")

            # -------------------------------------------------
            # Extract prefix and content.
            #
            # IMPORTANT:
            # The prefix is NOT hard-coded.
            # -------------------------------------------------

            prefix, content = split_prefixed_line(text)

            # -------------------------------------------------
            # Prefix-only line = END OF PARAGRAPH
            # -------------------------------------------------

            if prefix and not content.strip():

                if buffer:

                    handle_paragraph(
                        buffer,
                        current_prefix,
                        fout
                    )

                    buffer = []
                    buffer_chars = 0

                current_prefix = ""

                maybe_commit()

                continue

            # -------------------------------------------------
            # Normal prefixed content line
            # -------------------------------------------------

            if prefix and content.strip():

                # Remember the original prefix for this paragraph.
                if not current_prefix:

                    current_prefix = prefix

                # Store ONLY the content.
                buffer.append(content)

                buffer_chars += len(content)

            # -------------------------------------------------
            # Unexpected line without a prefix
            #
            # We keep it instead of silently deleting it.
            # -------------------------------------------------

            else:

                if text.strip():

                    buffer.append(text)

                    buffer_chars += len(text)

            # -------------------------------------------------
            # RAM safety valves
            # -------------------------------------------------

            if (
                len(buffer) >= MAX_BUFFER_LINES
                or buffer_chars >= MAX_BUFFER_CHARS
            ):

                handle_paragraph(
                    buffer,
                    current_prefix,
                    fout
                )

                buffer = []
                buffer_chars = 0
                current_prefix = ""

            maybe_commit()

        # -----------------------------------------------------
        # Flush final paragraph
        # -----------------------------------------------------

        if buffer:

            handle_paragraph(
                buffer,
                current_prefix,
                fout
            )

            buffer = []
            buffer_chars = 0
            current_prefix = ""

    # ---------------------------------------------------------
    # Final DB cleanup
    # ---------------------------------------------------------

    conn.commit()

    conn.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    )

    conn.close()

    if os.path.exists(db_path):
        os.remove(db_path)

    for ext in ("-wal", "-shm"):

        p = db_path + ext

        if os.path.exists(p):
            os.remove(p)

    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    print(f"\n{'=' * 60}")

    print(
        f"Total lines read   : {lines_read:,}"
    )

    print(
        f"Unique Paras Kept  : {stats['kept']:,}"
    )

    print(
        f"Exact Dupes Dropped: {stats['exact']:,}"
    )

    print(
        f"Near Dupes Dropped : {stats['near']:,}"
    )

    print(
        f"[pipeline] Written : {output_path}"
    )


def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input"
    )

    parser.add_argument(
        "output"
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=3
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=500_000
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    stream_clean_with_sqlite_and_near_dedup(
        args.input,
        args.output,
        args.min_words,
        args.report_every
    )