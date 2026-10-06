import bz2
import xml.etree.ElementTree as ET
from pathlib import Path
import mwparserfromhell

# Directories
INPUT_DIR = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data2/llm/wikimedia_dumps")
OUTPUT_DIR = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_H/data/raw/manual/data/wiki_dumps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def strip_namespace(tag: str) -> str:
    """Removes XML namespaces (e.g., {http://...}page -> page)."""
    return tag.split("}")[-1] if "}" in tag else tag


def process_dump_to_txt(bz2_path: Path):
    # Output file name matches the source dump (e.g. tewikisource.xml.bz2 -> tewikisource.txt)
    base_name = bz2_path.name
    for ext in [".xml.bz2", ".bz2", ".xml"]:
        if base_name.endswith(ext):
            base_name = base_name[: -len(ext)]

    output_txt_path = OUTPUT_DIR / f"{base_name}.txt"
    print(f"\n[Starting] Processing {bz2_path.name} -> {output_txt_path.name}")

    article_count = 0

    with (
        bz2.open(bz2_path, "rt", encoding="utf-8", errors="replace") as in_f,
        open(output_txt_path, "w", encoding="utf-8") as out_f,
    ):

        # Stream XML elements
        context = ET.iterparse(in_f, events=("end",))

        for _, elem in context:
            tag_name = strip_namespace(elem.tag)

            if tag_name == "page":
                # Find title, text, and redirect status
                title_elem = elem.find(".//{*}title")
                text_elem = elem.find(".//{*}text")
                redirect_elem = elem.find(".//{*}redirect")

                # Skip Wikipedia redirects, talk pages, and metadata namespaces (e.g., "Wikipedia:", "File:")
                if (
                    redirect_elem is None
                    and title_elem is not None
                    and text_elem is not None
                ):
                    title = title_elem.text or ""

                    # Filter out non-content namespaces
                    ignore_prefixes = (
                        "వికీపీడియా:",
                        "దస్త్రం:",
                        "మూస:",
                        "వర్గం:",
                        "సహాయం:",
                        "Wikipedia:",
                        "Template:",
                        "Category:",
                        "File:",
                        "Help:",
                        "MediaWiki:",
                    )
                    if not any(
                        title.startswith(prefix) for prefix in ignore_prefixes
                    ):
                        raw_wikicode = text_elem.text or ""

                        if raw_wikicode.strip():
                            # Strip wiki formatting to plain text
                            parsed_wikicode = mwparserfromhell.parse(
                                raw_wikicode
                            )
                            clean_text = parsed_wikicode.strip_code().strip()

                            if clean_text:
                                # Write article content to TXT
                                out_f.write(f"=== {title} ===\n\n")
                                out_f.write(clean_text)
                                out_f.write("\n\n" + "-" * 50 + "\n\n")
                                article_count += 1

                # Free element memory immediately
                elem.clear()

    print(
        f"[Done] {bz2_path.name}: Extracted {article_count:,} articles into {output_txt_path.name}"
    )


def main():
    dump_files = sorted(INPUT_DIR.glob("*.xml.bz2"))

    if not dump_files:
        print(f"No '.xml.bz2' files found in {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(dump_files)} dump file(s) to process.")

    for dump_file in dump_files:
        process_dump_to_txt(dump_file)

    print(
        f"\nAll files processed successfully! Text files saved in: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()