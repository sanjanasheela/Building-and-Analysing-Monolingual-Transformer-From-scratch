import subprocess
from pathlib import Path


SOURCES_FILE = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/code/nepali_sources.txt")
OUTPUT_DIR = Path("/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/raw/manual/data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(SOURCES_FILE, "r", encoding="utf-8") as f:
    sources = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]


print(f"Found {len(sources)} sources")


failed = []


for source in sources:

    name, url = source.split("|", 1)

    # Convert source name into a safe filename
    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".txt"
    )

    output_file = OUTPUT_DIR / filename

    print("\n" + "=" * 60)
    print(f"Source : {name}")
    print(f"URL    : {url}")
    print(f"Output : {output_file}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [
                "python3",
                "scrape.py",
                url,
                str(output_file)
            ],
            check=False
        )

        if result.returncode == 0:
            print(f"✓ Finished: {name}")
        else:
            print(f"✗ FAILED: {name}")
            failed.append(name)

    except Exception as e:
        print(f"✗ ERROR: {name}")
        print(e)
        failed.append(name)


print("\n" + "=" * 60)
print("SCRAPING COMPLETE")
print("=" * 60)

if failed:
    print("\nFailed sources:")
    for name in failed:
        print(f"  - {name}")
else:
    print("All sources completed successfully.")