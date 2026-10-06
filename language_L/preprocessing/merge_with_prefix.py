"""
merge_with_prefix.py

Merges two large text files (A and B) into a single output file.
Each line in the output is prefixed with the dataset label.

Format:
    datasetA\t<line from file A>
    datasetB\t<line from file B>

Files are processed line-by-line (streaming) so memory usage stays O(1)
regardless of file size.

Disk-space mode (--delete_after):
    Each source file is deleted immediately after it has been fully streamed
    to the output, so you never need space for all three files at once.
    Peak usage = max(2*size_A + size_B,  size_A + 2*size_B)
    instead of the naive  2*(size_A + size_B).

    WARNING: this permanently deletes the source files. Make sure you have
    backups or are certain you no longer need the originals.

Usage:
    python merge_with_prefix.py \
        --file_a  /path/to/fileA.txt \
        --file_b  /path/to/fileB.txt \
        --output  /path/to/output.txt \
        --label_a datasetA \
        --label_b datasetB \
        [--separator "\t"]          # default: tab
        [--skip_empty]              # skip blank lines
        [--delete_after]            # delete each source after streaming it
"""

import argparse
import os
import sys


def stream_with_prefix(path: str, label: str, separator: str, skip_empty: bool, out,
                       delete_after: bool = False):
    """
    Stream lines from `path`, write each one prefixed with `label` to `out`.
    If delete_after=True, the source file is removed once streaming completes.
    """
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if skip_empty and not line.strip():
                continue
            out.write(f"{label}{separator}{line}\n")
            count += 1

    if delete_after:
        os.remove(path)
        print(f"  Deleted source file: {path}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Merge two large text files with per-line dataset prefixes."
    )
    parser.add_argument("--file_a",    required=True,  help="Path to text file A")
    parser.add_argument("--file_b",    required=True,  help="Path to text file B")
    parser.add_argument("--output",    required=True,  help="Path to output file")
    parser.add_argument("--label_a",   default="datasetA", help="Prefix label for file A (default: datasetA)")
    parser.add_argument("--label_b",   default="datasetB", help="Prefix label for file B (default: datasetB)")
    parser.add_argument("--separator", default="\t",   help="Separator between label and line (default: tab)")
    parser.add_argument("--skip_empty", action="store_true",
                        help="Skip blank / whitespace-only lines")
    parser.add_argument("--delete_after", action="store_true",
                        help=("Delete each source file immediately after streaming it "
                              "to save disk space. WARNING: permanent deletion."))
    args = parser.parse_args()

    # Resolve escape sequences entered on the command line (e.g. \\t -> \t)
    separator = args.separator.encode().decode("unicode_escape")

    # Safety: make sure we are not overwriting a source file
    out_real = os.path.realpath(args.output)
    for label, path in [(args.label_a, args.file_a), (args.label_b, args.file_b)]:
        if os.path.realpath(path) == out_real:
            sys.exit(f"ERROR: output path is the same as source file ({label}: {path}). "
                     "Choose a different output path.")

    print(f"Output      : {args.output}")
    print(f"Label A     : {args.label_a!r}  ->  {args.file_a}")
    print(f"Label B     : {args.label_b!r}  ->  {args.file_b}")
    print(f"Separator   : {separator!r}")
    print(f"Skip empty  : {args.skip_empty}")
    print(f"Delete after: {args.delete_after}")
    if args.delete_after:
        print("  *** WARNING: source files will be permanently deleted after streaming ***")
    print()

    with open(args.output, "w", encoding="utf-8") as out:
        print("Processing file A …")
        n_a = stream_with_prefix(args.file_a, args.label_a, separator,
                                 args.skip_empty, out, args.delete_after)
        print(f"  Written {n_a:,} lines from A")

        print("Processing file B …")
        n_b = stream_with_prefix(args.file_b, args.label_b, separator,
                                 args.skip_empty, out, args.delete_after)
        print(f"  Written {n_b:,} lines from B")

    print(f"\nDone! Total lines written: {n_a + n_b:,}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
