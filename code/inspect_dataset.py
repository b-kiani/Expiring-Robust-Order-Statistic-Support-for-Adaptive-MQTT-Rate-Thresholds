#!/usr/bin/env python3
"""Inventory extracted CICIoT CSV headers and supporting files.

Usage (from the project folder):
    python code/inspect_dataset.py
    python code/inspect_dataset.py "C:/path/to/extracted/CSV"

Uses only Python's standard library. Reads CSV headers; never executes
supporting scripts or assigns traffic labels. Reports are written to results/.
A label-like column is a candidate, not verified ground truth.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LABEL_NAMES = {"label", "labels", "class", "attack_cat", "attack_type", "is_attack"}
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules"}
HEADER_LIMIT = 65536


def inspect_header(path):
    opener = gzip.open if path.name.lower().endswith(".csv.gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        line = stream.readline(HEADER_LIMIT + 1)
    if not line:
        raise ValueError("empty file")
    if len(line) > HEADER_LIMIT:
        raise ValueError("header exceeds 64 KiB; inspect file format")
    if "\x00" in line:
        raise ValueError("binary-looking file; not a plain CSV header")
    header = next(csv.reader(io.StringIO(line), strict=True))
    columns = [name.strip() for name in header]
    if not columns or any(not name for name in columns):
        raise ValueError("empty column name in header")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate column names after trimming spaces")
    return columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", help="Extracted dataset folder")
    parser.add_argument("--output", default="results", help="Report directory")
    args = parser.parse_args()
    root_text = args.root or input("Paste the full path to your extracted dataset folder: ")
    root_text = root_text.strip().strip('"').strip("'")
    if not root_text:
        parser.error("Enter the path to the extracted dataset folder.")
    root = Path(root_text).expanduser()
    if not root.is_dir():
        parser.error(f"Folder not found: {root}")
    root = root.resolve()
    output = Path(args.output).resolve()
    rows, supporting, archives, traversal_errors = [], [], [], []
    schemas = Counter()

    def walk_error(error):
        traversal_errors.append(str(error))

    for base, directories, filenames in os.walk(root, onerror=walk_error):
        directories[:] = sorted(name for name in directories if name not in SKIP_DIRS
                                and (Path(base) / name).resolve() != output)
        for name in sorted(filenames):
            path = Path(base) / name
            relative = path.relative_to(root).as_posix()
            lower = name.lower()
            if lower.endswith((".csv", ".csv.gz")):
                row = {"relative_path": relative, "size_bytes": "", "column_count": "",
                       "candidate_label_columns": "", "schema_sha256": "",
                       "columns_json": "", "status": "error", "error": ""}
                try:
                    row["size_bytes"] = path.stat().st_size
                    columns = inspect_header(path)
                    candidates = [c for c in columns if c.casefold() in LABEL_NAMES]
                    row.update(column_count=len(columns),
                               candidate_label_columns=" | ".join(candidates),
                               schema_sha256=hashlib.sha256(
                                   json.dumps(columns, ensure_ascii=False).encode("utf-8")
                               ).hexdigest(),
                               columns_json=json.dumps(columns, ensure_ascii=False),
                               status="header_read")
                    schemas[(len(columns), tuple(candidates), row["schema_sha256"])] += 1
                except (OSError, UnicodeError, csv.Error, ValueError, EOFError) as error:
                    row["error"] = str(error)
                rows.append(row)
            elif lower.endswith((".py", ".ipynb", ".md", ".rst", ".txt", ".json")) or (
                    "readme" in lower and lower.endswith(".pdf")):
                supporting.append(relative)
            elif lower.endswith((".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tgz")):
                archives.append(relative)

    rows.sort(key=lambda row: row["relative_path"])
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "dataset_inventory.csv"
    fields = ["relative_path", "size_bytes", "column_count", "candidate_label_columns",
              "schema_sha256", "columns_json", "status", "error"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    candidates = [row for row in rows if row["candidate_label_columns"]]
    errors = [row for row in rows if row["status"] == "error"]
    summary = {
        "checker_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "python_version": sys.version.split()[0],
        "csv_files_found": len(rows),
        "headers_read": len(rows) - len(errors),
        "files_with_candidate_label_columns": len(candidates),
        "schema_groups": [
            {"column_count": count, "candidate_label_columns": list(labels),
             "schema_sha256": schema_hash, "files": total}
            for (count, labels, schema_hash), total in sorted(schemas.items())
        ],
        "supporting_files": sorted(supporting),
        "archives_not_opened": sorted(archives),
        "traversal_errors": traversal_errors,
        "label_status": "Not verified. Header names and folder names do not establish row-level labels.",
        "scope": "Header inventory only; no row counts, values, full-file hashes or chronology verified.",
    }
    summary_path = output / "dataset_inventory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDATASET INVENTORY")
    print("CSV files found:", len(rows))
    print("Headers read:", len(rows) - len(errors))
    print("Files with candidate label columns:", len(candidates))
    for group in summary["schema_groups"]:
        labels = ", ".join(group["candidate_label_columns"]) or "none"
        print(f"  {group['files']} file(s): {group['column_count']} columns; "
              f"label-like fields: {labels}; header ID: {group['schema_sha256'][:12]}")
    if candidates:
        print("\nCandidate files (first 10):")
        for row in candidates[:10]:
            print(" ", row["relative_path"])
    print("\nSupporting files (first 20):")
    for name in sorted(supporting)[:20]:
        print(" ", name)
    if not supporting:
        print("  None under the selected folder.")
    if archives:
        print("\nArchives found but not opened:", len(archives))
        for name in sorted(archives)[:5]:
            print(" ", name)
    print("\nUnreadable CSV headers:", len(errors))
    for row in errors[:5]:
        print(" ", row["relative_path"], ":", row["error"])
    if traversal_errors:
        print("Folder traversal errors:", len(traversal_errors))
    if not rows:
        print("No CSVs found. Select the extracted folder containing Benign_Final and the attack folders.")
    print("\nLabel columns are candidates only; label provenance still needs verification.")
    print("Reports saved:")
    print(" ", csv_path)
    print(" ", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

