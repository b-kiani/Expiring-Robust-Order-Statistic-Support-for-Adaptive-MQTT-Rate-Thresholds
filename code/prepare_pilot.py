#!/usr/bin/env python3
"""Prepare a bounded CICIoT2023 scenario pilot on a CPU.

Run from the project directory:
    .venv/Scripts/python.exe code/prepare_pilot.py

This is data preparation, not a labelled intrusion-detection evaluation.
Folder names are preserved as scenario metadata; row-level attack labels
are not established by the supplied extractor. No model is trained here.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP",
    "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]
SCENARIOS = ("Benign_Final", "Recon-PortScan", "DDoS-SYN_Flood", "Mirai-udpplain")
METADATA = [
    "scenario_name", "label_source", "row_label_verified",
    "source_relative_path", "source_row_index",
]


def stable_seed(seed, text):
    digest = hashlib.sha256((str(seed) + "|" + text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_one(path, root, maximum, chunksize, seed):
    relative = path.relative_to(root).as_posix()
    before = path.stat()
    header = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    if header != FEATURES:
        raise ValueError(f"Unexpected feature names/order in {relative}. Recheck its inventory.")
    rng = np.random.default_rng(stable_seed(seed, relative))
    missing = np.zeros(len(FEATURES), dtype=np.int64)
    infinite = missing.copy()
    non_numeric = missing.copy()
    rows_seen, partial_count, no_finite_count = 0, 0, 0
    reservoir = None
    for raw in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        raw.columns = raw.columns.str.strip()
        raw = raw[FEATURES]
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        missing += raw.isna().sum().to_numpy(dtype=np.int64)
        non_numeric += (numeric.isna() & raw.notna()).sum().to_numpy(dtype=np.int64)
        array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        infinite += np.isinf(array).sum(axis=0)
        no_finite_count += int((~np.isfinite(array)).all(axis=1).sum())
        partial_count += int((numeric["Number"] < 10).sum())
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric["source_row_index"] = np.arange(rows_seen, rows_seen + len(raw), dtype=np.int64)
        numeric["_sampling_priority"] = rng.random(len(raw))
        rows_seen += len(raw)
        pool = numeric if reservoir is None else pd.concat([reservoir, numeric], ignore_index=True)
        if len(pool) > maximum:
            positions = np.argpartition(pool["_sampling_priority"].to_numpy(), maximum - 1)[:maximum]
            reservoir = pool.iloc[positions].copy().reset_index(drop=True)
        else:
            reservoir = pool.copy().reset_index(drop=True)
    if reservoir is None or reservoir.empty:
        raise ValueError(f"No data rows in {relative}.")
    content_hash = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during preparation: {relative}. Run again after downloads finish.")
    reservoir = reservoir.drop(columns="_sampling_priority").sort_values("source_row_index")
    reservoir["scenario_name"] = path.relative_to(root).parts[0]
    reservoir["label_source"] = "folder_scenario"
    reservoir["row_label_verified"] = False
    reservoir["source_relative_path"] = relative
    reservoir = reservoir[FEATURES + METADATA].reset_index(drop=True)
    audit = {
        "source_relative_path": relative,
        "size_bytes": before.st_size,
        "sha256": content_hash,
        "source_rows_scanned": rows_seen,
        "pilot_rows_selected": len(reservoir),
        "sample_seed": stable_seed(seed, relative),
        "missing_cells": int(missing.sum()),
        "infinite_cells": int(infinite.sum()),
        "non_numeric_cells_coerced": int(non_numeric.sum()),
        "rows_with_no_finite_features": no_finite_count,
        "rows_with_Number_below_10": partial_count,
        "column_quality": {
            name: {"missing": int(missing[i]), "infinite": int(infinite[i]),
                   "non_numeric": int(non_numeric[i])}
            for i, name in enumerate(FEATURES)
            if missing[i] or infinite[i] or non_numeric[i]
        },
    }
    return reservoir, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/CSV")
    parser.add_argument("--files-per-scenario", type=int, default=3)
    parser.add_argument("--rows-per-file", type=int, default=10000)
    parser.add_argument("--chunksize", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/scenario_pilot")
    args = parser.parse_args()
    for name in ("files_per_scenario", "rows_per_file", "chunksize"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset folder not found: {root}")
    output = Path(args.output).expanduser().resolve()
    if output == root or root in output.parents:
        parser.error("Choose an output directory outside the raw dataset folder.")
    selections, selected_paths = [], []
    for scenario in SCENARIOS:
        directory = root / scenario
        files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
        if not files:
            parser.error(f"No CSV files found in {directory}")
        rng = np.random.default_rng(stable_seed(args.seed, scenario))
        positions = sorted(rng.choice(len(files), min(args.files_per_scenario, len(files)), replace=False))
        chosen = [files[int(index)] for index in positions]
        selected_paths.extend(chosen)
        selections.append({
            "scenario_name": scenario, "available_files": len(files),
            "selected_files": [p.relative_to(root).as_posix() for p in chosen],
        })
    print("Preparing a technical scenario pilot. Row-level attack labels remain unverified.", flush=True)
    print(f"At most {len(selected_paths) * args.rows_per_file:,} sampled rows.", flush=True)
    frames, audits = [], []
    for index, path in enumerate(selected_paths, 1):
        print(f"[{index}/{len(selected_paths)}] {path.relative_to(root)}", flush=True)
        frame, audit = prepare_one(path, root, args.rows_per_file, args.chunksize, args.seed)
        frames.append(frame)
        audits.append(audit)
        print(f"  Scanned {audit['source_rows_scanned']:,}; selected {len(frame):,}.", flush=True)
    pilot = pd.concat(frames, ignore_index=True)
    counts = {str(k): int(v) for k, v in pilot["scenario_name"].value_counts().sort_index().items()}
    duplicates = int(pilot.duplicated(subset=FEATURES, keep=False).sum())
    manifest = {
        "program_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "seed": args.seed, "selection": selections,
        "files_per_scenario_limit": args.files_per_scenario, "chunksize": args.chunksize,
        "sampling": "Seeded random selection of up to N files per scenario; independent uniform priority sampling over all parsed rows in each selected file.",
        "maximum_rows_per_file": args.rows_per_file,
        "source_row_index_definition": "Zero-based parsed CSV data-record index; excludes the header and blank lines, not a timestamp or packet ID.",
        "features": FEATURES, "metadata_columns": METADATA,
        "model_input_rule": "Use ONLY the explicit features list. Never use metadata as model inputs.",
        "label_policy": {"source": "folder_scenario", "row_label_verified": False,
                         "interpretation": "Source scenario metadata, not verified per-window malicious/benign truth."},
        "processing": {"infinity": "Converted to missing values in the pilot copy; counts recorded.",
                       "non_numeric": "Coerced to missing values in the pilot copy; counts recorded.",
                       "imputation": "None. Fit any imputer on the later training partition only.",
                       "scaling": "None.", "rows_dropped_for_quality": 0,
                       "sampling_reductions": "Recorded per source file as rows scanned versus rows selected."},
        "split_status": "No training/calibration/test split created.",
        "provenance_limits": [
            "Source-file separation is not verified device/session independence.",
            "CSV row order is not established chronology.",
            "The supplied extractor matches the schema, but its exact use for every file is not independently proven.",
            "Device drift, per-window attack recall and genuine zero-day detection are not established by this preparation.",
        ],
        "pilot_rows": len(pilot), "rows_per_scenario": counts,
        "sample_rows_sharing_exact_feature_vectors": duplicates,
        "duplicate_handling": "Retained and counted; check overlap before any later model evaluation.",
        "source_file_audits": audits,
    }
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "scenario_pilot.csv"
    manifest_path = output / "pilot_manifest.json"
    pilot.to_csv(data_path, index=False)
    manifest["prepared_csv_sha256"] = sha256_file(data_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nPILOT PREPARATION COMPLETE")
    print("Rows:", f"{len(pilot):,}")
    print("Feature columns:", len(FEATURES))
    print("Metadata columns:", len(METADATA))
    for scenario, count in counts.items():
        print(f"  {scenario}: {count:,}")
    for key in ("missing_cells", "infinite_cells", "non_numeric_cells_coerced"):
        print(f"Source {key}: {sum(a[key] for a in audits):,}")
    print("Sample rows sharing feature vectors:", f"{duplicates:,}")
    print("Verified row-level labels: NO")
    print("Saved:", data_path)
    print("Saved:", manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise SystemExit(f"Preparation stopped: {error}")

