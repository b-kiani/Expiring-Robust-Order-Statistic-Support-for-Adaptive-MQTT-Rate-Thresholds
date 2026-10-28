#!/usr/bin/env python3
"""Repeat fixed baseline settings across seeds, reusing the reference seed.

Run from the project folder:
    .venv/Scripts/python.exe code/check_seed_robustness.py
Keep train_baseline.py and diagnose_calibration.py beside this file.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0"
FEATURES = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min",
    "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance"]
META = ["scenario_name", "label_source", "row_label_verified", "source_relative_path", "source_row_index"]
ROLE_KEYS = ["rotation", "training_source_files", "calibration_source_file", "evaluation_benign_source_file",
             "training_rows", "training_rows_used", "calibration_rows", "calibration_rows_scored"]
ASSIGNMENT_KEYS = ["assignment", "model_rotation", "assignment_kind", "training_source_file",
    "calibration_source_file", "evaluation_benign_source_file", "calibration_rank_one_based", "calibration_rows_scored"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def roles(summary):
    return [{key: record[key] for key in ROLE_KEYS}
            for record in sorted(summary["rotations"], key=lambda row: row["rotation"])]


def assignments(summary):
    return [{key: record[key] for key in ASSIGNMENT_KEYS}
            for record in sorted(summary["role_assignments"], key=lambda row: row["assignment"])]


def settings(summary):
    return {key: value for key, value in summary["model_settings"].items() if key != "seed"}


def run_helper(script, arguments, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(script), *map(str, arguments)],
                                   stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f"  Still working; details: {log_path}", flush=True)
    if code:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
        raise ValueError(f"{script.name} failed (exit {code}). Log: {log_path}\n{tail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="data/processed/scenario_pilot")
    parser.add_argument("--baseline", default="results/baseline_pilot")
    parser.add_argument("--output", default="results/seed_robustness")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    require(len(args.seeds) >= 2 and len(set(args.seeds)) == len(args.seeds)
            and all(0 <= seed <= 2**32 - 1 for seed in args.seeds),
            "Provide at least two distinct seeds in 0..2**32-1.")
    pilot, baseline, output = (Path(value).resolve() for value in (args.pilot, args.baseline, args.output))
    manifest_path, pilot_path = pilot / "pilot_manifest.json", pilot / "scenario_pilot.csv"
    reference_path = baseline / "baseline_summary.json"
    manifest, reference = read_json(manifest_path), read_json(reference_path)
    reference_hash = sha256(reference_path)
    protected = [pilot, baseline, Path(manifest["dataset_root"]).resolve()]
    require(all(output != path and path not in output.parents for path in protected),
            "Keep output outside the pilot, reference baseline and raw dataset folders.")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Output must be a fresh or empty folder. Use --output with a new path; existing runs are not refitted.")
    pilot_hash, manifest_hash = sha256(pilot_path), sha256(manifest_path)
    require(pilot_hash == manifest["prepared_csv_sha256"] == reference["pilot_sha256"]
            and manifest_hash == reference["pilot_manifest_sha256"], "Pilot/manifest hashes differ from reference.")
    require(manifest["features"] == reference["features"] == FEATURES
            and manifest["metadata_columns"] == reference["metadata_excluded_from_model"] == META,
            "Expected the audited 39-feature/five-metadata schema.")
    require(pd.read_csv(pilot_path, nrows=0).columns.tolist() == FEATURES + META, "Pilot header differs.")
    metadata = pd.read_csv(pilot_path, usecols=META, low_memory=False)
    require(len(metadata) == manifest["pilot_rows"] == reference["pilot_rows"], "Pilot row counts differ.")
    require(metadata["label_source"].eq("folder_scenario").all()
            and metadata["row_label_verified"].astype(str).str.lower().isin(["false", "0"]).all(),
            "Unexpected pilot label policy.")
    benign_files = set(metadata.loc[metadata["scenario_name"].eq("Benign_Final"), "source_relative_path"])
    require(len(benign_files) == 3 and len(reference["rotations"]) == 3, "Exactly three benign files are required.")
    require(len({row["rotation"] for row in reference["rotations"]}) == 3
            and all(len(row["training_source_files"]) == 1
                    and {row["training_source_files"][0], row["calibration_source_file"],
                         row["evaluation_benign_source_file"]} == benign_files for row in reference["rotations"])
            and {row["training_source_files"][0] for row in reference["rotations"]} == benign_files,
            "Reference source assignments are incompatible with the pilot.")
    trees, reference_seed = reference["model_settings"]["trees"], reference["model_settings"]["seed"]
    require(isinstance(trees, int) and trees > 0 and isinstance(reference_seed, int)
            and 0 <= reference_seed <= 2**32 - 1
            and reference["model_settings"]["model"] == "IsolationForest"
            and reference["model_settings"]["scaling"] == "None"
            and reference["model_settings"]["imputation_fit"] == "Training only", "Unexpected reference settings.")
    helpers = {name: Path(__file__).resolve().with_name(name)
               for name in ["train_baseline.py", "diagnose_calibration.py"]}
    helper_hashes = {name: sha256(path) for name, path in helpers.items()}
    output.mkdir(parents=True, exist_ok=True)
    ref_case = output / f"seed_{reference_seed}"
    ref_case.mkdir()
    print(f"Validate existing seed {reference_seed}; no reference models will be refitted.", flush=True)
    run_helper(helpers["diagnose_calibration.py"], ["--baseline", baseline, "--output", ref_case / "calibration"],
               ref_case / "calibration.log")
    reference_diag = read_json(ref_case / "calibration" / "diagnostic_summary.json")
    all_rates, all_changes, all_sources, cases = [], [], [], []
    reference_counts, reference_source_counts = None, None
    for index, seed in enumerate(args.seeds, 1):
        case = output / f"seed_{seed}"
        reused = seed == reference_seed
        print(f"[{index}/{len(args.seeds)}] Seed {seed}: " + ("reuse verified scores" if reused else "fit three source rotations"), flush=True)
        current_baseline, diagnostic = (baseline if reused else case / "baseline"), case / "calibration"
        if not reused:
            case.mkdir()
            run_helper(helpers["train_baseline.py"], ["--pilot", pilot, "--output", current_baseline,
                "--seed", seed, "--trees", trees, "--alpha", reference["calibration_target_alpha"]], case / "baseline.log")
        current = read_json(current_baseline / "baseline_summary.json")
        require(current["pilot_sha256"] == pilot_hash and current["pilot_manifest_sha256"] == manifest_hash
                and current["features"] == FEATURES and current["metadata_excluded_from_model"] == META
                and current["pilot_rows"] == reference["pilot_rows"] and settings(current) == settings(reference)
                and current["model_settings"]["seed"] == seed and roles(current) == roles(reference)
                and current["calibration_target_alpha"] == reference["calibration_target_alpha"],
                f"Seed {seed}: changed inputs, settings or source assignments.")
        if not reused:
            run_helper(helpers["diagnose_calibration.py"], ["--baseline", current_baseline, "--output", diagnostic],
                       case / "calibration.log")
        diag = read_json(diagnostic / "diagnostic_summary.json")
        require(diag["pilot_rows"] == reference["pilot_rows"]
                and diag["baseline_pilot_sha256_recorded"] == pilot_hash
                and diag["calibration_alpha"] == reference["calibration_target_alpha"]
                and len(diag["role_assignments"]) == 6 and assignments(diag) == assignments(reference_diag),
                f"Seed {seed}: diagnostic role assignments changed.")
        rates = pd.read_csv(diagnostic / "flag_rates_by_assignment.csv", float_precision="round_trip")
        counts = rates[["assignment", "scenario_name", "rows_total", "rows_scored", "rows_unscored"]].sort_values(
            ["assignment", "scenario_name"]).reset_index(drop=True)
        require(len(rates) == 24 and not counts.duplicated(["assignment", "scenario_name"]).any(), "Unexpected assignment-rate rows.")
        require(reference_counts is None or counts.equals(reference_counts), "Evaluation rows/availability changed across seeds.")
        reference_counts = counts
        rates.insert(0, "seed", seed)
        all_rates.append(rates)
        sources = pd.read_csv(diagnostic / "flag_rates_by_source.csv", float_precision="round_trip")
        source_counts = sources[["assignment", "source_relative_path", "rows_total", "rows_scored", "rows_unscored"]].sort_values(
            ["assignment", "source_relative_path"]).reset_index(drop=True)
        require(not source_counts.duplicated(["assignment", "source_relative_path"]).any()
                and (reference_source_counts is None or source_counts.equals(reference_source_counts)),
                "Source rows/availability changed across seeds.")
        reference_source_counts = source_counts
        sources.insert(0, "seed", seed)
        all_sources.append(sources)
        changes = pd.read_csv(diagnostic / "calibration_sensitivity.csv", float_precision="round_trip")
        require(len(changes) == 9 and changes["evaluation_rows_identical"].astype(str).str.lower().eq("true").all(),
                "Paired scenario rows changed.")
        training = {row["rotation"]: row["training_source_files"][0] for row in current["rotations"]}
        changes.insert(0, "seed", seed)
        changes["training_source_file"] = changes["model_rotation"].map(training)
        all_changes.append(changes)
        cases.append({"seed": seed, "reference_scores_reused": reused,
            "baseline_summary_sha256": sha256(current_baseline / "baseline_summary.json"),
            "diagnostic_summary_sha256": sha256(diagnostic / "diagnostic_summary.json"),
            "diagnostic_input_sha256": diag["input_sha256"]})
    require(sha256(pilot_path) == pilot_hash and sha256(manifest_path) == manifest_hash
            and sha256(reference_path) == reference_hash == reference_diag["input_sha256"]["baseline_summary.json"]
            and all(sha256(path) == helper_hashes[name] for name, path in helpers.items()), "Inputs/helpers changed during the run.")
    rates, changes = pd.concat(all_rates, ignore_index=True), pd.concat(all_changes, ignore_index=True)
    group = ["training_source_file", "calibration_source_file", "evaluation_benign_source_file", "scenario_name"]
    role_summary = rates.groupby(group, dropna=False).agg(seed_count=("seed", "nunique"),
        scored_seed_count=("flag_fraction_scored", "count"), median_flag_fraction=("flag_fraction_scored", "median"),
        minimum_flag_fraction=("flag_fraction_scored", "min"), maximum_flag_fraction=("flag_fraction_scored", "max")).reset_index()
    effect_summary = changes.groupby(["training_source_file", "scenario_name"]).agg(seed_count=("seed", "nunique"),
        median_change_percentage_points=("change_percentage_points", "median"),
        minimum_change_percentage_points=("change_percentage_points", "min"),
        maximum_change_percentage_points=("change_percentage_points", "max")).reset_index()
    report = {"program_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "seeds": args.seeds, "reference_seed": reference_seed, "pilot_sha256": pilot_hash,
        "pilot_manifest_sha256": manifest_hash, "helper_sha256": helper_hashes, "script_sha256": sha256(Path(__file__)),
        "fixed_model_settings": settings(reference), "calibration_alpha": reference["calibration_target_alpha"],
        "new_fits": 3 * sum(seed != reference_seed for seed in args.seeds),
        "existing_fits_reused_for_analysis": 3 if reference_seed in args.seeds else 0,
        "total_analyzed_models": 3 * len(args.seeds), "role_assignments": 6 * len(args.seeds), "cases": cases,
        "label_policy": "Folder scenarios only; no verified per-window attack labels.",
        "limits": ["Seeds measure computational repeatability, not independent traffic samples.",
            "Role assignments reuse observations; they are not independent replications. No confidence intervals are computed.",
            "No best seed, model, threshold or assignment is selected; settings and pilot rows are unchanged.",
            "Fractions are not accuracy, attack recall, deployment FPR, or evidence of temporal drift or poisoning.",
            "CSV-file separation does not establish independent devices, sessions or parent captures.",
            "Reference models are not refitted. Reference score consistency is checked, but original score CSVs were not cryptographically bound to the saved models.",
            "Diagnostic secondary/overlap masks are not reused; only primary fractions are summarized."]}
    filenames = ["seed_assignment_rates.csv", "seed_calibration_sensitivity.csv", "seed_role_summary.csv",
                 "calibration_effect_summary.csv", "seed_source_rates.csv"]
    for name, frame in zip(filenames, [rates, changes, role_summary, effect_summary, pd.concat(all_sources, ignore_index=True)]):
        frame.to_csv(output / name, index=False)
    summary_name = "seed_robustness_summary.json"
    (output / summary_name).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    archive = output / "seed_robustness_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in [*filenames, summary_name]:
            bundle.write(output / name, arcname=name)
    print("SEED ROBUSTNESS CHECK COMPLETE")
    print(f"New fits: {report['new_fits']}; existing fits reused: {report['existing_fits_reused_for_analysis']}; assignments: {report['role_assignments']}")
    print("Reports ZIP:", archive)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Seed check stopped: {error}")
