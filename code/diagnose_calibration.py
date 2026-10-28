#!/usr/bin/env python3
"""Check calibration sensitivity without fitting models or changing alpha.

Run from the project folder:
    .venv/Scripts/python.exe code/diagnose_calibration.py
Requires the original baseline summary, per-source report and three score CSVs.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

VERSION = "1.0"
META = ["scenario_name", "label_source", "row_label_verified",
        "source_relative_path", "source_row_index"]
SCENARIOS = ("Benign_Final", "Recon-PortScan", "DDoS-SYN_Flood", "Mirai-udpplain")
SCORE_COLUMNS = ["fold"] + META + ["role", "score_available", "anomaly_score", "flagged"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_checked(path, hashes, csv=False, usecols=None):
    before = sha256(path)
    value = (pd.read_csv(path, usecols=usecols, low_memory=False, float_precision="round_trip")
             if csv else json.loads(path.read_text(encoding="utf-8-sig")))
    require(before == sha256(path), f"Input changed while reading: {path.name}")
    hashes[path.name] = before
    return value


def booleans(series, name):
    text = series.astype(str).str.lower()
    require(text.isin(["true", "false", "1", "0"]).all(), f"Invalid boolean: {name}")
    return text.isin(["true", "1"]).to_numpy()


def integers(series, name):
    values = pd.to_numeric(series, errors="raise")
    require(np.isfinite(values).all() and (values >= 0).all()
            and (values % 1 == 0).all() and (values < 2**63).all(), f"Invalid {name}")
    return values.astype(np.int64)


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def close(actual, expected):
    return (actual is None and pd.isna(expected)) or (
        actual is not None and pd.notna(expected)
        and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12))


def threshold_from(scores, mask, alpha):
    ordered = np.sort(scores[mask])
    require(len(ordered) >= 2, "Fewer than two scored calibration rows.")
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    return (float(ordered[rank - 1]) if rank <= len(ordered) else math.inf), rank


def threshold_fields(value, prefix=""):
    return {prefix + "threshold": value if math.isfinite(value) else None,
            prefix + "threshold_is_infinite": not math.isfinite(value)}


def rate(frame, mask, flags):
    available = frame["score_available"].to_numpy()
    scored = mask & available
    total, count = int(mask.sum()), int(scored.sum())
    flagged = int((scored & flags).sum())
    return {"rows_total": total, "rows_scored": count, "rows_unscored": total - count,
            "flagged_rows": flagged, "flag_fraction_scored": fraction(flagged, count),
            "scoring_coverage": fraction(count, total),
            "review_fraction_including_unscored": fraction(flagged + total - count, total)}


def validate_scores(frame, record, summary, expected_metadata):
    fold = record["rotation"]
    require(len(frame) == summary["pilot_rows"], f"Rotation {fold}: pilot row count differs.")
    require(not frame[META + ["role"]].isna().any().any(), "Missing provenance or role.")
    require(integers(frame["fold"], "fold").eq(fold).all(), "Incorrect fold column.")
    frame["source_row_index"] = integers(frame["source_row_index"], "source row index")
    require(not frame.duplicated(["source_relative_path", "source_row_index"]).any(),
            "Repeated source-file/row IDs.")
    require(frame["label_source"].eq("folder_scenario").all(), "Unexpected label provenance.")
    verified = booleans(frame["row_label_verified"], "row_label_verified")
    require(not verified.any(), "Verified labels are outside this diagnostic's protocol.")
    frame["row_label_verified"] = verified
    require(set(frame["scenario_name"]) == set(SCENARIOS), "Unexpected/missing scenarios.")
    for path, group in frame.groupby("source_relative_path", sort=False):
        posix = PurePosixPath(str(path))
        require(not posix.is_absolute() and ".." not in posix.parts and "\\" not in str(path)
                and len(posix.parts) >= 2, "Invalid relative source path.")
        require(group["scenario_name"].nunique() == 1
                and group["scenario_name"].iloc[0] == posix.parts[0], "Source/scenario mismatch.")
    frame["score_available"] = booleans(frame["score_available"], "score_available")
    frame["flagged"] = booleans(frame["flagged"], "flagged")
    frame["anomaly_score"] = pd.to_numeric(frame["anomaly_score"], errors="raise")
    scores = frame["anomaly_score"].to_numpy(dtype=float)
    require(not np.isinf(scores).any(), "Infinite anomaly scores.")
    require(np.array_equal(np.isfinite(scores), frame["score_available"]),
            "Scores must be finite exactly for available rows and missing otherwise.")
    frame = frame.sort_values(["source_relative_path", "source_row_index"]).reset_index(drop=True)
    metadata = frame[META + ["score_available"]]
    if expected_metadata is not None:
        require(metadata.equals(expected_metadata), "Metadata IDs/availability differ across rotations.")
    paths = frame["source_relative_path"].to_numpy()
    benign_files = set(frame.loc[frame["scenario_name"].eq("Benign_Final"), "source_relative_path"])
    train_file, cal_file, test_file = (record["training_source_files"][0],
                                      record["calibration_source_file"],
                                      record["evaluation_benign_source_file"])
    require(benign_files == {train_file, cal_file, test_file} and len(benign_files) == 3,
            "Expected exactly three distinct benign files with separate roles.")
    expected_roles = np.where(paths == train_file, "training",
                              np.where(paths == cal_file, "calibration", "evaluation"))
    require(np.array_equal(frame["role"], expected_roles), "Stored roles disagree with summary.")
    available = frame["score_available"].to_numpy()
    scores = frame["anomaly_score"].to_numpy()
    threshold, rank = threshold_from(scores, (paths == cal_file) & available,
                                     summary["calibration_target_alpha"])
    stored = math.inf if record["threshold_infinite_due_to_calibration_size"] else record["threshold"]
    require((math.isinf(threshold) and stored == math.inf) or close(threshold, stored),
            "Original calibration threshold disagrees with saved scores.")
    require(rank == record["calibration_rank_one_based"], "Original calibration rank differs.")
    require(np.array_equal(frame["flagged"], available & (scores > stored)),
            "Stored flags disagree with saved threshold and scores.")
    require(int((paths == train_file).sum()) == record["training_rows"]
            and int(((paths == train_file) & available).sum()) == record["training_rows_used"]
            and int((paths == cal_file).sum()) == record["calibration_rows"]
            and int(((paths == cal_file) & available).sum()) == record["calibration_rows_scored"],
            "Original role counts disagree with summary.")
    observed = float(frame.loc[(paths == cal_file) & available, "flagged"].mean())
    require(close(observed, record["observed_calibration_flag_fraction"]),
            "Original calibration flag fraction differs.")
    return frame, metadata.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="results/baseline_pilot")
    parser.add_argument("--output", default="results/calibration_diagnostic")
    args = parser.parse_args()
    baseline, output = Path(args.baseline).resolve(), Path(args.output).resolve()
    require(output != baseline and baseline not in output.parents,
            "Keep diagnostic outputs outside the baseline folder.")
    hashes = {}
    summary = read_checked(baseline / "baseline_summary.json", hashes)
    alpha = summary["calibration_target_alpha"]
    require(isinstance(alpha, (int, float)) and 0 < alpha < 1, "Invalid calibration alpha.")
    records = summary["rotations"]
    require(len(records) == 3 and len({r["rotation"] for r in records}) == 3
            and all(len(r["training_source_files"]) == 1 for r in records)
            and len({r["training_source_files"][0] for r in records}) == 3,
            "Expected three rotations with three distinct single-file training sets.")
    original = read_checked(baseline / "flag_rates_by_source.csv", hashes, csv=True)
    require(not original.duplicated(["fold", "source_relative_path"]).any(),
            "Repeated per-source report keys.")
    frames, metadata, expected_rows = [], None, []
    for record in records:
        fold = record["rotation"]
        frame = read_checked(baseline / f"scores_rotation_{fold}.csv", hashes,
                             csv=True, usecols=SCORE_COLUMNS)
        frame, metadata = validate_scores(frame, record, summary, metadata)
        frames.append((record, frame))
        for source, group in frame[frame["role"].eq("evaluation")].groupby("source_relative_path"):
            expected_rows.append((fold, source))
            saved = original[(original["fold"] == fold) & (original["source_relative_path"] == source)]
            require(len(saved) == 1, "Missing original per-source report row.")
            actual = rate(group, np.ones(len(group), dtype=bool), group["flagged"].to_numpy())
            require(saved.iloc[0]["scenario_name"] == group["scenario_name"].iloc[0]
                    and all(close(value, saved.iloc[0][name]) for name, value in actual.items()),
                    "Original per-source counts/fractions disagree with score CSV.")
    require(set(expected_rows) == set(zip(original["fold"], original["source_relative_path"])),
            "Unexpected original per-source report rows.")
    rates, source_rates, quantiles, changes, assignments = [], [], [], [], []
    for record, frame in frames:
        fold = record["rotation"]
        paths, scenarios = frame["source_relative_path"].to_numpy(), frame["scenario_name"].to_numpy()
        scores, available = frame["anomaly_score"].to_numpy(), frame["score_available"].to_numpy()
        for source, group in frame.groupby("source_relative_path"):
            values = group.loc[group["score_available"], "anomaly_score"].to_numpy()
            row = {"model_rotation": fold, "source_relative_path": source,
                   "scenario_name": group["scenario_name"].iloc[0], "original_role": group["role"].iloc[0],
                   "rows_total": len(group), "rows_scored": len(values), "rows_unscored": len(group)-len(values)}
            names = ["score_min", "score_q50", "score_q90", "score_q95", "score_q99", "score_q999", "score_max"]
            row.update(zip(names, np.quantile(values, [0, .5, .9, .95, .99, .999, 1]).tolist()
                           if len(values) else [None] * len(names)))
            quantiles.append(row)
        paired = []
        for kind, cal_file, test_file in (
            ("original", record["calibration_source_file"], record["evaluation_benign_source_file"]),
            ("alternate", record["evaluation_benign_source_file"], record["calibration_source_file"])):
            threshold, rank = threshold_from(scores, (paths == cal_file) & available, alpha)
            flags = available & (scores > threshold)
            if kind == "original":
                require(np.array_equal(flags, frame["flagged"].to_numpy()),
                        "Recomputed original assignment does not reproduce the saved flags.")
            evaluation = (paths == test_file) | (scenarios != "Benign_Final")
            context = {"assignment": f"rotation_{fold}_{kind}", "model_rotation": fold,
                       "assignment_kind": kind, "training_source_file": record["training_source_files"][0],
                       "calibration_source_file": cal_file, "evaluation_benign_source_file": test_file,
                       **threshold_fields(threshold)}
            assignments.append({**context, "calibration_rank_one_based": rank,
                "calibration_rows_scored": int(((paths == cal_file) & available).sum()),
                "observed_calibration_flag_fraction": float(flags[(paths == cal_file) & available].mean())})
            by_scenario = {}
            for scenario in SCENARIOS:
                mask = evaluation & (scenarios == scenario)
                result = rate(frame, mask, flags)
                rates.append({**context, "scenario_name": scenario, **result})
                by_scenario[scenario] = (mask, result)
            for source in sorted(set(paths[evaluation])):
                mask = evaluation & (paths == source)
                source_rates.append({**context, "source_relative_path": source,
                    "scenario_name": scenarios[np.flatnonzero(mask)[0]], **rate(frame, mask, flags)})
            paired.append((context, by_scenario))
        for scenario in SCENARIOS[1:]:
            old, new = paired[0][1][scenario], paired[1][1][scenario]
            require(np.array_equal(old[0], new[0]), "Attack-scenario evaluation rows changed.")
            a, b = old[1], new[1]
            changes.append({"model_rotation": fold, "scenario_name": scenario,
                "rows_total": a["rows_total"], "rows_scored": a["rows_scored"],
                "original_flag_fraction": a["flag_fraction_scored"],
                "alternate_flag_fraction": b["flag_fraction_scored"],
                "change_percentage_points": None if a["flag_fraction_scored"] is None else
                    100 * (b["flag_fraction_scored"] - a["flag_fraction_scored"]),
                "evaluation_rows_identical": True})
    report = {"program_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "input_sha256": hashes, "baseline_pilot_sha256_recorded": summary["pilot_sha256"],
        "pilot_rows": summary["pilot_rows"], "calibration_alpha": alpha,
        "threshold_rule": "Sorted scored calibration rank ceil((n+1)*(1-alpha)); strict score > threshold.",
        "quantile_method": "NumPy linear interpolation; score quantiles are within each fixed model.",
        "fitted_models_reused": 3, "role_assignments": assignments, "paired_changes": changes,
        "limits": ["No models refitted; alpha unchanged; no best assignment selected.",
            "Six assignments are descriptive role swaps, not independent replications or new untouched tests.",
            "Folder scenarios are not verified per-window labels; fractions are not accuracy, recall or deployment FPR.",
            "Scores and thresholds from different models are not directly comparable.",
            "Stored secondary/overlap masks are not reused; only primary fractions are reported.",
            "Hashes record the score files inspected now. Baseline did not bind them cryptographically; saved models/pilot were not loaded or independently verified.",
            "CSV-file separation does not establish independent devices, sessions, parent captures or chronology."]}
    output.mkdir(parents=True, exist_ok=True)
    filenames = ["diagnostic_summary.json", "flag_rates_by_assignment.csv", "flag_rates_by_source.csv",
                 "score_quantiles_by_source.csv", "calibration_sensitivity.csv"]
    (output / filenames[0]).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for name, rows in zip(filenames[1:], [rates, source_rates, quantiles, changes]):
        pd.DataFrame(rows).to_csv(output / name, index=False)
    archive = output / "calibration_diagnostic_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in filenames:
            bundle.write(output / name, arcname=name)
    print("CALIBRATION DIAGNOSTIC COMPLETE")
    print("Paired changes: alternate minus original; percentage points; no refitting")
    for row in changes:
        delta = row["change_percentage_points"]
        print(f"  Model {row['model_rotation']} | {row['scenario_name']}: "
              + (f"{delta:+.2f} pp" if delta is not None else "no scored rows"))
    print("Reports ZIP:", archive)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"Diagnostic stopped: {error}")
