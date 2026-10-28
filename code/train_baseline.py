#!/usr/bin/env python3
"""Train a CPU Isolation Forest technical baseline for the prepared scenario pilot.

Run from the project folder:
    .venv/Scripts/python.exe code/train_baseline.py

No verified row-level attack labels are assumed. Results are scenario flag
fractions, not accuracy, attack recall, or deployment false-positive rates.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP",
    "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]
METADATA = ["scenario_name", "label_source", "row_label_verified",
            "source_relative_path", "source_row_index"]
SCENARIOS = ("Benign_Final", "Recon-PortScan", "DDoS-SYN_Flood", "Mirai-udpplain")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_fingerprints(array):
    """SHA-256 of canonical float64 rows; normalize NaNs and signed zero."""
    canonical = np.array(array, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0] = 0.0
    canonical[np.isnan(canonical)] = np.nan
    return np.array([hashlib.sha256(row.tobytes()).digest()
                     for row in canonical], dtype="S32")


def membership(keys, reference_mask):
    reference = set(keys[reference_mask].tolist())
    return np.fromiter((key in reference for key in keys.tolist()),
                       dtype=bool, count=len(keys))


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def load_pilot(directory):
    csv_path = directory / "scenario_pilot.csv"
    manifest_path = directory / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("features") != FEATURES or manifest.get("metadata_columns") != METADATA:
        raise ValueError("Manifest schema differs from the audited 39-feature pilot.")
    if sha256_file(csv_path) != manifest.get("prepared_csv_sha256"):
        raise ValueError("Pilot CSV checksum does not match its manifest. Re-run preparation.")
    data = pd.read_csv(csv_path, low_memory=False)
    if data.columns.tolist() != FEATURES + METADATA:
        raise ValueError("Unexpected CSV columns/order.")
    if len(data) != manifest.get("pilot_rows"):
        raise ValueError("CSV row count differs from its manifest.")
    if data[METADATA].isna().any().any():
        raise ValueError("Missing provenance metadata.")
    if not data["label_source"].eq("folder_scenario").all():
        raise ValueError("Unexpected label provenance.")
    if not data["row_label_verified"].astype(str).str.lower().isin(["false", "0"]).all():
        raise ValueError("This program expects the unverified scenario pilot.")
    if not set(data["scenario_name"]).issubset(SCENARIOS):
        raise ValueError("Unexpected scenario names.")
    for path, group in data.groupby("source_relative_path", sort=True):
        parts = PurePosixPath(str(path)).parts
        if not parts or ".." in parts or PurePosixPath(str(path)).is_absolute():
            raise ValueError("Invalid relative source path.")
        if group["scenario_name"].nunique() != 1 or group["scenario_name"].iloc[0] != parts[0]:
            raise ValueError("Source path/scenario mismatch.")
    indices = pd.to_numeric(data["source_row_index"], errors="raise")
    if not np.isfinite(indices).all() or (indices < 0).any() or (indices % 1 != 0).any():
        raise ValueError("Invalid source row indices.")
    data["source_row_index"] = indices.astype(np.int64)
    if data.duplicated(["source_relative_path", "source_row_index"]).any():
        raise ValueError("Repeated source-file/row IDs in the prepared data.")
    actual = {str(k): int(v) for k, v in data["scenario_name"].value_counts().items()}
    if actual != manifest.get("rows_per_scenario"):
        raise ValueError("Scenario counts differ from the preparation record.")
    audits = manifest.get("source_file_audits", [])
    expected_files = {a["source_relative_path"]: a["pilot_rows_selected"] for a in audits}
    actual_files = data["source_relative_path"].value_counts().to_dict()
    if actual_files != expected_files:
        raise ValueError("Per-file row counts differ from the preparation record.")
    values = data[FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Infinite values remain in the pilot. Re-run preparation.")
    if np.nanmax(np.abs(values), initial=0) > np.finfo(np.float32).max:
        raise ValueError("A feature exceeds the estimator's float32 range; inspect it before fitting.")
    return data, values, manifest, sha256_file(manifest_path)


def rate_record(mask, scores, flags, valid, unseen, fold, scenario, source):
    scored = mask & valid
    novel = scored & unseen
    total = int(mask.sum())
    count = int(scored.sum())
    flagged = int((flags & scored).sum())
    unscored = total - count
    return {
        "fold": fold, "scenario_name": scenario, "source_relative_path": source,
        "rows_total": total, "rows_scored": count, "rows_unscored": unscored,
        "flagged_rows": flagged, "flag_fraction_scored": fraction(flagged, count),
        "scoring_coverage": fraction(count, total),
        "review_fraction_including_unscored": fraction(flagged + unscored, total),
        "unseen_vector_rows_scored": int(novel.sum()),
        "excluded_seen_vector_rows_scored": int((scored & ~unseen).sum()),
        "unseen_vector_flagged_rows": int((novel & flags).sum()),
        "unseen_vector_flag_fraction": fraction(int((novel & flags).sum()), int(novel.sum())),
    }


def save_plot(rates, alpha, destination):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    folds = sorted(rates["fold"].unique())
    markers = ["o", "s", "^", "D", "v"]
    for i, fold in enumerate(folds):
        group = rates[rates["fold"] == fold].set_index("scenario_name")
        y = [group.loc[name, "flag_fraction_scored"] if name in group.index else np.nan
             for name in SCENARIOS]
        offset = (i - (len(folds) - 1) / 2) * 0.10
        ax.scatter(np.arange(len(SCENARIOS)) + offset, y, marker=markers[i % len(markers)],
                   facecolors="white", edgecolors="black", s=48, label=f"Rotation {fold}")
    ax.axhline(alpha, color="0.55", linestyle="--", linewidth=1,
               label="Calibration target")
    ax.set_xticks(range(len(SCENARIOS)),
                  ["Benign\nheld-out file", "Port-scan\nscenario",
                   "SYN-flood\nscenario", "Mirai UDP\nscenario"])
    ax.set_ylim(-0.025, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Flag fraction among scored rows")
    ax.set_title("Technical pilot: scenario flag fractions")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.text(0.06, 0.025,
             "CSV-file rotations are descriptive. Scenario names are not verified per-window attack labels.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="data/processed/scenario_pilot")
    parser.add_argument("--output", default="results/baseline_pilot")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.trees <= 0:
        parser.error("Use 0 < alpha < 1 and a positive tree count.")
    pilot_dir = Path(args.pilot).resolve()
    output = Path(args.output).resolve()
    if output == pilot_dir or pilot_dir in output.parents:
        parser.error("Keep baseline outputs outside the prepared pilot folder.")
    data, values, manifest, manifest_hash = load_pilot(pilot_dir)
    raw_root = Path(manifest["dataset_root"]).resolve()
    if output == raw_root or raw_root in output.parents:
        parser.error("Keep baseline outputs outside the raw dataset folder.")
    paths = data["source_relative_path"].to_numpy(dtype=str)
    scenarios = data["scenario_name"].to_numpy(dtype=str)
    benign = scenarios == "Benign_Final"
    benign_files = sorted(set(paths[benign]))
    if len(benign_files) < 3:
        raise ValueError("At least three benign source files are needed for separate roles.")
    valid = np.isfinite(values).any(axis=1)
    raw_keys = vector_fingerprints(values)
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)
    all_source_rates, all_scenario_rates, overlaps, fold_records = [], [], [], []
    missing_by_column = {name: int(np.isnan(values[:, i]).sum())
                         for i, name in enumerate(FEATURES) if np.isnan(values[:, i]).any()}
    print("Training a technical, CSV-file-separated anomaly baseline.", flush=True)
    print("Pilot rows:", f"{len(data):,}", "| Pilot missing cells:", int(np.isnan(values).sum()), flush=True)

    for fold_index, test_file in enumerate(benign_files):
        fold = fold_index + 1
        cal_file = benign_files[(fold_index + 1) % len(benign_files)]
        train_files = [p for p in benign_files if p not in (test_file, cal_file)]
        train = np.isin(paths, train_files)
        calibration = paths == cal_file
        evaluation = (paths == test_file) | ~benign
        train_valid = train & valid
        cal_valid = calibration & valid
        if train_valid.sum() < 32 or cal_valid.sum() < 2:
            raise ValueError("Insufficient usable training or calibration rows.")
        print(f"[{fold}/{len(benign_files)}] Fit on {int(train_valid.sum()):,} benign-scenario rows.", flush=True)
        empty_features = [FEATURES[i] for i in range(len(FEATURES))
                          if np.isnan(values[train_valid, i]).all()]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=False)
        imputer.fit(values[train_valid])
        transformed = imputer.transform(values)
        if transformed.shape != values.shape or not np.isfinite(transformed).all():
            raise ValueError("Imputation did not produce the expected finite 39-feature matrix.")
        model_inputs = transformed.astype(np.float32)
        model = IsolationForest(n_estimators=args.trees, max_samples=min(256, int(train_valid.sum())),
                                max_features=1.0, contamination="auto", bootstrap=False,
                                random_state=args.seed, n_jobs=-1)
        model.fit(model_inputs[train_valid])
        scores = np.full(len(data), np.nan)
        scores[valid] = -model.score_samples(model_inputs[valid])
        cal_scores = np.sort(scores[cal_valid])
        rank = math.ceil((len(cal_scores) + 1) * (1 - args.alpha))
        threshold = float(cal_scores[rank - 1]) if rank <= len(cal_scores) else math.inf
        flags = valid & (scores > threshold)

        model_keys = vector_fingerprints(model_inputs)
        raw_train_seen, raw_cal_seen = membership(raw_keys, train), membership(raw_keys, calibration)
        model_train_seen = membership(model_keys, train_valid)
        model_cal_seen = membership(model_keys, cal_valid)
        unseen = ~(raw_train_seen | raw_cal_seen | model_train_seen | model_cal_seen)
        for representation, keys in (("raw_39_features", raw_keys),
                                     ("imputed_float32_estimator_inputs", model_keys)):
            for label, target, reference in (
                ("calibration_seen_in_training", calibration, train_valid),
                ("evaluation_seen_in_training", evaluation, train_valid),
                ("evaluation_seen_in_calibration", evaluation, cal_valid),
            ):
                seen = membership(keys, reference)
                overlaps.append({"fold": fold, "representation": representation, "comparison": label,
                                 "target_rows": int(target.sum()),
                                 "overlapping_rows": int((target & seen).sum()),
                                 "overlap_fraction": fraction(int((target & seen).sum()), int(target.sum()))})
        for scenario in SCENARIOS:
            mask = evaluation & (scenarios == scenario)
            all_scenario_rates.append(rate_record(mask, scores, flags, valid, unseen,
                                                   fold, scenario, "ALL_EVALUATION_FILES"))
        for source in sorted(set(paths[evaluation])):
            mask = evaluation & (paths == source)
            all_source_rates.append(rate_record(mask, scores, flags, valid, unseen,
                                                fold, scenarios[np.flatnonzero(mask)[0]], source))
        roles = np.full(len(data), "evaluation", dtype=object)
        roles[train] = "training"
        roles[calibration] = "calibration"
        scored_data = data[METADATA].copy()
        scored_data.insert(0, "fold", fold)
        scored_data["role"] = roles
        scored_data["score_available"] = valid
        scored_data["anomaly_score"] = scores
        scored_data["flagged"] = flags
        scored_data["raw_vector_seen_in_training"] = raw_train_seen
        scored_data["raw_vector_seen_in_calibration"] = raw_cal_seen
        scored_data["model_input_seen_in_training"] = model_train_seen
        scored_data["model_input_seen_in_calibration"] = model_cal_seen
        scored_data["secondary_unseen_evaluation_subset"] = evaluation & valid & unseen
        scored_data.to_csv(output / f"scores_rotation_{fold}.csv", index=False)
        model_path = models_dir / f"isolation_forest_rotation_{fold}.joblib"
        joblib.dump({"features": FEATURES, "imputer": imputer, "model": model,
                     "estimator_input_dtype": "float32",
                     "threshold": threshold, "alpha": args.alpha,
                     "training_source_files": train_files, "calibration_source_file": cal_file,
                     "pilot_sha256": manifest["prepared_csv_sha256"]}, model_path, compress=3)
        record = {
            "rotation": fold, "training_source_files": train_files,
            "calibration_source_file": cal_file, "evaluation_benign_source_file": test_file,
            "training_rows": int(train.sum()), "training_rows_used": int(train_valid.sum()),
            "calibration_rows": int(calibration.sum()), "calibration_rows_scored": int(cal_valid.sum()),
            "entirely_missing_training_features": empty_features,
            "empty_feature_policy": "Retain width; training-empty feature uses fixed zero imputation.",
            "threshold": threshold if math.isfinite(threshold) else None,
            "threshold_infinite_due_to_calibration_size": not math.isfinite(threshold),
            "calibration_rank_one_based": rank,
            "observed_calibration_flag_fraction": float(flags[cal_valid].mean()),
            "calibration_scores_equal_to_threshold": int((cal_scores == threshold).sum()),
            "model_sha256": sha256_file(model_path),
        }
        fold_records.append(record)
        print(f"  Calibration flag fraction: {record['observed_calibration_flag_fraction']:.2%}", flush=True)

    rates = pd.DataFrame(all_scenario_rates)
    pd.DataFrame(all_source_rates).to_csv(output / "flag_rates_by_source.csv", index=False)
    rates.to_csv(output / "flag_rates_by_scenario.csv", index=False)
    pd.DataFrame(overlaps).to_csv(output / "feature_overlap.csv", index=False)
    aggregate = []
    for scenario in SCENARIOS:
        part = rates[rates["scenario_name"] == scenario]["flag_fraction_scored"].dropna()
        aggregate.append({"scenario_name": scenario, "rotation_count": int(len(part)),
                          "mean_flag_fraction": float(part.mean()) if len(part) else None,
                          "minimum_flag_fraction": float(part.min()) if len(part) else None,
                          "maximum_flag_fraction": float(part.max()) if len(part) else None})
    summary = {
        "program_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_sha256": manifest["prepared_csv_sha256"], "pilot_manifest_sha256": manifest_hash,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "pilot_rows": len(data), "pilot_missing_cells": int(np.isnan(values).sum()),
        "pilot_missing_by_column": missing_by_column,
        "pilot_rows_with_no_finite_features": int((~valid).sum()),
        "pilot_rows_sharing_raw_feature_vectors": int(data.duplicated(FEATURES, keep=False).sum()),
        "features": FEATURES, "metadata_excluded_from_model": METADATA,
        "model_settings": {"model": "IsolationForest", "trees": args.trees, "max_samples": "min(256, training_rows)",
                           "seed": args.seed, "scaling": "None", "imputation_fit": "Training only"},
        "threshold_rule": "Higher score = -score_samples. Sorted calibration rank ceil((n+1)*(1-alpha)); strict score > threshold.",
        "calibration_target_alpha": args.alpha, "rotations": fold_records,
        "primary_analysis": "All scored evaluation rows; naturally repeated vectors retained.",
        "unscored_policy": "Rows with no finite raw feature are unscored and separately counted as review cases.",
        "fingerprint_rule": "Raw inputs use canonical float64; imputed estimator inputs are rounded to float32 then encoded as float64 for SHA-256. Normalize NaNs and signed zero.",
        "secondary_analysis": "Same fitted imputer/model/threshold; evaluate only rows whose raw float64 and imputed float32 estimator-input fingerprints were absent from training and calibration. This selects a changed population, not an unbiased correction.",
        "scenario_flag_fraction_summary": aggregate,
        "limitations": [
            "Labels are folder-derived scenarios, not verified per-window attack labels.",
            "Results are flag fractions, not accuracy, attack recall, deployment FPR, or zero-day detection.",
            "CSV-file separation does not establish independent devices, sessions, parent captures, or chronology.",
            "Rotations reuse attack-scenario rows; they are descriptive role swaps, not independent replications.",
            "The calibration target is not a guarantee under dependence or distribution shift.",
            "Duplicate vectors can reflect natural repeated traffic; overlap alone does not prove leakage.",
            "The pilot samples equally per selected file, not uniformly over the whole released dataset.",
        ],
    }
    (output / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save_plot(rates, args.alpha, output / "scenario_flag_rates.png")
    print("\nBASELINE PILOT COMPLETE")
    print("Scenario flag fractions: mean [minimum, maximum] across CSV-file rotations")
    for row in aggregate:
        if row["mean_flag_fraction"] is None:
            print(f"  {row['scenario_name']}: no scorable rows")
        else:
            print(f"  {row['scenario_name']}: {row['mean_flag_fraction']:.2%} "
                  f"[{row['minimum_flag_fraction']:.2%}, {row['maximum_flag_fraction']:.2%}]")
    print("Pilot missing cells:", summary["pilot_missing_cells"])
    print("Verified row-level attack labels: NO")
    print("Full report:", output / "baseline_summary.json")
    print("Per-source results:", output / "flag_rates_by_source.csv")
    print("Overlap diagnostics:", output / "feature_overlap.csv")
    print("Plot:", output / "scenario_flag_rates.png")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        raise SystemExit(f"Baseline stopped: {error}")

