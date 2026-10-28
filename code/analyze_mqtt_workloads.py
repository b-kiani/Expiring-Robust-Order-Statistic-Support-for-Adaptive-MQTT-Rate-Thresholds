#!/usr/bin/env python3
"""Static diagnostic of authorized MQTT development workloads; no adaptation.

From the project folder:
    .venv/Scripts/python.exe code/analyze_mqtt_workloads.py
Or add --input PATH_TO_MQTT_REPORTS_ZIP_OR_BATCH_DIRECTORY.
Requires NumPy and scikit-learn in this Python environment; no MQTT dependency.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import statistics
import sys
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

try:
    import numpy as np
    import sklearn
    from sklearn.ensemble import IsolationForest
except ImportError as error:
    raise SystemExit("Use the ML environment with NumPy and scikit-learn installed. " + str(error))

VERSION, ALPHA, WIDTH = "1.0", .01, 200_000_000
FEATURES = ["message_count", "mean_payload_bytes", "std_payload_bytes"]
USERS = [f"sensor{i:02d}" for i in range(1, 9)]
CASES = [(101, "stationary"), (101, "rate_shift"), (102, "rate_shift"), (102, "stationary")]
PHASES = [{"name": name, "start_ns": begin * 1_000_000_000, "end_ns": end * 1_000_000_000}
          for name, begin, end in [("warmup", 0, 2), ("train", 2, 12), ("calibration", 12, 27),
                                   ("baseline_eval", 27, 32), ("online_eval", 32, 45)]]
FILES = ["events.jsonl", "ground_truth.jsonl", "protocol.json", "run_summary.json",
         "windows_features.csv", "windows_metadata.csv"]
META = ["row_id", "run_id", "seed", "condition", "authenticated_username", "bin_index",
        "window_start_monotonic_ns", "window_end_monotonic_ns", "nominal_phase", "oracle_label",
        "boundary_guard", "mixed_planned_phases", "arrival_phase_mismatch", "complete_bin", "planned_phases_present"]
IF_SETTINGS = {"n_estimators": 300, "max_samples": 256, "random_state": 42, "n_jobs": 1,
               "max_features": 1.0, "bootstrap": False, "contamination": "auto"}
LIMITS = ["Development data already inspected; this is not a blinded final evaluation.",
          "Only two paired seeds, eight emulated publishers and one host/process; not independent physical-device trials.",
          "All traffic is authorized normal workload. Alerts are false alarms here; attack detection is not measured.",
          "No adaptation, poisoning, review budget or proposed defense is implemented.",
          "Correlated time windows do not support an IID or distribution-free 1% false-alarm guarantee.",
          "The first two publishers change rate only in the shifted arm. Group membership is output metadata, never a model input.",
          "Mean/std payload bytes depend on messages per window even when the per-message size distribution stays unchanged.",
          "Isolation Forest cannot split on a training-constant feature and may ignore the count change.",
          "The guard sensitivity refits on separately filtered stationary data; variants are not independent replications."]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), "Cannot write an empty analysis table.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Source:
    """Read only the 25 named report files; never extract archive members."""
    def __init__(self, requested):
        self.requested, self.path = str(requested), Path(requested).resolve()
        self.rule, self.cache, self.archive = "explicit batch or ZIP", {}, None
        if self.path.is_dir() and not (self.path / "batch_summary.json").is_file():
            candidates = sorted(path for path in self.path.iterdir() if path.is_dir()
                                and re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{12}", path.name))
            require(bool(candidates), "No dated batch found. Give --input the reports ZIP or exact batch folder.")
            self.path, self.rule = candidates[-1], "lexicographically latest dated batch, regardless of pass/fail"
        prefixes = [f"run_{i}_seed_{seed}_{arm}" for i, (seed, arm) in enumerate(CASES, 1)]
        self.allowed = {"batch_summary.json"} | {f"{prefix}/{name}" for prefix in prefixes for name in FILES}
        require(self.path.exists(), f"Input not found: {self.path}")
        if self.path.is_file():
            require(zipfile.is_zipfile(self.path), "File input must be the MQTT reports ZIP.")
            self.archive = zipfile.ZipFile(self.path)
            names = self.archive.namelist()
            require(len(names) == len(set(names)) and set(names) == self.allowed, "ZIP must contain exactly the 25 expected report files.")
            for info in self.archive.infolist():
                name = PurePosixPath(info.filename)
                require(not name.is_absolute() and ".." not in name.parts and "\\" not in info.filename
                        and not (info.external_attr >> 16 & 0o170000) == 0o120000,
                        "Unsafe ZIP member.")
            require(sum(info.file_size for info in self.archive.infolist()) < 200_000_000, "Unexpectedly large reports ZIP.")
        print(f"Selected input: {self.path}\nSelection: {self.rule}", flush=True)

    def read(self, name):
        require(name in self.allowed, "Refusing to read an unlisted source file.")
        if name not in self.cache:
            path = (self.path / name).resolve()
            require(self.archive is not None or self.path in path.parents, "Source file resolves outside batch folder.")
            self.cache[name] = self.archive.read(name) if self.archive else path.read_bytes()
        return self.cache[name]

    def json(self, name):
        return json.loads(self.read(name).decode("utf-8-sig"))

    def table(self, name, columns):
        reader = csv.DictReader(io.StringIO(self.read(name).decode("utf-8-sig"), newline=""))
        require(reader.fieldnames == columns, f"Unexpected CSV schema: {name}")
        rows = list(reader)
        require(all(None not in row and None not in row.values() for row in rows), f"Malformed CSV: {name}")
        return rows


def validate_source(source):
    batch = source.json("batch_summary.json")
    require(batch.get("status") == "passed" and batch.get("quick_engineering_only") is False,
            "Require a fully passed default batch; quick or failed batches cannot be analyzed.")
    require(batch.get("planned_order") == [list(case) for case in CASES] and len(batch.get("runs", [])) == 4,
            "Unexpected paired-run order or count.")
    runs, identities, shared_protocol = {}, set(), None
    for index, (seed, arm) in enumerate(CASES, 1):
        prefix = f"run_{index}_seed_{seed}_{arm}"
        get = lambda name: source.json(f"{prefix}/{name}")
        summary, protocol = get("run_summary.json"), get("protocol.json")
        require(summary == batch["runs"][index - 1], f"Batch/run summary disagreement: {prefix}")
        require(summary.get("status") == "passed" and summary.get("quick_engineering_only") is False
                and summary.get("schedule_quality_evaluable") is True and summary.get("schedule_quality_passed") is True,
                f"Run or schedule quality did not pass: {prefix}")
        require(not summary["issues"] and not summary["cleanup_errors"] and summary["original_error"] is None
                and summary["credential_file_removed"] is True, f"Unresolved run errors: {prefix}")
        require(summary["schedule_skipped_fraction"] <= .01 and summary["maximum_source_skipped_fraction"] <= .01
                and all(row["skipped_fraction"] <= .01 for row in summary["per_source_counts"].values()), "Schedule skip limit exceeded.")
        run_id = summary["run_id"]
        require(run_id not in identities and re.fullmatch(r"[0-9a-f]{32}", run_id), "Duplicate/invalid run identity.")
        identities.add(run_id)
        require((summary["seed"], summary["condition"]) == (seed, arm)
                and (protocol["seed"], protocol["condition"], protocol["run_id"]) == (seed, arm, run_id), "Run identity mismatch.")
        require(protocol["program_version"] == summary["program_version"] == "1.0"
                and protocol["quick_engineering_only"] is False and protocol["phases"] == PHASES
                and protocol["features"] == FEATURES and protocol["publishers"] == USERS
                and protocol["window_width_ns"] == WIDTH and protocol["normal_rate_hz"] == 10
                and protocol["authorized_online_rate_hz_first_two_publishers"] == (20 if arm == "rate_shift" else 10)
                and protocol["qos"] == 1 and protocol["retain"] is False, "Unsupported collection protocol.")
        normalized = {key: value for key, value in protocol.items() if key not in
                      {"run_id", "seed", "condition", "client_ids", "authorized_online_rate_hz_first_two_publishers"}}
        if shared_protocol is None:
            shared_protocol = normalized
        require(normalized == shared_protocol, "Paired runs differ in collection settings beyond the planned rate intervention.")
        require(set(summary["file_sha256"]) == set(FILES) - {"run_summary.json"}, "Unexpected source hash manifest.")
        for name, expected in summary["file_sha256"].items():
            require(digest(source.read(f"{prefix}/{name}")) == expected, f"Source hash mismatch: {prefix}/{name}")
        truth = [json.loads(line) for line in source.read(f"{prefix}/ground_truth.jsonl").decode().splitlines()]
        gt = {row["correlation_id"]: row for row in truth}
        require(len(gt) == len(truth) and all(row["run_id"] == run_id and row["planned_label"] == "normal_authorized_workload"
                and row["username"] in USERS and row["client_id"] == protocol["client_ids"][row["username"]]
                and row["phase"] in [phase["name"] for phase in PHASES] for row in truth), "Invalid planned normal-workload ledger.")
        events = [json.loads(line) for line in source.read(f"{prefix}/events.jsonl").decode().splitlines()]
        require([row["event_index"] for row in events] == list(range(len(events)))
                and all(row["run_id"] == run_id for row in events), "Invalid event chronology/identity.")
        ids, broker = defaultdict(set), {}
        for row in events:
            if "correlation_id" not in row:
                continue
            ident, kind = row["correlation_id"], row["event"]
            require(ident in gt, "Event ID absent from planned ledger.")
            ids[kind].add(ident)
            if kind in ("broker_received", "collector_received"):
                planned = gt[ident]
                require(row["payload_sha256"] == planned["payload_sha256"] and row["payload_bytes"] == planned["payload_bytes"]
                        and row["topic"] == planned["topic"] and row["qos"] == 1 and row["retain"] is False, "Message content ledger mismatch.")
                if kind == "broker_received":
                    require(row["authenticated_username"] == planned["username"] and row["observed_client_id"] == planned["client_id"], "Broker authenticated identity mismatch.")
                    broker.setdefault(ident, row)
        attempted = ids["publisher_attempt"]
        require(attempted == ids["publisher_puback"] == ids["broker_received"] == ids["collector_received"]
                and not attempted & ids["schedule_skipped"] and attempted | ids["schedule_skipped"] == set(gt), "Raw message reconciliation failed.")
        reconciliation = summary["reconciliation"]
        require(reconciliation["scheduled_accounted"] and len(gt) == reconciliation["scheduled"]
                and len(attempted) == reconciliation["attempted"] and len(ids["schedule_skipped"]) == reconciliation["skipped"], "Summary/raw accounting mismatch.")
        feature_rows = source.table(f"{prefix}/windows_features.csv", ["row_id", *FEATURES])
        metadata = source.table(f"{prefix}/windows_metadata.csv", META)
        require(len(feature_rows) == len(metadata) == summary["window_rows"] == 1800, "Incomplete feature horizon.")
        epoch = summary["measurement_start_monotonic_ns"]
        require(summary["observation_stop_monotonic_ns"] >= epoch + PHASES[-1]["end_ns"], "Incomplete observation horizon.")
        buckets, rows = defaultdict(list), []
        for ident, row in broker.items():
            offset = row["monotonic_ns"] - epoch
            if 0 <= offset < PHASES[-1]["end_ns"]:
                buckets[(row["authenticated_username"], offset // WIDTH)].append((row["payload_bytes"], gt[ident]["phase"]))
        for position, (feature, meta) in enumerate(zip(feature_rows, metadata)):
            bin_index, user = position // 8, USERS[position % 8]
            begin, end = bin_index * WIDTH, (bin_index + 1) * WIDTH
            phase = next(row for row in PHASES if row["start_ns"] <= begin < row["end_ns"])
            expected_id = f"{run_id}:{user}:{bin_index:05d}"
            require(feature["row_id"] == meta["row_id"] == expected_id and meta["run_id"] == run_id
                    and meta["seed"] == str(seed) and meta["condition"] == arm and meta["authenticated_username"] == user
                    and meta["bin_index"] == str(bin_index) and meta["nominal_phase"] == phase["name"]
                    and meta["oracle_label"] == "normal_authorized_workload" and meta["complete_bin"] == "True"
                    and int(meta["window_start_monotonic_ns"]) == epoch + begin
                    and int(meta["window_end_monotonic_ns"]) == epoch + end, "Row join, chronology, phase or oracle mismatch.")
            values = buckets[(user, bin_index)]
            sizes, present = [row[0] for row in values], sorted({row[1] for row in values})
            guard = begin == phase["start_ns"] or end == phase["end_ns"]
            require(meta["boundary_guard"] == str(guard) and meta["mixed_planned_phases"] == str(len(present) > 1)
                    and meta["arrival_phase_mismatch"] == str(any(name != phase["name"] for name in present))
                    and meta["planned_phases_present"] == "|".join(present), "Phase guard metadata mismatch.")
            vector = [float(feature[name]) for name in FEATURES]
            expected = [len(sizes), statistics.mean(sizes) if sizes else 0., statistics.pstdev(sizes) if sizes else 0.]
            require(all(math.isfinite(value) and value >= 0 and math.isclose(value, target, abs_tol=1e-9)
                        for value, target in zip(vector, expected)) and vector[0].is_integer(), "Feature differs from authenticated broker observations.")
            rows.append({**meta, "guard": guard, "x": vector})
        runs[(seed, arm)] = rows
    return runs


def threshold(scores):
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    return (float(np.sort(scores)[rank - 1]) if rank <= len(scores) else math.inf), rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/mqtt_workloads")
    parser.add_argument("--output", default="results/mqtt_baseline_diagnostic")
    args = parser.parse_args()
    source = Source(args.input)
    try:
        runs = validate_source(source)
    finally:
        if source.archive:
            source.archive.close()
    output_root = Path(args.output).resolve()
    require(not (source.path.is_dir() and (output_root == source.path or source.path in output_root.parents)), "Keep output outside the source batch.")
    output = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"program_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Static engineering diagnostic on known authorized development data; no adaptation or attack experiment.",
        "input_requested": source.requested, "input_selected": str(source.path), "input_selection_rule": source.rule,
        "source_sha256": {name: digest(raw) for name, raw in sorted(source.cache.items())},
        "features": FEATURES, "alpha": ALPHA, "threshold_rule": "sorted calibration score at one-based ceil((n+1)*(1-alpha)); strict score > threshold; rank > n gives infinity (JSON null)",
        "models": {"count_upper_tail": {"score": "raw message_count"}, "isolation_forest": {**IF_SETTINGS, "input_dtype": "float32", "score": "negative score_samples"}},
        "variants": {"primary_all_windows": "All complete bins.", "phase_guard_sensitivity": "Exclude first/last bin of every phase; fit a separate stationary-only model."},
        "fitting": "For each seed and variant, fit on stationary train and calibrate on stationary calibration only; reuse frozen model and threshold for both paired arms.",
        "evaluation_phases": ["baseline_eval", "online_eval"], "groups": {"overall": USERS, "first_two_publishers": USERS[:2], "remaining_six_publishers": USERS[2:]},
        "metadata_policy": "Only the three allowlisted feature values enter Isolation Forest; count baseline uses the first feature. Identity/phase/group/labels are reporting metadata.",
        "limits": LIMITS, "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "script_sha256": digest(Path(__file__).read_bytes())}
    write_json(output / "analysis_protocol.json", protocol)  # Saved before any model fit.
    summaries, scored_rows, model_records = [], [], []
    for seed in [101, 102]:
        for variant in protocol["variants"]:
            filtered = lambda arm: [row for row in runs[(seed, arm)] if variant == "primary_all_windows" or not row["guard"]]
            stationary = filtered("stationary")
            training = np.asarray([row["x"] for row in stationary if row["nominal_phase"] == "train"], dtype=np.float32)
            calibration = np.asarray([row["x"] for row in stationary if row["nominal_phase"] == "calibration"], dtype=np.float32)
            require(len(training) >= 256 and len(calibration) > 0, "Insufficient stationary training/calibration rows.")
            forest = IsolationForest(**IF_SETTINGS).fit(training)
            split_counts = Counter(int(feature) for tree in forest.estimators_ for feature in tree.tree_.feature if feature >= 0)
            count_probe = calibration.copy()
            count_probe[:, 0] += 2
            probe_difference = float(np.max(np.abs(forest.score_samples(calibration) - forest.score_samples(count_probe))))
            for model in ["count_upper_tail", "isolation_forest"]:
                score = (lambda matrix: np.asarray(matrix[:, 0], dtype=np.float64)) if model == "count_upper_tail" else (lambda matrix: -forest.score_samples(np.asarray(matrix, dtype=np.float32)))
                cal_scores = score(calibration)
                cutoff, rank = threshold(cal_scores)
                finite_cutoff = cutoff if math.isfinite(cutoff) else None
                model_records.append({"seed": seed, "variant": variant, "model": model,
                    "training_run_id": stationary[0]["run_id"], "training_rows": len(training), "calibration_rows": len(calibration),
                    "model_input_features": ["message_count"] if model == "count_upper_tail" else FEATURES,
                    "training_constant_features": [FEATURES[i] for i in range(3) if np.all(training[:, i] == training[0, i])],
                    "forest_split_feature_counts": {name: split_counts[i] for i, name in enumerate(FEATURES)} if model == "isolation_forest" else None,
                    "count_only_calibration_probe_max_absolute_score_change": probe_difference if model == "isolation_forest" else None,
                    "calibration_rank_one_based": rank, "threshold": finite_cutoff, "threshold_is_infinite": not math.isfinite(cutoff),
                    "calibration_flagged": int(np.count_nonzero(cal_scores > cutoff)), "calibration_flag_fraction": float(np.mean(cal_scores > cutoff)),
                    "retained_rows_by_arm_phase": {arm: dict(Counter(row["nominal_phase"] for row in filtered(arm))) for arm in ["stationary", "rate_shift"]}})
                for arm in ["stationary", "rate_shift"]:
                    for phase in protocol["evaluation_phases"]:
                        rows = [row for row in filtered(arm) if row["nominal_phase"] == phase]
                        values = score(np.asarray([row["x"] for row in rows], dtype=np.float32))
                        flags = values > cutoff
                        common = {"seed": seed, "variant": variant, "model": model, "arm": arm, "phase": phase}
                        for group, users in protocol["groups"].items():
                            mask = np.asarray([row["authenticated_username"] in users for row in rows])
                            summaries.append({**common, "group": group, "rows": int(mask.sum()), "flagged": int(np.count_nonzero(flags[mask])),
                                "flag_fraction": float(flags[mask].mean()), "threshold": finite_cutoff, "threshold_is_infinite": not math.isfinite(cutoff)})
                        for row, value, flag in zip(rows, values, flags):
                            scored_rows.append({**common, "row_id": row["row_id"], "run_id": row["run_id"],
                                "authenticated_username": row["authenticated_username"], "bin_index": int(row["bin_index"]),
                                "source_group": "first_two_publishers" if row["authenticated_username"] in USERS[:2] else "remaining_six_publishers",
                                "score": float(value), "threshold": finite_cutoff, "threshold_is_infinite": not math.isfinite(cutoff), "flagged": bool(flag)})
            print(f"Finished seed {seed}, {variant}: train={len(training)}, calibration={len(calibration)}", flush=True)
    write_csv(output / "alarm_summary.csv", summaries)
    write_csv(output / "evaluation_scores.csv", scored_rows)
    write_json(output / "diagnostic_report.json", {"program_version": VERSION, "status": "completed", "models": model_records,
        "summary_rows": len(summaries), "scored_evaluation_rows": len(scored_rows), "limits": LIMITS,
        "file_sha256": {name: digest((output / name).read_bytes()) for name in ["analysis_protocol.json", "alarm_summary.csv", "evaluation_scores.csv"]}})
    names = ["analysis_protocol.json", "alarm_summary.csv", "evaluation_scores.csv", "diagnostic_report.json"]
    with zipfile.ZipFile(output / "mqtt_baseline_diagnostic_reports.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(output / name, arcname=name)
    print("Static development diagnostic complete. All traffic is authorized; no attack performance was measured.")
    print("Reports ZIP:", output / "mqtt_baseline_diagnostic_reports.zip")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Analysis stopped: {type(error).__name__}: {error}")
