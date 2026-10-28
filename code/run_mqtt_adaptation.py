#!/usr/bin/env python3
"""Collect six authenticated development streams and replay count calibration.

    .venv-mqtt/Scripts/python.exe code/run_mqtt_adaptation.py
Keep the original collect_mqtt_workloads.py beside this file. No ML packages needed.
Default duration is about eight minutes. --quick is an engineering check only.
"""
import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import random
import sys
import uuid
import zipfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

CORE_SHA = "40984ae9dec1872e633738cfbcdd478b85de6fb34052ac4a9b2732972dc2f918"
CORE_PATH = Path(__file__).resolve().with_name("collect_mqtt_workloads.py")
if not CORE_PATH.is_file() or hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA:
    raise SystemExit("Keep the unchanged collect_mqtt_workloads.py supplied with this script beside it (core hash mismatch).")
import collect_mqtt_workloads as core

VERSION, ALPHA = "2.0", .01
PHASES = ("warmup", "train", "calibration", "baseline_eval", "adaptation", "probe_eval")
CASES = [(201, "stationary"), (201, "authorized_shift"), (201, "compromised_shift"),
         (202, "compromised_shift"), (202, "authorized_shift"), (202, "stationary")]
USERS = [f"sensor{i:02d}" for i in range(1, 9)]
POLICIES = ("frozen_pooled", "pooled_fifo", "per_source_fifo")
RAW_NAMES = ["events.jsonl", "ground_truth.jsonl", "protocol.json", "run_summary.json",
             "windows_features.csv", "windows_metadata.csv", "windows_authorization.csv"]
AUTH_COLUMNS = ["row_id", "run_id", "seed", "condition", "authenticated_username", "bin_index",
                "nominal_phase", "scenario_authorized", "normal_message_count", "unauthorized_message_count",
                "oracle_label", "mixed_authorization", "complete_bin"]
LIMITS = ["Development mechanism test with two paired seeds; not a proposed defense or final blinded evaluation.",
          "Eight authenticated emulated accounts share one localhost process. Identity means broker-authenticated account, not a physical device.",
          "Unauthorized means an experimental workload-policy violation by a valid account; no credential theft, exploit or real malware is executed.",
          "Authorized and compromised rate shifts are intentionally observationally identical. Label-blind policies must behave identically on matched observations.",
          "Policy replay is causal offline replay of collected observations, not live detector latency or an intervention on the broker.",
          "Source-local calibration can isolate cross-source reference contamination but cannot protect a source whose own buffer is contaminated.",
          "Time windows are dependent; the 1% quantile setting is not an IID or distribution-free guarantee.",
          "The 125-row source cutoff uses its maximum at alpha=1%; a single high admission can move it. Pooled1000 uses rank991.",
          "FIFO admission is unconditional during adaptation, including alerted windows; all policies freeze during the probe.",
          "Labels are separate audit joins. There is no review feedback, label-aware admission, retraining or new defense.",
          "First-two ongoing workloads persist into the probe. Two additional accounts violate policy during every probe, up to four concurrent unauthorized accounts.",
          "Count score uses observed messages per 200ms window. Payload byte statistics remain collection outputs, not replay inputs."]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def authorized(condition, user, phase):
    return not ((phase == "probe_eval" and user in USERS[-2:]) or
                (condition == "compromised_shift" and user in USERS[:2] and phase in ("adaptation", "probe_eval")))


class AdaptationRun(core.WorkloadRun):
    def make_plan(self):
        with (self.directory / "ground_truth.jsonl").open("w", encoding="utf-8") as stream:
            for user_index, user in enumerate(self.users):
                sequence = 0
                topic = f"t/{self.run_id}/{uuid.uuid5(uuid.UUID(self.run_id), user).hex}"
                for phase_index, phase in enumerate(self.phases):
                    # Independent phase streams keep future payload-size sequences paired.
                    rng = random.Random(f"{self.seed}:{user_index}:{phase_index}")
                    shifted = (user_index < 2 and self.condition != "stationary" and phase["name"] in ("adaptation", "probe_eval"))
                    probe = user_index >= 6 and phase["name"] == "probe_eval"
                    hz, permitted = (20 if shifted or probe else 10), authorized(self.condition, user, phase["name"])
                    interval = 1_000_000_000 // hz
                    for slot in range((phase["end_ns"] - phase["start_ns"]) // interval):
                        opaque_id = uuid.uuid5(uuid.UUID(self.run_id), f"{user}:{sequence}").hex
                        body = {"opaque_id": opaque_id, "counter": f"{sequence:08d}",
                                "reading": f"{rng.uniform(18, 28):09.6f}", "padding": "x" * rng.randint(16, 48)}
                        payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                        record = {"correlation_id": opaque_id, "username": user, "client_id": self.client_ids[user],
                            "sequence": sequence, "phase": phase["name"], "scheduled_offset_ns": phase["start_ns"] + slot * interval,
                            "interval_ns": interval, "topic": topic, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                            "payload_bytes": len(payload), "payload": payload, "authorized": permitted}
                        self.planned[opaque_id] = record
                        stream.write(json.dumps({"run_id": self.run_id,
                            "planned_label": "normal_authorized_workload" if permitted else "unauthorized_policy_violation",
                            **{key: value for key, value in record.items() if key != "payload"}}) + "\n")
                        sequence += 1

    def protocol(self):
        result = super().protocol()
        for key in ["authorized_online_rate_hz_first_two_publishers", "future_poisoning_definition", "future_source_counts"]:
            result.pop(key, None)
        result.update({"experiment": "authenticated workload shift and unconditional calibration-buffer admission",
            "adaptation_rate_first_two_hz": 10 if self.condition == "stationary" else 20,
            "probe_rate_last_two_hz": 20, "ongoing_first_two_rate_persists_during_probe": True,
            "authorization": "Only experimental truth metadata distinguishes authorized and compromised first-two shifts; broker authentication stays valid. Last two are unauthorized during every probe.",
            "payload_rng": "Independent deterministic stream per seed, account and phase; no authorization, phase or condition in network payload/topic.",
            "policy_execution": "Causal offline count-only replay after all six collections pass; trusted calibration from each run itself; no future observations or oracle labels in policy input.",
            "candidate_admission": "All eight windows are scored before any admission; unconditional admission in adaptation only; no admission in baseline or probe.",
            "limits": LIMITS})
        return result

    def windows(self):
        features, metadata, counts = super().windows()
        actual = defaultdict(Counter)
        horizon = max(0, min(self.span_ns, (self.stop_ns or self.start_ns or 0) - (self.start_ns or 0)))
        if self.start_ns is not None:
            for ident, observation in self.observed["broker"].items():
                offset = observation["monotonic_ns"] - self.start_ns
                if 0 <= offset < horizon:
                    actual[(observation["username"], offset // core.BIN_NS)]["normal" if self.planned[ident]["authorized"] else "unauthorized"] += 1
        self.authorization_rows = []
        for feature, meta in zip(features, metadata):
            counts_here = actual[(meta["authenticated_username"], meta["bin_index"])]
            normal, unauthorized = counts_here["normal"], counts_here["unauthorized"]
            require(normal + unauthorized == feature["message_count"], "Authorization count/feature mismatch.")
            label = ("mixed_authorization_messages" if normal and unauthorized else "unauthorized_policy_violation" if unauthorized
                     else "normal_authorized_workload" if normal else "no_observed_messages")
            meta["oracle_label"] = label
            self.authorization_rows.append({key: meta[key] for key in AUTH_COLUMNS if key in meta} | {
                "scenario_authorized": authorized(self.condition, meta["authenticated_username"], meta["nominal_phase"]),
                "normal_message_count": normal, "unauthorized_message_count": unauthorized,
                "mixed_authorization": bool(normal and unauthorized)})
        return features, metadata, counts

    def finish(self):
        summary = super().finish()
        core.write_csv(self.directory / "windows_authorization.csv", AUTH_COLUMNS, self.authorization_rows)
        summary["file_sha256"]["windows_authorization.csv"] = core.sha256_file(self.directory / "windows_authorization.csv")
        summary["oracle_authorization_message_counts"] = {name: sum(row[name] for row in self.authorization_rows)
            for name in ["normal_message_count", "unauthorized_message_count"]}
        core.write_json(self.directory / "run_summary.json", summary)
        return summary


def cutoff(entries):
    scores = sorted(entry["score"] for entry in entries)
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    return (scores[rank - 1] if rank <= len(scores) else None), rank


def replay_policy(rows, policy):
    """Pure label-free replay. Input has exactly five allowlisted fields per row."""
    require(policy in POLICIES and bool(rows), "Unknown policy or empty replay.")
    require(all(set(row) == {"row_id", "source", "bin_index", "phase", "score"}
                and row["source"] in USERS and row["phase"] in PHASES and math.isfinite(row["score"])
                and row["score"] >= 0 for row in rows), "Replay accepts only count/source/clock/correlation fields.")
    require(len({row["row_id"] for row in rows}) == len(rows), "Duplicate replay row ID.")
    groups = [(index, list(group)) for index, group in groupby(rows, key=lambda row: row["bin_index"])]
    require([index for index, group in groups] == list(range(len(groups)))
            and all([row["source"] for row in group] == USERS and len({row["phase"] for row in group}) == 1 for index, group in groups),
            "Replay requires chronological complete eight-source bins.")
    phase_order = [PHASES.index(group[0]["phase"]) for index, group in groups]
    require(phase_order == sorted(phase_order)
            and all(PHASES.index(name) in phase_order for name in ("calibration", "baseline_eval", "adaptation", "probe_eval")),
            "Trusted calibration, baseline, adaptation and probe must occur in that chronological order.")
    calibration = [row for row in rows if row["phase"] == "calibration"]
    require(calibration and max(row["bin_index"] for row in calibration) < min(row["bin_index"] for row in rows if row["phase"] == "baseline_eval"), "Calibration is not strictly before evaluation.")
    entry = lambda row: {key: row[key] for key in ("row_id", "source", "score")}
    by_source = {user: [entry(row) for row in calibration if row["source"] == user] for user in USERS}
    require(len({len(values) for values in by_source.values()}) == 1 and len(by_source[USERS[0]]) > 0, "Unequal/empty source calibration.")
    buffers = ({user: deque(values, maxlen=len(values)) for user, values in by_source.items()} if policy == "per_source_fifo"
               else {"pooled": deque([entry(row) for row in calibration], maxlen=len(calibration))})
    initial = {key: {"rows": len(values), "rank_one_based": cutoff(values)[1], "threshold": cutoff(values)[0],
                "count_distribution": dict(sorted(Counter(value["score"] for value in values).items())),
                "minimum": min(value["score"] for value in values), "maximum": max(value["score"] for value in values)} for key, values in buffers.items()}
    decisions, traces, snapshots, first_moves = [], [], [], {}
    state = lambda: {key: list(values) for key, values in buffers.items()}
    thresholds = lambda: {key: cutoff(values)[0] for key, values in buffers.items()}
    def snapshot(stage, bin_index):
        current = state()
        snapshots.append({"stage": stage, "bin_index": bin_index, "thresholds": thresholds(), "state_sha256": digest(current), "buffers": current})
    selected = [(index, group) for index, group in groups if group[0]["phase"] in ("baseline_eval", "adaptation", "probe_eval")]
    for position, (bin_index, group) in enumerate(selected):
        phase = group[0]["phase"]
        if phase == "adaptation" and (position == 0 or selected[position - 1][1][0]["phase"] != phase):
            snapshot("adaptation_start", bin_index)
        before, before_hash = thresholds(), digest(state())
        admitted = phase == "adaptation" and policy != "frozen_pooled"
        # All decisions use the same pre-update state, regardless of source order.
        for row in group:
            threshold = before[row["source"] if policy == "per_source_fifo" else "pooled"]
            decisions.append({"policy": policy, **row, "threshold": threshold, "threshold_is_infinite": threshold is None,
                              "flagged": threshold is not None and row["score"] > threshold, "admitted": admitted})
        if admitted:
            for row in group:
                buffers[row["source"] if policy == "per_source_fifo" else "pooled"].append(entry(row))
        after, after_hash = thresholds(), digest(state())
        for key in before:
            if before[key] != after[key] and key not in first_moves:
                first_moves[key] = {"bin_index": bin_index, "threshold_before": before[key], "threshold_after": after[key]}
        traces.append({"policy": policy, "bin_index": bin_index, "phase": phase,
            "thresholds_before": canonical(before), "thresholds_after": canonical(after),
            "buffer_size": sum(len(values) for values in buffers.values()), "state_before_sha256": before_hash,
            "state_after_sha256": after_hash, "admitted_row_ids": canonical([row["row_id"] for row in group] if admitted else [])})
        if phase != "adaptation":
            require(before_hash == after_hash, "A frozen phase changed policy state.")
        if phase == "adaptation" and (position + 1 == len(selected) or selected[position + 1][1][0]["phase"] != phase):
            snapshot("adaptation_end", bin_index)
    snapshot("probe_end", selected[-1][0])
    require(snapshots[-1]["state_sha256"] == snapshots[-2]["state_sha256"], "Probe changed the final adaptation state.")
    return {"policy": policy, "initial_calibration": initial, "decisions": decisions, "traces": traces,
            "snapshots": snapshots, "first_threshold_moves": first_moves}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def analyze_run(directory, summary):
    for name, expected in summary["file_sha256"].items():
        require(core.sha256_file(directory / name) == expected, f"Source hash mismatch: {name}")
    features, metadata, authorization = [read_csv(directory / name) for name in
        ["windows_features.csv", "windows_metadata.csv", "windows_authorization.csv"]]
    require(len(features) == len(metadata) == len(authorization) == summary["window_rows"], "Window join length mismatch.")
    rows, audit = [], {}
    for feature, meta, truth in zip(features, metadata, authorization):
        require(feature["row_id"] == meta["row_id"] == truth["row_id"] and meta["complete_bin"] == "True", "Window join/full-bin mismatch.")
        rows.append({"row_id": feature["row_id"], "source": meta["authenticated_username"], "bin_index": int(meta["bin_index"]),
                     "phase": meta["nominal_phase"], "score": int(feature["message_count"])})
        audit[feature["row_id"]] = truth
    outputs = []
    for policy in POLICIES:
        result = replay_policy(rows, policy)
        # Audit joins occur only AFTER the pure policy has returned all decisions/states.
        result["admission_audit"] = [{"policy": policy, "row_id": row["row_id"], "source": row["source"],
            "bin_index": row["bin_index"], "oracle_label": audit[row["row_id"]]["oracle_label"],
            "normal_message_count": int(audit[row["row_id"]]["normal_message_count"]),
            "unauthorized_message_count": int(audit[row["row_id"]]["unauthorized_message_count"])} for row in result["decisions"] if row["admitted"]]
        result["snapshot_audit"] = []
        for snapshot in result["snapshots"]:
            for buffer_key, entries in snapshot["buffers"].items():
                for position, value in enumerate(entries):
                    truth = audit[value["row_id"]]
                    result["snapshot_audit"].append({"policy": policy, "stage": snapshot["stage"], "buffer_key": buffer_key,
                        "entry_position": position, **value, "oracle_label": truth["oracle_label"],
                        "normal_message_count": int(truth["normal_message_count"]), "unauthorized_message_count": int(truth["unauthorized_message_count"])})
        groups = {"overall": USERS, "first_two": USERS[:2], "middle_four": USERS[2:6], "probe_last_two": USERS[-2:]}
        result["alarm_summary"] = []
        for phase in ("baseline_eval", "adaptation", "probe_eval"):
            for group, users in groups.items():
                selected = [row for row in result["decisions"] if row["phase"] == phase and row["source"] in users]
                for category in ["all_windows", *sorted({audit[row["row_id"]]["oracle_label"] for row in selected})]:
                    subset = selected if category == "all_windows" else [row for row in selected if audit[row["row_id"]]["oracle_label"] == category]
                    flagged = sum(row["flagged"] for row in subset)
                    result["alarm_summary"].append({"policy": policy, "phase": phase, "source_group": group,
                        "observed_authorization": category, "rows": len(subset), "flagged": flagged, "flag_fraction": flagged / len(subset)})
        end_audit = [row for row in result["snapshot_audit"] if row["stage"] == "adaptation_end"]
        result["contamination_audit"] = {"admitted_windows": len(result["admission_audit"]),
            "unauthorized_admitted_windows": sum(row["unauthorized_message_count"] > 0 for row in result["admission_audit"]),
            "unauthorized_messages_in_admitted_windows": sum(row["unauthorized_message_count"] for row in result["admission_audit"]),
            "end_buffer_windows": len(end_audit), "end_unauthorized_windows": sum(row["unauthorized_message_count"] > 0 for row in end_audit),
            "end_unauthorized_window_fraction": sum(row["unauthorized_message_count"] > 0 for row in end_audit) / len(end_audit),
            "per_buffer_end": {key: {"windows": sum(row["buffer_key"] == key for row in end_audit),
                "unauthorized_windows": sum(row["buffer_key"] == key and row["unauthorized_message_count"] > 0 for row in end_audit)}
                for key in result["snapshots"][1]["buffers"]}}
        outputs.append(result)
    return outputs


def write_table(path, rows, columns):
    core.write_csv(path, columns if not rows else list(rows[0]), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/mqtt_adaptation")
    parser.add_argument("--port", type=int, default=18883)
    parser.add_argument("--quick", action="store_true", help="Short engineering collection; infinite small-sample cutoffs expected.")
    args = parser.parse_args()
    require(1 <= args.port <= 65535, "Port must be in 1..65535.")
    logging.basicConfig(handlers=[logging.NullHandler()], level=logging.WARNING)
    logging.getLogger("amqtt").addHandler(logging.NullHandler())
    logging.getLogger("amqtt").propagate = False
    core.PHASES, core.VERSION = PHASES, VERSION
    durations = (.2, .4, .8, .4, 1., .6) if args.quick else (2, 10, 25, 3, 20, 10)
    batch = Path(args.output).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12])
    batch.mkdir(parents=True, exist_ok=False)
    protocol = {"program_version": VERSION, "planned_order": CASES, "phases": dict(zip(PHASES, durations)),
        "quick_engineering_only": args.quick, "window_width_ns": core.BIN_NS, "policies": POLICIES, "alpha": ALPHA,
        "default_capacities": {"pooled": 1000, "per_source": 125, "sources": 8},
        "quantile_rule": "One-based ceil((n+1)*.99), strict score > threshold; rank>n is infinity represented as JSON null.",
        "fitting": "Raw count score needs no training fit. Each run uses only its own trusted pre-adaptation calibration; no pooling across runs or seeds.",
        "update_rule": "Score all eight rows with the pre-bin state, then unconditionally append all eight during adaptation only. Baseline/probe are frozen.",
        "policy_inputs": ["row_id", "source", "bin_index", "phase", "score"],
        "truth_policy_separation": "Authorization enters audit joins after replay only; source is a buffer key solely for per_source_fifo.",
        "identity_and_threat": "Valid first-two accounts exceed experimental policy from adaptation in compromised_shift; last-two accounts additionally violate policy in every probe. No topic ACL or exploit is claimed.",
        "state_digest_definition": "Canonical ordered buffer entries contain row_id/source/score only, never authorization labels.",
        "contamination_definition": "Unauthorized window entries / all buffer window entries. Actual unauthorized/normal message counts are separate audit fields.",
        "reserved_evaluation_seeds": [1001, 1002, 1003, 1004, 1005], "core_sha256": CORE_SHA,
        "script_sha256": core.sha256_file(Path(__file__)), "limits": LIMITS}
    core.write_json(batch / "experiment_protocol.json", protocol)
    summaries, collected, analysis_records, errors = [], [], [], []
    decisions, traces, admissions, snapshots, alarms = [], [], [], [], []
    print("Six authenticated development runs; roughly eight minutes. Analysis is causal offline replay.", flush=True)
    try:
        for index, (seed, condition) in enumerate(CASES, 1):
            directory = batch / f"run_{index}_seed_{seed}_{condition}"
            directory.mkdir()
            collected.append(directory)
            print(f"[{index}/6] seed {seed}: {condition}" + (" (QUICK engineering only)" if args.quick else ""), flush=True)
            async def execute():
                return await AdaptationRun(directory, seed, condition, durations, args.port, args.quick).run()
            summary = asyncio.run(execute())
            summaries.append(summary)
            print(f"  transport={summary['status']}; schedule_quality={summary['quality']}; messages={summary['reconciliation']['attempted']}", flush=True)
            if summary["original_error"] and summary["original_error"].startswith("CancelledError"):
                break
        passed = len(summaries) == 6 and all(row["status"] == "passed" and row["schedule_quality_passed"] for row in summaries)
        if passed:
            for directory, summary in zip(collected, summaries):
                print(f"Replaying policies: seed {summary['seed']} {summary['condition']}", flush=True)
                common = {"run_id": summary["run_id"], "seed": summary["seed"], "condition": summary["condition"]}
                for result in analyze_run(directory, summary):
                    for target, key in [(decisions, "decisions"), (traces, "traces"), (admissions, "admission_audit"),
                                        (snapshots, "snapshot_audit"), (alarms, "alarm_summary")]:
                        target.extend({**common, **row} for row in result[key])
                    analysis_records.append({**common, **{key: value for key, value in result.items()
                        if key not in ["decisions", "traces", "admission_audit", "snapshot_audit", "alarm_summary"]}})
        else:
            errors.append("At least one of six collections failed transport/schedule quality or collection was interrupted. ALL policy analysis skipped; no clean-run selection or retry.")
    except (Exception, KeyboardInterrupt) as error:
        errors.append(f"{type(error).__name__}: {error}")
    passed = len(summaries) == 6 and not errors
    report_names = ["experiment_protocol.json", "batch_summary.json", "policy_report.json",
                    "policy_decisions.csv", "policy_trace.csv", "admission_audit.csv", "buffer_snapshot_audit.csv", "alarm_summary.csv"]
    for filename, rows, columns in [("policy_decisions.csv", decisions, ["run_id", "policy", "row_id", "score", "threshold", "flagged", "admitted"]),
            ("policy_trace.csv", traces, ["run_id", "policy", "bin_index"]), ("admission_audit.csv", admissions, ["run_id", "policy", "row_id", "oracle_label"]),
            ("buffer_snapshot_audit.csv", snapshots, ["run_id", "policy", "stage", "row_id", "oracle_label"]),
            ("alarm_summary.csv", alarms, ["run_id", "policy", "phase", "rows", "flagged"])]:
        write_table(batch / filename, rows, columns)
    core.write_json(batch / "policy_report.json", {"program_version": VERSION, "analysis_status": "completed" if passed else "skipped_or_failed",
        "quick_engineering_only": args.quick, "errors": errors, "runs_and_policies": analysis_records, "limits": LIMITS})
    core.write_json(batch / "batch_summary.json", {"program_version": VERSION, "created_utc": core.utc_now(),
        "status": "passed" if passed else "failed", "quick_engineering_only": args.quick, "planned_order": CASES,
        "runs": summaries, "errors": errors, "core_sha256": CORE_SHA, "script_sha256": core.sha256_file(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "amqtt": importlib.metadata.version("amqtt")},
        "report_sha256": {name: core.sha256_file(batch / name) for name in report_names if name != "batch_summary.json"}})
    archive = batch / "mqtt_adaptation_reports.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in report_names:
            bundle.write(batch / name, arcname=name)
        for directory in collected:
            for name in RAW_NAMES:
                if (directory / name).is_file():
                    bundle.write(directory / name, arcname=f"{directory.name}/{name}")
    print("ADAPTATION DEVELOPMENT EXPERIMENT COMPLETE" if passed else "EXPERIMENT FAILED; inspect failure reports. No failed-run analysis is accepted.")
    print("No proposed defense or final attack-performance claim is established. Reports ZIP:", archive)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"Experiment stopped: {type(error).__name__}: {error}")
