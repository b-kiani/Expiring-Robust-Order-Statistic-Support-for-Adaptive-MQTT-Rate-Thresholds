#!/usr/bin/env python3
"""Collect authenticated localhost MQTT development workloads; no ML or poisoning.

    .venv-mqtt/Scripts/python.exe code/collect_mqtt_workloads.py
Use --quick only to check the runner. Requires amqtt==0.12.1 and its dependencies.
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
import re
import secrets
import statistics
import sys
import tempfile
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from amqtt.broker import Broker
from amqtt.client import MQTTClient
from amqtt.plugins.base import BasePlugin
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

VERSION, HOST, BIN_NS = "1.0", "127.0.0.1", 200_000_000
PHASES = ("warmup", "train", "calibration", "baseline_eval", "online_eval")
FEATURES = ["message_count", "mean_payload_bytes", "std_payload_bytes"]
CASES = [(101, "stationary"), (101, "rate_shift"), (102, "rate_shift"), (102, "stationary")]
ACTIVE_RUN = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class BrokerAuditPlugin(BasePlugin):
    async def on_broker_message_received(self, *, client_id, message):
        observed_ns, observed_utc = time.perf_counter_ns(), utc_now()
        if ACTIVE_RUN is None or message is None:
            return
        try:
            session = self.context.get_session(client_id)
            if session is None or session.is_anonymous:
                raise RuntimeError("Broker audit has no authenticated session.")
            ACTIVE_RUN.observe("broker", message, observed_ns, observed_utc,
                               username=session.username, client_id=client_id)
        except Exception as error:
            ACTIVE_RUN.problem(f"Broker audit: {type(error).__name__}: {error}")


class WorkloadRun:
    def __init__(self, directory, seed, condition, durations, port, quick):
        self.directory, self.seed, self.condition = directory, seed, condition
        self.port, self.quick, self.run_id = port, quick, uuid.uuid4().hex
        self.users = [f"sensor{i:02d}" for i in range(1, 9)]
        self.client_ids = {user: "p" + uuid.uuid4().hex[:18] for user in self.users}
        self.client_ids["collector"] = "c" + uuid.uuid4().hex[:18]
        self.passwords = {user: secrets.token_hex(24) for user in self.client_ids}
        self.phases, cursor = [], 0
        for name, duration in zip(PHASES, durations):
            end = cursor + round(duration * 1e9)
            self.phases.append({"name": name, "start_ns": cursor, "end_ns": end})
            cursor = end
        self.span_ns, self.start_ns, self.stop_ns = cursor, None, None
        self.planned, self.attempted, self.skipped, self.acked = {}, set(), set(), set()
        self.observed = {"broker": {}, "collector": {}}
        self.counts = {"broker": Counter(), "collector": Counter()}
        self.issues, self.cleanup_errors, self.clients, self.jobs = [], [], [], []
        self.broker, self.broker_started, self.error = None, False, None
        self.credential_file_removed = False
        self.event_index, self.measurement_started = 0, False
        self.fatal = asyncio.Event()
        self.events = (directory / "events.jsonl").open("w", encoding="utf-8")
        self.make_plan()

    def safe(self, text):
        text = re.sub(r"mqtt://[^/@\s]+:[^/@\s]+@", "mqtt://[REDACTED]@", str(text))
        for password in self.passwords.values():
            text = text.replace(password, "[REDACTED]")
        return text

    def event(self, kind, *, observed_ns=None, observed_utc=None, **fields):
        record = {"event_index": self.event_index, "event": kind, "run_id": self.run_id,
            "utc": observed_utc or utc_now(), "monotonic_ns": observed_ns if observed_ns is not None else time.perf_counter_ns(),
            **fields}
        self.event_index += 1
        self.events.write(json.dumps(record, allow_nan=False) + "\n")
        self.events.flush()

    def problem(self, message):
        message = self.safe(message)
        self.issues.append(message)
        self.event("validation_problem", message=message)
        self.fatal.set()

    def make_plan(self):
        with (self.directory / "ground_truth.jsonl").open("w", encoding="utf-8") as stream:
            for user_index, user in enumerate(self.users):
                rng, sequence = random.Random(self.seed * 100 + user_index), 0
                topic = f"t/{self.run_id}/{uuid.uuid5(uuid.UUID(self.run_id), user).hex}"
                for phase in self.phases:
                    hz = 20 if self.condition == "rate_shift" and user_index < 2 and phase["name"] == "online_eval" else 10
                    interval_ns = 1_000_000_000 // hz
                    for slot in range((phase["end_ns"] - phase["start_ns"]) // interval_ns):
                        opaque_id = uuid.uuid5(uuid.UUID(self.run_id), f"{user}:{sequence}").hex
                        body = {"opaque_id": opaque_id, "counter": f"{sequence:08d}",
                            "reading": f"{rng.uniform(18, 28):09.6f}", "padding": "x" * rng.randint(16, 48)}
                        payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
                        record = {"correlation_id": opaque_id, "username": user, "client_id": self.client_ids[user],
                            "sequence": sequence, "phase": phase["name"], "scheduled_offset_ns": phase["start_ns"] + slot * interval_ns,
                            "interval_ns": interval_ns, "topic": topic, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                            "payload_bytes": len(payload), "payload": payload}
                        self.planned[opaque_id] = record
                        stream.write(json.dumps({"run_id": self.run_id, "planned_label": "normal_authorized_workload",
                            **{key: value for key, value in record.items() if key != "payload"}}) + "\n")
                        sequence += 1

    def protocol(self):
        return {"program_version": VERSION, "run_id": self.run_id, "seed": self.seed,
            "condition": self.condition, "quick_engineering_only": self.quick, "host": HOST, "port": self.port,
            "phases": self.phases, "window_width_ns": BIN_NS, "features": FEATURES,
            "publishers": self.users, "client_ids": self.client_ids, "qos": 1, "retain": False,
            "normal_rate_hz": 10, "authorized_online_rate_hz_first_two_publishers": 20 if self.condition == "rate_shift" else 10,
            "authentication": "FileAuthPlugin only; unique per-account random passwords; temporary Argon2 file removed after cleanup. No topic ACL is claimed.",
            "broker_observation": "MESSAGE_RECEIVED callback entry timestamps and context.get_session authenticated username; asynchronous callback observation, not wire-arrival time.",
            "scheduling": "Common monotonic epoch after all connections/subscription; skip slots more than one interval late; no catch-up loop.",
            "window_definition": "Fixed half-open 200ms bins per broker-authenticated username, unique valid application messages, empty bins retained; empty-bin size mean/std are 0 by definition.",
            "boundary_guard_definition": "The first and last 200ms bins of every phase are marked, retained and available for sensitivity analysis; no rows are silently dropped.",
            "schedule_quality_definition": "Evaluable only after the full measurement horizon and all scheduled slots are accounted for. Aggregate and every publisher skip fraction must be at most 1%; quality is separate from attempted-message transport integrity.",
            "future_poisoning_definition": "Future attacker controls a declared fraction of online update candidates, not initial trusted data or feedback. Poisoning requires logged admission to adaptive model or calibration state, a resulting state change, and evaluation of its effect on later traffic; ordinary attack traffic alone is insufficient.",
            "future_source_counts": "Vary legitimate changed-source counts and attacker-controlled-source counts; do not assume known source cardinalities distinguish drift from poisoning.",
            "reserved_evaluation_seeds": [1001, 1002, 1003, 1004, 1005],
            "limits": ["Development data only; no model, adaptive buffer or poisoning experiment is implemented.",
                "Eight emulated clients share one host/process; not physical IoT, packet capture or the CIC 39-feature schema.",
                "Time windows are correlated, not IID; 1% empirical calibration is not a distribution-free guarantee.",
                "Phase/identity/condition/labels are metadata, never the three model features.",
                "Collector publisher assignment is a planned-ledger join; only the broker callback supplies authenticated provenance.",
                "A missing/late schedule slot is logged, not replaced with synthetic traffic."]}

    def observe(self, source, message, observed_ns, observed_utc, username=None, client_id=None):
        payload = bytes(message.data)
        digest = hashlib.sha256(payload).hexdigest()
        try:
            opaque_id = json.loads(payload.decode("utf-8"))["opaque_id"]
            expected = self.planned.get(opaque_id)
            if not isinstance(opaque_id, str) or expected is None or opaque_id not in self.attempted:
                raise ValueError("Unexpected application ID.")
            self.counts[source][opaque_id] += 1
            if message.topic != expected["topic"] or digest != expected["payload_sha256"] or len(payload) != expected["payload_bytes"]:
                raise ValueError("Topic/payload hash/length mismatch.")
            if message.qos != 1 or message.retain:
                raise ValueError("Unexpected QoS or retained message.")
            if source == "broker" and (username != expected["username"] or client_id != expected["client_id"]):
                raise ValueError("Authenticated broker identity mismatch.")
            self.event(source + "_received", observed_ns=observed_ns, observed_utc=observed_utc,
                correlation_id=opaque_id, topic=message.topic, payload_sha256=digest, payload_bytes=len(payload),
                qos=message.qos, retain=bool(message.retain), authenticated_username=username,
                observed_client_id=client_id, planned_publisher_username=expected["username"],
                duplicate_application_id=self.counts[source][opaque_id] > 1)
            if opaque_id not in self.observed[source]:
                self.observed[source][opaque_id] = {"monotonic_ns": observed_ns, "utc": observed_utc,
                    "username": username, "payload_bytes": len(payload)}
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
            self.problem(f"{source} observation: {type(error).__name__}: {error}")

    async def connect(self, user):
        client = MQTTClient(client_id=self.client_ids[user], config={"plugins": {},
            "auto_reconnect": False, "connection_timeout": 5, "default_retain": False})
        self.clients.append(client)
        uri = f"mqtt://{user}:{self.passwords[user]}@{HOST}:{self.port}"
        result = await asyncio.wait_for(client.connect(uri, cleansession=True), 8)
        if result != 0:
            raise RuntimeError(f"Authenticated connection rejected: {user}")
        self.event("client_connected", username=user, client_id=self.client_ids[user])
        return client

    async def collect(self, collector):
        while True:
            message = await collector.deliver_message()
            if message is None:
                raise RuntimeError("Collector stopped unexpectedly.")
            self.observe("collector", message, time.perf_counter_ns(), utc_now())

    async def publish(self, user, client):
        for opaque_id, record in self.planned.items():
            if record["username"] != user:
                continue
            due_ns = self.start_ns + record["scheduled_offset_ns"]
            delay = (due_ns - time.perf_counter_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            now = time.perf_counter_ns()
            lateness = now - due_ns
            if lateness > record["interval_ns"] or now >= self.start_ns + self.span_ns:
                self.skipped.add(opaque_id)
                self.event("schedule_skipped", correlation_id=opaque_id, username=user, lateness_ns=lateness)
                continue
            self.attempted.add(opaque_id)
            self.event("publisher_attempt", correlation_id=opaque_id, username=user, client_id=self.client_ids[user],
                lateness_ns=lateness, scheduled_offset_ns=record["scheduled_offset_ns"])
            sent = await asyncio.wait_for(client.publish(record["topic"], record["payload"], qos=1, retain=False), 5)
            if sent.puback_packet is None or sent.packet_id is None or sent.puback_packet.packet_id != sent.packet_id:
                raise RuntimeError("Missing or mismatched publisher PUBACK.")
            self.acked.add(opaque_id)
            self.event("publisher_puback", correlation_id=opaque_id, username=user, packet_id=sent.packet_id)

    async def progress(self):
        for phase in self.phases:
            delay = (self.start_ns + phase["start_ns"] - time.perf_counter_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            print(f"  seed {self.seed} {self.condition}: {phase['name']}", flush=True)
            self.event("phase_clock", phase=phase["name"])

    async def workload(self, password_file):
        global ACTIVE_RUN
        ACTIVE_RUN = self
        self.broker = Broker({"listeners": {"default": {"type": "tcp", "bind": f"{HOST}:{self.port}",
            "max_connections": 12}}, "plugins": {
            "amqtt.plugins.authentication.FileAuthPlugin": {"password_file": str(password_file)},
            f"{__name__}.BrokerAuditPlugin": {}}})
        await asyncio.wait_for(self.broker.start(), 8)
        self.broker_started = True
        collector = await self.connect("collector")
        grants = await asyncio.wait_for(collector.subscribe([(f"t/{self.run_id}/+", 1)]), 5)
        if grants != [1]:
            raise RuntimeError("Collector QoS 1 subscription was not granted.")
        publishers = await asyncio.gather(*(self.connect(user) for user in self.users))
        self.start_ns = time.perf_counter_ns() + 200_000_000
        self.measurement_started = True
        self.event("measurement_epoch", planned_monotonic_ns=self.start_ns, span_ns=self.span_ns)
        consumer, fatal = asyncio.create_task(self.collect(collector)), asyncio.create_task(self.fatal.wait())
        senders = [asyncio.create_task(self.publish(user, client)) for user, client in zip(self.users, publishers)]
        progress = asyncio.create_task(self.progress())
        sending = asyncio.gather(*senders)
        self.jobs.extend([consumer, fatal, *senders, progress, sending])
        done, _ = await asyncio.wait([consumer, fatal, sending], timeout=self.span_ns / 1e9 + 3.2,
                                     return_when=asyncio.FIRST_COMPLETED)
        if fatal in done:
            raise RuntimeError("Observation validation failed; see events.")
        if consumer in done:
            await consumer
            raise RuntimeError("Collector stopped prematurely.")
        if sending not in done:
            raise TimeoutError("Publisher schedule exceeded measurement/drain deadline.")
        await sending
        remaining = (self.start_ns + self.span_ns - time.perf_counter_ns()) / 1e9 + 3
        done, _ = await asyncio.wait([consumer, fatal], timeout=max(0, remaining), return_when=asyncio.FIRST_COMPLETED)
        if fatal in done:
            raise RuntimeError("Observation validation failed during drain.")
        if consumer in done:
            await consumer
        self.stop_ns = time.perf_counter_ns()

    async def cleanup(self):
        for task in self.jobs:
            if not task.done():
                task.cancel()
        if self.jobs:
            try:
                await asyncio.wait_for(asyncio.gather(*self.jobs, return_exceptions=True), 3)
            except Exception as error:
                self.cleanup_errors.append(self.safe(f"Task cleanup: {error}"))
        async def close(client):
            try:
                await asyncio.wait_for(client.disconnect(), 3)
                await asyncio.wait_for(client.cancel_tasks(), 1)
            except Exception as error:
                self.cleanup_errors.append(self.safe(f"Client cleanup: {error}"))
        await asyncio.gather(*(close(client) for client in self.clients))
        if self.broker_started:
            try:
                await asyncio.wait_for(self.broker.shutdown(), 6)
            except Exception as error:
                self.cleanup_errors.append(self.safe(f"Broker cleanup: {error}"))

    async def run(self):
        global ACTIVE_RUN
        write_json(self.directory / "protocol.json", self.protocol())
        self.event("run_started", protocol_sha256=sha256_file(self.directory / "protocol.json"))
        try:
            with tempfile.TemporaryDirectory(prefix="mqtt_credentials_") as temporary:
                password_file = Path(temporary) / "passwords.txt"
                hasher = PasswordHash((Argon2Hasher(),))
                password_file.write_text("".join(f"{user}:{hasher.hash(password)}\n"
                    for user, password in self.passwords.items()), encoding="utf-8")
                try:
                    await asyncio.wait_for(self.workload(password_file), self.span_ns / 1e9 + 30)
                except (Exception, asyncio.CancelledError) as error:
                    self.error = self.safe(f"{type(error).__name__}: {error}")
                    self.event("run_error", message=self.error)
                finally:
                    if self.stop_ns is None:
                        self.stop_ns = time.perf_counter_ns()
                    try:
                        await asyncio.wait_for(self.cleanup(), 14)
                    except (Exception, asyncio.CancelledError) as error:
                        self.cleanup_errors.append(self.safe(f"Cleanup deadline: {type(error).__name__}: {error}"))
            self.credential_file_removed = True
        except Exception as error:
            self.error = self.error or self.safe(f"Credential/setup failure: {type(error).__name__}: {error}")
        finally:
            ACTIVE_RUN = None
            self.event("cleanup_complete", errors=self.cleanup_errors, credential_file_removed=self.credential_file_removed)
            self.events.close()
        return self.finish()

    def finish(self):
        attempted = self.attempted
        reconciliation = {"scheduled": len(self.planned), "attempted": len(attempted), "skipped": len(self.skipped),
            "acked": len(self.acked), "broker_unique": len(self.observed["broker"]),
            "collector_unique": len(self.observed["collector"]),
            "scheduled_accounted": set(self.planned) == attempted | self.skipped and not attempted & self.skipped,
            "missing_puback": sorted(attempted - self.acked),
            "missing_broker": sorted(attempted - set(self.observed["broker"])),
            "missing_collector": sorted(attempted - set(self.observed["collector"])),
            "duplicates": {source: sum(max(0, value - 1) for value in counts.values()) for source, counts in self.counts.items()}}
        complete_sets = all(attempted == set(self.observed[source]) for source in self.observed) and attempted == self.acked
        passed = bool(attempted) and complete_sets and reconciliation["scheduled_accounted"] and not self.issues and self.error is None and not self.cleanup_errors and self.credential_file_removed
        feature_rows, metadata_rows, bin_counts = self.windows()
        write_csv(self.directory / "windows_features.csv", ["row_id", *FEATURES], feature_rows)
        meta_columns = ["row_id", "run_id", "seed", "condition", "authenticated_username", "bin_index",
            "window_start_monotonic_ns", "window_end_monotonic_ns", "nominal_phase", "oracle_label",
            "boundary_guard", "mixed_planned_phases", "arrival_phase_mismatch", "complete_bin", "planned_phases_present"]
        write_csv(self.directory / "windows_metadata.csv", meta_columns, metadata_rows)
        ratio = len(self.skipped) / len(self.planned) if self.planned else 0
        outside = {"before_epoch": 0, "at_or_after_planned_end": 0}
        if self.start_ns is not None:
            for observation in self.observed["broker"].values():
                outside["before_epoch"] += observation["monotonic_ns"] < self.start_ns
                outside["at_or_after_planned_end"] += observation["monotonic_ns"] >= self.start_ns + self.span_ns
        quality_evaluable = (reconciliation["scheduled_accounted"] and self.measurement_started
            and self.start_ns is not None and self.stop_ns is not None
            and self.stop_ns >= self.start_ns + self.span_ns)
        per_source = {}
        for user in self.users:
            ids = {key for key, row in self.planned.items() if row["username"] == user}
            per_source[user] = {"scheduled": len(ids), "attempted": len(ids & attempted),
                "skipped": len(ids & self.skipped), "skipped_fraction": len(ids & self.skipped) / len(ids),
                "schedule_quality_passed": bool(quality_evaluable and len(ids & self.skipped) / len(ids) <= .01),
                "acked": len(ids & self.acked), "broker_unique": len(ids & set(self.observed["broker"])),
                "collector_unique": len(ids & set(self.observed["collector"]))}
        worst_source_ratio = max(row["skipped_fraction"] for row in per_source.values())
        quality_passed = bool(quality_evaluable and ratio <= .01 and worst_source_ratio <= .01)
        summary = {"program_version": VERSION, "run_id": self.run_id, "seed": self.seed, "condition": self.condition,
            "status": "passed" if passed else "failed", "quick_engineering_only": self.quick,
            "measurement_start_monotonic_ns": self.start_ns, "observation_stop_monotonic_ns": self.stop_ns,
            "reconciliation": reconciliation, "schedule_skipped_fraction": ratio,
            "schedule_quality_evaluable": bool(quality_evaluable),
            "schedule_quality_passed": quality_passed, "maximum_source_skipped_fraction": worst_source_ratio,
            "quality": ("schedule_not_evaluable_incomplete_collection" if not quality_evaluable else
                "schedule_within_1_percent_skip_limit" if quality_passed else "aggregate_or_source_schedule_skip_above_1_percent"),
            "window_rows": len(feature_rows), "window_counts_by_nominal_phase": bin_counts,
            "broker_observations_outside_feature_horizon": outside, "per_source_counts": per_source,
            "issues": self.issues, "original_error": self.error, "cleanup_errors": self.cleanup_errors,
            "credential_file_removed": self.credential_file_removed, "file_sha256": {name: sha256_file(self.directory / name) for name in
                ["events.jsonl", "ground_truth.jsonl", "protocol.json", "windows_features.csv", "windows_metadata.csv"]}}
        write_json(self.directory / "run_summary.json", summary)
        return summary

    def windows(self):
        if self.start_ns is None:
            return [], [], {}
        observed_span = max(0, min(self.span_ns, (self.stop_ns or self.start_ns) - self.start_ns))
        bins = math.ceil(observed_span / BIN_NS)
        contents = defaultdict(list)
        for opaque_id, observation in self.observed["broker"].items():
            offset = observation["monotonic_ns"] - self.start_ns
            if 0 <= offset < observed_span:
                contents[(observation["username"], offset // BIN_NS)].append((observation["payload_bytes"], self.planned[opaque_id]["phase"]))
        feature_rows, metadata_rows, counts = [], [], Counter()
        for index in range(bins):
            for user in self.users:
                begin, end = index * BIN_NS, min((index + 1) * BIN_NS, self.span_ns)
                phase = next(row for row in self.phases if row["start_ns"] <= begin < row["end_ns"])
                values = contents[(user, index)]
                sizes, present = [value[0] for value in values], sorted({value[1] for value in values})
                row_id = f"{self.run_id}:{user}:{index:05d}"
                feature_rows.append({"row_id": row_id, "message_count": len(sizes),
                    "mean_payload_bytes": statistics.mean(sizes) if sizes else 0.0,
                    "std_payload_bytes": statistics.pstdev(sizes) if sizes else 0.0})
                metadata_rows.append({"row_id": row_id, "run_id": self.run_id, "seed": self.seed,
                    "condition": self.condition, "authenticated_username": user, "bin_index": index,
                    "window_start_monotonic_ns": self.start_ns + begin, "window_end_monotonic_ns": self.start_ns + end,
                    "nominal_phase": phase["name"], "oracle_label": "normal_authorized_workload",
                    "boundary_guard": begin == phase["start_ns"] or end == phase["end_ns"],
                    "mixed_planned_phases": len(present) > 1, "arrival_phase_mismatch": any(name != phase["name"] for name in present),
                    "complete_bin": (index + 1) * BIN_NS <= observed_span, "planned_phases_present": "|".join(present)})
                counts[phase["name"]] += 1
        return feature_rows, metadata_rows, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/mqtt_workloads")
    parser.add_argument("--port", type=int, default=18883)
    parser.add_argument("--quick", action="store_true", help="Short engineering check; not research data.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in 1..65535.")
    logging.basicConfig(handlers=[logging.NullHandler()], level=logging.WARNING)
    logging.getLogger("amqtt").addHandler(logging.NullHandler())
    logging.getLogger("amqtt").propagate = False
    durations = (.2, .4, .6, .4, .6) if args.quick else (2, 10, 15, 5, 13)
    batch = Path(args.output).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12])
    batch.mkdir(parents=True, exist_ok=False)
    summaries = []
    print("Collecting authenticated MQTT development workloads; no model or poisoning test.", flush=True)
    for number, (seed, condition) in enumerate(CASES, 1):
        directory = batch / f"run_{number}_seed_{seed}_{condition}"
        directory.mkdir()
        print(f"[{number}/4] Seed {seed}: {condition}" + (" (QUICK engineering only)" if args.quick else ""), flush=True)
        async def execute():
            return await WorkloadRun(directory, seed, condition, durations, args.port, args.quick).run()
        summary = asyncio.run(execute())
        summaries.append(summary)
        quality = "passed" if summary["schedule_quality_passed"] else summary["quality"]
        print(f"  transport={summary['status']}, schedule_quality={quality}: attempts={summary['reconciliation']['attempted']}, skips={summary['reconciliation']['skipped']}, windows={summary['window_rows']}", flush=True)
        if summary["original_error"] and summary["original_error"].startswith("CancelledError"):
            break
    passed = len(summaries) == 4 and all(row["status"] == "passed" for row in summaries)
    write_json(batch / "batch_summary.json", {"program_version": VERSION, "created_utc": utc_now(),
        "status": "passed" if passed else "failed", "quick_engineering_only": args.quick,
        "schedule_quality_evaluable": len(summaries) == 4 and all(row["schedule_quality_evaluable"] for row in summaries),
        "schedule_quality_passed": len(summaries) == 4 and all(row["schedule_quality_passed"] for row in summaries),
        "planned_order": CASES, "reserved_evaluation_seeds": [1001, 1002, 1003, 1004, 1005],
        "script_sha256": sha256_file(Path(__file__)), "runs": summaries,
        "environment": {"python": sys.version, "platform": platform.platform(), "amqtt": importlib.metadata.version("amqtt")}})
    archive = batch / "mqtt_workload_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(batch.rglob("*")):
            if path.is_file() and path != archive:
                bundle.write(path, arcname=path.relative_to(batch).as_posix())
    print("MQTT WORKLOAD COLLECTION COMPLETE" if passed else "MQTT WORKLOAD COLLECTION FAILED")
    print("Development data only; inspect schedule quality and phase guards before analysis.")
    print("Reports ZIP:", archive)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyboardInterrupt) as error:
        raise SystemExit(f"Collection stopped: {type(error).__name__}: {error}")
