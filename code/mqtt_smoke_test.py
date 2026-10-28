#!/usr/bin/env python3
"""Check local MQTT QoS 1 acknowledgements and collector delivery; no ML.

    .venv-mqtt/Scripts/python.exe code/mqtt_smoke_test.py
Requires amqtt==0.12.1. The broker binds only to 127.0.0.1.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import sys
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.0"
HOST = "127.0.0.1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def reconcile(expected_ids, attempted, acknowledged, valid_received, counts, issues):
    """Reconcile application correlation IDs, never MQTT packet IDs across sessions."""
    expected_ids = set(expected_ids)
    missing_ack = sorted(expected_ids - acknowledged)
    missing_delivery = sorted(expected_ids - valid_received)
    unexpected = sorted(set(counts) - expected_ids)
    return {"expected": len(expected_ids), "publish_attempts": len(attempted),
        "puback_confirmed": len(acknowledged), "unique_valid_deliveries": len(valid_received),
        "collector_messages_with_parseable_ids": sum(counts.values()),
        "duplicate_deliveries": sum(max(0, count - 1) for count in counts.values()),
        "missing_puback_ids": missing_ack, "missing_delivery_ids": missing_delivery,
        "unexpected_ids": unexpected, "validation_issues": issues,
        "passed": attempted == expected_ids and acknowledged == expected_ids
            and valid_received == expected_ids and not unexpected and not issues}


class SmokeRun:
    def __init__(self, protocol, planned, stream):
        self.protocol, self.planned, self.stream = protocol, planned, stream
        self.attempted, self.acknowledged, self.received = set(), set(), set()
        self.counts, self.issues, self.cleanup_errors = Counter(), [], []
        self.clients, self.jobs, self.broker = [], [], None
        self.broker_started, self.clean_shutdown, self.error = False, False, None
        self.event_index = 0
        self.complete = asyncio.Event()

    def event(self, kind, **fields):
        record = {"event_index": self.event_index, "utc": utc_now(),
            "monotonic_ns": time.perf_counter_ns(), "run_id": self.protocol["run_id"],
            "event": kind, **fields}
        self.event_index += 1
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()

    async def connect(self, client_id):
        from amqtt.client import MQTTClient
        client = MQTTClient(client_id=client_id, config={"auto_reconnect": False,
            "connection_timeout": 5, "default_retain": False, "plugins": {}})
        self.clients.append(client)
        result = await asyncio.wait_for(client.connect(self.protocol["broker_uri"], cleansession=True), 5)
        if result != 0:
            raise RuntimeError(f"MQTT connection rejected for {client_id}: {result}")
        self.event("client_connected", client_session_id=client_id)
        return client

    async def collect(self, client):
        while True:
            message = await client.deliver_message()
            if message is None:
                raise RuntimeError("Collector delivery stream ended unexpectedly.")
            payload = bytes(message.data)
            digest = hashlib.sha256(payload).hexdigest()
            try:
                body = json.loads(payload.decode("utf-8"))
                correlation_id = body["opaque_id"]
                if not isinstance(correlation_id, str):
                    raise ValueError("Opaque ID is not a string.")
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
                self.issues.append("Unparseable collector payload")
                self.event("collector_invalid_payload", payload_sha256=digest, payload_bytes=len(payload))
                raise RuntimeError("Unparseable collector payload.") from error
            self.counts[correlation_id] += 1
            expected = self.planned.get(correlation_id)
            self.event("collector_received", collector_session_id=self.protocol["collector_session_id"],
                correlation_id=correlation_id, topic=message.topic, qos=message.qos,
                retain=bool(message.retain), payload_sha256=digest, payload_bytes=len(payload),
                publisher_session_from_plan=expected["publisher_session_id"] if expected else None,
                duplicate_application_id=self.counts[correlation_id] > 1)
            valid = expected is not None and message.topic == expected["topic"] and message.qos == 1
            valid = valid and not message.retain and digest == expected["payload_sha256"]
            valid = valid and len(payload) == expected["payload_bytes"]
            if not valid:
                self.issues.append(f"Unexpected ID/topic/QoS/retain or conflicting payload: {correlation_id}")
                raise RuntimeError(self.issues[-1])
            self.received.add(correlation_id)
            if len(self.received) == len(self.planned):
                self.complete.set()

    async def publish(self, session_id):
        client = await self.connect(session_id)
        messages = [row for row in self.planned.values() if row["publisher_session_id"] == session_id]
        start = asyncio.get_running_loop().time()
        for row in messages:
            delay = start + row["sequence"] * self.protocol["interval_seconds"] - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
            correlation_id = row["correlation_id"]
            self.attempted.add(correlation_id)
            self.event("publisher_attempt", client_session_id=session_id, correlation_id=correlation_id,
                sequence=row["sequence"], topic=row["topic"], payload_sha256=row["payload_sha256"],
                payload_bytes=row["payload_bytes"])
            sent = await asyncio.wait_for(client.publish(row["topic"], row["payload"], qos=1, retain=False), 5)
            ack = sent.puback_packet
            if ack is None or sent.packet_id is None or ack.packet_id != sent.packet_id:
                raise RuntimeError(f"Missing or mismatched PUBACK: {correlation_id}")
            self.acknowledged.add(correlation_id)
            self.event("publisher_puback", client_session_id=session_id, correlation_id=correlation_id,
                       packet_id=ack.packet_id)

    async def workload(self):
        from amqtt.broker import Broker
        self.broker = Broker({"listeners": {"default": {"type": "tcp", "bind": f"{HOST}:{self.protocol['port']}",
            "max_connections": 8}}, "plugins": {
            "amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}}})
        await asyncio.wait_for(self.broker.start(), 5)
        self.broker_started = True
        self.event("broker_ready", host=HOST, port=self.protocol["port"])
        collector = await self.connect(self.protocol["collector_session_id"])
        grants = await asyncio.wait_for(collector.subscribe([(self.protocol["subscription_filter"], 1)]), 5)
        if grants != [1]:
            raise RuntimeError(f"Collector QoS 1 subscription rejected/downgraded: {grants}")
        self.event("collector_subscribed", qos_granted=grants)
        consumer = asyncio.create_task(self.collect(collector))
        publishers = [asyncio.create_task(self.publish(session)) for session in self.protocol["publisher_session_ids"]]
        self.jobs.extend([consumer, *publishers])
        publishing = asyncio.gather(*publishers)
        self.jobs.append(publishing)
        done, _ = await asyncio.wait([publishing, consumer], return_when=asyncio.FIRST_COMPLETED)
        if consumer in done:
            await consumer
            raise RuntimeError("Collector stopped prematurely.")
        await publishing
        delivered = asyncio.create_task(self.complete.wait())
        self.jobs.append(delivered)
        done, _ = await asyncio.wait([delivered, consumer], timeout=5, return_when=asyncio.FIRST_COMPLETED)
        if consumer in done:
            await consumer
        if delivered not in done:
            raise TimeoutError("Collector did not receive every expected message within the delivery deadline.")
        done, _ = await asyncio.wait([consumer], timeout=3)
        if consumer in done:
            await consumer
        self.event("delivery_drain_complete", seconds=3)

    async def cleanup(self):
        for job in self.jobs:
            if not job.done():
                job.cancel()
        if self.jobs:
            try:
                await asyncio.wait_for(asyncio.gather(*self.jobs, return_exceptions=True), 3)
            except Exception as error:
                self.cleanup_errors.append(f"Task cleanup: {type(error).__name__}: {error}")
        async def close_client(client):
            try:
                await asyncio.wait_for(client.disconnect(), 3)
                await asyncio.wait_for(client.cancel_tasks(), 1)
            except Exception as error:
                self.cleanup_errors.append(f"Client cleanup: {type(error).__name__}: {error}")
        await asyncio.gather(*(close_client(client) for client in self.clients))
        if self.broker_started:
            try:
                await asyncio.wait_for(self.broker.shutdown(), 5)
                self.event("broker_stopped")
            except Exception as error:
                self.cleanup_errors.append(f"Broker cleanup: {type(error).__name__}: {error}")
        self.clean_shutdown = not self.cleanup_errors

    async def run(self):
        self.event("run_started")
        try:
            await asyncio.wait_for(self.workload(), 30)
        except (Exception, asyncio.CancelledError) as error:
            self.error = f"{type(error).__name__}: {error}"
            self.event("run_error", error=self.error)
        finally:
            try:
                await asyncio.wait_for(self.cleanup(), 12)
            except (Exception, asyncio.CancelledError) as error:
                self.cleanup_errors.append(f"Cleanup failed: {type(error).__name__}: {error}")
                self.clean_shutdown = False
            self.event("cleanup_complete", successful=self.clean_shutdown, errors=self.cleanup_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18883)
    parser.add_argument("--output", default="results/mqtt_smoke")
    parser.add_argument("--messages", type=int, default=25)
    parser.add_argument("--interval", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.messages <= 100 or not .02 <= args.interval <= 1:
        parser.error("Use port 1..65535, messages 1..100 and interval .02..1 seconds.")
    if not 0 <= args.seed <= 2**32 - 1 or (args.messages - 1) * args.interval > 20:
        parser.error("Use seed 0..2**32-1 and a scheduled publication span of at most 20 seconds.")
    run_id = uuid.uuid4().hex
    directory = Path(args.output).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + run_id[:12])
    directory.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(handlers=[logging.NullHandler()], level=logging.WARNING)
    publishers, collector = ["p" + uuid.uuid4().hex[:18] for _ in range(4)], "c" + uuid.uuid4().hex[:18]
    protocol = {"program_version": VERSION, "run_id": run_id, "created_utc": utc_now(), "host": HOST,
        "port": args.port, "broker_uri": f"mqtt://{HOST}:{args.port}", "publisher_session_ids": publishers,
        "collector_session_id": collector, "subscription_filter": f"t/{run_id}/+", "qos": 1,
        "publishers": 4, "messages_per_publisher": args.messages, "interval_seconds": args.interval,
        "seed": args.seed, "retain": False, "condition": "planned normal connectivity smoke only",
        "authentication": "AnonymousAuthPlugin; localhost only; no production security claim",
        "telemetry": "Application-level publisher and collector observations; no broker-receive timestamps or authenticated publisher identity inferred.",
        "timeouts_seconds": {"operation": 5, "overall_workload": 30, "delivery_wait": 5, "drain": 3, "cleanup": 12},
        "limits": ["No ML training or 39-feature extraction; no attack, drift, poisoning or throughput evaluation.",
            "Emulated clients share one process/host; they are not physical IoT devices.",
            "PUBACK and collector delivery are separate observations; neither temporal order is required.",
            "QoS 1 duplicates are allowed and reconciled by application ID and payload hash.",
            "Publisher assignment is joined from the planned ledger, not authenticated by subscriber observations."],
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}}}
    planned = {}
    for index, session in enumerate(publishers):
        rng = random.Random(args.seed + index)
        for sequence in range(args.messages):
            opaque_id = uuid.uuid5(uuid.UUID(run_id), f"{session}:{sequence}").hex
            body = {"opaque_id": opaque_id, "seq": sequence, "reading": round(rng.uniform(18, 28), 6),
                    "padding": "x" * rng.randint(8, 40)}
            payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            planned[opaque_id] = {"correlation_id": opaque_id, "publisher_session_id": session,
                "sequence": sequence, "topic": f"t/{run_id}/{session}", "payload": payload,
                "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_bytes": len(payload)}
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with (directory / "ground_truth.jsonl").open("w", encoding="utf-8") as stream:
        for row in planned.values():
            stream.write(json.dumps({"run_id": run_id, "planned_label": "normal_smoke",
                **{key: value for key, value in row.items() if key != "payload"}}) + "\n")
    with (directory / "events.jsonl").open("w", encoding="utf-8") as stream:
        async def execute():
            state = SmokeRun(protocol, planned, stream)
            await state.run()
            return state
        state = asyncio.run(execute())
    results = reconcile(planned, state.attempted, state.acknowledged, state.received, state.counts, state.issues)
    success = results["passed"] and state.error is None and state.clean_shutdown
    logging.shutdown()
    summary = {"program_version": VERSION, "run_id": run_id, "completed_utc": utc_now(),
        "status": "passed" if success else "failed", "results": results, "original_error": state.error,
        "cleanup_successful": state.clean_shutdown, "cleanup_errors": state.cleanup_errors,
        "script_sha256": sha256_file(Path(__file__)), "file_sha256": {
            name: sha256_file(directory / name) for name in ["protocol.json", "ground_truth.jsonl", "events.jsonl"]}}
    (directory / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    archive = directory / "mqtt_smoke_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in ["protocol.json", "ground_truth.jsonl", "events.jsonl", "run_summary.json"]:
            bundle.write(directory / name, arcname=name)
    print("MQTT SMOKE TEST COMPLETE" if success else "MQTT SMOKE TEST FAILED")
    print(f"Expected {results['expected']} | Acked {results['puback_confirmed']} | Unique valid {results['unique_valid_deliveries']} | Duplicates {results['duplicate_deliveries']}")
    if state.error or state.cleanup_errors:
        print("Error:", state.error or "; ".join(state.cleanup_errors))
    print("Reports ZIP:", archive)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyboardInterrupt) as error:
        raise SystemExit(f"MQTT smoke stopped: {type(error).__name__}: {error}")
