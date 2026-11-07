# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Publish initial evidence and guard a paired run before Policy inference.

Only evidence writes, reference reads and waiting happen here. The caller has
finished every native reset and has not called Policy.reset or Policy.act.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, NoReturn

from roboscope_kit.runtime.benchmark.queue import atomic_json

from .protocol import TASKS_BY_NAME

_SCHEMA_VERSION = 1
_HEADER_KEYS = ("schema_version", "task", "stream_episode", "runtime_profile", "identifiers")
_SUMMARY_LIMIT = 16


def _canonical(value: Any) -> str:
    # Preserve numeric types: Python equality treats True, 1 and 1.0 as equal.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _differences(actual: dict[str, Any], reference: Any) -> list[str]:
    if not isinstance(reference, dict):
        return ["reference payload is not an object"]
    differences = []
    if set(reference) != {*_HEADER_KEYS, "fingerprints"}:
        differences.append("header field set differs")
    for key in _HEADER_KEYS:
        if key not in reference:
            differences.append(f"header.{key} is missing")
        elif _canonical(actual[key]) != _canonical(reference[key]):
            differences.append(f"header.{key} differs")
    reference_rows = reference.get("fingerprints")
    if not isinstance(reference_rows, list):
        return [*differences, "reference fingerprints is not an array"]
    actual_rows = actual["fingerprints"]
    if len(actual_rows) != len(reference_rows):
        differences.append(f"fingerprint slot counts differ ({len(actual_rows)} vs {len(reference_rows)})")
    for slot, (row, expected) in enumerate(zip(actual_rows, reference_rows, strict=False)):
        if not isinstance(expected, dict):
            differences.append(f"slot {slot}: reference fingerprint is not an object")
            continue
        for key in sorted(set(row) | set(expected)):
            if key not in row or key not in expected:
                differences.append(f"slot {slot}: {key} is missing on one side")
            elif _canonical(row[key]) != _canonical(expected[key]):
                differences.append(f"slot {slot}: {key} differs")
    return differences


def _reject(args, taskname: str, stream_episode: int, differences: list[str]) -> NoReturn:
    atomic_json(Path(args.output_dir) / "pairing-rejected.json", {
        "schema_version": _SCHEMA_VERSION,
        "task": taskname,
        "stream_episode": stream_episode,
        "differences": differences,
    })
    summary = "; ".join(differences[:_SUMMARY_LIMIT])
    extra = len(differences) - _SUMMARY_LIMIT
    if extra > 0:
        summary += f"; {extra} additional slot/key differences"
    raise RuntimeError(f"Paired initial-state guard rejected {taskname} group {stream_episode}: {summary}")


def publish_and_check_initial(
    args, taskname: str, stream_episode: int, identifiers, fingerprints,
) -> None:
    """Publish this group and require exact initial agreement with a K=1 reference."""
    if taskname not in TASKS_BY_NAME:
        raise ValueError(f"Unknown RoboCasa365 task: {taskname!r}")
    if type(stream_episode) is not int or stream_episode < 0:
        raise ValueError("Paired initial-state group index must be a non-negative integer")
    worker_id = args.worker_id
    if type(worker_id) is not int or worker_id < 0:
        raise ValueError("Paired initial-state worker id must be a non-negative integer")
    names = list(identifiers)
    rows = list(fingerprints)
    if (
        not names or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names)) or len(names) != len(rows)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("Initial-state identifiers and fingerprint objects must match unique slots")
    filename = f"{taskname}-group-{stream_episode:02d}-initial.json"
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "task": taskname,
        "stream_episode": stream_episode,
        "runtime_profile": args.runtime_profile,
        "identifiers": names,
        "fingerprints": rows,
    }
    _canonical(payload)  # Reject non-JSON or non-finite evidence before publishing.
    destination = Path(args.output_dir) / "evidence" / f"worker-{worker_id:03d}" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, payload)
    reference_output = args.pair_reference_output
    if reference_output is None:
        return
    if args.runtime_profile != "paired-stable-regions-v1" or not args.audit_trajectories:
        _reject(args, taskname, stream_episode, [
            "reference checking requires runtime_profile=paired-stable-regions-v1 and auditing",
        ])
    if worker_id != 0:
        _reject(args, taskname, stream_episode, ["reference checking requires fixed K=1 (worker 000)"])
    reference_output = Path(reference_output)
    if not reference_output.is_absolute():
        _reject(args, taskname, stream_episode, ["reference output directory must be absolute"])
    timeout = float(getattr(args, "pair_reference_timeout_s", 900))
    if not math.isfinite(timeout) or timeout <= 0:
        _reject(args, taskname, stream_episode, ["reference timeout must be finite and positive"])
    reference_path = reference_output / "evidence" / "worker-000" / filename
    if reference_path.resolve() == destination.resolve():
        _reject(args, taskname, stream_episode, ["reference must use a different output directory"])
    if (Path(args.output_dir) / "pairing-rejected.json").exists():
        raise RuntimeError(
            "This paired run already rejected initial-state evidence; retries cannot be accepted"
        )
    deadline = time.monotonic() + timeout
    while not reference_path.is_file():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _reject(args, taskname, stream_episode, [f"timed out after {timeout:g}s waiting for {filename}"])
        time.sleep(min(0.1, remaining))
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        differences = _differences(payload, reference)
    except (OSError, ValueError, TypeError):
        _reject(args, taskname, stream_episode, [f"invalid reference evidence: {filename}"])
    if differences:
        _reject(args, taskname, stream_episode, differences)
