#!/usr/bin/env python3
"""Apply the reviewed RoboCasa365 simulator patch to its exact source file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError(f"Invalid reviewed patch path: {value!r}")
    return path


def selected_profile(inputs: dict, profile: str) -> tuple[dict, list[dict]]:
    descriptor = inputs.get("profiles", {}).get(profile)
    if not isinstance(descriptor, dict):
        raise ValueError(f"Unknown reviewed runtime profile: {profile}")
    patches = {spec["id"]: spec for spec in inputs["patches"]}
    if len(patches) != len(inputs["patches"]):
        raise ValueError("Reviewed runtime patch IDs must be unique")
    selected = [patches[identifier] for identifier in descriptor["patches"]]
    if not selected or "camera-rgb-reuse" not in descriptor["patches"]:
        raise ValueError("Both runtime profiles require the reviewed common camera patch")
    return descriptor, selected


def patch_identity(spec: dict) -> dict:
    return {
        key: spec[key]
        for key in (
            "source",
            "revision",
            "path",
            "patch",
            "patch_sha256",
            "original_sha256",
            "patched_sha256",
        )
    }


def check_profile(source_root: Path, inputs: dict, profile: str, selected: list[dict]) -> None:
    record = source_root / "roboscope-runtime-patches.json"
    if record.exists():
        existing = json.loads(record.read_text(encoding="utf-8"))
        # The former camera-only builder had no explicit profile stamp.
        existing_profile = existing.get("profile", "default")
        if existing_profile != profile:
            raise RuntimeError(
                f"Runtime prefix belongs to {existing_profile!r}, requested {profile!r}; use a fresh prefix"
            )
        if [patch_identity(spec) for spec in existing["patches"]] != [
            patch_identity(spec) for spec in selected
        ]:
            raise RuntimeError("Existing runtime patch provenance differs; use a fresh prefix")
    selected_ids = {spec["id"] for spec in selected}
    for spec in inputs["patches"]:
        target = source_root / spec["source"] / relative_path(spec["path"])
        if not target.exists():
            continue
        actual = file_sha256(target)
        allowed = {spec["original_sha256"]}
        if spec["id"] in selected_ids:
            allowed.add(spec["patched_sha256"])
        if target.is_symlink() or actual not in allowed:
            raise RuntimeError(f"Runtime profile {profile!r} refuses source {target}; use a fresh prefix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("runtime-patches.json"))
    parser.add_argument("--profile", choices=("default", "paired"), default="default")
    parser.add_argument("--check-profile", action="store_true", help="Read-only prefix/profile preflight")
    args = parser.parse_args()
    inputs_path = args.inputs.resolve()
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    sources = json.loads(inputs_path.with_name("source-inputs.json").read_text(encoding="utf-8"))["sources"]
    if inputs.get("schema_version") != 1 or not inputs.get("patches"):
        raise ValueError("Expected a non-empty reviewed runtime patch list")
    descriptor, selected = selected_profile(inputs, args.profile)
    source_root = args.source_root.resolve()
    check_profile(source_root, inputs, args.profile, selected)
    for spec in inputs["patches"]:
        source = spec["source"]
        if source not in sources or sources[source]["revision"] != spec["revision"]:
            raise ValueError("Runtime patch source revision differs from the source lock")
        patch = inputs_path.parent.joinpath(*relative_path(spec["patch"]).parts)
        target_root = source_root / source
        target = target_root.joinpath(*relative_path(spec["path"]).parts)
        if patch.is_symlink() or target.is_symlink() or file_sha256(patch) != spec["patch_sha256"]:
            raise RuntimeError("Reviewed runtime patch file or target failed integrity verification")
    if args.check_profile:
        print(f"Runtime profile preflight passed: {descriptor['runtime_profile']}", flush=True)
        return
    for spec in selected:
        source = spec["source"]
        patch = inputs_path.parent.joinpath(*relative_path(spec["patch"]).parts)
        target_root = source_root / source
        target = target_root.joinpath(*relative_path(spec["path"]).parts)
        actual = file_sha256(target)
        if actual == spec["patched_sha256"]:
            print(f"Verified existing runtime patch: {source}/{spec['path']}", flush=True)
            continue
        if actual != spec["original_sha256"]:
            raise RuntimeError(f"Runtime patch refuses an unknown source file: {target}")
        subprocess.run(
            (
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "apply",
                "--check",
                "--whitespace=error-all",
                str(patch),
            ),
            cwd=target_root,
            check=True,
        )
        subprocess.run(
            (
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "apply",
                "--whitespace=error-all",
                str(patch),
            ),
            cwd=target_root,
            check=True,
        )
        if file_sha256(target) != spec["patched_sha256"]:
            raise RuntimeError(f"Runtime patch produced an unexpected source file: {target}")
        print(f"Applied reviewed runtime patch: {source}/{spec['path']}", flush=True)
    record = source_root / "roboscope-runtime-patches.json"
    provenance = {
        "schema_version": 1,
        "profile": args.profile,
        "runtime_profile": descriptor["runtime_profile"],
        "validation_only": descriptor.get("validation_only", False),
        "patches": selected,
    }
    if descriptor.get("validation_only"):
        provenance["full_suite"] = False
    record.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
