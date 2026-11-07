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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("runtime-patches.json"))
    args = parser.parse_args()
    inputs_path = args.inputs.resolve()
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    sources = json.loads(inputs_path.with_name("source-inputs.json").read_text(encoding="utf-8"))["sources"]
    if inputs.get("schema_version") != 1 or not inputs.get("patches"):
        raise ValueError("Expected a non-empty reviewed runtime patch list")
    for spec in inputs["patches"]:
        source = spec["source"]
        if source not in sources or sources[source]["revision"] != spec["revision"]:
            raise ValueError("Runtime patch source revision differs from the source lock")
        patch = inputs_path.parent.joinpath(*relative_path(spec["patch"]).parts)
        target_root = args.source_root.resolve() / source
        target = target_root.joinpath(*relative_path(spec["path"]).parts)
        if patch.is_symlink() or target.is_symlink() or file_sha256(patch) != spec["patch_sha256"]:
            raise RuntimeError("Reviewed runtime patch file or target failed integrity verification")
        actual = file_sha256(target)
        if actual == spec["patched_sha256"]:
            print(f"Verified existing runtime patch: {source}/{spec['path']}", flush=True)
            continue
        if actual != spec["original_sha256"]:
            raise RuntimeError(f"Runtime patch refuses an unknown source file: {target}")
        subprocess.run(
            ("git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
             "apply", "--check", "--whitespace=error-all", str(patch)),
            cwd=target_root, check=True,
        )
        subprocess.run(
            ("git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
             "apply", "--whitespace=error-all", str(patch)),
            cwd=target_root, check=True,
        )
        if file_sha256(target) != spec["patched_sha256"]:
            raise RuntimeError(f"Runtime patch produced an unexpected source file: {target}")
        print(f"Applied reviewed runtime patch: {source}/{spec['path']}", flush=True)
    record = args.source_root.resolve() / "roboscope-runtime-patches.json"
    record.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
