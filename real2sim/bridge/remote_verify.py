#!/usr/bin/env python3
"""Remote-side verifier for one staged real2sim bundle.

Run this on the simulation host after rsync has uploaded a bundle to
``real2sim/experiment/.staging/<bundle>.partial-*``.  It:

1. rejects symlinks and verifies every hash in ``BUNDLE.json`` / ``SHA256SUMS``;
2. requires the manifest's CSV hash to match;
3. runs the project's own strict dataset validator (``./run.sh data --strict``);
4. writes ``READY.json`` only after all checks pass;
5. atomically renames the staging directory to ``experiment/<bundle>``.

It never edits the bundle payload and never starts PhysX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_EXCLUDES = {"READY.json", "SHA256SUMS", "VERIFY_FAILED.json", "BUNDLE.json", ".DS_Store", "Thumbs.db"}
PROTOCOL = "botyard-real2sim-bridge/v1"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"bundle contains a symlink: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if len(relative.parts) == 1 and relative.name in ROOT_EXCLUDES:
            continue
        if item.name in {".DS_Store", "Thumbs.db"} or "__pycache__" in relative.parts:
            continue
        if relative.parts and relative.parts[0] == "results":
            continue
        files.append(item)
    return files


def parse_sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        digest, separator, name = raw.partition("  ")
        if not separator or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"SHA256SUMS:{number}: expected '<64-hex>  <path>'")
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"SHA256SUMS:{number}: unsafe path {name!r}")
        if name in entries:
            raise ValueError(f"SHA256SUMS:{number}: duplicate path {name!r}")
        entries[name] = digest.lower()
    return entries


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fail(staging: Path, bundle_id: str, errors: list[str], details: dict[str, Any] | None = None) -> int:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "ok": False,
        "bundle_id": bundle_id,
        "verified_at_utc": utc_now(),
        "staging": str(staging),
        "errors": errors,
        "details": details or {},
    }
    try:
        write_json(staging / "VERIFY_FAILED.json", receipt)
    except OSError:
        pass
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 2


def project_dataset(project_root: Path, manifest: Path) -> dict[str, Any]:
    """Invoke the project's strict validator with its configured interpreter."""

    with tempfile.NamedTemporaryFile(prefix="r2s-dataset-", suffix=".json", delete=False) as handle:
        report_path = Path(handle.name)
    try:
        runner = project_root / "run.sh"
        if runner.is_file():
            command = [str(runner), "data", "--manifest", str(manifest), "--strict", "--out", str(report_path)]
            environment = None
        else:
            command = [
                sys.executable,
                "-m",
                "botyard_real2sim.cli.data",
                "--manifest",
                str(manifest),
                "--strict",
                "--out",
                str(report_path),
            ]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root / "src") + (
                os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
            )
        result = subprocess.run(
            command,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env=environment,
        )
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {}
        report["_validator_exit_code"] = result.returncode
        report["_validator_stdout"] = result.stdout[-4000:]
        report["_validator_stderr"] = result.stderr[-4000:]
        return report
    finally:
        try:
            report_path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path("/home/cyj/botyard_real2sim"))
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="move an existing final bundle aside instead of failing")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    experiment_root = (project_root / "real2sim" / "experiment").resolve()
    staging_root = (experiment_root / ".staging").resolve()
    staging = args.staging.resolve()
    bundle_id = args.bundle_id.strip()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    if not bundle_id or any(c not in allowed for c in bundle_id):
        return fail(staging, bundle_id, ["invalid bundle_id"])
    if not staging.is_dir():
        return fail(staging, bundle_id, [f"staging directory does not exist: {staging}"])
    try:
        staging.relative_to(staging_root)
    except ValueError:
        return fail(staging, bundle_id, [f"staging must live under {staging_root}"])

    errors: list[str] = []
    details: dict[str, Any] = {}
    try:
        files = payload_files(staging)
    except ValueError as error:
        return fail(staging, bundle_id, [str(error)])
    actual = {item.relative_to(staging).as_posix() for item in files}
    for required in ("manifest.json", "TASK.json", "BUNDLE.json", "SHA256SUMS"):
        if not (staging / required).is_file():
            errors.append(f"missing required file: {required}")
    if errors:
        return fail(staging, bundle_id, errors)

    try:
        bundle = json.loads((staging / "BUNDLE.json").read_text(encoding="utf-8"))
        task = json.loads((staging / "TASK.json").read_text(encoding="utf-8"))
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        sums = parse_sums(staging / "SHA256SUMS")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return fail(staging, bundle_id, [f"cannot parse bundle metadata: {error}"])
    if not isinstance(bundle, dict) or not isinstance(task, dict) or not isinstance(manifest, dict):
        return fail(staging, bundle_id, ["BUNDLE.json, TASK.json, manifest.json must be objects"])
    if bundle.get("bundle_id") != bundle_id:
        errors.append("BUNDLE.json bundle_id does not match --bundle-id")
    if task.get("bundle_id") != bundle_id:
        errors.append("TASK.json bundle_id does not match --bundle-id")
    if manifest.get("dataset_id") != bundle_id:
        errors.append("manifest.dataset_id does not match --bundle-id")
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("protocol") != PROTOCOL:
        errors.append("unsupported BUNDLE.json schema/protocol")

    expected_sums = set(actual) | {"BUNDLE.json"}
    if set(sums) != expected_sums:
        missing = sorted(expected_sums - set(sums))
        extra = sorted(set(sums) - expected_sums)
        if missing:
            errors.append("SHA256SUMS missing: " + ", ".join(missing))
        if extra:
            errors.append("SHA256SUMS has unknown entries: " + ", ".join(extra))
    for name, digest in sums.items():
        item = staging / name
        if not item.is_file():
            errors.append(f"SHA256SUMS entry missing on disk: {name}")
        elif sha256_file(item) != digest:
            errors.append(f"SHA256 mismatch: {name}")

    bundle_files = bundle.get("files") or {}
    if not isinstance(bundle_files, dict):
        errors.append("BUNDLE.json files must be an object")
        bundle_files = {}
    expected_bundle_files = actual
    if set(bundle_files) != expected_bundle_files:
        missing = sorted(expected_bundle_files - set(bundle_files))
        extra = sorted(set(bundle_files) - expected_bundle_files)
        if missing:
            errors.append("BUNDLE.json files missing: " + ", ".join(missing))
        if extra:
            errors.append("BUNDLE.json files has unknown entries: " + ", ".join(extra))
    for name, entry in bundle_files.items():
        item = staging / name
        if not isinstance(entry, dict) or not item.is_file():
            errors.append(f"invalid BUNDLE.json entry: {name}")
            continue
        if entry.get("sha256") != sha256_file(item):
            errors.append(f"BUNDLE.json hash mismatch: {name}")
        if entry.get("bytes") != item.stat().st_size:
            errors.append(f"BUNDLE.json size mismatch: {name}")

    manifest_sha = sha256_file(staging / "manifest.json")
    csv_name = str((manifest.get("recording") or {}).get("csv") or "capture.csv")
    csv_path = staging / csv_name
    csv_sha = sha256_file(csv_path) if csv_path.is_file() else None
    if csv_sha is None:
        errors.append(f"manifest CSV does not exist: {csv_name}")
    else:
        if (manifest.get("recording") or {}).get("sha256", "").lower() != csv_sha:
            errors.append("manifest.recording.sha256 does not match the CSV")
        if bundle.get("csv_sha256") != csv_sha:
            errors.append("BUNDLE.json csv_sha256 does not match the CSV")
    if bundle.get("manifest_sha256") != manifest_sha:
        errors.append("BUNDLE.json manifest_sha256 does not match manifest.json")

    if errors:
        details["checks"] = {"files": len(actual), "checksums": len(sums)}
        return fail(staging, bundle_id, errors, details)

    dataset = project_dataset(project_root, staging / "manifest.json")
    details["dataset"] = {
        "status": dataset.get("status"),
        "data_ready": dataset.get("data_ready"),
        "fit_ready": dataset.get("fit_ready"),
        "evaluation_ready": dataset.get("evaluation_ready"),
        "errors": dataset.get("errors"),
        "blocking_reasons": dataset.get("blocking_reasons"),
        "metrics": dataset.get("metrics"),
    }
    if dataset.get("_validator_exit_code") != 0 or not dataset.get("data_ready"):
        errors.append("project strict dataset validation did not return data_ready=true")
        errors.extend(str(item) for item in (dataset.get("errors") or []))
        return fail(staging, bundle_id, errors, details)

    final = experiment_root / bundle_id
    if final.exists():
        if not args.replace:
            return fail(staging, bundle_id, [f"final bundle already exists: {final}"], details)
        backup_root = experiment_root / ".replaced"
        backup_root.mkdir(parents=True, exist_ok=True)
        moved = backup_root / f"{bundle_id}.replaced-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        os.replace(final, moved)
        details["replaced_previous"] = str(moved)

    ready = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "ok": True,
        "bundle_id": bundle_id,
        "ready_at_utc": utc_now(),
        "verifier": "real2sim/bridge/remote_verify.py",
        "project_root": str(project_root),
        "staging": str(staging),
        "final_path": str(final),
        "manifest_sha256": manifest_sha,
        "csv_sha256": csv_sha,
        "bundle_record_sha256": sha256_file(staging / "BUNDLE.json"),
        "file_count": len(actual),
        "dataset_status": dataset.get("status"),
        "dataset_report": details["dataset"],
        "checks": {
            "checksums": True,
            "bundle_record": True,
            "manifest_csv_hash": True,
            "strict_dataset": True,
            "no_symlinks": True,
        },
    }
    write_json(staging / "READY.json", ready)
    try:
        os.replace(staging, final)
    except OSError as error:
        ready["ok"] = False
        ready["errors"] = [f"atomic rename failed: {error}"]
        write_json(staging / "READY.json", ready)
        return fail(staging, bundle_id, [f"atomic rename failed: {error}"], details)
    print(json.dumps(ready, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
