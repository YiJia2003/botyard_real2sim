#!/usr/bin/env python3
"""Remote-side atomic publisher for one real2sim result packet.

The remote agent writes into ``real2sim/results/.staging/<bundle>.partial`` and
then runs this script.  It validates ``RESULT.json``, hashes every artifact,
writes ``RESULT_READY.json``, and atomically renames the staging directory to
``real2sim/results/<bundle>``.

It never runs PhysX and never edits the result payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL = "botyard-real2sim-bridge/v1"
SCHEMA_VERSION = 1
RESULT_EXCLUDES = {"RESULT_READY.json", "PUBLISH_FAILED.json", ".DS_Store", "Thumbs.db"}
ALLOWED_STATUS = {"completed", "partial", "blocked", "failed", "acknowledged"}


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
            raise ValueError(f"result contains a symlink: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if len(relative.parts) == 1 and relative.name in RESULT_EXCLUDES:
            continue
        if item.name in {".DS_Store", "Thumbs.db"} or "__pycache__" in relative.parts:
            continue
        files.append(item)
    return files


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fail(staging: Path, bundle_id: str, errors: list[str]) -> int:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "ok": False,
        "bundle_id": bundle_id,
        "published_at_utc": utc_now(),
        "staging": str(staging),
        "errors": errors,
    }
    try:
        write_json(staging / "PUBLISH_FAILED.json", receipt)
    except OSError:
        pass
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path("/home/cyj/botyard_real2sim"))
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    results_root = (project_root / "real2sim" / "results").resolve()
    staging_root = (results_root / ".staging").resolve()
    bundle_id = args.bundle_id.strip()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    if not bundle_id or any(c not in allowed for c in bundle_id):
        print(json.dumps({"ok": False, "errors": ["invalid bundle_id"]}))
        return 2
    staging = (args.staging or (staging_root / f"{bundle_id}.partial")).resolve()
    if not staging.is_dir():
        return fail(staging, bundle_id, [f"result staging directory does not exist: {staging}"])
    try:
        staging.relative_to(staging_root)
    except ValueError:
        return fail(staging, bundle_id, [f"staging must live under {staging_root}"])

    try:
        files = payload_files(staging)
    except ValueError as error:
        return fail(staging, bundle_id, [str(error)])
    names = {item.relative_to(staging).as_posix() for item in files}
    errors: list[str] = []
    for required in ("RESULT.json", "report.md"):
        if required not in names:
            errors.append(f"missing required result file: {required}")
    if errors:
        return fail(staging, bundle_id, errors)
    try:
        result = json.loads((staging / "RESULT.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return fail(staging, bundle_id, [f"cannot parse RESULT.json: {error}"])
    if not isinstance(result, dict):
        return fail(staging, bundle_id, ["RESULT.json root must be an object"])
    if result.get("bundle_id") != bundle_id:
        errors.append("RESULT.json bundle_id does not match --bundle-id")
    if result.get("schema_version") != SCHEMA_VERSION or result.get("protocol") != PROTOCOL:
        errors.append("RESULT.json has unsupported schema/protocol")
    if result.get("status") not in ALLOWED_STATUS:
        errors.append(f"RESULT.json status must be one of {sorted(ALLOWED_STATUS)}")
    if result.get("claims_scope") not in (None, "curve_fit_candidate_only"):
        errors.append("RESULT.json claims_scope must be curve_fit_candidate_only when present")
    next_request = staging / "next_data_request.json"
    if next_request.is_file():
        try:
            request = json.loads(next_request.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot parse next_data_request.json: {error}")
        else:
            if not isinstance(request, dict) or request.get("bundle_id") != bundle_id:
                errors.append("next_data_request.json bundle_id does not match")
    if errors:
        return fail(staging, bundle_id, errors)

    files = payload_files(staging)
    table = {
        item.relative_to(staging).as_posix(): {
            "sha256": sha256_file(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    }
    final = results_root / bundle_id
    if final.exists():
        if not args.replace:
            return fail(staging, bundle_id, [f"final result already exists: {final}"])
        backup_root = results_root / ".replaced"
        backup_root.mkdir(parents=True, exist_ok=True)
        moved = backup_root / f"{bundle_id}.replaced-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        os.replace(final, moved)

    ready = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "ok": True,
        "bundle_id": bundle_id,
        "published_at_utc": utc_now(),
        "publisher": "real2sim/bridge/remote_publish_result.py",
        "staging": str(staging),
        "final_path": str(final),
        "result_status": result.get("status"),
        "result_sha256": sha256_file(staging / "RESULT.json"),
        "report_sha256": sha256_file(staging / "report.md"),
        "next_data_request": "next_data_request.json" if next_request.is_file() else None,
        "file_count": len(table),
        "files": dict(sorted(table.items())),
    }
    write_json(staging / "RESULT_READY.json", ready)
    try:
        os.replace(staging, final)
    except OSError as error:
        ready["ok"] = False
        ready["errors"] = [f"atomic rename failed: {error}"]
        write_json(staging / "RESULT_READY.json", ready)
        return fail(staging, bundle_id, [f"atomic rename failed: {error}"])
    print(json.dumps(ready, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
