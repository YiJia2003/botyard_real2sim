"""Local/remote data bridge for the real2sim experiment loop.

This module owns the *transport-independent* part of the bridge:

* bundle directory layout and integrity manifests (``BUNDLE.json`` / ``SHA256SUMS``);
* staging -> verified -> atomic-rename publication;
* construction of a canonical dataset manifest and CSV from a raw capture;
* the JSON command/result/next-data contracts shared with the remote agent;
* a tiny client for the remote DeepSeek Harness web API and SSH/rsync helpers.

It intentionally imports only the standard library.  Dataset validation is
performed by :mod:`botyard_real2sim.datasets` through a lazy import so that
CI, packaging, and the remote side can use the bundle tools without PyTorch,
Isaac Lab, or a GPU.

中文说明：
  本模块只负责“数据包”本身，不启动仿真、不控制硬件。
  数据包在本地构建到 staging 目录，写入 BUNDLE.json 和 SHA256SUMS；
  上传到远程后由远程脚本复核哈希、调用项目自带 manifest 校验器，
  通过后写 READY.json，并原子改名为正式数据接口目录。
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

BRIDGE_PROTOCOL = "botyard-real2sim-bridge/v1"
BRIDGE_VERSION = "1.0.0"
SCHEMA_VERSION = 1

BUNDLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROOT_EXCLUDES = frozenset(
    {
        "READY.json",
        "SHA256SUMS",
        "VERIFY_FAILED.json",
        "BUNDLE.json",
        ".DS_Store",
        "Thumbs.db",
    }
)
_RESULT_EXCLUDES = frozenset({"RESULT_READY.json", "PUBLISH_FAILED.json", ".DS_Store", "Thumbs.db"})

#: Raw-capture board alias for each joint supported by the current S1 rig.
BOARD_FOR_JOINT = {"MFJ2": "j2", "MFJ3": "j3"}
SUPPORTED_CAPTURE_JOINTS = tuple(BOARD_FOR_JOINT)
DEFAULT_THIRD_FIELD_SCALE_N_PER_UNIT = 0.01


class BridgeError(RuntimeError):
    """A user-actionable bridge failure."""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_bundle_id(value: str) -> str:
    """Validate a bundle/result identifier safe for both filesystems and rsync."""

    if not isinstance(value, str) or not BUNDLE_ID_PATTERN.fullmatch(value):
        raise BridgeError("bundle_id must start with a letter/digit and contain only letters, digits, '.', '_' or '-'")
    return value


def slug(value: str, *, fallback: str = "run") -> str:
    """Make a short filesystem-safe fragment for auto-generated bundle ids."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("._-")
    return (cleaned or fallback)[:48]


def auto_bundle_id(label: str = "run", *, now: datetime | None = None) -> str:
    """Build a sortable bundle id, for example ``r2s-20261120T081500Z-mfj2-step``."""

    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return ensure_bundle_id(f"r2s-{stamp}-{slug(label)}")


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(paths: Iterable[str | Path]) -> str:
    """Hash an ordered set of files plus their names, for raw-capture provenance."""

    digest = hashlib.sha256()
    for path in paths:
        item = Path(path)
        if not item.is_file():
            continue
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    """Read JSON with a path-specific error message."""

    item = Path(path)
    try:
        return json.loads(item.read_text(encoding="utf-8"))
    except OSError as error:
        raise BridgeError(f"cannot read {item}: {error}") from error
    except json.JSONDecodeError as error:
        raise BridgeError(f"{item}: invalid JSON: {error}") from error


def write_json_atomic(path: str | Path, value: Any, *, mode: int = 0o644) -> None:
    """Write JSON via a same-directory temporary file and ``os.replace``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_text_atomic(path: str | Path, text: str, *, mode: int = 0o644) -> None:
    """Write UTF-8 text atomically, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _walk_regular_files(root: Path, *, exclude_root: frozenset[str]) -> list[Path]:
    if not root.is_dir():
        raise BridgeError(f"not a directory: {root}")
    files: list[Path] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise BridgeError(f"bundle contains a symlink, refusing to publish: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if relative.parts and relative.parts[0] == ".bridge":
            continue
        if len(relative.parts) == 1 and relative.name in exclude_root:
            continue
        if item.name in {".DS_Store", "Thumbs.db"} or "__pycache__" in relative.parts:
            continue
        files.append(item)
    return files


def payload_files(root: str | Path) -> list[Path]:
    """List files that belong to a bundle's content (root markers/results excluded)."""

    base = Path(root)
    return [
        item
        for item in _walk_regular_files(base, exclude_root=_ROOT_EXCLUDES)
        if item.relative_to(base).parts[0] != "results"
    ]


def result_files(root: str | Path) -> list[Path]:
    """List files that belong to a result packet (result markers excluded)."""

    return _walk_regular_files(Path(root), exclude_root=_RESULT_EXCLUDES)


def relative_file_table(root: str | Path, files: Iterable[Path] | None = None) -> dict[str, dict[str, Any]]:
    """Build the ``{relative_path: {sha256, bytes}}`` table used in BUNDLE.json."""

    base = Path(root)
    table: dict[str, dict[str, Any]] = {}
    for item in files if files is not None else payload_files(base):
        relative = item.relative_to(base).as_posix()
        table[relative] = {"sha256": sha256_file(item), "bytes": item.stat().st_size}
    return dict(sorted(table.items()))


def write_sha256sums(
    root: str | Path,
    *,
    exclude: Iterable[str] = ("SHA256SUMS", "READY.json", "VERIFY_FAILED.json"),
) -> Path:
    """Write a deterministic ``sha256  relative/path`` file for payload files.

    ``BUNDLE.json`` is included explicitly: it is not part of its own ``files``
    table, but the checksum file must pin the record that describes the rest.
    """

    base = Path(root)
    excluded = set(exclude)
    lines: list[str] = []
    for item in payload_files(base):
        relative = item.relative_to(base).as_posix()
        if relative in excluded:
            continue
        lines.append(f"{sha256_file(item)}  {relative}")
    bundle_record = base / "BUNDLE.json"
    if bundle_record.is_file() and "BUNDLE.json" not in excluded:
        lines.append(f"{sha256_file(bundle_record)}  BUNDLE.json")
    lines.sort(key=lambda line: line.split("  ", 1)[1])
    destination = base / "SHA256SUMS"
    write_text_atomic(destination, "\n".join(lines) + "\n")
    return destination


def parse_sha256sums(path: str | Path) -> dict[str, str]:
    """Parse a SHA256SUMS file, rejecting malformed or duplicate entries."""

    item = Path(path)
    entries: dict[str, str] = {}
    try:
        lines = item.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BridgeError(f"cannot read {item}: {error}") from error
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        digest, separator, name = raw.partition("  ")
        if not separator or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise BridgeError(f"{item}:{line_number}: expected '<64-hex>  <path>'")
        if name.startswith("/") or ".." in Path(name).parts:
            raise BridgeError(f"{item}:{line_number}: unsafe relative path {name!r}")
        if name in entries:
            raise BridgeError(f"{item}:{line_number}: duplicate entry {name!r}")
        entries[name] = digest.lower()
    return entries


def build_bundle_record(
    root: str | Path,
    *,
    bundle_id: str | None = None,
    intent: str | None = None,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``BUNDLE.json`` record for an already-written bundle directory."""

    base = Path(root)
    manifest = read_json(base / "manifest.json")
    if not isinstance(manifest, dict):
        raise BridgeError("manifest.json root must be an object")
    task_record = dict(task or read_json(base / "TASK.json"))
    record_bundle_id = ensure_bundle_id(bundle_id or str(task_record.get("bundle_id") or manifest.get("dataset_id")))
    files = relative_file_table(base)
    files.pop("BUNDLE.json", None)
    csv_name = str((manifest.get("recording") or {}).get("csv") or "capture.csv")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": BRIDGE_PROTOCOL,
        "bridge_version": BRIDGE_VERSION,
        "bundle_id": record_bundle_id,
        "created_at_utc": str(task_record.get("created_at_utc") or utc_now()),
        "intent": str(intent or task_record.get("intent") or "calibration"),
        "dataset_id": manifest.get("dataset_id"),
        "split": manifest.get("split"),
        "manifest_sha256": sha256_file(base / "manifest.json"),
        "csv_sha256": sha256_file(base / csv_name),
        "csv": csv_name,
        "files": files,
        "file_count": len(files),
        "ready_marker": "READY.json",
        "task_sha256": sha256_file(base / "TASK.json"),
    }


def verify_bundle_integrity(root: str | Path) -> dict[str, Any]:
    """Verify BUNDLE.json, SHA256SUMS, and required files without a simulator."""

    base = Path(root)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": BRIDGE_PROTOCOL,
        "bundle_dir": str(base.resolve()),
        "bundle_id": None,
        "ok": False,
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    def fail(message: str) -> None:
        report["errors"].append(message)

    for required in ("manifest.json", "TASK.json", "BUNDLE.json", "SHA256SUMS"):
        if not (base / required).is_file():
            fail(f"missing required file: {required}")
    if report["errors"]:
        return report

    try:
        bundle = read_json(base / "BUNDLE.json")
        task = read_json(base / "TASK.json")
        manifest = read_json(base / "manifest.json")
        sums = parse_sha256sums(base / "SHA256SUMS")
    except BridgeError as error:
        fail(str(error))
        return report
    if not isinstance(bundle, dict) or not isinstance(task, dict) or not isinstance(manifest, dict):
        fail("BUNDLE.json, TASK.json, and manifest.json roots must be objects")
        return report

    bundle_id = bundle.get("bundle_id")
    try:
        ensure_bundle_id(str(bundle_id))
    except BridgeError as error:
        fail(str(error))
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("protocol") != BRIDGE_PROTOCOL:
        fail("BUNDLE.json has an unsupported schema_version/protocol")
    if task.get("bundle_id") != bundle_id:
        fail("TASK.json bundle_id does not match BUNDLE.json")
    if manifest.get("dataset_id") != bundle_id:
        fail("manifest.json dataset_id does not match bundle_id")
    if bundle.get("manifest_sha256") != sha256_file(base / "manifest.json"):
        fail("BUNDLE.json manifest_sha256 does not match manifest.json")
    csv_name = str((manifest.get("recording") or {}).get("csv") or "capture.csv")
    csv_path = base / csv_name
    if not csv_path.is_file():
        fail(f"manifest recording.csv does not exist: {csv_name}")
    else:
        csv_hash = sha256_file(csv_path)
        if (manifest.get("recording") or {}).get("sha256", "").lower() != csv_hash:
            fail("manifest recording.sha256 does not match the CSV")
        if bundle.get("csv_sha256") != csv_hash:
            fail("BUNDLE.json csv_sha256 does not match the CSV")

    try:
        actual = {item.relative_to(base).as_posix() for item in payload_files(base)}
    except BridgeError as error:
        fail(str(error))
        return report
    expected_sum_names = set(actual)
    if (base / "BUNDLE.json").is_file():
        expected_sum_names.add("BUNDLE.json")
    if set(sums) != expected_sum_names:
        missing = sorted(expected_sum_names - set(sums))
        extra = sorted(set(sums) - expected_sum_names)
        if missing:
            fail(f"SHA256SUMS is missing: {', '.join(missing)}")
        if extra:
            fail(f"SHA256SUMS lists unknown files: {', '.join(extra)}")
    for name, expected in sums.items():
        item = base / name
        if not item.is_file():
            fail(f"SHA256SUMS entry is missing on disk: {name}")
            continue
        if sha256_file(item) != expected:
            fail(f"SHA256 mismatch: {name}")
    bundle_files = bundle.get("files") or {}
    if not isinstance(bundle_files, dict):
        fail("BUNDLE.json files must be an object")
    else:
        expected_bundle_names = actual - {"BUNDLE.json"}
        if set(bundle_files) != expected_bundle_names:
            missing = sorted(expected_bundle_names - set(bundle_files))
            extra = sorted(set(bundle_files) - expected_bundle_names)
            if missing:
                fail(f"BUNDLE.json files is missing: {', '.join(missing)}")
            if extra:
                fail(f"BUNDLE.json files lists unknown paths: {', '.join(extra)}")
        for name, entry in bundle_files.items():
            item = base / name
            if not item.is_file() or not isinstance(entry, dict):
                fail(f"BUNDLE.json entry is invalid: {name}")
                continue
            if entry.get("sha256") != sha256_file(item):
                fail(f"BUNDLE.json hash mismatch: {name}")
            if entry.get("bytes") != item.stat().st_size:
                fail(f"BUNDLE.json size mismatch: {name}")
    if bundle.get("file_count") != len(bundle_files):
        fail("BUNDLE.json file_count does not match its files table")

    ready_path = base / "READY.json"
    if ready_path.is_file():
        try:
            ready = read_json(ready_path)
        except BridgeError as error:
            fail(str(error))
        else:
            if not isinstance(ready, dict) or ready.get("bundle_id") != bundle_id:
                fail("READY.json bundle_id does not match")
            elif ready.get("manifest_sha256") != bundle.get("manifest_sha256"):
                fail("READY.json manifest_sha256 does not match BUNDLE.json")
            elif ready.get("csv_sha256") != bundle.get("csv_sha256"):
                fail("READY.json csv_sha256 does not match BUNDLE.json")
            else:
                report["checks"]["ready_marker"] = True

    report["bundle_id"] = bundle_id
    report["checks"].update(
        {
            "files": len(actual),
            "checksum_entries": len(sums),
            "manifest_sha256": bundle.get("manifest_sha256"),
            "csv_sha256": bundle.get("csv_sha256"),
        }
    )
    report["ok"] = not report["errors"]
    return report


def finalize_bundle(
    staging: str | Path,
    final: str | Path,
    *,
    bundle_id: str | None = None,
    intent: str | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Write BUNDLE.json/SHA256SUMS, verify, then atomically rename staging to final."""

    source = Path(staging)
    destination = Path(final)
    if not source.is_dir():
        raise BridgeError(f"staging directory does not exist: {source}")
    if destination.exists() and not replace:
        raise BridgeError(f"destination already exists (use --replace only after review): {destination}")
    bundle = build_bundle_record(source, bundle_id=bundle_id, intent=intent)
    write_json_atomic(source / "BUNDLE.json", bundle)
    write_sha256sums(source)
    report = verify_bundle_integrity(source)
    if not report["ok"]:
        raise BridgeError("local bundle verification failed: " + "; ".join(report["errors"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not replace:
            raise BridgeError(f"refusing to replace existing bundle: {destination}")
        backup_root = destination.parent / ".replaced"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"{destination.name}.replaced-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        os.replace(destination, backup)
        report["replaced_previous"] = str(backup)
    os.replace(source, destination)
    report["bundle_dir"] = str(destination.resolve())
    report["renamed_from"] = str(source.resolve())
    return report


def _make_bundle_staging(out_root: str | Path, bundle_id: str) -> Path:
    root = Path(out_root)
    staging_root = root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{bundle_id}.partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=False, exist_ok=False)
    return staging


def _task_record(
    *,
    bundle_id: str,
    intent: str,
    split: str,
    joints: Sequence[str],
    tolerances: Mapping[str, float],
    requested_actions: Sequence[str],
    notes: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": BRIDGE_PROTOCOL,
        "bundle_id": bundle_id,
        "intent": intent,
        "created_at_utc": created_at_utc or utc_now(),
        "split": split,
        "joints": list(joints),
        "requested_actions": list(requested_actions),
        "tolerances": {key: float(value) for key, value in tolerances.items()},
        "remote": {
            "read_only_experiment": True,
            "result_dir": f"real2sim/results/{bundle_id}",
            "staging_dir": f"real2sim/results/.staging/{bundle_id}.partial",
            "publish_with": "real2sim/bridge/remote_publish_result.py",
            "ready_marker": "RESULT_READY.json",
            "next_data_request": "next_data_request.json",
        },
        "result_contract": {
            "required_files": ["RESULT.json", "report.md"],
            "allowed_status": ["completed", "partial", "blocked", "failed", "acknowledged"],
            "claims_scope": "curve_fit_candidate_only",
            "next_data_request": {
                "path": "next_data_request.json",
                "required_fields": [
                    "schema_version",
                    "protocol",
                    "bundle_id",
                    "status",
                    "reason",
                    "experiments",
                ],
                "experiment_fields": [
                    "id",
                    "joints",
                    "repeats",
                    "purpose",
                ],
                "execution_modes": {
                    "separate": {
                        "default": True,
                        "required_fields": [
                            "profile",
                            "center_deg or center_rad",
                            "amplitude_deg or amplitude_rad",
                            "speed_cap_deg_s or speed_limit_rad_s",
                            "duration_s",
                        ],
                        "intent_fields": ["directions", "dwell_s"],
                        "semantics": "One local run per joint; profile stages determine executed direction and dwell.",
                    },
                    "paired": {
                        "required_fields": ["execution_mode", "pair_plan"],
                        "pair_plan_schema": "real2sim-pair-plan-v1",
                        "joint_order": ["MFJ2", "MFJ3"],
                        "semantics": (
                            "Both joint targets and independent speed caps are encoded in each point; "
                            "the hardware host validates and executes the plan."
                        ),
                    },
                },
                "semantic_request": {
                    "optional_fields": ["requested_conditions", "local_preparation_required"],
                    "local_preparation_required": True,
                    "semantics": (
                        "If a full trajectory is unavailable, describe required observations and conditions. "
                        "The local agent must prepare and validate an executable request before plan-requests; "
                        "the remote agent does not need embedding_ws."
                    ),
                },
            },
        },
        "notes": notes,
    }


def _readme_text(task: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    return f"""# real2sim data bundle `{task.get("bundle_id")}`

- intent: `{task.get("intent")}`
- split: `{manifest.get("split")}`
- joints: {", ".join(manifest.get("joints") or [])}
- dataset_id: `{manifest.get("dataset_id")}`
- protocol: `{BRIDGE_PROTOCOL}`

This directory was staged, checksum-verified, and atomically renamed by the
real2sim bridge.  Do not edit files in place after `READY.json` exists: create a
new bundle instead.

Remote work must read this directory only.  Write results under
`real2sim/results/.staging/<bundle_id>.partial/`, then publish with
`real2sim/bridge/remote_publish_result.py`.
"""


def write_bundle_files(
    staging: str | Path,
    *,
    manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    csv_text: str,
    readme_text: str | None = None,
) -> None:
    """Write the canonical manifest/TASK/CSV files into a staging directory."""

    base = Path(staging)
    base.mkdir(parents=True, exist_ok=True)
    write_text_atomic(base / "capture.csv", csv_text)
    # Keep recording.sha256 in sync with the freshly written CSV.
    manifest = json.loads(json.dumps(manifest))
    manifest.setdefault("recording", {})["csv"] = "capture.csv"
    manifest["recording"]["sha256"] = sha256_file(base / "capture.csv")
    write_json_atomic(base / "manifest.json", manifest)
    write_json_atomic(base / "TASK.json", task)
    write_text_atomic(base / "README.md", readme_text or _readme_text(task, manifest))


def copy_raw_artifacts(capture_dir: str | Path, staging: str | Path) -> dict[str, dict[str, Any]]:
    """Copy the supported raw capture files beside the canonical CSV."""

    source = Path(capture_dir)
    destination = Path(staging) / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        "samples.csv",
        "commands.csv",
        "boards.csv",
        "board_samples_aligned.csv",
        "clock_fit.json",
        "joint_state_quality.json",
        "metadata.json",
        "command_plan.csv",
    )
    table: dict[str, dict[str, Any]] = {}
    for name in names:
        item = source / name
        if not item.is_file():
            continue
        target = destination / name
        shutil.copy2(item, target)
        table[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
    return table


# ── capture -> canonical CSV ──


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    except OSError as error:
        raise BridgeError(f"cannot read {path}: {error}") from error


def _number(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _unwrap(values: Sequence[float]) -> list[float]:
    """Unwrap a radian series to a continuous curve (identity for already-continuous data)."""

    result: list[float] = []
    offset = 0.0
    previous: float | None = None
    for value in values:
        if previous is not None:
            delta = value - previous
            if delta > math.pi:
                offset -= 2.0 * math.pi
            elif delta < -math.pi:
                offset += 2.0 * math.pi
        result.append(value + offset)
        previous = value
    return result


def _interp(
    targets: Sequence[float],
    xs: Sequence[float],
    ys: Sequence[float | None],
    max_gap: float,
) -> list[float | None]:
    """Linear interpolation with an explicit maximum neighbour gap; no extrapolation."""

    if not xs:
        return [None] * len(targets)
    result: list[float | None] = []
    for target in targets:
        if target < xs[0] or target > xs[-1]:
            result.append(None)
            continue
        right = bisect.bisect_left(xs, target)
        if right < len(xs) and xs[right] == target:
            result.append(ys[right])
            continue
        left = right - 1
        if left < 0 or right >= len(xs):
            result.append(None)
            continue
        span = xs[right] - xs[left]
        if span <= 0 or span > max_gap:
            result.append(None)
            continue
        fraction = (target - xs[left]) / span
        left_value, right_value = ys[left], ys[right]
        if left_value is None or right_value is None:
            result.append(None)
            continue
        result.append(left_value + fraction * (right_value - left_value))
    return result


def _series_from_rows(
    rows: Sequence[Mapping[str, str]],
    time_column: str,
    value_column: str,
) -> list[tuple[float, float]]:
    series: list[tuple[float, float]] = []
    for row in rows:
        stamp = _number(row.get(time_column))
        value = _number(row.get(value_column))
        if stamp is not None and value is not None:
            series.append((stamp, value))
    series.sort(key=lambda item: item[0])
    # Enforce strict time order for interpolation inputs; keep the last duplicate.
    deduped: list[tuple[float, float]] = []
    for stamp, value in series:
        if deduped and stamp <= deduped[-1][0]:
            deduped[-1] = (stamp, value)
        else:
            deduped.append((stamp, value))
    return deduped


def _board_series(
    rows: Sequence[Mapping[str, str]],
    *,
    board: str,
    time_column: str,
    unwrap_motor: bool,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    motor: list[tuple[float, float]] = []
    tension: list[tuple[float, float]] = []
    for row in rows:
        if row.get("board") != board:
            continue
        source = (row.get("source") or "sample").strip().lower()
        if source and source != "sample":
            continue
        stamp = _number(row.get(time_column))
        if stamp is None:
            continue
        motor_value = _number(row.get("module_position_rad"))
        if motor_value is None:
            motor_value = _number(row.get("motor_shaft_position_rad"))
        tension_value = _number(row.get("tension_N_from_firmware_scale"))
        if tension_value is None:
            raw = _number(row.get("tension_raw_cN"))
            tension_value = None if raw is None else raw / 100.0
        if motor_value is not None:
            motor.append((stamp, motor_value))
        if tension_value is not None:
            tension.append((stamp, tension_value))
    motor.sort(key=lambda item: item[0])
    tension.sort(key=lambda item: item[0])
    if unwrap_motor and motor:
        values = _unwrap([value for _, value in motor])
        motor = [(stamp, value) for (stamp, _), value in zip(motor, values, strict=True)]
    # Max nearest-sample distance for this board, as a measured skew proxy.
    stamps = [stamp for stamp, _ in motor] or [stamp for stamp, _ in tension]
    if len(stamps) < 2:
        skew = 0.0
    else:
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        skew = max(gaps) / 2.0 / 1_000_000_000.0
    return motor, tension, skew


def _load_capture_meta(path: str | Path | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    if path is None:
        meta: dict[str, Any] = {}
    else:
        meta = read_json(path)
        if not isinstance(meta, dict):
            raise BridgeError("capture metadata root must be an object")
    for key, value in overrides.items():
        if value is not None:
            meta[key] = value
    return meta


def _meta_get(meta: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = meta
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _calibration_block(source: str | None, joints: Sequence[str]) -> dict[str, Any]:
    applied = bool(source)
    return {
        "by_joint": {
            joint: {
                "status": "applied" if applied else "unknown",
                "source": source or "not provided by capture metadata",
            }
            for joint in joints
        }
    }


def _manifest_from_meta(
    *,
    bundle_id: str,
    split: str,
    joints: Sequence[str],
    meta: Mapping[str, Any],
    csv_sha256: str,
    source_recording_id: str,
    source_sha256: str,
    max_sensor_skew_s: float,
    sensor_sync_method: str,
    raw_files: Mapping[str, Any],
    data_kind: str,
    command_semantics: str = "send_events",
) -> dict[str, Any]:
    robot = dict(meta.get("robot") or {})
    firmware = dict(meta.get("firmware") or {})
    experiment = dict(meta.get("experiment") or {})
    calibration = dict(meta.get("calibration") or {})
    gates = dict(meta.get("quality_gates") or {})
    motor_cal = dict(calibration.get("motor_position") or {})
    gear_ratios = dict(motor_cal.get("gear_ratio") or {})
    encoder_signs = dict(motor_cal.get("encoder_sign") or {})
    motor_source = motor_cal.get("source")
    for joint in joints:
        gear_ratios.setdefault(joint, 1.0)
        encoder_signs.setdefault(joint, 1)
    if "max_sensor_skew_s" not in gates:
        raise BridgeError(
            "capture metadata must set quality_gates.max_sensor_skew_s "
            "(and max_sensor_skew_source); do not infer a hardware tolerance"
        )
    if not str(gates.get("max_sensor_skew_source") or "").strip():
        raise BridgeError("capture metadata must set quality_gates.max_sensor_skew_source")
    root_pose = experiment.get("root_pose")
    if not isinstance(root_pose, dict):
        raise BridgeError("capture metadata must set experiment.root_pose {position_m[3], quaternion_xyzw[4]}")
    prehistory = experiment.get("prehistory")
    if not isinstance(prehistory, dict):
        raise BridgeError("capture metadata must set experiment.prehistory")
    if not robot.get("id") or not robot.get("hardware_revision"):
        raise BridgeError("capture metadata must set robot.id and robot.hardware_revision")
    if (
        not firmware.get("version")
        or not firmware.get("control_mode")
        or not isinstance(firmware.get("fallback_confirmed"), bool)
    ):
        raise BridgeError("capture metadata must set firmware.version/control_mode/fallback_confirmed")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": bundle_id,
        "split": split,
        "robot": {
            "id": str(robot["id"]),
            "hardware_revision": str(robot["hardware_revision"]),
        },
        "firmware": {
            "version": str(firmware["version"]),
            "control_mode": str(firmware["control_mode"]),
            "fallback_confirmed": bool(firmware["fallback_confirmed"]),
            "fallback_evidence": str(firmware.get("fallback_evidence") or ""),
        },
        "recording": {
            "csv": "capture.csv",
            "sha256": csv_sha256,
            "source_recording_id": source_recording_id,
            "source_sha256": source_sha256,
            "timestamp_column": "timestamp_ns",
            "clock_id": str(meta.get("clock_id") or "CLOCK_MONOTONIC_RAW"),
            "command_semantics": command_semantics,
            "command_time_source": "explicit_send_time",
            "sensor_sync_method": sensor_sync_method,
            "max_sensor_skew_s": float(max_sensor_skew_s),
            "max_sensor_skew_source": str(gates.get("max_sensor_skew_source")),
        },
        "joints": list(joints),
        "signals": {
            "position": {
                "suffix": "position_rad",
                "unit": "rad",
                "calibration": _calibration_block(calibration.get("position_source"), joints),
            },
            "motor_position": {
                "suffix": "motor_position_rad",
                "unit": "rad",
                "semantics": "unwrapped_spool_output_angle",
                "calibration": _calibration_block(motor_source, joints),
                "transforms_by_joint": {
                    joint: {
                        "gear_ratio": float(gear_ratios[joint]),
                        "encoder_sign": int(encoder_signs[joint]),
                        "unwrapped": True,
                        "gear_ratio_applied": True,
                    }
                    for joint in joints
                },
            },
            "tension": {
                "suffix": "tension_N",
                "unit": "N",
                "sensor_location": "motor_side_before_routing",
                "calibration": _calibration_block(calibration.get("tension_source"), joints),
            },
            "commands": {
                "angle_suffix": "cmd_rad",
                "speed_suffix": "cmd_speed_rad_s",
                "third_field_suffix": "cmd_tension_N",
                "third_field_semantics": "fallback_tension_command_not_enforced_limit",
            },
        },
        "experiment": {
            "protocol": str(
                experiment.get("protocol")
                or "H7 A0 10-DOF Fallback; third command field is telemetry, not an enforced tension limit"
            ),
            "root_pose": root_pose,
            "prehistory": prehistory,
            "load_condition": str(experiment.get("load_condition") or "unloaded"),
            "temperature_c": experiment.get("temperature_c"),
            "notes": str(experiment.get("notes") or ""),
        },
        "quality_gates": {
            "min_samples": int(gates.get("min_samples", 50)),
            "min_duration_s": float(gates.get("min_duration_s", 0.5)),
            "min_channel_coverage": float(gates.get("min_channel_coverage", 0.8)),
            "max_gap_s": float(gates.get("max_gap_s", 0.1)),
            "max_sensor_skew_s": float(gates["max_sensor_skew_s"]),
            "max_sensor_skew_source": str(gates["max_sensor_skew_source"]),
        },
        "provenance": {
            "data_kind": data_kind,
            "bridge_version": BRIDGE_VERSION,
            "generated_at_utc": utc_now(),
            "raw_files": dict(raw_files),
            "tool": "botyard_real2sim.bridge.export_capture",
        },
    }


def export_capture(
    capture_dir: str | Path,
    *,
    meta_path: str | Path | None,
    out_root: str | Path,
    bundle_id: str,
    joints: Sequence[str] | None = None,
    split: str = "train",
    intent: str = "calibration",
    tolerances: Mapping[str, float] | None = None,
    replace: bool = False,
    max_interpolation_gap_s: float = 0.05,
    unwrap_motor: bool = True,
    third_field_scale_n_per_unit: float = DEFAULT_THIRD_FIELD_SCALE_N_PER_UNIT,
    notes: str = "",
    data_kind: str = "hardware",
) -> dict[str, Any]:
    """Convert legacy host-aligned capture tables into a bridge bundle.

    Required raw files: ``samples.csv``, ``commands.csv``,
    ``board_samples_aligned.csv`` and a separate capture metadata file.
    Board timestamps must already be in host ``CLOCK_MONOTONIC_RAW``
    nanoseconds. The source's alignment producer, clock mapping and sensor-age
    evidence require independent qualification; this exporter does not infer
    or verify them. Current SWD RAM recordings require a separate adapter.
    The canonical grid combines host samples and command events, with
    ``max_interpolation_gap_s`` passed to the interpolation helper. Raw source
    files are kept under ``raw/`` and their hashes are recorded.
    """

    ensure_bundle_id(bundle_id)
    source = Path(capture_dir).resolve()
    for required in ("samples.csv", "commands.csv", "board_samples_aligned.csv"):
        if not (source / required).is_file():
            raise BridgeError(f"capture is missing {required}: {source / required}")
    meta = _load_capture_meta(meta_path, {})
    if not isinstance(tolerances, Mapping) or not tolerances:
        raise BridgeError("explicit tolerances are required (angle_deg, motor_angle_deg, tension_N)")
    for key in ("angle_deg", "motor_angle_deg", "tension_N"):
        if key not in tolerances or not math.isfinite(float(tolerances[key])) or float(tolerances[key]) <= 0:
            raise BridgeError(f"tolerance {key} must be a positive finite number")
    sample_rows = _read_rows(source / "samples.csv")
    command_rows = _read_rows(source / "commands.csv")
    board_rows = _read_rows(source / "board_samples_aligned.csv")
    if not sample_rows:
        raise BridgeError("samples.csv has no rows")
    if not command_rows:
        raise BridgeError("commands.csv has no rows")

    if joints is None:
        joint_list = sorted(
            {
                str(row.get("joint")).strip()
                for row in command_rows
                if str(row.get("joint") or "").strip() in BOARD_FOR_JOINT
            }
        )
        if not joint_list:
            raise BridgeError("commands.csv has no events for a supported capture joint (MFJ2/MFJ3)")
    else:
        joint_list = [str(item) for item in joints]
    unknown = [item for item in joint_list if item not in BOARD_FOR_JOINT]
    if unknown:
        raise BridgeError(f"unsupported capture joints: {', '.join(unknown)}")
    absent = sorted(set(joint_list) - {str(row.get("joint") or "").strip() for row in command_rows})
    if absent:
        raise BridgeError(
            "commands.csv has no command events for requested joint(s): "
            + ", ".join(absent)
            + "; build one bundle per joint or capture both joints in one run"
        )
    if split not in {"train", "validation", "test"}:
        raise BridgeError("split must be train, validation, or test")

    sample_series: dict[str, dict[str, list[tuple[float, float]]]] = {}
    board_skews: list[float] = []
    for joint in joint_list:
        position_series = _series_from_rows(sample_rows, "host_receive_monotonic_raw_ns", f"{joint}/position_rad")
        velocity_series = _series_from_rows(sample_rows, "host_receive_monotonic_raw_ns", f"{joint}/velocity_rad_s")
        motor_series, tension_series, skew = _board_series(
            board_rows,
            board=BOARD_FOR_JOINT[joint],
            time_column="timestamp_ns",
            unwrap_motor=unwrap_motor,
        )
        motor_cal = _meta_get(meta, "calibration.motor_position", {}) or {}
        angle_basis = str(motor_cal.get("angle_basis") or "output_shaft")
        gear_ratio = float((motor_cal.get("gear_ratio") or {}).get(joint, 1.0))
        encoder_sign = int((motor_cal.get("encoder_sign") or {}).get(joint, 1))
        if angle_basis == "motor_shaft":
            if gear_ratio <= 0 or encoder_sign not in (-1, 1):
                raise BridgeError(f"{joint}: motor_shaft basis requires gear_ratio>0 and encoder_sign=±1")
            motor_series = [(stamp, encoder_sign * value / gear_ratio) for stamp, value in motor_series]
        elif angle_basis != "output_shaft":
            raise BridgeError(
                f"{joint}: calibration.motor_position.angle_basis must be 'output_shaft' or 'motor_shaft'"
            )
        if not motor_series:
            raise BridgeError(f"board_samples_aligned.csv has no motor samples for {joint} ({BOARD_FOR_JOINT[joint]})")
        if not tension_series:
            raise BridgeError(
                f"board_samples_aligned.csv has no tension samples for {joint} ({BOARD_FOR_JOINT[joint]})"
            )
        board_skews.append(skew)
        sample_series[joint] = {
            "position": position_series,
            "velocity": velocity_series,
            "motor": motor_series,
            "tension": tension_series,
        }

    # Commands: one ordered event per CSV row; keep only requested joints.
    events: list[dict[str, Any]] = []
    max_adjustment_ns = 0
    for row in command_rows:
        joint = (row.get("joint") or "").strip()
        if joint not in joint_list:
            continue
        stamp = _number(row.get("host_publish_monotonic_raw_ns"))
        angle = _number(row.get("position_rad"))
        speed = _number(row.get("speed_limit_rad_s"))
        effort = _number(row.get("effort_raw"))
        if stamp is None or angle is None or speed is None or effort is None:
            continue
        if effort < 0:
            raise BridgeError(
                "commands.csv contains a negative effort_raw third field; the canonical CSV contract "
                "requires a non-negative cmd_tension_N. Record the actual Fallback tension command or "
                "extend the contract explicitly instead of taking an absolute value silently."
            )
        events.append(
            {
                "timestamp_ns": int(round(stamp)),
                "joint": joint,
                "angle_rad": angle,
                "speed_rad_s": speed,
                "third_raw": effort,
                "third_n": effort * third_field_scale_n_per_unit,
                "sequence": _number(row.get("publish_attempt_seq")) or _number(row.get("plan_index")) or 0,
            }
        )
    if not events:
        raise BridgeError("commands.csv has no usable command events for the selected joints")
    events.sort(key=lambda item: (item["timestamp_ns"], item["sequence"]))
    # Merge events with identical timestamps into a single row when they target
    # different joints; same-joint duplicates keep the last one.
    event_by_time: dict[int, dict[str, Any]] = {}
    collisions = 0
    for event in events:
        stamp = event["timestamp_ns"]
        existing = event_by_time.get(stamp)
        if existing is None:
            event_by_time[stamp] = event
            continue
        collisions += 1
        # Move the later same-time event by 1 ns so the CSV stays strictly ordered.
        adjusted = stamp + 1
        while adjusted in event_by_time:
            adjusted += 1
        event["timestamp_ns"] = adjusted
        max_adjustment_ns = max(max_adjustment_ns, adjusted - stamp)
        event_by_time[adjusted] = event

    sample_times = sorted({int(round(stamp)) for series in sample_series.values() for stamp, _ in series["position"]})
    if len(sample_times) < 2:
        raise BridgeError("samples.csv has fewer than two finite position samples")
    grid = sorted(set(sample_times) | set(event_by_time))

    # Interpolate feedback onto any command-only grid times, bounded by gap.
    interpolated: dict[str, dict[str, list[float | None]]] = {}
    for joint, series in sample_series.items():
        interpolated[joint] = {
            key: _interp(
                grid,
                [stamp for stamp, _ in series[key]],
                [value for _, value in series[key]],
                max_interpolation_gap_s,
            )
            for key in ("position", "velocity", "motor", "tension")
        }
    columns = [
        "timestamp_ns",
        "command_event",
        "command_seq",
        "command_timestamp_ns",
        "is_fallback",
    ]
    for joint in joint_list:
        columns.extend(
            [
                f"{joint}/position_rad",
                f"{joint}/velocity_rad_s",
                f"{joint}/motor_position_rad",
                f"{joint}/motor_velocity_rad_s",
                f"{joint}/tension_N",
                f"{joint}/cmd_rad",
                f"{joint}/cmd_speed_rad_s",
                f"{joint}/cmd_tension_N",
                f"{joint}/command_seq",
            ]
        )
    fallback_confirmed = bool(_meta_get(meta, "firmware.fallback_confirmed", False))
    fallback_text = "1" if fallback_confirmed else "0"

    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for index, stamp in enumerate(grid):
        row: dict[str, Any] = {name: "" for name in columns}
        row["timestamp_ns"] = stamp
        row["is_fallback"] = fallback_text
        event = event_by_time.get(stamp)
        if event is not None:
            row["command_event"] = "1"
            row["command_seq"] = int(event["sequence"])
            row["command_timestamp_ns"] = stamp
        else:
            row["command_event"] = "0"
        channel_columns = {
            "position": "position_rad",
            "velocity": "velocity_rad_s",
            "motor": "motor_position_rad",
            "tension": "tension_N",
        }
        for joint in joint_list:
            values = interpolated[joint]
            for key, suffix in channel_columns.items():
                value = values[key][index]
                if value is not None:
                    row[f"{joint}/{suffix}"] = f"{value:.12g}"
        if event is not None:
            joint = event["joint"]
            row[f"{joint}/cmd_rad"] = f"{event['angle_rad']:.12g}"
            row[f"{joint}/cmd_speed_rad_s"] = f"{event['speed_rad_s']:.12g}"
            row[f"{joint}/cmd_tension_N"] = f"{event['third_n']:.12g}"
            row[f"{joint}/command_seq"] = int(event["sequence"])
        writer.writerow(row)
    csv_text = buffer.getvalue()

    out_root_path = Path(out_root)
    staging = _make_bundle_staging(out_root_path, bundle_id)
    write_text_atomic(staging / "capture.csv", csv_text)
    csv_sha = sha256_file(staging / "capture.csv")
    raw_files = copy_raw_artifacts(source, staging)
    raw_order = [source / name for name in ("samples.csv", "commands.csv", "board_samples_aligned.csv")]
    source_sha = combined_sha256(raw_order)
    max_skew = max(board_skews) if board_skews else 0.0
    sensor_sync_method = (
        "Legacy host-aligned input: board_samples_aligned.csv timestamps are supplied in host "
        "CLOCK_MONOTONIC_RAW; the source alignment producer is not inferred or verified by this "
        "exporter. Source timing evidence declared by capture metadata: "
        f"{_meta_get(meta, 'quality_gates.max_sensor_skew_source', 'unspecified')}. "
        "Canonical grid is the union of host samples and command events; configured interpolation "
        f"gap is {max_interpolation_gap_s:g}s, no extrapolation. Retained source evidence must "
        "establish clock mapping, transport bias and sensor acquisition age independently."
    )
    source_recording_id = str(
        _meta_get(meta, "source_recording_id")
        or (read_json(source / "metadata.json").get("run_id") if (source / "metadata.json").is_file() else None)
        or source.name
    )
    manifest = _manifest_from_meta(
        bundle_id=bundle_id,
        split=split,
        joints=joint_list,
        meta=meta,
        csv_sha256=csv_sha,
        source_recording_id=source_recording_id,
        source_sha256=source_sha,
        max_sensor_skew_s=max_skew,
        sensor_sync_method=sensor_sync_method,
        raw_files=raw_files,
        data_kind=data_kind,
    )
    manifest["provenance"]["timebase"] = {
        "clock_id": manifest["recording"]["clock_id"],
        "duplicate_command_timestamp_adjustments_ns": max_adjustment_ns,
        "duplicate_command_timestamp_collisions": collisions,
    }
    manifest["provenance"]["third_field"] = {
        "source": "commands.csv effort_raw",
        "scale_n_per_unit": third_field_scale_n_per_unit,
        "semantics": "telemetry only; Fallback does not enforce it as a tension limit",
    }
    task = _task_record(
        bundle_id=bundle_id,
        intent=intent,
        split=split,
        joints=joint_list,
        tolerances=tolerances,
        requested_actions=(
            (
                ["validate_dataset", "replay_baseline", "evaluate", "diagnose", "fit_if_validation_ready"]
                if split == "train"
                else ["validate_dataset", "replay_baseline", "evaluate", "diagnose"]
            )
            if intent == "calibration"
            else ["validate_dataset", "acknowledge_transport"]
        ),
        notes=notes,
    )
    write_json_atomic(staging / "manifest.json", manifest)
    write_json_atomic(staging / "TASK.json", task)
    write_text_atomic(staging / "README.md", _readme_text(task, manifest))
    report = finalize_bundle(staging, out_root_path / bundle_id, bundle_id=bundle_id, intent=intent, replace=replace)
    report["manifest"] = manifest
    report["task"] = task
    return report


# ── self-test bundle ──


def make_selftest_bundle(
    out_root: str | Path,
    *,
    bundle_id: str | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Create a synthetic, strict-valid bundle for transport verification only."""

    bundle = ensure_bundle_id(bundle_id or auto_bundle_id("selftest"))
    duration = 2.0
    rate = 500
    rows = int(duration * rate) + 1
    joints = ["MFJ2", "MFJ3"]
    columns = ["timestamp_ns", "command_event", "command_seq", "command_timestamp_ns", "is_fallback"]
    for joint in joints:
        columns.extend(
            [
                f"{joint}/position_rad",
                f"{joint}/velocity_rad_s",
                f"{joint}/motor_position_rad",
                f"{joint}/motor_velocity_rad_s",
                f"{joint}/tension_N",
                f"{joint}/cmd_rad",
                f"{joint}/cmd_speed_rad_s",
                f"{joint}/cmd_tension_N",
            ]
        )
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for index in range(rows):
        t = index / rate
        row: dict[str, Any] = {name: "" for name in columns}
        stamp = index * 2_000_000  # 2 ms, host monotonic raw style
        row["timestamp_ns"] = stamp
        row["is_fallback"] = "1"
        command = index % 50 == 0
        if command:
            row["command_event"] = "1"
            row["command_seq"] = index // 50
            row["command_timestamp_ns"] = stamp
        else:
            row["command_event"] = "0"
        for offset, joint in enumerate(joints):
            angle = 0.20 + 0.05 * math.sin(2 * math.pi * 0.5 * t + offset)
            motor = 0.20 + 0.05 * math.sin(2 * math.pi * 0.5 * t + offset) * 4.0
            tension = 2.0 + 0.4 * math.sin(2 * math.pi * 0.5 * t + offset)
            row[f"{joint}/position_rad"] = f"{angle:.9f}"
            row[f"{joint}/velocity_rad_s"] = f"{0.05 * math.pi * math.cos(2 * math.pi * 0.5 * t + offset):.9f}"
            row[f"{joint}/motor_position_rad"] = f"{motor:.9f}"
            row[f"{joint}/motor_velocity_rad_s"] = f"{0.2 * math.pi * math.cos(2 * math.pi * 0.5 * t + offset):.9f}"
            row[f"{joint}/tension_N"] = f"{tension:.9f}"
            if command:
                row[f"{joint}/cmd_rad"] = f"{angle + 0.05:.9f}"
                row[f"{joint}/cmd_speed_rad_s"] = "0.2"
                row[f"{joint}/cmd_tension_N"] = "20.0"
        writer.writerow(row)
    csv_text = buffer.getvalue()
    meta = {
        "robot": {"id": "botyard-s1-selftest", "hardware_revision": "synthetic"},
        "firmware": {
            "version": "selftest-fixture",
            "control_mode": "synthetic_fixture",
            "fallback_confirmed": True,
            "fallback_evidence": "Synthetic bridge self-test fixture; not hardware evidence.",
        },
        "calibration": {
            "position_source": "synthetic identity fixture; not a hardware calibration",
            "tension_source": "synthetic identity fixture; not a hardware calibration",
            "motor_position": {
                "source": "synthetic identity fixture; output-shaft angle",
                "gear_ratio": {joint: 1.0 for joint in joints},
                "encoder_sign": {joint: 1 for joint in joints},
            },
        },
        "experiment": {
            "protocol": "synthetic transport self-test; no hardware",
            "root_pose": {"position_m": [0.0, 0.0, 0.5], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "prehistory": {
                "recorded": True,
                "duration_s": 0.2,
                "initial_state_source": "synthetic fixture initialization; not hardware",
            },
            "load_condition": "unloaded synthetic",
            "temperature_c": 25.0,
            "notes": "TRANSPORT SELF-TEST ONLY. Do not use for fitting or as hardware evidence.",
        },
        "quality_gates": {
            "max_sensor_skew_s": 0.002,
            "max_sensor_skew_source": "synthetic fixture; self-test only",
            "min_samples": 50,
            "min_duration_s": 0.5,
            "min_channel_coverage": 0.8,
            "max_gap_s": 0.1,
        },
        "source_recording_id": "synthetic-selftest",
    }
    out_root_path = Path(out_root)
    staging = _make_bundle_staging(out_root_path, bundle)
    write_text_atomic(staging / "capture.csv", csv_text)
    csv_sha = sha256_file(staging / "capture.csv")
    source_sha = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    manifest = _manifest_from_meta(
        bundle_id=bundle,
        split="test",
        joints=joints,
        meta=meta,
        csv_sha256=csv_sha,
        source_recording_id="synthetic-selftest",
        source_sha256=source_sha,
        max_sensor_skew_s=0.001,
        sensor_sync_method="synthetic fixture; all channels share the same generated clock",
        raw_files={},
        data_kind="synthetic_transport_selftest",
    )
    manifest["provenance"]["notes"] = "Synthetic transport self-test; not hardware evidence; not fit data."
    task = _task_record(
        bundle_id=bundle,
        intent="transport_selftest",
        split="test",
        joints=joints,
        tolerances={"angle_deg": 2.0, "motor_angle_deg": 3.0, "tension_N": 1.0},
        requested_actions=["validate_dataset", "acknowledge_transport"],
        notes=(
            "Transport self-test only. Verify READY/checksums and publish an acknowledgement result; "
            "do not run PhysX or fit."
        ),
    )
    write_json_atomic(staging / "manifest.json", manifest)
    write_json_atomic(staging / "TASK.json", task)
    write_text_atomic(staging / "README.md", _readme_text(task, manifest))
    report = finalize_bundle(
        staging,
        out_root_path / bundle,
        bundle_id=bundle,
        intent="transport_selftest",
        replace=replace,
    )
    report["manifest"] = manifest
    report["task"] = task
    return report


# ── dataset validation (lazy, simulator-free) ──


def validate_dataset_bundle(root: str | Path) -> dict[str, Any]:
    """Run the project's strict manifest/CSV validator on a bundle."""

    from .datasets import validate_dataset

    base = Path(root)
    return validate_dataset(base / "manifest.json", strict=True)


# ── bridge configuration and transport ──


@dataclass(frozen=True)
class BridgeConfig:
    """Connection settings for the local -> remote bridge."""

    remote_host: str = "cyj@10.249.133.227"
    remote_key: Path = Path("~/.ssh/real2sim_bridge_ed25519").expanduser()
    remote_root: Path = Path("/home/cyj/botyard_real2sim")
    local_experiment: Path = Path("/home/botyard/real2sim/experiment")
    dsh_url: str = "http://127.0.0.1:3081"
    dsh_token: str | None = None
    dsh_session: str | None = None
    ssh_options: tuple[str, ...] = (
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
    )

    @property
    def remote_experiment(self) -> str:
        return str(self.remote_root / "real2sim" / "experiment")

    @property
    def remote_results(self) -> str:
        return str(self.remote_root / "real2sim" / "results")

    @property
    def remote_bridge(self) -> str:
        return str(self.remote_root / "real2sim" / "bridge")

    def ssh_args(self) -> list[str]:
        args = [
            "ssh",
            "-i",
            str(self.remote_key),
            "-o",
            "IdentitiesOnly=yes",
            *self.ssh_options,
            self.remote_host,
        ]
        return args

    def rsync_ssh(self) -> str:
        return " ".join(
            shlex.quote(part)
            for part in ["ssh", "-i", str(self.remote_key), "-o", "IdentitiesOnly=yes", *self.ssh_options]
        )


def default_config_path() -> Path:
    return Path("~/.config/botyard-real2sim/bridge.env").expanduser()


def load_config(path: str | Path | None = None) -> BridgeConfig:
    """Load KEY=VALUE settings from env/config; environment wins over the file."""

    values: dict[str, str] = {}
    config_path = Path(path).expanduser() if path else default_config_path()
    if config_path.is_file():
        for raw in config_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in (
        "R2S_REMOTE_HOST",
        "R2S_REMOTE_KEY",
        "R2S_REMOTE_ROOT",
        "R2S_LOCAL_EXPERIMENT",
        "R2S_REMOTE_DSH_URL",
        "R2S_REMOTE_DSH_TOKEN",
        "R2S_REMOTE_SESSION",
    ):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return BridgeConfig(
        remote_host=values.get("R2S_REMOTE_HOST", "cyj@10.249.133.227"),
        remote_key=Path(values.get("R2S_REMOTE_KEY", "~/.ssh/real2sim_bridge_ed25519")).expanduser(),
        remote_root=Path(values.get("R2S_REMOTE_ROOT", "/home/cyj/botyard_real2sim")),
        local_experiment=Path(values.get("R2S_LOCAL_EXPERIMENT", "/home/botyard/real2sim/experiment")).expanduser(),
        dsh_url=values.get("R2S_REMOTE_DSH_URL", "http://127.0.0.1:3081"),
        dsh_token=values.get("R2S_REMOTE_DSH_TOKEN") or None,
        dsh_session=values.get("R2S_REMOTE_SESSION") or None,
    )


def save_config(config: BridgeConfig, path: str | Path | None = None) -> Path:
    """Persist the non-secret settings and, when present, the DSH token (mode 600)."""

    target = Path(path).expanduser() if path else default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# real2sim local -> remote bridge settings",
        f"R2S_REMOTE_HOST={config.remote_host}",
        f"R2S_REMOTE_KEY={config.remote_key}",
        f"R2S_REMOTE_ROOT={config.remote_root}",
        f"R2S_LOCAL_EXPERIMENT={config.local_experiment}",
        f"R2S_REMOTE_DSH_URL={config.dsh_url}",
    ]
    if config.dsh_token:
        lines.append(f"R2S_REMOTE_DSH_TOKEN={config.dsh_token}")
    if config.dsh_session:
        lines.append(f"R2S_REMOTE_SESSION={config.dsh_session}")
    write_text_atomic(target, "\n".join(lines) + "\n", mode=0o600)
    return target


_TRANSIENT_SSH_MARKERS = (
    "connection timed out",
    "banner exchange",
    "connection reset",
    "broken pipe",
    "connection closed",
    "kex_exchange_identification",
)


def _is_transient_ssh_failure(result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode != 255:
        return False
    text = (result.stderr or "").lower()
    return any(marker in text for marker in _TRANSIENT_SSH_MARKERS)


def run_ssh(
    config: BridgeConfig,
    command: str,
    *,
    check: bool = True,
    timeout: float = 120.0,
    retries: int = 2,
) -> subprocess.CompletedProcess[str]:
    """Run one shell command on the remote host over the dedicated key.

    Connection-level failures (banner timeout, reset) are retried a bounded
    number of times.  Callers that are not idempotent, such as the ready
    verifier, pass ``retries=0``.
    """

    args = [*config.ssh_args(), command]
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BridgeError(f"ssh failed: {error}") from error
        if attempt < retries and _is_transient_ssh_failure(result):
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    assert result is not None
    if check and result.returncode != 0:
        raise BridgeError(
            f"remote command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_rsync(
    config: BridgeConfig,
    source: str | Path,
    destination: str,
    *,
    delete: bool = True,
    timeout: float = 600.0,
    retries: int = 1,
) -> subprocess.CompletedProcess[str]:
    """Upload a directory tree to the remote host with checksum comparison."""

    args = [
        "rsync",
        "-a",
        "--checksum",
        "--human-readable",
        "--itemize-changes",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        *(["--delete", "--delete-excluded"] if delete else []),
        "-e",
        config.rsync_ssh(),
        str(Path(source).resolve()) + "/",
        f"{config.remote_host}:{destination.rstrip('/')}/",
    ]
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BridgeError(f"rsync failed: {error}") from error
        if attempt < retries and result.returncode in {12, 255}:
            time.sleep(2.0 * (attempt + 1))
            continue
        break
    assert result is not None
    if result.returncode != 0:
        raise BridgeError(
            f"rsync failed ({result.returncode}) for {source} -> {destination}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_rsync_download(
    config: BridgeConfig,
    remote_source: str,
    destination: str | Path,
    *,
    delete: bool = False,
    timeout: float = 600.0,
    retries: int = 1,
) -> subprocess.CompletedProcess[str]:
    """Download a remote directory tree to a local destination."""

    args = [
        "rsync",
        "-a",
        "--checksum",
        "--human-readable",
        "--itemize-changes",
        *(["--delete"] if delete else []),
        "-e",
        config.rsync_ssh(),
        f"{config.remote_host}:{remote_source.rstrip('/')}/",
        str(Path(destination).resolve()) + "/",
    ]
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BridgeError(f"rsync download failed: {error}") from error
        if attempt < retries and result.returncode in {12, 255}:
            time.sleep(2.0 * (attempt + 1))
            continue
        break
    assert result is not None
    if result.returncode != 0:
        raise BridgeError(
            f"rsync download failed ({result.returncode}) for {remote_source} -> {destination}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def ensure_remote_layout(config: BridgeConfig) -> None:
    """Create the remote experiment/results/staging/bridge directories."""

    command = (
        "set -e; "
        f"mkdir -p {shlex.quote(config.remote_experiment)} "
        f"{shlex.quote(config.remote_experiment + '/.staging')} "
        f"{shlex.quote(config.remote_results)} "
        f"{shlex.quote(config.remote_results + '/.staging')} "
        f"{shlex.quote(config.remote_bridge)} "
        f"{shlex.quote(str(config.remote_root / 'real2sim' / 'logs'))}; "
        f"chmod 775 {shlex.quote(str(config.remote_root / 'real2sim'))} || true"
    )
    run_ssh(config, command)


def sync_bridge_scripts(config: BridgeConfig, source_dir: str | Path) -> None:
    """Upload standalone remote bridge scripts to ``real2sim/bridge``."""

    ensure_remote_layout(config)
    run_rsync(config, source_dir, config.remote_bridge, delete=True)


# ── DSH remote agent client ──


class DshClient:
    """Minimal authenticated client for the remote DeepSeek Harness web API.

    The remote Host is reached through the SSH local-forward (default
    ``127.0.0.1:3081``).  Authentication uses the launch token from its printed
    URL; the token is exchanged for a browser cookie on first use.
    """

    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        import http.cookiejar

        self._cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._cookies))
        self._exchange()

    def _exchange(self) -> None:
        url = f"{self.base_url}/?token={urllib.parse.quote(self.token, safe='')}"
        try:
            with self._opener.open(url, timeout=self.timeout) as response:
                response.read()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise BridgeError(
                    "remote DSH token was rejected; reopen the remote URL printed by `dsh web` and update "
                    "R2S_REMOTE_DSH_TOKEN"
                ) from error
            raise BridgeError(f"remote DSH token exchange failed: HTTP {error.code}") from error
        except OSError as error:
            raise BridgeError(f"cannot reach remote DSH at {self.base_url}: {error}") from error

    def call(self, method: str, args: Mapping[str, Any]) -> Any:
        body = {
            "type": "client-request",
            "rpcId": str(uuid.uuid4()),
            "method": method,
            "payload": {"args": dict(args)},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/{method}",
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise BridgeError(f"remote DSH does not expose {method}") from error
            raise BridgeError(f"remote DSH {method} failed: HTTP {error.code}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise BridgeError(f"remote DSH {method} failed: {error}") from error
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or result.get("ok") is not True:
            detail = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(payload)
            raise BridgeError(f"remote DSH {method} returned an error: {detail}")
        return result.get("value")

    def list_sessions(self) -> list[dict[str, Any]]:
        value = self.call("session/list", {"_request": {}})
        items = value.get("items") if isinstance(value, dict) else None
        return list(items or [])

    def latest_session(self, cwd: str) -> dict[str, Any]:
        candidates = [
            item
            for item in self.list_sessions()
            if item.get("cwd") == cwd and not item.get("parentSessionId") and item.get("origin") != "subagent"
        ]
        if not candidates:
            raise BridgeError(f"no main DSH session found with cwd={cwd}")
        return max(candidates, key=lambda item: float(item.get("updatedAt") or 0))

    def prompt(self, session_id: str, text: str) -> Any:
        request = {
            "requestId": str(uuid.uuid4()),
            "sessionId": session_id,
            "mode": "queue",
            "content": [{"type": "text", "text": text}],
            "clientTimeZone": "Asia/Shanghai",
        }
        return self.call("session/prompt", {"request": request})

    def page(self, session_id: str, *, through_seq: int, max_messages: int = 50) -> dict[str, Any]:
        value = self.call(
            "session/page",
            {
                "request": {
                    "address": {"kind": "session", "sessionId": session_id},
                    "throughSeq": int(through_seq),
                    "maxMessages": int(max_messages),
                }
            },
        )
        return value if isinstance(value, dict) else {}

    def session_running(self, session_id: str) -> bool:
        for item in self.list_sessions():
            if item.get("sessionId") == session_id:
                return bool(item.get("running"))
        return False

    def session_cursor(self, session_id: str) -> int:
        for item in self.list_sessions():
            if item.get("sessionId") == session_id:
                projections = item.get("projections") or {}
                value = projections.get("asOfSeq")
                return int(value) if isinstance(value, (int, float)) else 0
        return 0

    def latest_assistant_text(self, session_id: str, *, since_seq: int | None = None) -> str:
        cursor = self.session_cursor(session_id)
        page = self.page(session_id, through_seq=cursor, max_messages=200)
        texts: list[str] = []
        for record in page.get("records") or []:
            event = record.get("event") if isinstance(record, dict) else None
            if not isinstance(event, dict) or event.get("type") != "assistant/message":
                continue
            if since_seq is not None and int(event.get("seq") or 0) < since_seq:
                continue
            message = (event.get("data") or {}).get("message") or {}
            for block in message.get("content") or []:
                if block.get("type") == "text":
                    texts.append(str(block.get("text") or ""))
        return "".join(texts).strip()


# ── prompt templates ──


def build_remote_task_prompt(
    *,
    bundle_id: str,
    remote_bundle_dir: str,
    remote_result_dir: str,
    intent: str,
    split: str,
    tolerances: Mapping[str, float],
    notes: str = "",
) -> str:
    """Compose the stable instruction sent to the remote DSH session."""

    tolerance_text = ", ".join(f"{key}={value:g}" for key, value in tolerances.items())
    remote_staging_dir = str(PurePosixPath(remote_result_dir).parent / ".staging" / f"{bundle_id}.partial")
    if intent == "transport_selftest":
        body = (
            "这是 **transport_selftest**：只验证数据桥，不要运行 PhysX、不要拟合、"
            "不要修改实验目录。\n"
            "请执行：\n"
            "1. 读取该目录的 READY.json、TASK.json、manifest.json、README.md；\n"
            "2. 复核 READY.json 与 SHA256SUMS（如与本消息不一致，立即停止并报告）；\n"
            f"3. 在 `{remote_staging_dir}/` 写 `RESULT.json`、`report.md`，\n"
            "   内容只需记录 transport 校验结论和 `next_data_request.json`（可为空请求）；\n"
            "4. 用 `real2sim/bridge/remote_publish_result.py` 原子发布；\n"
            "5. 最后用简短中文回复发布路径和校验结论。\n"
        )
    else:
        body = (
            "这是 **calibration** 数据包。请按项目既有标定流程执行：\n"
            "1. 先跑 `./run.sh data --manifest ... --strict`；不通过就停止，"
            "并把 blocking_reasons 写入结果；\n"
            "2. 建立 baseline replay，并用 TASK.json 的显式容差跑 evaluate；\n"
            "3. 只有 TASK.json 要求时才在明确边界内 fit；不得从预设推断容差/边界；\n"
            f"4. 结果写入 `{remote_staging_dir}/`，必须包含 RESULT.json、report.md、\n"
            "   baseline/、evaluation.json、可选 fit/；若数据不足，写 next_data_request.json，\n"
            "   逐项列出需要的关节、幅度、速度、方向、停留、重复次数与目的；\n"
            "5. 用 `real2sim/bridge/remote_publish_result.py` 原子发布；\n"
            "6. 最后回复结论摘要、是否接受、以及下一步需要的数据。\n"
            "如果 split=train：只有对应的独立 validation bundle 已 ready 时才允许 fit；"
            "否则只做 baseline/evaluate/diagnose，并在 next_data_request 中申请独立 validation 录制。\n"
            "如果 split=validation/test：只做评估/对比，不得用它训练或拟合。\n"
            "next_data_request.json 使用 `experiments` 数组，共同字段为 id、joints、repeats、purpose。\n"
            "单关节分开采集使用 execution_mode=separate（默认），另写 profile、center_deg/center_rad、\n"
            "amplitude_deg/amplitude_rad、speed_cap_deg_s/speed_limit_rad_s、duration_s；\n"
            "该模式的 directions/dwell_s 只描述意图，实际阶段由本机profile决定。\n"
            "搭档定姿或同时运动使用 execution_mode=paired、joints=[MFJ2,MFJ3]，\n"
            "并附 real2sim-pair-plan-v1 的完整 pair_plan；两关节目标、独立速度与停留写进points。\n"
            "无法给出完整轨迹时，可用 purpose 和 requested_conditions 描述需要的状态、响应及原因，\n"
            "标记 local_preparation_required=true，由本机agent补齐可执行计划后再调用plan-requests；\n"
            "不要用两个独立run替代要求的同时运动。完整请求契约见项目docs/REAL2SIM_BRIDGE.md第7节。\n"
            "status 用 `needs_more_data` 或 `none`，不要用未定义的字段名。\n"
        )
    return (
        f"[R2S-BRIDGE {BRIDGE_PROTOCOL}] 新数据包已通过远程 ready 校验。\n"
        f"bundle_id: {bundle_id}\n"
        f"intent: {intent}  split: {split}\n"
        f"只读数据目录: {remote_bundle_dir}\n"
        f"最终结果目录（由发布器生成）: {remote_result_dir}\n"
        f"结果暂存目录（agent写入）: {remote_staging_dir}\n"
        f"显式容差: {tolerance_text or '未提供（不得自行推断）'}\n"
        f"备注: {notes or '无'}\n\n"
        f"{body}\n"
        "分工：你在远程运行botyard_real2sim，负责数据复核、仿真回放与曲线对比，"
        "根据实测残差改进模型结构和参数，完成独立验证并返回结果与补采请求。\n"
        "远程无需安装embedding_ws、ROS或访问烧录器；采集计划的生成、验证、烧录及实机执行由本机agent负责。\n"
        "约束：real2sim/experiment/ 输入目录只读，模型、代码和参数按任务修改；"
        "不要改动输入包的 READY.json、SHA256SUMS、manifest.json、capture.csv；"
        "不要用仿真 CSV 冒充实机数据；结果必须通过 remote_publish_result.py 发布。"
    )
