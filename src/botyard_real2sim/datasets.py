"""Manifest and CSV validation for reproducible real2sim data sets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import JOINT_NAMES

SCHEMA_VERSION = 1
_CHANNELS = {
    "position": ("position_rad", "rad"),
    "motor_position": ("motor_position_rad", "rad"),
    "tension": ("tension_N", "N"),
}
_COMMAND_SUFFIXES = {
    "angle_suffix": ("cmd_rad",),
    "speed_suffix": ("cmd_speed_rad_s",),
    "third_field_suffix": ("cmd_tension_N", "cmd_tension_command_N"),
}
_DEFAULT_GATES = {
    "min_samples": 50,
    "min_duration_s": 0.5,
    "min_channel_coverage": 0.8,
    "max_gap_s": 0.1,
}


def _problem(report: dict[str, Any], code: str, message: str, *, strict: bool, fit_blocker: bool = True) -> None:
    item = {"code": code, "message": message}
    if strict and fit_blocker:
        report["errors"].append(item)
    elif fit_blocker:
        report["blocking_reasons"].append(item)
        report["warnings"].append(item)
    else:
        report["warnings"].append(item)


def _read_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read JSON manifest {path}: {exc}"]
    if not isinstance(value, dict):
        return None, ["manifest root must be a JSON object"]
    return value, []


def _validate_manifest_shape(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "dataset_id",
        "split",
        "robot",
        "firmware",
        "recording",
        "joints",
        "signals",
        "experiment",
    )
    for key in required:
        if key not in manifest:
            errors.append(f"manifest is missing required field {key!r}")
    if errors:
        return errors
    if manifest["schema_version"] != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {manifest['schema_version']!r}; expected {SCHEMA_VERSION}")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"].strip():
        errors.append("dataset_id must be a non-empty string")
    if not isinstance(manifest["split"], str) or manifest["split"] not in {"train", "validation", "test"}:
        errors.append("split must be train, validation, or test")
    for section, keys in (
        ("robot", ("id", "hardware_revision")),
        ("firmware", ("version", "control_mode", "fallback_confirmed")),
    ):
        value = manifest.get(section)
        if not isinstance(value, dict):
            errors.append(f"{section} must be an object")
        else:
            for key in keys:
                if key not in value:
                    errors.append(f"{section} is missing required field {key!r}")
            if section == "robot" and any(
                not isinstance(value.get(key), str) or not value[key].strip() for key in keys
            ):
                errors.append("robot.id and robot.hardware_revision must be non-empty strings")
            if section == "firmware":
                if any(
                    not isinstance(value.get(key), str) or not value[key].strip() for key in ("version", "control_mode")
                ):
                    errors.append("firmware.version and firmware.control_mode must be non-empty strings")
                if not isinstance(value.get("fallback_confirmed"), bool):
                    errors.append("firmware.fallback_confirmed must be boolean")
    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        errors.append("recording must be an object")
    else:
        for key in (
            "csv",
            "sha256",
            "source_recording_id",
            "source_sha256",
            "timestamp_column",
            "clock_id",
            "command_semantics",
            "command_time_source",
        ):
            if key not in recording:
                errors.append(f"recording is missing required field {key!r}")
        if recording.get("timestamp_column") not in ("timestamp_ns", "time_s"):
            errors.append("recording.timestamp_column must be timestamp_ns or time_s")
        if recording.get("command_semantics") not in ("send_events", "sampled_state"):
            errors.append("recording.command_semantics must be send_events or sampled_state")
        if recording.get("command_time_source") not in ("explicit_send_time", "row_timestamp"):
            errors.append("recording.command_time_source must be explicit_send_time or row_timestamp")
        for key in ("csv", "clock_id"):
            if not isinstance(recording.get(key), str) or not recording[key].strip():
                errors.append(f"recording.{key} must be a non-empty string")
        for key in ("sha256", "source_sha256"):
            digest = recording.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in digest)
            ):
                errors.append(f"recording.{key} must be 64 hexadecimal characters")
        if not isinstance(recording.get("source_recording_id"), str) or not recording["source_recording_id"].strip():
            errors.append("recording.source_recording_id must be a non-empty string")
        skew = recording.get("max_sensor_skew_s")
        if skew is not None and (
            not isinstance(skew, (int, float)) or isinstance(skew, bool) or not math.isfinite(skew) or skew < 0
        ):
            errors.append("recording.max_sensor_skew_s must be a finite nonnegative number")
    joints = manifest.get("joints")
    if (
        not isinstance(joints, list)
        or not joints
        or any(not isinstance(name, str) for name in joints)
        or len(set(joints)) != len(joints)
    ):
        errors.append("joints must be a non-empty array of unique joint names")
    elif any(name not in JOINT_NAMES for name in joints):
        errors.append(
            f"joints contains unsupported name(s): {', '.join(str(n) for n in joints if n not in JOINT_NAMES)}"
        )
    signals = manifest.get("signals")
    if not isinstance(signals, dict):
        errors.append("signals must be an object")
    else:
        for name in (*_CHANNELS, "commands"):
            if name not in signals or not isinstance(signals.get(name), dict):
                errors.append(f"signals.{name} must be an object")
        for channel, (suffix, unit) in _CHANNELS.items():
            descriptor = signals.get(channel)
            if isinstance(descriptor, dict) and (descriptor.get("suffix") != suffix or descriptor.get("unit") != unit):
                errors.append(f"signals.{channel} must declare suffix {suffix!r} and unit {unit!r}")
            if isinstance(descriptor, dict) and descriptor.get("calibration") is not None:
                calibration = descriptor["calibration"]
                if not isinstance(calibration, dict) or not isinstance(calibration.get("by_joint"), dict):
                    errors.append(f"signals.{channel}.calibration must map by_joint sensor calibration records")
        motor = signals.get("motor_position")
        if isinstance(motor, dict):
            if motor.get("semantics") not in (None, "unwrapped_spool_output_angle"):
                errors.append("signals.motor_position.semantics must be unwrapped_spool_output_angle")
            if motor.get("transforms_by_joint") is not None and not isinstance(motor.get("transforms_by_joint"), dict):
                errors.append("signals.motor_position.transforms_by_joint must be an object")
        tension = signals.get("tension")
        if isinstance(tension, dict) and tension.get("sensor_location") not in (None, "motor_side_before_routing"):
            errors.append("signals.tension.sensor_location must be motor_side_before_routing")
        commands = signals.get("commands")
        if isinstance(commands, dict):
            for key, allowed in _COMMAND_SUFFIXES.items():
                if commands.get(key) not in allowed:
                    errors.append(f"signals.commands.{key} must be one of {allowed}")
            semantics = commands.get("third_field_semantics")
            if semantics != "fallback_tension_command_not_enforced_limit":
                errors.append("signals.commands.third_field_semantics must document Fallback semantics")
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        errors.append("experiment must be an object")
    else:
        for key in ("protocol", "root_pose", "prehistory"):
            if key not in experiment:
                errors.append(f"experiment is missing required field {key!r}")
        pose = experiment.get("root_pose")
        if isinstance(pose, dict) and (
            not isinstance(pose.get("position_m"), list)
            or not isinstance(pose.get("quaternion_xyzw"), list)
            or len(pose["position_m"]) != 3
            or len(pose["quaternion_xyzw"]) != 4
        ):
            errors.append("experiment.root_pose needs position_m[3] and quaternion_xyzw[4]")
        if (
            isinstance(pose, dict)
            and isinstance(pose.get("position_m"), list)
            and isinstance(pose.get("quaternion_xyzw"), list)
        ):
            pose_values = [*pose["position_m"], *pose["quaternion_xyzw"]]
            if any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                for value in pose_values
            ):
                errors.append("experiment.root_pose values must be finite numbers")
            else:
                norm = math.sqrt(sum(value * value for value in pose["quaternion_xyzw"]))
                if abs(norm - 1.0) > 1e-3:
                    errors.append("experiment.root_pose.quaternion_xyzw must be normalized")
        history = experiment.get("prehistory")
        if isinstance(history, dict):
            for key in ("recorded", "duration_s", "initial_state_source"):
                if key not in history:
                    errors.append(f"experiment.prehistory is missing required field {key!r}")
            duration = history.get("duration_s")
            if (
                not isinstance(history.get("recorded"), bool)
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(duration)
                or duration < 0
                or not isinstance(history.get("initial_state_source"), str)
                or not history["initial_state_source"].strip()
            ):
                errors.append("experiment.prehistory fields have invalid types or values")
    if "quality_gates" in manifest and not isinstance(manifest["quality_gates"], dict):
        errors.append("quality_gates must be an object")
    quality_gates = manifest.get("quality_gates")
    if isinstance(quality_gates, dict) and "max_sensor_skew_s" in quality_gates:
        skew_limit = quality_gates["max_sensor_skew_s"]
        if (
            not isinstance(skew_limit, (int, float))
            or isinstance(skew_limit, bool)
            or not math.isfinite(skew_limit)
            or skew_limit < 0
        ):
            errors.append("quality_gates.max_sensor_skew_s must be a finite nonnegative number")
    if isinstance(quality_gates, dict) and "max_sensor_skew_source" in quality_gates:
        skew_source = quality_gates["max_sensor_skew_source"]
        if not isinstance(skew_source, str) or not skew_source.strip():
            errors.append("quality_gates.max_sensor_skew_source must be a non-empty string")
    return errors


def _number(raw: str | None) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _timestamp(raw: str | None, column: str) -> int | float | None:
    if raw is None or not raw.strip():
        return None
    try:
        if column == "timestamp_ns":
            # Preserve integer precision until subtracting the first timestamp.
            return int(raw)
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _sensor_metadata_blockers(manifest: dict[str, Any], report: dict[str, Any], *, strict: bool) -> None:
    firmware = manifest.get("firmware", {})
    if firmware.get("fallback_confirmed") is not True or not firmware.get("fallback_evidence"):
        _problem(report, "fallback_unconfirmed", "Firmware Fallback mode lacks explicit evidence.", strict=strict)
    recording = manifest.get("recording", {})
    if not recording.get("sensor_sync_method"):
        _problem(report, "sync_method_missing", "Sensor clock synchronization method is not documented.", strict=strict)
    max_skew = recording.get("max_sensor_skew_s")
    if not isinstance(max_skew, (int, float)) or isinstance(max_skew, bool) or not math.isfinite(max_skew):
        _problem(report, "sensor_skew_unknown", "Measured maximum sensor clock skew is not documented.", strict=strict)
    else:
        report["metrics"]["max_sensor_skew_s"] = max_skew
    quality_gates = manifest.get("quality_gates") or {}
    skew_limit = quality_gates.get("max_sensor_skew_s")
    skew_source = quality_gates.get("max_sensor_skew_source")
    if (
        not isinstance(skew_limit, (int, float))
        or isinstance(skew_limit, bool)
        or not math.isfinite(skew_limit)
        or skew_limit < 0
    ):
        _problem(
            report,
            "sensor_skew_limit_missing",
            "A maximum acceptable sensor clock skew must be explicitly set in quality_gates.",
            strict=strict,
        )
    else:
        report["metrics"]["max_allowed_sensor_skew_s"] = skew_limit
        if (
            isinstance(max_skew, (int, float))
            and not isinstance(max_skew, bool)
            and math.isfinite(max_skew)
            and max_skew > skew_limit
        ):
            _problem(
                report,
                "sensor_skew_exceeds_limit",
                f"Measured sensor clock skew {max_skew:g}s exceeds the configured limit {skew_limit:g}s.",
                strict=strict,
            )
    if not isinstance(skew_source, str) or not skew_source.strip():
        _problem(
            report,
            "sensor_skew_source_missing",
            "The source for the maximum acceptable sensor clock skew must be documented.",
            strict=strict,
        )
    else:
        report["metrics"]["max_sensor_skew_source"] = skew_source
    if recording.get("command_time_source") != "explicit_send_time":
        _problem(
            report,
            "command_time_sampled",
            "Command times use sampled row timestamps; event-level send timing is unproven.",
            strict=strict,
        )
    if recording.get("command_semantics") != "send_events":
        _problem(
            report,
            "command_semantics_sampled",
            "The recording stores sampled command state rather than explicit send events.",
            strict=strict,
        )
    signals = manifest.get("signals", {})
    joints = manifest.get("joints", [])
    for channel, (suffix, unit) in _CHANNELS.items():
        descriptor = signals.get(channel, {})
        if descriptor.get("suffix") != suffix or descriptor.get("unit") != unit:
            _problem(
                report, f"{channel}_definition", f"signals.{channel} must declare {suffix} in {unit}.", strict=strict
            )
        calibration = descriptor.get("calibration")
        by_joint = calibration.get("by_joint", {}) if isinstance(calibration, dict) else {}
        missing_calibrations = [
            joint
            for joint in joints
            if not isinstance(by_joint.get(joint), dict)
            or by_joint[joint].get("status") != "applied"
            or not by_joint[joint].get("source")
        ]
        if missing_calibrations:
            _problem(
                report,
                f"{channel}_calibration_missing",
                f"Applied calibration provenance for {channel} is missing.",
                strict=strict,
            )
        if channel == "motor_position" and (
            descriptor.get("semantics") != "unwrapped_spool_output_angle"
            or not isinstance(descriptor.get("transforms_by_joint"), dict)
            or any(
                not isinstance(descriptor["transforms_by_joint"].get(joint), dict)
                or not isinstance(descriptor["transforms_by_joint"][joint].get("gear_ratio"), (int, float))
                or descriptor["transforms_by_joint"][joint].get("gear_ratio", 0) <= 0
                or descriptor["transforms_by_joint"][joint].get("encoder_sign") not in (-1, 1)
                or descriptor["transforms_by_joint"][joint].get("unwrapped") is not True
                or descriptor["transforms_by_joint"][joint].get("gear_ratio_applied") is not True
                for joint in joints
            )
        ):
            _problem(
                report,
                "motor_angle_semantics",
                "Motor feedback must be a continuous, gear-ratio-corrected winding-spool output angle.",
                strict=strict,
            )
        if channel == "tension" and descriptor.get("sensor_location") != "motor_side_before_routing":
            _problem(
                report,
                "tension_sensor_location",
                "Tension channel must identify the motor-side sensor location.",
                strict=strict,
            )
    commands = signals.get("commands", {})
    for key, allowed in _COMMAND_SUFFIXES.items():
        if commands.get(key) not in allowed:
            _problem(
                report, f"command_definition_{key}", f"signals.commands.{key} must be one of {allowed}.", strict=strict
            )
    if commands.get("third_field_semantics") != "fallback_tension_command_not_enforced_limit":
        _problem(
            report,
            "command_third_field_semantics",
            "Third command field must be documented as telemetry; Fallback does not enforce it as a tension limit.",
            strict=strict,
        )
    history = manifest.get("experiment", {}).get("prehistory", {})
    if history.get("recorded") is not True or not history.get("initial_state_source"):
        _problem(
            report,
            "prehistory_missing",
            "Recorded loading prehistory and initial-state source are required for fitting.",
            strict=strict,
        )


def validate_dataset(
    manifest_path: str | Path,
    *,
    csv_path: str | Path | None = None,
    strict: bool = False,
    overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Validate a dataset manifest and canonical wide CSV; no simulator is loaded.

    In permissive mode, absent calibration/synchronization evidence and quality
    gate failures are returned as ``blocking_reasons`` so legacy records remain
    inspectable. ``fit_ready`` is still false. ``strict=True`` promotes these
    blockers to errors and rejects the data set for fitting.
    """
    manifest_file = Path(manifest_path).resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest_file),
        "strict": strict,
        "valid": False,
        "data_ready": False,
        "fit_ready": False,
        "evaluation_ready": False,
        "errors": [],
        "warnings": [],
        "blocking_reasons": [],
        "metrics": {},
    }
    manifest, read_errors = _read_manifest(manifest_file)
    report["errors"].extend({"code": "manifest_read", "message": item} for item in read_errors)
    if manifest is None:
        report["status"] = "invalid"
        return report
    report["dataset_id"] = manifest.get("dataset_id")
    report["split"] = manifest.get("split")
    shape_errors = _validate_manifest_shape(manifest)
    report["errors"].extend({"code": "manifest_schema", "message": item} for item in shape_errors)
    if shape_errors:
        report["status"] = "invalid"
        return report
    recording = manifest["recording"]
    data_path = (
        Path(csv_path).resolve() if csv_path is not None else (manifest_file.parent / recording["csv"]).resolve()
    )
    report["csv"] = str(data_path)
    if not data_path.is_file():
        report["errors"].append({"code": "csv_missing", "message": f"CSV file does not exist: {data_path}"})
        report["status"] = "invalid"
        return report
    expected_sha = recording.get("sha256")
    if expected_sha:
        actual_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
        report["metrics"]["csv_sha256"] = actual_sha
        if actual_sha.lower() != expected_sha.lower():
            report["errors"].append(
                {"code": "csv_hash_mismatch", "message": "CSV SHA-256 does not match the manifest."}
            )
            report["status"] = "invalid"
            return report
    else:
        _problem(
            report, "csv_hash_missing", "CSV SHA-256 is absent; the recording is not content-pinned.", strict=strict
        )
    _sensor_metadata_blockers(manifest, report, strict=strict)

    time_column = recording["timestamp_column"]
    joint_names = manifest["joints"]
    signals = manifest["signals"]
    commands = signals["commands"]
    gates = {**_DEFAULT_GATES, **(manifest.get("quality_gates") or {}), **(overrides or {})}
    if not isinstance(gates["min_samples"], int) or gates["min_samples"] < 2:
        report["errors"].append({"code": "quality_gate_invalid", "message": "min_samples must be an integer >= 2"})
        report["status"] = "invalid"
        return report
    for key in ("min_duration_s", "min_channel_coverage", "max_gap_s"):
        if not isinstance(gates[key], (int, float)) or not math.isfinite(float(gates[key])):
            report["errors"].append({"code": "quality_gate_invalid", "message": f"{key} must be finite"})
    if not (0 <= gates["min_channel_coverage"] <= 1) or gates["min_duration_s"] <= 0 or gates["max_gap_s"] <= 0:
        report["errors"].append(
            {
                "code": "quality_gate_invalid",
                "message": "coverage must be in [0, 1]; duration and max gap must be positive",
            }
        )
    if report["errors"]:
        report["status"] = "invalid"
        return report

    channel_metrics: dict[str, dict[str, float | int | None]] = {}
    command_metrics: dict[str, dict[str, int]] = {}
    fallback_count = 0
    fallback_rows = 0
    timestamps: list[float] = []
    value_series: dict[str, list[float | None]] = defaultdict(list)
    command_complete: dict[str, int] = defaultdict(int)
    command_event_times: list[int | float] = []
    ns_origin: int | None = None
    csv_errors: list[dict[str, str]] = []
    try:
        with data_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if time_column not in header:
                csv_errors.append(
                    {"code": "time_column_missing", "message": f"CSV lacks declared time column {time_column!r}"}
                )
            required_columns = {
                f"{joint}/{signals[channel]['suffix']}" for joint in joint_names for channel in _CHANNELS
            }
            command_columns = {
                "angle_suffix": commands.get("angle_suffix"),
                "speed_suffix": commands.get("speed_suffix"),
                "third_field_suffix": commands.get("third_field_suffix"),
            }
            for joint in joint_names:
                required_columns.update(f"{joint}/{suffix}" for suffix in command_columns.values() if suffix)
            missing = sorted(required_columns - set(header))
            if missing:
                csv_errors.append(
                    {"code": "required_columns_missing", "message": f"Missing required columns: {', '.join(missing)}"}
                )
            if "is_fallback" not in header:
                _problem(
                    report, "fallback_column_missing", "CSV has no row-level is_fallback telemetry.", strict=strict
                )
            event_time_column = "command_timestamp_ns" if time_column == "timestamp_ns" else "command_time_s"
            if recording["command_semantics"] == "send_events" and (
                "command_event" not in header or event_time_column not in header
            ):
                _problem(
                    report,
                    "command_event_columns_missing",
                    f"Explicit send-event timing requires command_event and {event_time_column} columns.",
                    strict=strict,
                )
            if not csv_errors:
                previous: int | float | None = None
                for row_number, row in enumerate(reader, start=2):
                    raw_stamp = row.get(time_column)
                    if time_column == "timestamp_ns":
                        try:
                            stamp_int = int(raw_stamp or "")
                        except ValueError:
                            stamp_int = None
                        stamp = stamp_int
                    else:
                        stamp_int = None
                        stamp = _timestamp(raw_stamp, time_column)
                    if stamp is None:
                        csv_errors.append(
                            {"code": "timestamp_invalid", "message": f"Invalid timestamp at CSV row {row_number}"}
                        )
                        continue
                    if previous is not None and stamp <= previous:
                        csv_errors.append(
                            {
                                "code": "timestamps_not_increasing",
                                "message": (
                                    f"Timestamps must strictly increase; row {row_number} is not after its predecessor"
                                ),
                            }
                        )
                    previous = stamp
                    if time_column == "timestamp_ns":
                        ns_origin = stamp_int if ns_origin is None else ns_origin
                        timestamps.append((stamp_int - ns_origin) * 1e-9)
                    else:
                        timestamps.append(float(stamp))
                    for joint in joint_names:
                        for channel in _CHANNELS:
                            column = f"{joint}/{signals[channel]['suffix']}"
                            value_series[f"{joint}/{channel}"].append(_number(row.get(column)))
                        suffixes = [command_columns[part] for part in command_columns]
                        complete = all(_number(row.get(f"{joint}/{suffix}")) is not None for suffix in suffixes)
                        command_complete[joint] += int(complete)
                        if complete and recording["command_semantics"] == "send_events":
                            event_flag = (row.get("command_event") or "").strip().lower()
                            event_time = _timestamp(row.get(event_time_column), event_time_column)
                            if event_flag not in {"1", "1.0", "true"} or event_time is None:
                                _problem(
                                    report,
                                    "command_event_evidence_missing",
                                    f"{joint} has a complete command triple without an explicit event flag "
                                    f"and timestamp at row {row_number}.",
                                    strict=strict,
                                )
                            else:
                                command_event_times.append(event_time)
                    if "is_fallback" in header:
                        raw_flag = (row.get("is_fallback") or "").strip().lower()
                        if raw_flag in {"1", "1.0", "true"}:
                            fallback_count += 1
                        elif raw_flag not in {"0", "0.0", "false", ""}:
                            csv_errors.append(
                                {
                                    "code": "fallback_value_invalid",
                                    "message": f"Invalid is_fallback value at CSV row {row_number}",
                                }
                            )
                        fallback_rows += 1
    except OSError as exc:
        csv_errors.append({"code": "csv_read_error", "message": str(exc)})
    report["errors"].extend(csv_errors)
    if csv_errors:
        report["status"] = "invalid"
        report["metrics"]["rows"] = len(timestamps)
        return report
    if len(timestamps) < 2:
        report["errors"].append({"code": "too_few_rows", "message": "At least two timestamped rows are required."})
        report["status"] = "invalid"
        return report
    duration_s = timestamps[-1] - timestamps[0]
    report["metrics"].update(
        rows=len(timestamps),
        duration_s=duration_s,
        fallback_fraction=(fallback_count / fallback_rows if fallback_rows else None),
        quality_gates=gates,
    )
    if command_event_times:
        command_times_relative = [
            (stamp - ns_origin) * 1e-9 if time_column == "timestamp_ns" else stamp - timestamps[0]
            for stamp in command_event_times
        ]
        if any(t < 0 for t in command_times_relative):
            _problem(
                report,
                "command_event_before_capture",
                "A command event precedes the first feedback timestamp.",
                strict=strict,
            )
        if any(
            later < earlier for earlier, later in zip(command_times_relative, command_times_relative[1:], strict=False)
        ):
            _problem(
                report,
                "command_events_not_ordered",
                "Command event timestamps must be non-decreasing in CSV row order.",
                strict=strict,
            )
        report["metrics"]["command_event_count"] = len(command_event_times)
    if fallback_rows != len(timestamps) or fallback_count != len(timestamps):
        _problem(
            report,
            "fallback_rows_not_confirmed",
            "Every row used for fitting must have is_fallback=true.",
            strict=strict,
        )
    for joint in joint_names:
        command_metrics[joint] = {"complete_triples": command_complete[joint]}
        if command_complete[joint] == 0:
            _problem(
                report, "command_triple_missing", f"{joint} has no complete recorded command triple.", strict=strict
            )
    for key, values in value_series.items():
        finite = [v is not None for v in values]
        count = sum(finite)
        covered = 0.0
        for index in range(1, len(values)):
            dt = timestamps[index] - timestamps[index - 1]
            if dt <= gates["max_gap_s"] and finite[index - 1] and finite[index]:
                covered += dt
        coverage = covered / duration_s if duration_s > 0 else 0.0
        item = {"finite_samples": count, "sample_fraction": count / len(values), "time_coverage": coverage}
        channel_metrics[key] = item
        if count < gates["min_samples"]:
            _problem(
                report,
                "channel_too_few_samples",
                f"{key} has {count} finite samples; requires {gates['min_samples']}.",
                strict=strict,
            )
        if coverage < gates["min_channel_coverage"]:
            _problem(
                report,
                "channel_coverage_low",
                f"{key} covers {coverage:.3f} of the recording; requires {gates['min_channel_coverage']:.3f}.",
                strict=strict,
            )
    if duration_s < gates["min_duration_s"]:
        _problem(
            report,
            "duration_too_short",
            f"Recording lasts {duration_s:.6g}s; requires {gates['min_duration_s']:.6g}s.",
            strict=strict,
        )
    report["metrics"]["channels"] = channel_metrics
    report["metrics"]["commands"] = command_metrics
    report["valid"] = not report["errors"]
    report["data_ready"] = report["valid"] and not report["blocking_reasons"]
    report["fit_ready"] = report["data_ready"] and report["split"] == "train"
    report["evaluation_ready"] = report["data_ready"] and report["split"] in {"validation", "test"}
    report["status"] = (
        "invalid"
        if not report["valid"]
        else "valid_with_blockers"
        if not report["data_ready"]
        else "fit_ready"
        if report["fit_ready"]
        else "evaluation_ready"
    )
    return report


def write_report(report: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
