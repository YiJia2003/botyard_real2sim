import csv
import hashlib
import json

from botyard_real2sim.datasets import validate_dataset


def _write_recording(
    tmp_path,
    *,
    timestamps=(0.0, 0.1, 0.2, 0.3),
    time_column="time_s",
    include_motor=True,
    include_fallback=True,
):
    event_time_column = "command_timestamp_ns" if time_column == "timestamp_ns" else "command_time_s"
    fields = [time_column, "command_event", event_time_column]
    if include_fallback:
        fields.append("is_fallback")
    fields.extend(("THJ1/position_rad", "THJ1/tension_N"))
    if include_motor:
        fields.append("THJ1/motor_position_rad")
    fields.extend(("THJ1/cmd_rad", "THJ1/cmd_speed_rad_s", "THJ1/cmd_tension_N"))
    path = tmp_path / "capture.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, time_s in enumerate(timestamps):
            stamp = 1_800_000_000_000_000_000 + index if time_column == "timestamp_ns" else time_s
            row = {
                time_column: stamp,
                "command_event": "true" if index == 0 else "false",
                event_time_column: stamp if index == 0 else "",
                "THJ1/position_rad": 0.1 * index,
                "THJ1/tension_N": 2.0 + index,
                "THJ1/cmd_rad": 0.0 if index == 0 else "",
                "THJ1/cmd_speed_rad_s": 1.0 if index == 0 else "",
                "THJ1/cmd_tension_N": 20.0 if index == 0 else "",
            }
            if include_fallback:
                row["is_fallback"] = "true"
            if include_motor:
                row["THJ1/motor_position_rad"] = 0.2 * index
            writer.writerow(row)
    return path


def _manifest(csv_path, *, quality_gates=None, split="train"):
    def calibration(source):
        return {"by_joint": {"THJ1": {"status": "applied", "source": source}}}

    return {
        "schema_version": 1,
        "dataset_id": "run-001",
        "split": split,
        "robot": {"id": "hand-1", "hardware_revision": "v3.2"},
        "firmware": {
            "version": "fw-1.0",
            "control_mode": "Fallback_pidControl",
            "fallback_confirmed": True,
            "fallback_evidence": "recorded is_fallback telemetry",
        },
        "recording": {
            "csv": csv_path.name,
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "source_recording_id": "source-run-001",
            "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "timestamp_column": "time_s",
            "clock_id": "monotonic_clock_1",
            "command_semantics": "send_events",
            "command_time_source": "explicit_send_time",
            "sensor_sync_method": "shared monotonic clock",
            "max_sensor_skew_s": 0.001,
        },
        "joints": ["THJ1"],
        "signals": {
            "position": {
                "suffix": "position_rad",
                "unit": "rad",
                "calibration": calibration("encoder calibration cert A"),
            },
            "motor_position": {
                "suffix": "motor_position_rad",
                "unit": "rad",
                "semantics": "unwrapped_spool_output_angle",
                "transforms_by_joint": {
                    "THJ1": {"gear_ratio": 1.0, "encoder_sign": 1, "unwrapped": True, "gear_ratio_applied": True}
                },
                "calibration": calibration("spool encoder calibration"),
            },
            "tension": {
                "suffix": "tension_N",
                "unit": "N",
                "sensor_location": "motor_side_before_routing",
                "calibration": calibration("load cell calibration cert B"),
            },
            "commands": {
                "angle_suffix": "cmd_rad",
                "speed_suffix": "cmd_speed_rad_s",
                "third_field_suffix": "cmd_tension_N",
                "third_field_semantics": "fallback_tension_command_not_enforced_limit",
            },
        },
        "experiment": {
            "protocol": "single finger step",
            "root_pose": {"position_m": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "prehistory": {"recorded": True, "duration_s": 0.2, "initial_state_source": "first feedback row"},
        },
        "quality_gates": {
            "min_samples": 4,
            "min_duration_s": 0.2,
            "min_channel_coverage": 0.8,
            "max_gap_s": 0.11,
            "max_sensor_skew_s": 0.002,
            "max_sensor_skew_source": "sensor timing specification and synchronization test",
            **(quality_gates or {}),
        },
    }


def _write_manifest(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_complete_manifest_and_capture_are_fit_ready(tmp_path):
    capture = _write_recording(tmp_path)
    report = validate_dataset(_write_manifest(tmp_path, _manifest(capture)))

    assert report["valid"] is True
    assert report["fit_ready"] is True
    assert report["status"] == "fit_ready"
    assert report["metrics"]["channels"]["THJ1/motor_position"]["time_coverage"] == 1.0
    assert report["metrics"]["max_sensor_skew_s"] == 0.001
    assert report["metrics"]["max_allowed_sensor_skew_s"] == 0.002
    assert report["metrics"]["max_sensor_skew_source"] == "sensor timing specification and synchronization test"
    json.dumps(report, allow_nan=False)


def test_complete_validation_manifest_is_evaluation_ready_not_fit_ready(tmp_path):
    capture = _write_recording(tmp_path)
    manifest = _manifest(capture, split="validation")

    report = validate_dataset(_write_manifest(tmp_path, manifest), strict=True)

    assert report["valid"] is True
    assert report["data_ready"] is True
    assert report["fit_ready"] is False
    assert report["evaluation_ready"] is True
    assert report["status"] == "evaluation_ready"


def test_validation_split_is_for_evaluation_not_fitting(tmp_path):
    capture = _write_recording(tmp_path)
    manifest = _manifest(capture)
    manifest["split"] = "validation"
    report = validate_dataset(_write_manifest(tmp_path, manifest), strict=True)

    assert report["fit_ready"] is False
    assert report["evaluation_ready"] is True
    assert report["status"] == "evaluation_ready"


def test_missing_required_feedback_column_is_invalid(tmp_path):
    capture = _write_recording(tmp_path, include_motor=False)
    report = validate_dataset(_write_manifest(tmp_path, _manifest(capture)))

    assert report["valid"] is False
    assert report["fit_ready"] is False
    assert any(item["code"] == "required_columns_missing" for item in report["errors"])


def test_legacy_metadata_is_inspectable_but_not_fit_ready_and_strict_rejects(tmp_path):
    capture = _write_recording(tmp_path, include_fallback=False)
    manifest = _manifest(capture)
    manifest["firmware"]["fallback_confirmed"] = False
    manifest["firmware"].pop("fallback_evidence")
    manifest["recording"]["command_time_source"] = "row_timestamp"
    manifest["recording"].pop("sensor_sync_method")
    manifest["signals"]["motor_position"].pop("calibration")
    manifest["signals"]["tension"]["calibration"]["by_joint"]["THJ1"]["status"] = "unknown"
    manifest["experiment"]["prehistory"]["recorded"] = False
    path = _write_manifest(tmp_path, manifest)

    permissive = validate_dataset(path)
    strict = validate_dataset(path, strict=True)

    assert permissive["valid"] is True
    assert permissive["fit_ready"] is False
    assert permissive["blocking_reasons"]
    assert permissive["warnings"]
    assert not permissive["errors"]
    assert strict["valid"] is False
    assert strict["errors"]


def test_non_monotonic_timestamps_are_rejected(tmp_path):
    capture = _write_recording(tmp_path, timestamps=(0.0, 0.1, 0.1, 0.3))
    report = validate_dataset(_write_manifest(tmp_path, _manifest(capture)))

    assert report["valid"] is False
    assert any(item["code"] == "timestamps_not_increasing" for item in report["errors"])


def test_nanosecond_monotonicity_preserves_integer_precision(tmp_path):
    capture = _write_recording(tmp_path, time_column="timestamp_ns")
    manifest = _manifest(capture)
    manifest["recording"]["timestamp_column"] = "timestamp_ns"
    report = validate_dataset(_write_manifest(tmp_path, manifest))

    assert not any(item["code"] == "timestamps_not_increasing" for item in report["errors"])


def test_hash_mismatch_is_rejected(tmp_path):
    capture = _write_recording(tmp_path)
    manifest = _manifest(capture)
    manifest["recording"]["sha256"] = "0" * 64
    report = validate_dataset(_write_manifest(tmp_path, manifest))

    assert report["valid"] is False
    assert any(item["code"] == "csv_hash_mismatch" for item in report["errors"])


def test_missing_source_recording_identity_is_invalid(tmp_path):
    capture = _write_recording(tmp_path)
    manifest = _manifest(capture)
    manifest["recording"].pop("source_recording_id")
    report = validate_dataset(_write_manifest(tmp_path, manifest), strict=True)

    assert report["valid"] is False
    assert any("source_recording_id" in item["message"] for item in report["errors"])


def test_sparse_coverage_blocks_fitting(tmp_path):
    capture = _write_recording(tmp_path, timestamps=(0.0, 0.1, 1.0, 1.1))
    manifest = _manifest(capture, quality_gates={"max_gap_s": 0.2})
    report = validate_dataset(_write_manifest(tmp_path, manifest))

    assert report["valid"] is True
    assert report["fit_ready"] is False
    assert any(item["code"] == "channel_coverage_low" for item in report["blocking_reasons"])


def test_sensor_skew_above_explicit_limit_blocks_fitting(tmp_path):
    capture = _write_recording(tmp_path)
    manifest = _manifest(capture)
    manifest["recording"]["max_sensor_skew_s"] = 0.003
    report = validate_dataset(_write_manifest(tmp_path, manifest), strict=True)

    assert report["valid"] is False
    assert report["fit_ready"] is False
    assert any(item["code"] == "sensor_skew_exceeds_limit" for item in report["errors"])


def test_missing_sensor_skew_limit_blocks_strict_readiness(tmp_path):
    capture = _write_recording(tmp_path)
    manifest = _manifest(capture)
    manifest["quality_gates"].pop("max_sensor_skew_s")
    report = validate_dataset(_write_manifest(tmp_path, manifest), strict=True)

    assert report["valid"] is False
    assert any(item["code"] == "sensor_skew_limit_missing" for item in report["errors"])


def test_missing_sensor_skew_limit_source_blocks_strict_readiness(tmp_path):
    capture = _write_recording(tmp_path)
    manifest = _manifest(capture)
    manifest["quality_gates"].pop("max_sensor_skew_source")
    report = validate_dataset(_write_manifest(tmp_path, manifest), strict=True)

    assert report["valid"] is False
    assert any(item["code"] == "sensor_skew_source_missing" for item in report["errors"])
