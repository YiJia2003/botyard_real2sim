import csv
import json
import subprocess
import sys
from pathlib import Path

from botyard_real2sim import bridge
from botyard_real2sim.datasets import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_capture(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    samples = root / "samples.csv"
    commands = root / "commands.csv"
    boards = root / "board_samples_aligned.csv"
    with samples.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "host_receive_monotonic_raw_ns",
                "MFJ2/position_rad",
                "MFJ2/velocity_rad_s",
                "MFJ3/position_rad",
                "MFJ3/velocity_rad_s",
            )
        )
        for index in range(300):
            stamp = 1_000_000_000 + index * 2_000_000
            angle = 0.2 + 0.05 * (index / 299)
            writer.writerow((stamp, angle, 0.01, angle + 0.02, 0.01))
    with commands.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "host_publish_monotonic_raw_ns",
                "plan_index",
                "publish_attempt_seq",
                "joint",
                "position_rad",
                "speed_limit_rad_s",
                "effort_raw",
            )
        )
        for index in range(10):
            stamp = 1_000_000_000 + index * 40_000_000
            writer.writerow((stamp, index, index, "MFJ2", 0.2 + 0.01 * index, 0.2, 2000))
            writer.writerow((stamp + 1, index, index, "MFJ3", 0.22 + 0.01 * index, 0.2, 2000))
    with boards.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "timestamp_ns",
                "board",
                "source",
                "module_position_rad",
                "tension_N_from_firmware_scale",
            )
        )
        for index in range(700):
            stamp = 1_000_000_000 + index * 1_000_000
            for board in ("j2", "j3"):
                writer.writerow((stamp, board, "sample", 0.8 + 0.001 * index, 2.0 + 0.001 * index))
    meta = {
        "robot": {"id": "botyard-s1-test", "hardware_revision": "rev-test"},
        "firmware": {
            "version": "fw-test",
            "control_mode": "Fallback_pidControl",
            "fallback_confirmed": True,
            "fallback_evidence": "test fixture",
        },
        "calibration": {
            "position_source": "test fixture",
            "tension_source": "test fixture",
            "motor_position": {
                "source": "test fixture output shaft",
                "gear_ratio": {"MFJ2": 1.0, "MFJ3": 1.0},
                "encoder_sign": {"MFJ2": 1, "MFJ3": 1},
            },
        },
        "experiment": {
            "protocol": "test fixture",
            "root_pose": {"position_m": [0.0, 0.0, 0.5], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "prehistory": {
                "recorded": True,
                "duration_s": 0.2,
                "initial_state_source": "test fixture",
            },
        },
        "quality_gates": {
            "max_sensor_skew_s": 0.01,
            "max_sensor_skew_source": "test fixture",
        },
    }
    meta_path = root / "capture_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return {"root": root, "meta": meta_path}


def test_selftest_bundle_is_strict_valid(tmp_path):
    report = bridge.make_selftest_bundle(tmp_path)
    assert report["ok"] is True
    dataset = validate_dataset(Path(report["bundle_dir"]) / "manifest.json", strict=True)
    assert dataset["data_ready"] is True
    assert dataset["evaluation_ready"] is True
    integrity = bridge.verify_bundle_integrity(report["bundle_dir"])
    assert integrity["ok"] is True
    sums = bridge.parse_sha256sums(Path(report["bundle_dir"]) / "SHA256SUMS")
    assert "BUNDLE.json" in sums


def test_integrity_detects_tampering(tmp_path):
    report = bridge.make_selftest_bundle(tmp_path)
    bundle_dir = Path(report["bundle_dir"])
    csv_path = bundle_dir / "capture.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    integrity = bridge.verify_bundle_integrity(bundle_dir)
    assert integrity["ok"] is False
    assert any("SHA256 mismatch" in error for error in integrity["errors"])


def test_export_capture_to_bundle(tmp_path):
    fixture = _write_capture(tmp_path / "capture")
    bundle_id = "r2s-test-capture"
    report = bridge.export_capture(
        fixture["root"],
        meta_path=fixture["meta"],
        out_root=tmp_path / "experiment",
        bundle_id=bundle_id,
        joints=("MFJ2", "MFJ3"),
        split="train",
        tolerances={"angle_deg": 2.0, "motor_angle_deg": 3.0, "tension_N": 1.0},
    )
    bundle_dir = Path(report["bundle_dir"])
    assert bridge.verify_bundle_integrity(bundle_dir)["ok"] is True
    dataset = validate_dataset(bundle_dir / "manifest.json", strict=True)
    assert dataset["data_ready"] is True
    assert dataset["fit_ready"] is True
    with (bundle_dir / "capture.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 200
    assert any(row["command_event"] == "1" for row in rows)
    assert (bundle_dir / "raw" / "samples.csv").is_file()


def test_remote_verify_and_result_publish(tmp_path):
    report = bridge.make_selftest_bundle(tmp_path / "local")
    source = Path(report["bundle_dir"])
    fake_root = tmp_path / "remote"
    (fake_root / "real2sim" / "experiment" / ".staging").mkdir(parents=True)
    (fake_root / "src").symlink_to(PROJECT_ROOT / "src")
    staging = fake_root / "real2sim" / "experiment" / ".staging" / f"{source.name}.partial"
    staging.mkdir()
    for item in source.iterdir():
        if item.is_file():
            (staging / item.name).write_bytes(item.read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "bridge" / "remote_verify.py"),
            "--project-root",
            str(fake_root),
            "--bundle-id",
            source.name,
            "--staging",
            str(staging),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["ok"] is True
    final = fake_root / "real2sim" / "experiment" / source.name
    assert (final / "READY.json").is_file()

    result_staging = fake_root / "real2sim" / "results" / ".staging" / f"{source.name}.partial"
    result_staging.mkdir(parents=True)
    (result_staging / "RESULT.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": bridge.BRIDGE_PROTOCOL,
                "bundle_id": source.name,
                "status": "acknowledged",
                "claims_scope": "curve_fit_candidate_only",
                "summary": "test",
            }
        ),
        encoding="utf-8",
    )
    (result_staging / "report.md").write_text("# test\n", encoding="utf-8")
    published = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "bridge" / "remote_publish_result.py"),
            "--project-root",
            str(fake_root),
            "--bundle-id",
            source.name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert published.returncode == 0, published.stdout + published.stderr
    assert (fake_root / "real2sim" / "results" / source.name / "RESULT_READY.json").is_file()
