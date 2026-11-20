import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from botyard_real2sim import bridge
from botyard_real2sim.cli.bridge import _plan_requests
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
    sync_description = report["manifest"]["recording"]["sensor_sync_method"]
    assert "r2s_align.py" not in sync_description
    assert "test fixture" in sync_description


def test_generated_task_separates_remote_requests_from_local_paired_execution(tmp_path):
    fixture = _write_capture(tmp_path / "capture")
    report = bridge.export_capture(
        fixture["root"],
        meta_path=fixture["meta"],
        out_root=tmp_path / "experiment",
        bundle_id="r2s-task-contract",
        joints=("MFJ2", "MFJ3"),
        split="train",
        tolerances={"angle_deg": 2.0, "motor_angle_deg": 3.0, "tension_N": 1.0},
    )
    task = json.loads((Path(report["bundle_dir"]) / "TASK.json").read_text())
    assert task["remote"]["result_dir"] == "real2sim/results/r2s-task-contract"
    assert task["remote"]["staging_dir"] == "real2sim/results/.staging/r2s-task-contract.partial"
    contract = task["result_contract"]["next_data_request"]
    assert "profile" not in contract["experiment_fields"]
    assert set(contract["experiment_fields"]) == {"id", "joints", "repeats", "purpose"}
    separate = contract["execution_modes"]["separate"]
    paired = contract["execution_modes"]["paired"]
    assert "profile" in separate["required_fields"]
    assert "profile" not in paired["required_fields"]
    assert "pair_plan" in paired["required_fields"]
    assert paired["pair_plan_schema"] == "real2sim-pair-plan-v1"
    assert paired["joint_order"] == ["MFJ2", "MFJ3"]
    assert contract["semantic_request"]["local_preparation_required"] is True


@pytest.mark.parametrize("intent", ["transport_selftest", "calibration"])
def test_remote_prompt_uses_concrete_remote_staging_without_requiring_local_workspace(intent):
    prompt = bridge.build_remote_task_prompt(
        bundle_id="r2s-prompt",
        remote_bundle_dir="/srv/sim/real2sim/experiment/r2s-prompt",
        remote_result_dir="/srv/sim/real2sim/results/r2s-prompt",
        intent=intent,
        split="train",
        tolerances={"angle_deg": 2, "motor_angle_deg": 3, "tension_N": 1},
    )
    assert "/srv/sim/real2sim/results/.staging/r2s-prompt.partial" in prompt
    assert "<bundle_id>" not in prompt
    assert "embedding_ws" in prompt and "无需" in prompt
    if intent == "calibration":
        assert "execution_mode=paired" in prompt and "pair_plan" in prompt
        assert "requested_conditions" in prompt
        assert "每项至少包含 id、joints、profile" not in prompt


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


@pytest.mark.parametrize("speed_deg_s", [15.0, 360.0])
@pytest.mark.parametrize("center_deg", [2.5, 90.0])
def test_request_plan_preserves_float_boundaries(tmp_path, speed_deg_s, center_deg):
    request_path = tmp_path / "request.json"
    amplitude = math.nextafter(0.1, 0.0)
    request_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "joints": ["MFJ3"],
                        "profile": "hold",
                        "center_deg": center_deg,
                        "amplitude_rad": amplitude,
                        "speed_cap_deg_s": speed_deg_s,
                        "duration_s": 1.234567890123,
                        "frequency_hz": 0.123456789,
                        "end_frequency_hz": 1.23456789,
                        "rate_hz": 20.123456789,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = _plan_requests(request_path, tmp_path / "plan", r2s_root=tmp_path, effort_raw=500)
    script = Path(result["script"]).read_text(encoding="utf-8")
    command = shlex.split(next(line for line in script.splitlines() if line.startswith("python3 ")))
    assert "--execute" not in command

    def argument(option):
        return float(command[command.index(option) + 1])

    plan = result["plan"][0]
    assert argument("--speed-limit-rad-s") == plan["speed_limit_rad_s"]
    assert math.radians(15.0) <= argument("--speed-limit-rad-s") <= math.radians(360.0)
    assert argument("--center-rad") == plan["center_rad"]
    assert math.radians(2.5) <= argument("--center-rad") <= math.radians(90.0)
    assert argument("--amplitude-rad") == amplitude
    assert argument("--duration-s") == 1.234567890123
    assert argument("--frequency-hz") == 0.123456789
    assert argument("--end-frequency-hz") == 1.23456789
    assert argument("--rate-hz") == 20.123456789


def test_paired_request_preserves_one_atomic_command_and_plan(tmp_path):
    pair_plan = {
        "schema": "real2sim-pair-plan-v1",
        "joint_order": ["MFJ2", "MFJ3"],
        "metadata": {"experiment_id": "joint-grid-1", "family": "joint_grid", "split": "train"},
        "points": [
            {
                "offset_ns": 0,
                "phase": "start",
                "positions_rad": [0.2, 0.3],
                "speed_limits_rad_s": [math.radians(15), math.radians(360)],
            },
            {
                "offset_ns": 1000000000,
                "phase": "end",
                "positions_rad": [0.4, 0.5],
                "speed_limits_rad_s": [math.radians(15), math.radians(360)],
            },
        ],
    }
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "id": "joint-grid-1",
                        "joints": ["MFJ2", "MFJ3"],
                        "execution_mode": "paired",
                        "pair_plan": pair_plan,
                        "repeats": 2,
                    }
                ]
            }
        )
    )
    record = _plan_requests(request, tmp_path / "out", r2s_root=tmp_path / "embedding", effort_raw=500)
    lines = [line for line in Path(record["script"]).read_text().splitlines() if line.startswith("python3 ")]
    assert len(lines) == 2  # one paired message per repeat, never two single-joint commands
    for repeat, line in enumerate(lines, 1):
        command = shlex.split(line)
        assert command[1] == "tools/r2s_pair_run.py"
        assert "--execute" not in command and "--joint" not in command
        assert command[command.index("--effort-raw") + 1] == "500"
        saved = json.loads(Path(command[command.index("--plan") + 1]).read_text())
        assert saved["points"] == pair_plan["points"]
        assert saved["metadata"] == {
            **pair_plan["metadata"],
            "experiment_id": f"joint-grid-1-r{repeat:02d}",
            "repeat": repeat,
            "source_experiment_id": "joint-grid-1",
        }
    assert all(item["execution_mode"] == "paired" for item in record["plan"])
    assert json.loads(Path(record["plan"][0]["source_pair_plan"]).read_text()) == pair_plan


@pytest.mark.parametrize(
    "patch",
    [
        {"execution_mode": "paired"},
        {"execution_mode": "paired", "pair_plan": {"schema": "wrong"}},
        {"execution_mode": "unknown"},
    ],
)
def test_request_never_silently_downgrades_invalid_paired_plan(tmp_path, patch):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"experiments": [{"joints": ["MFJ2", "MFJ3"], **patch}]}))
    with pytest.raises(bridge.BridgeError):
        _plan_requests(request, tmp_path / "out", r2s_root=tmp_path / "embedding", effort_raw=0)


@pytest.mark.parametrize("repeats", [0, -1, 101, True, 1.5, "2"])
def test_paired_request_rejects_ambiguous_repeats(tmp_path, repeats):
    request = json.loads((PROJECT_ROOT / "bridge/next_data_request_pair.example.json").read_text())
    request["experiments"][0]["repeats"] = repeats
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    with pytest.raises(bridge.BridgeError, match="repeats must be an integer"):
        _plan_requests(path, tmp_path / "out", r2s_root=tmp_path, effort_raw=500)


def test_pair_payload_implies_paired_mode_but_forbids_separate_override(tmp_path):
    request = json.loads((PROJECT_ROOT / "bridge/next_data_request_pair.example.json").read_text())
    experiment = request["experiments"][0]
    del experiment["execution_mode"]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    record = _plan_requests(path, tmp_path / "auto", r2s_root=tmp_path, effort_raw=500)
    assert record["plan"][0]["execution_mode"] == "paired"
    experiment["execution_mode"] = "separate"
    path.write_text(json.dumps(request))
    with pytest.raises(bridge.BridgeError, match="cannot be split silently"):
        _plan_requests(path, tmp_path / "separate", r2s_root=tmp_path, effort_raw=500)


@pytest.mark.parametrize("mode", ["separate", "paired"])
def test_semantic_requests_require_local_preparation_before_any_plan_output(tmp_path, mode):
    request = json.loads((PROJECT_ROOT / "bridge/next_data_request_pair.example.json").read_text())
    semantic = {
        "id": "requested-state",
        "joints": ["MFJ2", "MFJ3"],
        "execution_mode": mode,
        "local_preparation_required": True,
        "requested_conditions": {"target_deg": [45, 45], "joint_speed_deg_s": [30, -60]},
        "purpose": "A missing coupled state needs local trajectory design",
    }
    request["experiments"].append(semantic)  # Do not even emit the earlier valid paired payload.
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    out = tmp_path / "plans"
    with pytest.raises(bridge.BridgeError, match="hardware-host agent.*prepare"):
        _plan_requests(path, out, r2s_root=tmp_path / "embedding", effort_raw=500)
    assert not out.exists()


@pytest.mark.parametrize("mode", ["separate", "paired"])
def test_relative_request_output_stays_at_requested_location_after_script_changes_directory(
    tmp_path, monkeypatch, mode
):
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "hardware workspace"
    (workspace / "tools").mkdir(parents=True)
    runner = """import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args, _ = parser.parse_known_args()
args.out.mkdir(parents=True, exist_ok=False)
(args.out / 'location.json').write_text(json.dumps({'cwd': str(Path.cwd()), 'out': str(args.out.resolve())}))
"""
    (workspace / "tools" / ("r2s_pair_run.py" if mode == "paired" else "r2s_run.py")).write_text(runner)
    if mode == "paired":
        request = json.loads((PROJECT_ROOT / "bridge/next_data_request_pair.example.json").read_text())
    else:
        request = {
            "experiments": [
                {"id": "hold", "joints": ["MFJ2"], "profile": "hold", "center_deg": 20, "speed_cap_deg_s": 15}
            ]
        }
    Path("request.json").write_text(json.dumps(request))
    record = _plan_requests(
        Path("request.json"), Path("output plans"), r2s_root=Path("hardware workspace"), effort_raw=500
    )
    other_cwd = tmp_path / "another caller"
    other_cwd.mkdir()
    result = subprocess.run(["bash", str(record["script"])], cwd=other_cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    output = Path(record["plan"][0]["output"])
    assert output.is_absolute() and output.parent == tmp_path / "output plans"
    observed = json.loads((output / "location.json").read_text())
    assert observed == {"cwd": str(workspace), "out": str(output)}


def test_distinct_request_ids_cannot_alias_the_same_output_path(tmp_path):
    request = json.loads((PROJECT_ROOT / "bridge/next_data_request_pair.example.json").read_text())
    first = request["experiments"][0]
    first["id"] = "joint/condition"
    second = json.loads(json.dumps(first))
    second["id"] = "joint condition"
    request["experiments"].append(second)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    out = tmp_path / "plans"
    with pytest.raises(bridge.BridgeError, match="IDs.*collide"):
        _plan_requests(path, out, r2s_root=tmp_path / "embedding", effort_raw=500)
    assert not out.exists()
