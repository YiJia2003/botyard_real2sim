"""Offline replay preparation validates experiments before Isaac is imported."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

from botyard_real2sim.config import JOINT_NAMES, Real2SimConfig, load_config
from botyard_real2sim.parameters import override_parameters

pytestmark = pytest.mark.unit
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "real2sim_replay.py"


@pytest.fixture(scope="module")
def replay():
    from botyard_real2sim.cli import replay

    return vars(replay)


def parse(replay, tmp_path, *extra):
    return replay["build_parser"]().parse_args(["--joint", "THJ1", "--out", str(tmp_path / "out.csv"), *extra])


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


def recorded_args(replay, tmp_path, path, *extra):
    return replay["build_parser"]().parse_args(["--commands", str(path), "--out", str(tmp_path / "out.csv"), *extra])


def test_export_cli_never_imports_isaac_or_torch(tmp_path):
    destination = tmp_path / "resolved.yaml"
    code = (
        "import runpy,sys; ns=runpy.run_path(sys.argv[1]); "
        "assert ns['main'](['--write-default-config',sys.argv[2],"
        "'--set','tendon.spring_preload_nm.THJ1=0.025','--cold-start'])==0; "
        "assert not any(k.startswith(('isaaclab','isaacsim','torch')) for k in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(destination)], check=True, capture_output=True, text=True
    )
    config = load_config(destination)
    assert config.tendon.spring_preload_nm[0] == 0.025
    assert config.tendon.spring_preload_nm[1] == 0
    assert config.controller.startup_pair_evaluations == 50
    assert config.controller.control_period_s == 0.003


def test_named_joint_finger_and_scalar_overrides_preserve_original():
    original = Real2SimConfig()
    updated = override_parameters(
        original,
        [
            "tendon.cable_stiffness_n_per_m.THJ1=12345",
            "tendon.crossing_moment_arm_m.thumb=-0.005",
            "controller.control_period_s=0.004",
            "curves.title=Calibration run",
        ],
    )
    assert updated.tendon.cable_stiffness_n_per_m[0] == 12345
    assert updated.tendon.cable_stiffness_n_per_m[1:] == original.tendon.cable_stiffness_n_per_m[1:]
    assert updated.tendon.crossing_moment_arm_m[0] == -0.005
    assert updated.controller.control_period_s == 0.004
    assert updated.curves["title"] == "Calibration run"
    assert original.tendon.cable_stiffness_n_per_m[0] == 8000


@pytest.mark.parametrize(
    "assignment",
    [
        "bad",
        "tendon.unknown=1",
        "wrong.field=2",
        "controller.control_period_s.THJ1=2",
        "tendon.cable_stiffness_n_per_m.BAD=2",
        "controller.control_period_s=.nan",
    ],
)
def test_bad_overrides_fail_before_runtime(assignment):
    with pytest.raises(ValueError):
        override_parameters(Real2SimConfig(), [assignment])


def test_recorded_initial_sample_is_earliest_same_row_and_override_is_explicit(replay, tmp_path):
    path = write_csv(
        tmp_path / "capture.csv",
        [
            "timestamp_ns",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "THJ1/position_rad",
            "THJ1/effort_raw",
            "THJ2/position_rad",
            "is_fallback",
        ],
        [
            [100_000_000, 0.6, 1, 20, 0.4, 900, 0.5, 1],
            [0, 0.5, 1, 20, 0.1, 200, "", 1],
        ],
    )
    args = recorded_args(replay, tmp_path, path, "--strict-commands", "--initial-joint-deg", "THJ1=15")
    _, frames, initial, overrides, duration = replay["prepare_experiment"](args)
    assert initial.time_s == 0
    assert initial.positions_rad == {"THJ1": 0.1}
    assert initial.tensions_n == {"THJ1": 2.0}
    assert "THJ2" in initial.missing_positions
    assert frames[1].time_s == pytest.approx(0.1)
    assert duration == pytest.approx(0.6)
    assert overrides == {"THJ1": math.radians(15)}


def test_missing_triple_nonfallback_and_delayed_initial_feedback_rejected(replay, tmp_path):
    missing = write_csv(tmp_path / "missing.csv", ["time_s", "THJ1/cmd_rad"], [[0, 0.5]])
    with pytest.raises(ValueError, match="missing command"):
        replay["prepare_experiment"](recorded_args(replay, tmp_path, missing, "--strict-commands"))
    nonfallback = write_csv(
        tmp_path / "force.csv",
        [
            "time_s",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "is_fallback",
        ],
        [[0, 0.5, 1, 20, 0]],
    )
    with pytest.raises(ValueError, match="non-Fallback"):
        replay["prepare_experiment"](recorded_args(replay, tmp_path, nonfallback))
    delayed = write_csv(
        tmp_path / "delayed.csv",
        [
            "time_s",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "THJ1/position_rad",
        ],
        [[0, 0.5, 1, 20, ""], [0.1, 0.5, 1, 20, 0.1]],
    )
    with pytest.raises(ValueError, match="after capture start"):
        replay["prepare_experiment"](recorded_args(replay, tmp_path, delayed))


@pytest.mark.parametrize(
    "option,value",
    [
        ("--duration", "0"),
        ("--duration", "nan"),
        ("--physics-dt", "nan"),
        ("--physics-dt", "0.01"),
        ("--sample-dt", "0.00001"),
        ("--effort-limit-nm", "inf"),
        ("--armature-scale", "-1"),
        ("--max-speed-deg-s", "-1"),
        ("--tension-command-n", "nan"),
        ("--target-deg", "inf"),
        ("--step-at", "-1"),
        ("--initial-joint-deg", "THJ1=nan"),
        ("--initial-joint-deg", "XX=10"),
    ],
)
def test_finite_range_and_duration_validation(replay, tmp_path, option, value):
    with pytest.raises(ValueError):
        replay["prepare_experiment"](parse(replay, tmp_path, option, value))


def test_duration_must_include_final_event_and_quaternion_must_be_unit(replay, tmp_path):
    with pytest.raises(ValueError, match="final command"):
        replay["prepare_experiment"](parse(replay, tmp_path, "--duration", "0.2"))
    with pytest.raises(ValueError, match="unit quaternion"):
        replay["prepare_experiment"](parse(replay, tmp_path, "--root-xyzw", "0", "0", "0", "2"))


def test_negative_initial_tension_rejected_before_simulator_launch(replay, tmp_path):
    path = write_csv(
        tmp_path / "negative.csv",
        [
            "time_s",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "THJ1/position_rad",
            "THJ1/tension_N",
        ],
        [[0, 0.5, 1, 20, 0.1, -1]],
    )
    with pytest.raises(ValueError, match="negative initial tension"):
        replay["prepare_experiment"](recorded_args(replay, tmp_path, path))


def test_explicit_empty_event_rejected_before_simulator_launch(replay, tmp_path):
    path = write_csv(
        tmp_path / "empty_event.csv",
        [
            "time_s",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "command_event",
            "dof_mask",
        ],
        [[0, 0.5, 1, 20, 1, 1], [0.1, "", "", "", 1, 0]],
    )
    with pytest.raises(ValueError, match="empty"):
        replay["prepare_experiment"](recorded_args(replay, tmp_path, path))


def test_pre_capture_command_timestamp_cannot_shift_initial_state(replay, tmp_path):
    path = write_csv(
        tmp_path / "pre_capture.csv",
        [
            "timestamp_ns",
            "command_timestamp_ns",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "THJ1/position_rad",
            "command_event",
        ],
        [[1_000_000_000, 900_000_000, 0.5, 1, 20, 0.1, 1]],
    )
    with pytest.raises(ValueError, match="predate|precedes capture"):
        replay["prepare_experiment"](recorded_args(replay, tmp_path, path))


def test_logged_row_aligns_requested_applied_effort_with_state(replay):
    import torch

    from botyard_real2sim.commands import CommandFrame, JointCommand
    from botyard_real2sim.runtime import Real2SimRuntime

    runtime = Real2SimRuntime(Real2SimConfig(), dtype=torch.float64)
    q = torch.full((1, 10), 0.1, dtype=torch.float64)
    runtime.reset(q, torch.full_like(q, 4.0))
    runtime.apply_command(CommandFrame(0, {"THJ1": JointCommand(0.5, 1.0, 20)}), q)
    runtime.update_controller(0, q)
    result = runtime.evaluate(q, torch.zeros_like(q))
    applied = result.joint_effort_nm.clamp(-0.01, 0.01)
    row = replay["_row"](0.1, q, torch.zeros_like(q), runtime, result, applied, q[:, [2, 4, 6, 8]], 0.0)
    values = dict(zip(replay["_header"](), row, strict=True))
    assert values["time_s"] == 0.1
    assert values["timestamp_ns"] == 100_000_000
    assert values["THJ1/position_rad"] == 0.1
    assert values["THJ1/requested_effort_Nm"] == result.joint_effort_nm[0, 0].item()
    assert values["THJ1/applied_effort_Nm"] == applied[0, 0].item()
    assert values["THJ1/effort_saturated"] == 1
    assert values["dof_mask"] == 3
    assert len([name for name in JOINT_NAMES if values[f"{name}/active"]]) == 2
    assert values["THJ1/motor_position_rad"] == result.motor_position_rad[0, 0].item()
    assert values["THJ1/motor_position_rad"] != values["THJ1/position_rad"]
    assert values["THJ1/length_balance_error_m"] == 0
    assert values["THJ1/routing_extension_m"] == 0  # Legacy API config disables this element.


def test_dynamic_routing_csv_fields_preserve_independent_tensions_and_state(replay):
    import torch

    from botyard_real2sim.config import ROUTING_DIAGNOSTIC_FIELDS
    from botyard_real2sim.runtime import Real2SimRuntime

    cfg = load_config(SCRIPT.parents[1] / "configs" / "real2sim" / "s1.yaml")
    runtime = Real2SimRuntime(cfg, dtype=torch.float64)
    q = torch.zeros((1, 10), dtype=torch.float64)
    runtime.reset(q)
    runtime.controller_speed_rad_s[:] = torch.arange(1, 11, dtype=q.dtype)
    runtime.advance(0.0001, q, q)
    result = runtime.evaluate(q, q)
    row = replay["_row"](0.0001, q, q, runtime, result, result.joint_effort_nm, q[:, [2, 4, 6, 8]], 0)
    header = replay["_header"]()
    assert len(header) == len(set(header))
    values = dict(zip(header, row, strict=True))
    for index, joint in enumerate(JOINT_NAMES):
        assert values[f"{joint}/tension_N"] == result.proximal_tension_n[0, index].item()
        assert values[f"{joint}/distal_tension_N"] == result.distal_tension_n[0, index].item()
        assert values[f"{joint}/tension_N"] > values[f"{joint}/distal_tension_N"]
        for field in ROUTING_DIAGNOSTIC_FIELDS:
            assert values[f"{joint}/{field}"] == result.routing_state[field][0, index].item()


def test_motor_initial_state_uses_same_row_and_keeps_missing_channels_explicit(replay, tmp_path):
    path = write_csv(
        tmp_path / "motor.csv",
        [
            "time_s",
            "THJ1/position_rad",
            "THJ1/motor_position_rad",
            "THJ1/velocity_rad_s",
            "THJ1/motor_velocity_rad_s",
            "THJ2/motor_position_rad",
        ],
        [[0, 0.1, 1.7, 0.2, 0.6, ""], [0.01, 0.2, 1.8, 0.3, 0.7, 2.0]],
    )
    state = replay["load_initial_state"](path)
    assert state.motor_positions_rad == {"THJ1": 1.7}
    assert state.velocities_rad_s == {"THJ1": 0.2}
    assert state.motor_velocities_rad_s == {"THJ1": 0.6}


def test_nonfallback_feedback_row_is_checked_even_without_new_command(replay, tmp_path):
    path = write_csv(
        tmp_path / "branch_change.csv",
        [
            "time_s",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "command_seq",
            "is_fallback",
        ],
        [[0, 0.5, 1, 20, 1, 1], [0.1, 0.5, 1, 20, 1, 0]],
    )
    with pytest.raises(ValueError, match="non-Fallback"):
        replay["prepare_experiment"](recorded_args(replay, tmp_path, path))
