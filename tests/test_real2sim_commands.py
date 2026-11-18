"""The recorded-command CSV reader can run without Isaac Sim or torch."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from botyard_real2sim.commands import JointCommand, load_command_frames


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.mark.unit
def test_epoch_nanoseconds_are_sorted_exactly_and_nan_rows_remain(tmp_path: Path) -> None:
    epoch = 1_800_000_000_000_000_000
    path = _write(
        tmp_path / "capture.csv",
        ["timestamp_ns", "THJ1/position_rad", "THJ1/effort_raw", "THJ1/cmd_rad"],
        [
            [epoch + 100_000, 0.2, 999, 0.5],
            [epoch, 0.1, 888, "nan"],
            [epoch + 100_000, 0.2, 777, 0.6],
            [epoch + 100, 0.1, 666, ""],
        ],
    )

    frames = load_command_frames(path)

    assert [frame.time_s for frame in frames] == pytest.approx([0, 1e-7, 1e-4, 1e-4])
    assert [frame.commands for frame in frames[:2]] == [{}, {}]
    assert frames[2].commands == {"THJ1": JointCommand(0.5, 1.5708, 20.0)}
    assert frames[3].commands == {"THJ1": JointCommand(0.6, 1.5708, 20.0)}


@pytest.mark.unit
def test_three_fields_raw_conversion_and_sparse_joints(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "commands.csv",
        [
            "time_s",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_limit_raw",
            "THJ2/cmd_rad",
            "THJ2/cmd_tension_limit_n",
            "THJ2/cmd_tension_limit_raw",
        ],
        [
            [5.25, 0.2, 0.7, 350, "nan", "", ""],
            [5.5, "", "", "", 0.3, 4.5, 900],
            [5.75, 0.4, "nan", "nan", 0.6, "", 150],
        ],
    )

    frames = load_command_frames(path, tension_raw_to_n=0.02)

    assert [frame.time_s for frame in frames] == pytest.approx([0.0, 0.25, 0.5])
    assert frames[0].commands == {"THJ1": JointCommand(0.2, 0.7, 7.0)}
    assert frames[1].commands == {"THJ2": JointCommand(0.3, 1.5708, 4.5)}
    assert frames[2].commands == {
        "THJ1": JointCommand(0.4, 1.5708, 20.0),
        "THJ2": JointCommand(0.6, 1.5708, 3.0),
    }


@pytest.mark.unit
def test_missing_command_columns_and_invalid_values_fail_clearly(tmp_path: Path) -> None:
    only_feedback = _write(tmp_path / "feedback.csv", ["time_s", "THJ1/position_rad", "THJ1/effort_raw"], [[0, 0, 500]])
    with pytest.raises(ValueError, match="cmd_rad"):
        load_command_frames(only_feedback)

    bad_time = _write(tmp_path / "bad_time.csv", ["timestamp_ns", "THJ1/cmd_rad"], [["nan", 1.0]])
    with pytest.raises(ValueError, match="timestamp"):
        load_command_frames(bad_time)

    bad_speed = _write(tmp_path / "bad_speed.csv", ["time_s", "THJ1/cmd_rad", "THJ1/cmd_speed_rad_s"], [[0, 1.0, -2.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        load_command_frames(bad_speed)

    with pytest.raises(ValueError, match="positive"):
        load_command_frames(bad_speed, tension_raw_to_n=0.0)


def test_canonical_and_legacy_tension_names_and_strict_missing(tmp_path):
    path = _write(
        tmp_path / "new.csv",
        ["time_s", "THJ1/cmd_rad", "THJ1/cmd_speed_rad_s", "THJ1/cmd_tension_command_N", "THJ1/cmd_tension_limit_raw"],
        [[0, 0.2, 0.3, 4, 900]],
    )
    cmd = load_command_frames(path, strict=True)[0].commands["THJ1"]
    assert cmd.tension_command_n == cmd.tension_limit_n == 4
    path = _write(tmp_path / "missing.csv", ["time_s", "THJ1/cmd_rad"], [[0, 0]])
    with pytest.raises(ValueError, match="missing command speed"):
        load_command_frames(path, strict=True)
    with pytest.raises(ValueError, match="missing command speed"):
        load_command_frames(path, default_speed_rad_s=None)
    with pytest.warns(UserWarning, match="not measured"):
        load_command_frames(path)


def test_initial_state_never_combines_feedback_from_different_rows(tmp_path):
    from botyard_real2sim.commands import load_initial_state

    path = _write(
        tmp_path / "state.csv",
        ["time_s", "THJ1/position_rad", "THJ1/tension_N", "THJ1/effort_raw", "THJ2/position_rad", "THJ2/effort_raw"],
        [[10, "", "", "", "", ""], [11, 0.2, 2, 999, "", ""], [12, 0.3, 3, 999, 0.4, 400]],
    )
    state = load_initial_state(path, joint_names=("THJ1", "THJ2"))
    assert state.time_s == 1
    assert state.positions_rad == {"THJ1": 0.2}
    assert state.tensions_n == {"THJ1": 2}
    assert state.missing_positions == state.missing_tensions == ("THJ2",)
    empty = _write(tmp_path / "empty.csv", ["time_s", "THJ1/position_rad"], [[0, "nan"]])
    with pytest.raises(ValueError, match="no valid"):
        load_initial_state(empty)


def test_explicit_events_preserve_repeated_sends_and_actual_send_times(tmp_path):
    path = _write(
        tmp_path / "events.csv",
        [
            "timestamp_ns",
            "command_timestamp_ns",
            "command_event",
            "sequence",
            "dof_mask",
            "is_fallback",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
            "THJ2/cmd_rad",
        ],
        [
            [1000000000, "", False, "", "", True, "", "", "", ""],
            [1100000000, 1050000000, True, 1, "0x1", True, 0.2, 0.3, 4, 0.9],
            [1200000000, 1150000000, True, 2, "0x1", False, 0.2, 0.3, 4, 0.9],
        ],
    )
    frames = load_command_frames(path, strict=True)
    assert [f.time_s for f in frames] == pytest.approx([0.05, 0.15])
    assert [f.sequence for f in frames] == [1, 2]
    assert all(f.is_event and f.dof_mask == 1 for f in frames)
    assert frames[0].commands == frames[1].commands
    assert set(frames[0].commands) == {"THJ1"}
    assert frames[0].is_fallback is True and frames[1].is_fallback is False


def test_sampled_sequence_metadata_is_not_repeated_as_new_events(tmp_path):
    path = _write(
        tmp_path / "sampled.csv",
        ["time_s", "sequence", "THJ1/cmd_rad", "THJ1/cmd_speed_rad_s", "THJ1/cmd_tension_N"],
        [[0, 3, 0.2, 0.3, 4], [0.1, 3, 0.2, 0.3, 4], [0.2, 4, 0.2, 0.3, 4]],
    )
    assert len(load_command_frames(path, strict=True)) == 2
    sampled = load_command_frames(path, strict=True, event_mode="sampled")
    assert len(sampled) == 3 and not any(f.is_event for f in sampled)


@pytest.mark.parametrize("field,value", [("dof_mask", "-1"), ("command_event", "yes"), ("THJ1/cmd_rad", "inf")])
def test_invalid_event_metadata_and_nonfinite_values(tmp_path, field, value):
    row = {"time_s": 0, "THJ1/cmd_rad": 0.2, "THJ1/cmd_speed_rad_s": 0.3, "THJ1/cmd_tension_N": 4, field: value}
    path = _write(tmp_path / "bad.csv", list(row), [list(row.values())])
    with pytest.raises(ValueError):
        load_command_frames(path, strict=True)


@pytest.mark.parametrize(
    "time_column,event_column,capture,event",
    [
        ("time_s", "command_time_s", 10.0, 9.5),
        ("timestamp_ns", "command_timestamp_ns", 1_800_000_000_000_000_000, 1_799_999_999_999_999_999),
    ],
)
def test_event_before_capture_requires_precommand_feedback(tmp_path, time_column, event_column, capture, event):
    path = _write(
        tmp_path / "early_event.csv",
        [time_column, event_column, "command_event", "THJ1/cmd_rad", "THJ1/cmd_speed_rad_s", "THJ1/cmd_tension_N"],
        [[capture, event, True, 0.2, 0.3, 4]],
    )
    with pytest.raises(ValueError, match="include pre-command feedback"):
        load_command_frames(path, strict=True)


def test_initial_state_and_delayed_events_share_capture_origin(tmp_path):
    from botyard_real2sim.commands import load_initial_state

    path = _write(
        tmp_path / "shared_origin.csv",
        [
            "time_s",
            "command_time_s",
            "command_event",
            "THJ1/position_rad",
            "THJ1/tension_N",
            "THJ1/cmd_rad",
            "THJ1/cmd_speed_rad_s",
            "THJ1/cmd_tension_N",
        ],
        [[10.1, 10.05, True, 0.1, 2, 0.2, 0.3, 4], [10, "", False, 0, 1, "", "", ""]],
    )
    initial = load_initial_state(path, joint_names=("THJ1",))
    frames = load_command_frames(path, strict=True)
    assert initial.time_s == 0
    assert frames[0].time_s == pytest.approx(0.05)


def test_fallback_validation_checks_duplicate_sequence_feedback(tmp_path):
    from botyard_real2sim.commands import validate_fallback_capture

    path = _write(tmp_path / "branch.csv", ["time_s", "command_seq", "is_fallback"], [[0, 1, True], [0.1, 1, False]])
    with pytest.raises(ValueError, match="row 3.*is_fallback=false"):
        validate_fallback_capture(path)


def test_fallback_validation_perjoint_inactive_exemptions(tmp_path):
    from botyard_real2sim.commands import validate_fallback_capture

    path = _write(
        tmp_path / "branch.csv",
        ["time_s", "dof_mask", "THJ1/is_fallback", "THJ2/is_fallback", "THJ2/active"],
        [[0, "0x1", True, False, ""], [0.1, "0x3", True, False, False]],
    )
    evidence = validate_fallback_capture(path)
    assert evidence["status"] == "validated"
    assert evidence["inactive_flags_ignored"] == 2
    path = _write(tmp_path / "active.csv", ["time_s", "THJ1/is_fallback", "THJ1/active"], [[0, False, True]])
    with pytest.raises(ValueError, match="THJ1/is_fallback=false"):
        validate_fallback_capture(path)
    path = _write(tmp_path / "relevant.csv", ["time_s", "THJ1/is_fallback"], [[0, False]])
    with pytest.raises(ValueError, match="THJ1/is_fallback=false"):
        validate_fallback_capture(path)


def test_fallback_validation_unknown_flags_are_unavailable_and_malformed_rejected(tmp_path):
    from botyard_real2sim.commands import validate_fallback_capture

    path = _write(
        tmp_path / "unknown.csv",
        ["time_s", "is_fallback", "THJ1/is_fallback", "dof_mask"],
        [[0, "", "nan", "nan"], [0.1, "NaN", "", ""]],
    )
    evidence = validate_fallback_capture(path)
    assert evidence["status"] == "unavailable" and evidence["rows_checked"] == 2
    path = _write(tmp_path / "malformed.csv", ["time_s", "THJ1/is_fallback", "THJ1/active"], [[0, "unknown", False]])
    with pytest.raises(ValueError, match="invalid boolean"):
        validate_fallback_capture(path)
    path = _write(tmp_path / "global_false.csv", ["time_s", "is_fallback", "dof_mask"], [[0, False, "0x0"]])
    with pytest.raises(ValueError, match="is_fallback=false"):
        validate_fallback_capture(path)
