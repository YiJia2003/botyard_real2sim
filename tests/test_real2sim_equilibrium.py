"""Checks for the measured equilibrium table and the derived initialization quantities."""

from __future__ import annotations

import pytest

from botyard_real2sim.equilibrium import (
    LOCK_GRADIENT_N,
    equilibrium_tension_n,
    initial_distal_tension_n,
    initial_slip_m,
)


def test_table_interpolates_measured_points_and_never_extrapolates() -> None:
    # MFJ3 卸载分支实测点：2.94 deg -> 6.43 N，3.74 deg -> 7.41 N。
    from botyard_real2sim.equilibrium import FLOOR_REPEATS_N

    floor = sum(FLOOR_REPEATS_N) / len(FLOOR_REPEATS_N)  # 地板两次实测 6.43 / 6.65
    assert equilibrium_tension_n(2.94, "unloaded") == pytest.approx(floor)
    assert equilibrium_tension_n(3.74, "unloaded") == pytest.approx(7.41)
    assert equilibrium_tension_n(3.4, "unloaded") == pytest.approx(
        floor + (7.41 - floor) * (3.4 - 2.94) / (3.74 - 2.94)
    )
    # 超出实测范围时钳到端点，不外推。
    assert equilibrium_tension_n(0.5, "unloaded") == pytest.approx(floor)
    assert equilibrium_tension_n(40.0, "loaded") == pytest.approx(16.09)


def test_branches_are_distinct_at_the_same_angle() -> None:
    # 实测量级：同一角度上行/下行差 6-7 N，这正是“角度→张力”不是单值函数的证据。
    up = equilibrium_tension_n(6.0, "loaded")
    down = equilibrium_tension_n(6.0, "unloaded")
    assert up - down > 6.0


def test_unknown_branch_is_rejected() -> None:
    with pytest.raises(ValueError):
        equilibrium_tension_n(5.0, "sideways")


def test_locked_state_table_matches_the_measured_start_poses() -> None:
    assert equilibrium_tension_n(5.27, "locked") == pytest.approx(11.66)
    assert equilibrium_tension_n(4.61, "locked") == pytest.approx(11.35)


def test_initial_distal_tension_adds_the_measured_gradient() -> None:
    assert initial_distal_tension_n(11.62) == pytest.approx(11.62 + LOCK_GRADIENT_N)
    assert initial_distal_tension_n(7.34) == pytest.approx(7.34 + LOCK_GRADIENT_N)
    # 非负钳位与非法输入。
    assert initial_distal_tension_n(0.0) == pytest.approx(LOCK_GRADIENT_N)
    with pytest.raises(ValueError):
        initial_distal_tension_n(-1.0)


def test_initial_slip_converts_the_gradient_through_routing_stiffness() -> None:
    assert initial_slip_m(11.62, routing_stiffness_n_per_m=1000.0) == pytest.approx(LOCK_GRADIENT_N / 1000.0)
    assert initial_slip_m(11.62, routing_stiffness_n_per_m=2000.0) == pytest.approx(LOCK_GRADIENT_N / 2000.0)
    with pytest.raises(ValueError):
        initial_slip_m(11.62, routing_stiffness_n_per_m=0.0)


def test_low_angle_commands_are_clamped_to_the_measured_floor() -> None:
    """Both 2.5 and 3.0 deg commands settle at the 2.94 deg floor."""
    from botyard_real2sim.equilibrium import JOINT_DEADBAND_DEG, JOINT_FLOOR_DEG, settled_angle_deg

    assert pytest.approx(2.94) == JOINT_FLOOR_DEG
    assert pytest.approx(0.7) == JOINT_DEADBAND_DEG
    # 地板：从下往上给 2.5 / 3.0 都停在 2.94。
    assert settled_angle_deg(2.5, approach="up") == pytest.approx(2.94)
    assert settled_angle_deg(3.0, approach="up") == pytest.approx(2.94)
    # 实测死区：上行 5.0 停在 4.30，下行 5.0→3.0 停在 3.70（实测 3.74）。
    assert settled_angle_deg(5.0, approach="up") == pytest.approx(4.30)
    assert settled_angle_deg(3.0, approach="down") == pytest.approx(3.70)
    # 不给方向时只钳地板，不猜死区。
    assert settled_angle_deg(5.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        settled_angle_deg(5.0, approach="sideways")


def test_table_uses_actual_angles_not_commands() -> None:
    """The measured table carries the authoritative local steady-state numbers."""
    # 指令 2.5 / 3.0 的实际停位张力，以及 5.0 指令的实际停位（上行分支）。
    from botyard_real2sim.equilibrium import FLOOR_REPEATS_N

    floor = sum(FLOOR_REPEATS_N) / len(FLOOR_REPEATS_N)
    assert equilibrium_tension_n(2.94, "unloaded") == pytest.approx(floor)
    assert equilibrium_tension_n(2.94, "loaded") == pytest.approx(13.79)
    assert equilibrium_tension_n(4.30, "loaded") == pytest.approx(13.79)
    # 下行分支：从 5.0 回到 3.0 停在 3.74 / 7.41 N。
    assert equilibrium_tension_n(3.74, "unloaded") == pytest.approx(7.41)


def test_table_based_initialisation_adds_the_measured_gradient() -> None:
    from botyard_real2sim.equilibrium import FLOOR_REPEATS_N, initial_distal_tension_n_from_table

    floor = sum(FLOOR_REPEATS_N) / len(FLOOR_REPEATS_N)
    assert initial_distal_tension_n_from_table(2.94, branch="unloaded") == pytest.approx(floor + LOCK_GRADIENT_N)
    assert initial_distal_tension_n_from_table(2.94, branch="loaded") == pytest.approx(13.79 + LOCK_GRADIENT_N)
