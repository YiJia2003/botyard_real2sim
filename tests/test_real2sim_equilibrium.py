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
    # MFJ3 卸载分支实测点：3.29 deg -> 6.15 N，3.74 deg -> 7.35 N。
    assert equilibrium_tension_n(3.29, "unloaded") == pytest.approx(6.15)
    assert equilibrium_tension_n(3.74, "unloaded") == pytest.approx(7.35)
    assert equilibrium_tension_n(3.5, "unloaded") == pytest.approx(6.15 + (7.35 - 6.15) * (3.5 - 3.29) / (3.74 - 3.29))
    # 超出实测范围时钳到端点，不外推。
    assert equilibrium_tension_n(0.5, "unloaded") == pytest.approx(6.15)
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
