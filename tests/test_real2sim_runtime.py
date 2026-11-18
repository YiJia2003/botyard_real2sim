"""Timing and command-event invariants independent of the simulator backend."""

import math

import pytest
import torch

from botyard_real2sim.commands import CommandFrame, JointCommand
from botyard_real2sim.config import Real2SimConfig
from botyard_real2sim.runtime import PeriodicClock, Real2SimRuntime


def test_nonintegral_period_has_bounded_lateness_without_drift():
    clock = PeriodicClock(0.002999)
    scheduled = []
    lateness = []
    for index in range(10001):
        time = index * 0.0001
        for tick in clock.pop_due(time):
            scheduled.append(tick)
            lateness.append(time - tick)
    assert len(scheduled) == math.floor(1.0 / 0.002999) + 1
    assert scheduled[-1] == pytest.approx((len(scheduled) - 1) * 0.002999)
    assert min(lateness) >= -1e-12
    assert max(lateness) < 0.0001 + 1e-12
    with pytest.raises(ValueError, match="monotonic"):
        clock.pop_due(0)


def test_command_partner_hold_is_captured_once_and_new_mask_replaces_old():
    runtime = Real2SimRuntime(Real2SimConfig(), dtype=torch.float64)
    q = torch.zeros((1, 10), dtype=torch.float64)
    q[0, 1] = math.radians(20)
    runtime.reset(q)
    runtime.apply_command(CommandFrame(0, {"THJ1": JointCommand(0.8, 1.0, 20)}), q)
    runtime.update_controller(0, q)
    q[0, 1] = math.radians(25)
    runtime.update_controller(0.003, q)
    assert runtime.target_rad[0, 1].item() == pytest.approx(math.radians(20))
    assert runtime.controller_speed_rad_s[0, 1] < 0
    runtime.apply_command(CommandFrame(0.003, {"FFJ2": JointCommand(0.8, 1.0, 20)}), q)
    assert not runtime.active[0, :2].any()
    assert runtime.active[0, 2:4].all()
    runtime.update_controller(0.006, q)
    assert torch.count_nonzero(runtime.controller_speed_rad_s[0, :2]) == 0
    with pytest.raises(ValueError, match="empty A0"):
        runtime.apply_command(CommandFrame(0.006, {}), q)


def test_evaluation_does_not_advance_initial_or_final_state():
    runtime = Real2SimRuntime(Real2SimConfig(), dtype=torch.float64)
    q = torch.zeros((1, 10), dtype=torch.float64)
    runtime.reset(q)
    runtime.apply_command(CommandFrame(0, {"THJ1": JointCommand(0.8, 1.0, 20)}), q)
    runtime.update_controller(0, q)
    initial = runtime.evaluate(q, q)
    again = runtime.evaluate(q, q)
    assert torch.equal(initial.cable_retraction_m, again.cable_retraction_m)
    assert torch.count_nonzero(initial.proximal_tension_n) == 0
    runtime.advance(0.0001)
    final = runtime.evaluate(q, q)
    assert final.proximal_tension_n[0, 0] > 0
    assert torch.equal(final.cable_retraction_m, runtime.evaluate(q, q).cable_retraction_m)


def test_runtime_rejects_skipped_controller_ticks():
    runtime = Real2SimRuntime(Real2SimConfig())
    q = torch.zeros((1, 10))
    runtime.reset(q)
    with pytest.raises(ValueError, match="skipped"):
        runtime.update_controller(0.1, q)
