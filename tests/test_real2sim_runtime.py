"""Timing and command-event invariants independent of the simulator backend."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from botyard_real2sim.commands import CommandFrame, JointCommand
from botyard_real2sim.config import FallbackControllerConfig, Real2SimConfig, load_config
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_motor_deadband_uses_motor_units_strict_boundary_and_per_joint_ratio(dtype):
    cfg = Real2SimConfig(
        controller=FallbackControllerConfig(
            motor_velocity_deadband_rad_s={"default": 200, "LFJ3": 0},
            motor_gear_ratio={"default": 800, "RFJ2": 400, "RFJ3": 1000},
        )
    )
    runtime = Real2SimRuntime(cfg, num_envs=2, dtype=dtype)
    q = torch.zeros((2, 10), dtype=dtype)
    runtime.reset(q)
    raw = torch.tensor([[0.249, 0.25, 0.251, -0.249, -0.25, -0.251, 0.4, 0.2, 0.0, 0.01]], dtype=dtype).expand(2, -1)
    runtime.controller_speed_rad_s.copy_(raw)
    expected = torch.tensor([[0, 0.25, 0.251, 0, -0.25, -0.251, 0, 0.2, 0, 0.01]], dtype=dtype).expand(2, -1)
    torch.testing.assert_close(runtime.motor_target_speed_rad_s, expected)
    assert torch.equal(runtime.controller_speed_rad_s, raw)
    runtime.advance(0.001)
    torch.testing.assert_close(runtime.evaluate(q, q).motor_position_rad, expected * 0.001)
    runtime.reset(q)
    assert not runtime.motor_target_speed_rad_s.any()


def test_motor_deadband_follows_h7_quantization_without_clearing_integral():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/real2sim/s1.yaml")
    # Quasistatic plant isolates the controller boundary from uncalibrated mechanics.
    cfg = Real2SimConfig(controller=cfg.controller)
    runtime = Real2SimRuntime(cfg, dtype=torch.float64)
    q = torch.full((1, 10), math.radians(10), dtype=torch.float64)
    runtime.reset(q)
    initial_motor = runtime.evaluate(q, torch.zeros_like(q)).motor_position_rad.clone()
    runtime.apply_command(CommandFrame(0, {"MFJ3": JointCommand(math.radians(11.2), 2.0, 0)}), q)
    runtime.update_controller(0, q)
    assert runtime.controller_speed_rad_s[0, 5].item() == 0.230
    assert runtime.motor_target_speed_rad_s[0, 5].item() == 0
    runtime.advance(0.003)
    assert torch.equal(runtime.evaluate(q, q).motor_position_rad, initial_motor)
    runtime.update_controller(0.003, q)
    assert runtime.controller.integral_deg_s[0, 5].item() == pytest.approx(2.4)
    assert runtime.motor_target_speed_rad_s[0, 5].item() == 0.251
    runtime.advance(0.003)
    displacement = runtime.evaluate(q, q).motor_position_rad - initial_motor
    assert displacement[0, 5].item() == pytest.approx(0.000753)


def test_legacy_config_without_driver_fields_preserves_speed_port():
    cfg = Real2SimConfig.from_dict({"controller": {"min_angle_deg": 0.5}})
    runtime = Real2SimRuntime(cfg, dtype=torch.float64)
    runtime.controller_speed_rad_s[:] = 0.01
    assert torch.equal(runtime.motor_target_speed_rad_s, runtime.controller_speed_rad_s)


@pytest.mark.parametrize("preset", ["s1.yaml", "default.yaml"])
def test_real2sim_presets_cap_the_spool_speed_after_h7_output(preset):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/real2sim" / preset)
    runtime = Real2SimRuntime(cfg, dtype=torch.float64)
    runtime.controller_speed_rad_s[0, :4] = torch.tensor([6.283, -6.282, 6.25, -6.25], dtype=torch.float64)
    assert runtime.motor_target_speed_rad_s[0, :4].tolist() == [6.25, -6.25, 6.25, -6.25]


def test_negative_h7_integer_conversion_precedes_motor_deadband():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/real2sim/s1.yaml")
    runtime = Real2SimRuntime(cfg, dtype=torch.float64)
    q = torch.full((1, 10), math.radians(30), dtype=torch.float64)
    runtime.reset(q)
    runtime.apply_command(
        CommandFrame(
            0,
            {
                "MFJ2": JointCommand(math.radians(40), math.radians(14.33), 0),
                "MFJ3": JointCommand(math.radians(20), math.radians(14.33), 0),
            },
        ),
        q,
    )
    runtime.update_controller(0, q)
    assert runtime.controller_speed_rad_s[0, 4:6].tolist() == [0.25, -0.249]
    assert runtime.motor_target_speed_rad_s[0, 4:6].tolist() == [0.25, 0]


def test_missing_middle_j3_partner_still_obeys_preload_target_floor():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/real2sim/s1.yaml")
    runtime = Real2SimRuntime(cfg, dtype=torch.float64)
    q = torch.full((1, 10), math.radians(0.5), dtype=torch.float64)
    runtime.reset(q)
    runtime.apply_command(CommandFrame(0, {"MFJ2": JointCommand(math.radians(20), 2, 0)}), q)
    assert runtime.target_rad[0, 5].item() == pytest.approx(math.radians(0.5))
    runtime.update_controller(0, q)
    assert runtime.motor_target_speed_rad_s[0, 5].item() == 0.349


@pytest.mark.parametrize(
    "changes",
    [
        {"motor_velocity_deadband_rad_s": -1},
        {"motor_velocity_deadband_rad_s": math.nan},
        {"motor_gear_ratio": 0},
        {"motor_gear_ratio": -800},
        {"motor_gear_ratio": math.inf},
        {"motor_gear_ratio": [800, 800]},
        {"motor_spool_speed_limit_rad_s": -1},
        {"motor_spool_speed_limit_rad_s": math.nan},
    ],
)
def test_driver_configuration_rejects_invalid_thresholds_and_ratios(changes):
    with pytest.raises(ValueError):
        replace(FallbackControllerConfig(), **changes)
