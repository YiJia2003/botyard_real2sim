"""Tick-level checks against the selected H7 fallback control semantics."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from botyard_real2sim.config import FallbackControllerConfig, load_config
from botyard_real2sim.controller import FallbackController, autopair_command

pytestmark = pytest.mark.unit


def vec(degrees):
    return torch.full((1, 10), math.radians(degrees), dtype=torch.float64)


def controller(**changes):
    cfg = replace(FallbackControllerConfig(), quantize_angle_feedback=False, **changes)
    return FallbackController(cfg, 1, "cpu", dtype=torch.float64)


def test_per_tick_integral_separation_deadband_and_clamp():
    ctl = controller(quantize_output=False)
    q, target, speed = vec(10), vec(11.5), vec(360)
    first = ctl.step(q, target, speed)
    assert ctl.integral_deg_s[0, 0].item() == pytest.approx(1.5)
    assert first[0, 0].item() == pytest.approx(math.radians(24.0))
    for _ in range(20):
        ctl.step(q, target, speed)
    assert ctl.integral_deg_s[0, 0].item() == 10
    # Outside the integration region the integral is retained but omitted.
    large = ctl.step(q, vec(13), speed)
    assert ctl.integral_deg_s[0, 0].item() == 10
    assert large[0, 0].item() == pytest.approx(math.radians(45))
    dead = ctl.step(q, vec(10.5), speed)
    assert not dead.any()
    assert not ctl.integral_deg_s.any()


def test_speed_defaults_abs_caps_angle_limits_and_asymmetric_integer_conversion():
    ctl = controller()
    positive = ctl.step(vec(10), vec(80), vec(0))
    negative = ctl.step(vec(80), vec(10), vec(-120))
    assert positive[0, 0].item() == 2.094
    assert negative[0, 0].item() == -2.093
    capped = ctl.step(vec(0), vec(1000), vec(1000))
    assert capped[0, 0].item() == 6.283
    ctl.reset()
    assert not ctl.step(vec(0.5), vec(-100), vec(360)).any()
    assert not ctl.step(vec(89.5), vec(500), vec(360)).any()


def test_tension_command_has_no_effect_and_alias_conflict_rejected():
    low, high = controller(), controller()
    q, cmd, speed = vec(10), vec(30), vec(120)
    assert torch.equal(
        low.step(q, cmd, speed, cmd_tension_command_n=torch.zeros_like(q)),
        high.step(q, cmd, speed, cmd_tension_command_n=torch.full_like(q, 100.0)),
    )
    with pytest.raises(ValueError, match="only one"):
        low.step(q, cmd, speed, q, cmd_tension_command_n=q)


def test_autopair_is_receipt_event_and_zero_field_does_not_disable_partner():
    actual, targets, speeds, tension = vec(20), vec(50), vec(90), torch.full((1, 10), 20.0, dtype=torch.float64)
    selected = torch.zeros((1, 10), dtype=torch.bool)
    selected[0, 4] = True
    targets, speeds, tension, mask = autopair_command(actual, targets, speeds, tension, selected)
    assert mask.tolist() == [[False, False, False, False, True, True, False, False, False, False]]
    assert targets[0, 5].item() == actual[0, 5].item()
    assert speeds[0, 5].item() == speeds[0, 4].item()
    assert tension[0, 5].item() == 0
    # The captured partner target remains fixed as actual position changes.
    moved = actual.clone()
    moved[0, 5] = math.radians(15)
    out = controller().step(moved, targets, speeds, tension, mask)
    assert out[0, 5] > 0
    assert torch.count_nonzero(out) == 2
    assert selected.sum() == 1  # no in-place mutation of event input


def test_autopair_raw_negative_speed_uses_default_for_missing_partner():
    selected = torch.zeros((1, 10), dtype=torch.bool)
    selected[0, 1] = True
    _, speed, _, _ = autopair_command(vec(20), vec(30), vec(-90), vec(0), selected)
    assert speed[0, 0].item() == pytest.approx(math.radians(120))
    assert speed[0, 1].item() == pytest.approx(math.radians(-90))


def test_cold_start_counts_active_pairs_in_order_and_warm_reset_runs_immediately():
    ctl = controller(startup_pair_evaluations=3)
    ctl.reset(cold_start=True)
    out = ctl.step(vec(10), vec(30), vec(120))
    assert not out[0, :6].any()
    assert (out[0, 6:] > 0).all()
    assert ctl.pair_evaluations.item() == 5
    ctl.reset(cold_start=True)
    mask = torch.zeros((1, 10), dtype=torch.bool)
    mask[:, 4:6] = True
    for _ in range(3):
        assert not ctl.step(vec(10), vec(30), vec(120), active_mask=mask).any()
    assert ctl.step(vec(10), vec(30), vec(120), active_mask=mask)[0, 4] > 0
    ctl.reset()
    assert ctl.step(vec(10), vec(30), vec(120), active_mask=mask)[0, 4] > 0


def test_inactive_mask_freezes_state_and_malformed_inputs_fail():
    ctl = controller()
    q = vec(10)
    mask = torch.zeros((1, 10), dtype=torch.bool)
    assert not ctl.step(q, vec(11.5), vec(120), active_mask=mask).any()
    assert not ctl.integral_deg_s.any()
    assert not ctl.previous_error_deg.any()
    mask[0, 0] = True
    with pytest.raises(ValueError, match="complete"):
        ctl.step(q, q, q, active_mask=mask)
    with pytest.raises(ValueError, match="shape"):
        ctl.step(q[:, :1], q, q)
    bad = q.clone()
    bad[0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        ctl.step(bad, q, q)
    with pytest.raises(ValueError, match="zero/one"):
        ctl.step(q, q, q, active_mask=torch.full_like(q, 2))
    with pytest.raises(ValueError, match="dtype"):
        ctl.step(q.float(), q, q)


@pytest.mark.parametrize("preset", ["s1.yaml", "default.yaml"])
def test_real2sim_presets_apply_middle_j3_preload_floor(preset):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/real2sim" / preset)
    ctl = FallbackController(cfg.controller, 1, "cpu", dtype=torch.float64)
    output = ctl.step(vec(0.5), vec(-10), vec(120))
    # Only MFJ3 clamps to 2.5 deg; its 2 deg error produces 20 deg/s.
    assert output[0, 5].item() == 0.349
    assert torch.count_nonzero(output) == 1
    q = vec(0.5)
    q[0, 5] = math.radians(2.5)
    assert not ctl.step(q, vec(-10), vec(120)).any()


@pytest.mark.parametrize("error_deg, expected", [(0.99, 0.0), (1.0, 0.279), (1.01, 0.282)])
def test_h7_angle_dead_zone_is_strict(error_deg, expected):
    # Zero feedback avoids subtractive roundoff at the exact 1 degree boundary.
    out = controller().step(vec(0), vec(error_deg), vec(120))
    assert out[0, 0].item() == expected
