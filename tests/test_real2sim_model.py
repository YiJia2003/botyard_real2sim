"""Behavioral checks for the unloaded tendon mechanics and firmware Fallback path."""

from __future__ import annotations

import math

import pytest
import torch

from botyard_real2sim.config import (
    FallbackControllerConfig,
    TendonModelConfig,
    load_config,
    save_config,
)
from botyard_real2sim.controller import FallbackController, autopair_command
from botyard_real2sim.model import TendonPlant


def _zeros() -> torch.Tensor:
    return torch.zeros((1, 10), dtype=torch.float64)


def test_cable_only_pulls_and_first_tendon_torques_both_crossed_joints() -> None:
    cfg = TendonModelConfig(
        spring_stiffness_nm_per_rad=(0.0,) * 10,
        viscous_friction_nm_s_per_rad=(0.0,) * 10,
        coulomb_friction_nm=(0.0,) * 10,
        static_friction_nm=(0.0,) * 10,
    )
    plant = TendonPlant(cfg, num_envs=1, device="cpu", dtype=torch.float64)
    zero = _zeros()
    plant.reset(zero)
    velocity = zero.clone()
    velocity[0, 0] = 1.0
    velocity[0, 2] = 1.0

    pulled = plant.step(zero, zero, velocity, dt_s=0.1)
    expected_tension = cfg.cable_stiffness_n_per_m[0] * cfg.drum_radius_m[0] * 0.1
    assert math.isclose(pulled.proximal_tension_n[0, 0].item(), expected_tension)
    assert math.isclose(pulled.proximal_tension_n[0, 2].item(), expected_tension)
    assert math.isclose(pulled.tendon_effort_nm[0, 1].item(), expected_tension * 0.005)
    assert math.isclose(pulled.tendon_effort_nm[0, 3].item(), expected_tension * 0.005)
    assert torch.all(pulled.proximal_tension_n >= 0.0)

    released = plant.step(zero, zero, -velocity, dt_s=0.1)
    assert torch.allclose(released.proximal_tension_n, zero, atol=1e-10)
    assert torch.allclose(released.tendon_effort_nm, zero, atol=1e-10)


def test_spring_returns_joint_and_friction_dissipates_in_both_directions() -> None:
    cfg = TendonModelConfig(
        spring_stiffness_nm_per_rad=(0.1,) * 10,
        viscous_friction_nm_s_per_rad=(0.01,) * 10,
    )
    plant = TendonPlant(cfg, num_envs=1, device="cpu", dtype=torch.float64)
    q = _zeros()
    q[0, 0] = 0.2
    plant.reset(q)
    rest = plant.step(q, _zeros(), _zeros(), dt_s=0.001)
    assert math.isclose(rest.spring_effort_nm[0, 0].item(), -0.02)
    assert math.isclose(rest.joint_effort_nm[0, 0].item(), -0.02)

    moving = _zeros()
    moving[0, 0] = 0.2
    forward = plant.step(q, moving, _zeros(), dt_s=0.001)
    assert forward.friction_effort_nm[0, 0] < 0.0
    backward = plant.step(q, -moving, _zeros(), dt_s=0.001)
    assert backward.friction_effort_nm[0, 0] > 0.0


def test_unvalidated_guide_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="segment geometry"):
        TendonModelConfig(crossing_guide_mu=0.2)


def test_fallback_speed_saturation_dead_zone_and_cold_start() -> None:
    cfg = FallbackControllerConfig(startup_pair_evaluations=50)
    controller = FallbackController(cfg, num_envs=1, device="cpu", dtype=torch.float64)
    controller.reset(cold_start=True)
    q = _zeros()
    target = torch.full_like(q, math.radians(80.0))
    speed_limit = torch.full_like(q, math.radians(360.0))
    for _ in range(10):
        assert torch.count_nonzero(controller.step(q, target, speed_limit)) == 0
    speed = controller.step(q, target, speed_limit)
    assert torch.all(speed > 0.0)
    assert torch.all(speed <= 7.0)  # Firmware OUTPUT_MAX_VALUE=7000 mrad/s.
    # When already at target, the 1-degree dead zone zeros the output and integral.
    controller.reset()
    assert torch.count_nonzero(controller.step(target, target, speed_limit)) == 0
    assert torch.count_nonzero(controller.integral_deg_s) == 0


def test_fallback_third_command_field_does_not_set_tension() -> None:
    cfg = FallbackControllerConfig()
    q = _zeros()
    target = torch.full_like(q, math.radians(20.0))
    speed_limit = torch.full_like(q, math.radians(60.0))
    low = FallbackController(cfg, 1, "cpu", torch.float64)
    high = FallbackController(cfg, 1, "cpu", torch.float64)
    speed_low = low.step(q, target, speed_limit, torch.full_like(q, 1.0))
    speed_high = high.step(q, target, speed_limit, torch.full_like(q, 100.0))
    assert torch.equal(speed_low, speed_high)


def test_autopair_holds_missing_partner_at_command_arrival() -> None:
    actual = _zeros()
    actual[0, 1] = 0.3
    target = _zeros()
    target[0, 0] = 0.5
    speed = torch.full_like(target, 0.5)
    limits = torch.full_like(target, 20.0)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 0] = True
    paired_target, paired_speed, paired_limit, paired_mask = autopair_command(actual, target, speed, limits, mask)
    assert paired_target[0, 1] == actual[0, 1]
    assert paired_speed[0, 1] == speed[0, 0]
    assert paired_limit[0, 1] == 0.0
    assert paired_mask[0, 0] and paired_mask[0, 1]


def test_shared_yaml_loads_model_and_preserves_curve_options(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("tendon:\n  cable_slack_m: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\ncurves:\n  window_s: 2.0\n")
    config = load_config(path)
    assert config.tendon.cable_slack_m == (0,) * 10
    save_config(path, config)
    assert "window_s: 2.0" in path.read_text()


def test_signed_routing_preserves_virtual_work() -> None:
    cfg = TendonModelConfig(crossing_moment_arm_m={"thumb": -0.005}, initial_tension_n=12.0)
    plant = TendonPlant(cfg, 1, "cpu", torch.float64)
    q = _zeros()
    plant.reset(q)
    velocity = torch.arange(10, dtype=torch.float64).unsqueeze(0) / 10
    result = plant.evaluate(q, velocity)
    path_velocity = velocity @ plant.routing_matrix_m.T
    assert torch.allclose((result.tendon_effort_nm * velocity).sum(), (result.proximal_tension_n * path_velocity).sum())
    assert plant.routing_matrix_m[0, 1] < 0
    assert abs(result.tendon_effort_nm[0, 1]) < 1e-14  # Equal opposing tendon moments.


def test_slack_cable_never_transmits_damping_force() -> None:
    cfg = TendonModelConfig(cable_slack_m=0.01, cable_damping_n_s_per_m=1000.0)
    plant = TendonPlant(cfg, 1, "cpu", torch.float64)
    q = _zeros()
    plant.reset(q)
    plant.advance(torch.ones_like(q), 0.001)
    result = plant.evaluate(q, q)
    assert torch.all(result.cable_extension_m < 0)
    assert torch.count_nonzero(result.proximal_tension_n) == 0
    plant.advance(torch.ones_like(q), 2.1)
    assert torch.all(plant.evaluate(q, q).proximal_tension_n > 0)


def test_slack_and_pretension_are_distinct_initial_states() -> None:
    with pytest.raises(ValueError, match="slack"):
        TendonModelConfig(cable_slack_m=0.01, initial_tension_n=1)
    q = _zeros()
    slack = TendonPlant(TendonModelConfig(cable_slack_m=0.01), 1, "cpu", torch.float64)
    slack.reset(q)
    assert torch.allclose(slack.evaluate(q, q).cable_extension_m, torch.full_like(q, -0.01))
    with pytest.raises(ValueError, match="slack"):
        slack.reset(q, initial_tension_n=2.0)
    taut = TendonPlant(TendonModelConfig(), 1, "cpu", torch.float64)
    taut.reset(q, initial_tension_n=2.0)
    assert torch.allclose(taut.evaluate(q, q).proximal_tension_n, torch.full_like(q, 2.0))
    slack.reset(q, cable_retraction_m=torch.full_like(q, 0.011))
    assert torch.allclose(slack.evaluate(q, q).proximal_tension_n, torch.full_like(q, 8.0))
    with pytest.raises(ValueError, match="not both"):
        slack.reset(q, initial_tension_n=0, cable_retraction_m=q)


def test_evaluate_does_not_integrate_and_lag_uses_exact_integral() -> None:
    plant = TendonPlant(TendonModelConfig(tendon_speed_lag_s=0.1), 1, "cpu", torch.float64)
    q = _zeros()
    plant.reset(q)
    before = plant.cable_retraction_m.clone()
    first = plant.evaluate(q, q)
    second = plant.evaluate(q, q)
    assert torch.equal(plant.cable_retraction_m, before)
    assert torch.equal(first.proximal_tension_n, second.proximal_tension_n)
    plant.advance(torch.ones_like(q), 0.1)
    expected = 0.005 * (0.1 - 0.1 * (1 - math.exp(-1)))
    assert torch.allclose(plant.cable_retraction_m, torch.full_like(q, expected))


@pytest.mark.parametrize(
    "key,value",
    [
        ("cable_stiffness_n_per_m", float("nan")),
        ("cable_damping_n_s_per_m", float("inf")),
        ("spring_rest_angle_rad", float("inf")),
        ("crossing_moment_arm_m", float("nan")),
        ("cable_stiffness_n_per_m", 0),
        ("stribeck_speed_rad_s", 0),
        ("guide_slip_smoothing_m_s", float("nan")),
    ],
)
def test_invalid_physics_parameters_fail_before_simulation(key, value) -> None:
    with pytest.raises(ValueError):
        TendonModelConfig.from_dict({key: value})


def test_config_named_overrides_and_curves_survive_new_path(tmp_path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        "tendon:\n  cable_stiffness_n_per_m: {default: 5000, FFJ2: 9000}\n"
        "  crossing_moment_arm_m: {thumb: -0.005}\ncurves:\n  alignment: start\n"
    )
    config = load_config(source)
    assert config.tendon.cable_stiffness_n_per_m[0] == 5000
    assert config.tendon.cable_stiffness_n_per_m[2] == 9000
    target = tmp_path / "resolved.yaml"
    save_config(target, config)
    assert load_config(target).to_dict() == config.to_dict()
    assert "alignment: start" in target.read_text()
    with pytest.raises(ValueError, match="unknown"):
        TendonModelConfig.from_dict({"cable_stiffness_n_per_m": {"FFJ22": 2}})
    with pytest.raises(ValueError, match="unknown"):
        TendonModelConfig.from_dict({"unknown_parameter": 1})


def test_runtime_nonfinite_inputs_are_rejected_without_state_change() -> None:
    plant = TendonPlant(TendonModelConfig(), 1, "cpu", torch.float64)
    q = _zeros()
    plant.reset(q)
    with pytest.raises(ValueError, match="finite"):
        plant.advance(q, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        plant.step(torch.full_like(q, float("nan")), q, q, 0.1)
    assert torch.count_nonzero(plant.cable_retraction_m) == 0


@pytest.mark.parametrize("text", ["[]", "false", "0", "tendon: []", "controller: null"])
def test_yaml_wrong_types_are_not_silently_replaced_with_defaults(tmp_path, text) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_partial_environment_reset_preserves_other_cable_states() -> None:
    plant = TendonPlant(TendonModelConfig(), 2, "cpu", torch.float64)
    q = torch.zeros((2, 10), dtype=torch.float64)
    plant.reset(q)
    plant.advance(torch.ones_like(q), 0.01)
    unchanged = plant.cable_retraction_m[0].clone()
    plant.reset(q, initial_tension_n=3.0, env_ids=torch.tensor([1]))
    assert torch.equal(plant.cable_retraction_m[0], unchanged)
    assert torch.allclose(plant.evaluate(q, q).proximal_tension_n[1], torch.full((10,), 3.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "values",
    [
        {"control_period_s": float("nan")},
        {"integral_limit_deg_s": -1},
        {"startup_pair_evaluations": 1.5},
        {"quantize_output": "false"},
        {"kp_deg_s_per_deg": {"THJ1": float("inf")}},
    ],
)
def test_controller_config_rejects_invalid_values(values) -> None:
    with pytest.raises(ValueError):
        FallbackControllerConfig.from_dict(values)


def test_controller_default_clock_uses_timer_arr_plus_one() -> None:
    cfg = FallbackControllerConfig()
    assert cfg.control_period_s == 0.003
    assert cfg.firmware_time_delta_s == 0.002


def test_plant_dtype_mismatch_is_clear_and_does_not_mutate_state() -> None:
    plant = TendonPlant(TendonModelConfig(), 1, "cpu", torch.float64)
    q = _zeros()
    with pytest.raises(ValueError, match="dtype"):
        plant.reset(q.float())
    plant.reset(q)
    before = plant.cable_retraction_m.clone()
    for invoke in (
        lambda: plant.evaluate(q.float(), q),
        lambda: plant.evaluate(q, q.float()),
        lambda: plant.advance(q.float(), 0.001),
    ):
        with pytest.raises(ValueError, match="dtype"):
            invoke()
    assert torch.equal(plant.cable_retraction_m, before)
    with pytest.raises(ValueError, match="dtype"):
        TendonPlant(TendonModelConfig(), 1, "cpu", torch.int64)
    with pytest.raises(ValueError, match="integer"):
        TendonPlant(TendonModelConfig(), True, "cpu")
