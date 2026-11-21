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
        # Isolate the viscous/Stribeck term: a nonzero static band would hold the
        # joint at rest and mask the spring-only assertion below.
        coulomb_friction_nm=(0.0,) * 10,
        static_friction_nm=(0.0,) * 10,
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


def test_static_friction_holds_joint_inside_the_band_and_slips_outside_it() -> None:
    cfg = TendonModelConfig(
        spring_stiffness_nm_per_rad=(0.1,) * 10,
        viscous_friction_nm_s_per_rad=(0.0,) * 10,
        coulomb_friction_nm=(0.01,) * 10,
        static_friction_nm=(0.05,) * 10,
        stribeck_speed_rad_s=(0.05,) * 10,
        friction_smoothing_speed_rad_s=(0.01,) * 10,
    )
    plant = TendonPlant(cfg, num_envs=1, device="cpu", dtype=torch.float64)

    # Inside the band: the spring asks for -0.02 N*m, the hold cancels it exactly,
    # so a joint at rest has no net effort and therefore no drift.
    for offset in (-0.4, -0.2, 0.0, 0.2, 0.4):
        q = _zeros()
        q[0, 0] = offset
        plant.reset(q)
        held = plant.evaluate(q, _zeros())
        assert abs(held.joint_effort_nm[0, 0].item()) < 1e-12
        assert math.isclose(held.friction_effort_nm[0, 0].item(), -held.spring_effort_nm[0, 0].item(), rel_tol=1e-9)

    # Outside the band: only the excess past +/-Static survives as sliding effort.
    q = _zeros()
    q[0, 0] = 0.8
    plant.reset(q)
    sliding = plant.evaluate(q, _zeros())
    assert math.isclose(sliding.spring_effort_nm[0, 0].item(), -0.08, rel_tol=1e-12)
    assert math.isclose(sliding.joint_effort_nm[0, 0].item(), -0.03, rel_tol=1e-12)

    # Above the smoothing speed the original Stribeck law is recovered verbatim.
    q = _zeros()
    q[0, 0] = 0.2
    plant.reset(q)
    velocity = _zeros()
    velocity[0, 0] = 1.0
    fast = plant.evaluate(q, velocity)
    dry = 0.01 + (0.05 - 0.01) * math.exp(-((1.0 / 0.05) ** 2))
    expected = -dry * math.tanh(1.0 / 0.01)
    assert math.isclose(fast.friction_effort_nm[0, 0].item(), expected, rel_tol=1e-9)
    assert math.isclose(fast.joint_effort_nm[0, 0].item(), -0.02 + expected, rel_tol=1e-9)

    # The blend is monotone in speed: a half-speed joint keeps half of the hold band.
    velocity[0, 0] = 0.005
    half = plant.evaluate(q, velocity)
    half_dry = 0.01 + (0.05 - 0.01) * math.exp(-((0.005 / 0.05) ** 2))
    hold_part = 0.5 * 0.02  # friction cancels the -0.02 N*m spring effort
    slip_part = 0.5 * (-half_dry * math.tanh(0.005 / 0.01))
    assert math.isclose(half.friction_effort_nm[0, 0].item(), hold_part + slip_part, rel_tol=1e-9)


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


def test_plant_reset_passes_distal_tension_to_dynamic_routing() -> None:
    """Measured held initialization feeds Tp and Td into the dynamic routing reset.

    中文：夹持态初始化——近端张力取实测传感器值，远端张力由调用方给定，
    两者都传给 DynamicRoutingTransmission.reset；缺省仍保持旧行为。
    """
    cfg = TendonModelConfig(routing_model="dynamic", routing_spring_max_length_m=0.01)
    plant = TendonPlant(cfg, 1, "cpu", dtype=torch.float64)
    q = _zeros().clone()
    q[0, 3] = math.radians(5.27)
    proximal = _zeros().clone()
    proximal[0, 3] = 11.66
    distal = _zeros().clone()
    distal[0, 3] = 10.66
    plant.reset(q, proximal, routing_initial_distal_tension_n=distal)
    dynamics = plant.routing_dynamics
    assert dynamics is not None
    torch.testing.assert_close(
        dynamics.friction_u[0, 3] - dynamics.friction_v[0, 3], torch.tensor(0.0, dtype=torch.float64)
    )
    # A held difference was accepted and preloaded: the two bristles carry it symmetrically
    # (each takes half of the 1 N held difference).
    torch.testing.assert_close(
        dynamics.friction_u[0, 3] + dynamics.friction_v[0, 3], torch.tensor(1.0, dtype=torch.float64)
    )


def test_plant_reset_distal_tension_defaults_to_proximal() -> None:
    """Omitting the distal override keeps the exact legacy equal split.

    中文：不给远端张力时，走线 reset 的行为与历史版本逐位一致。
    """
    cfg = TendonModelConfig(routing_model="dynamic", routing_spring_max_length_m=0.01)
    plant_new = TendonPlant(cfg, 1, "cpu", dtype=torch.float64)
    plant_old = TendonPlant(cfg, 1, "cpu", dtype=torch.float64)
    q = _zeros().clone()
    q[0, 3] = math.radians(5.27)
    tension = _zeros().clone()
    tension[0, 3] = 11.66
    plant_new.reset(q, tension, routing_initial_distal_tension_n=None)
    plant_old.reset(q, tension)
    torch.testing.assert_close(plant_new.routing_dynamics.u, plant_old.routing_dynamics.u)
    torch.testing.assert_close(plant_new.routing_dynamics.v, plant_old.routing_dynamics.v)
    torch.testing.assert_close(plant_new.routing_dynamics.zu, plant_old.routing_dynamics.zu)
    torch.testing.assert_close(plant_new.routing_dynamics.zv, plant_old.routing_dynamics.zv)


def test_return_spring_can_be_bilinear_without_changing_the_default_path() -> None:
    """A knee makes the return spring soft near free length and stiff beyond it.

    中文：回位拉簧的分段刚度——实测 3°/7.8 N 需要 ~500 N/m，而 5° 锁定态 12.7 N 需要
    ~1000 N/m；用单一段无法两全，故给一个拐点，缺省拐点为 inf（逐位等同旧行为）。
    """
    from botyard_real2sim.config import TendonModelConfig

    base = TendonModelConfig()
    assert all(math.isinf(v) for v in base.return_spring_knee_m)
    assert base.return_spring_stiffness_high_n_per_m == base.return_spring_stiffness_n_per_m


def test_bilinear_return_force_matches_measured_levels() -> None:
    """Check the two-segment force law directly against the measured levels."""
    import torch

    from botyard_real2sim.model import bilinear_return_force

    k_low, k_high, knee = 500.0, 1000.0, 0.0008
    zero = torch.zeros(())
    # 分段内仍是 k_low；跨过拐点后斜率变为 k_high。
    assert bilinear_return_force(
        torch.tensor(0.0004, dtype=torch.float64), 0.0, k_low, k_high, knee, zero
    ).item() == pytest.approx(0.0004 * k_low)
    assert bilinear_return_force(
        torch.tensor(knee, dtype=torch.float64), 0.0, k_low, k_high, knee, zero
    ).item() == pytest.approx(knee * k_low)
    assert bilinear_return_force(
        torch.tensor(knee + 0.001, dtype=torch.float64), 0.0, k_low, k_high, knee, zero
    ).item() == pytest.approx(knee * k_low + 0.001 * k_high)
    # 无拐点（inf）时退化为单段，与旧公式逐位一致。
    assert bilinear_return_force(
        torch.tensor(0.002, dtype=torch.float64), 0.0, 1000.0, 1000.0, math.inf, zero
    ).item() == pytest.approx(2.0)
    # 负拉伸被单边约束钳到 0（弹簧不能被压缩）。
    assert (
        bilinear_return_force(torch.tensor(-0.001, dtype=torch.float64), 0.0, k_low, k_high, knee, zero).item() == 0.0
    )
