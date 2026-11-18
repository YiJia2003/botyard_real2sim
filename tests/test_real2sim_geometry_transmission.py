"""Geometry, bounded transmission and encoder observables independent of Isaac."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from botyard_real2sim.config import TendonModelConfig, load_config, save_config
from botyard_real2sim.geometry import PlanarPath
from botyard_real2sim.model import TendonPlant
from botyard_real2sim.parameters import override_parameters
from botyard_real2sim.transmission import SeriesRoutingSpring

PRESET = Path(__file__).resolve().parents[1] / "configs/real2sim/s1.yaml"


def plant(cfg=None, n=1):
    return TendonPlant(cfg or load_config(PRESET).tendon, n, "cpu", torch.float64)


@pytest.mark.parametrize(
    "kind,name", [("tendons", "first"), ("tendons", "second"), ("returns", "first"), ("returns", "second")]
)
def test_path_gradient_matches_length_in_contact_and_detached_regimes(kind, name):
    spec = load_config(PRESET).tendon.geometry["index"]
    path = PlanarPath(spec, spec[kind][name], device="cpu", dtype=torch.float64)
    q = torch.tensor([[0.1, 0.1], [0.5, 0.8], [1.54, 1.54]], dtype=torch.float64)
    _, gradient = path.evaluate(q)
    for i in range(2):
        delta = torch.zeros_like(q)
        delta[:, i] = 1e-6
        finite_difference = (path.evaluate(q + delta)[0] - path.evaluate(q - delta)[0]) / 2e-6
        torch.testing.assert_close(gradient[:, i], finite_difference, atol=2e-10, rtol=1e-6)


def test_cad_routes_remain_continuous_and_restore_over_full_workspace():
    grid = torch.linspace(0, math.pi / 2, 91, dtype=torch.float64)
    pairs = torch.cartesian_prod(grid, grid)
    p = plant(n=len(pairs))
    q = pairs.repeat(1, 5)
    shortening, jacobian = p.geometry.evaluate(q)
    lengths, return_jacobian = p.geometry.return_lengths(q)
    assert torch.isfinite(jacobian).all()
    # Current sketch implies opposite sign at the crossed J3, unlike the old seed.
    assert torch.all(jacobian[:, 2, 2] > 0)
    assert torch.all(jacobian[:, 2, 3] < 0)
    assert torch.all(jacobian[:, 3, 3] > 0)
    for index in (2, 3):
        surface = shortening[:, index].reshape(91, 91)
        assert (surface[1:] - surface[:-1]).abs().max() < 0.0002
        assert (surface[:, 1:] - surface[:, :-1]).abs().max() < 0.0002
        assert torch.all(return_jacobian[:, index, index] > 0)
    # Variable force direction produces variable arms; no constant-radius surrogate.
    assert return_jacobian[:, 3, 3].max() - return_jacobian[:, 3, 3].min() > 0.005
    assert torch.isfinite(lengths).all()


def test_distal_mimic_is_included_in_virtual_work():
    cfg = load_config(PRESET)
    q = torch.zeros(1, 10, dtype=torch.float64)
    one = plant(cfg.tendon)
    two = plant(override_parameters(cfg, ["tendon.geometry.index.distal_mimic_ratio=0"]).tendon)
    r1, r0 = one.geometry.evaluate(q)[1], two.geometry.evaluate(q)[1]
    assert r1[0, 2, 2] == pytest.approx(0.010, abs=1e-9)
    assert r0[0, 2, 2] == pytest.approx(0.005, abs=1e-9)
    one.reset(q, initial_tension_n=4)
    velocity = torch.arange(10, dtype=torch.float64).unsqueeze(0) * 0.1
    result = one.evaluate(q, velocity)
    path_speed = torch.einsum("bij,bj->bi", result.moment_arm_m, velocity)
    torch.testing.assert_close(
        (result.tendon_effort_nm * velocity).sum(), (result.proximal_tension_n * path_speed).sum()
    )


def test_drawing_return_spring_preload_and_joint_direction():
    p = plant()
    q = torch.zeros(1, 10, dtype=torch.float64)
    p.reset(q)
    result = p.evaluate(q, q)
    torch.testing.assert_close(result.return_spring_length_m[:, 2:], torch.full((1, 8), 0.024, dtype=torch.float64))
    torch.testing.assert_close(result.return_spring_force_n[:, 2:], torch.full((1, 8), 4.0, dtype=torch.float64))
    assert torch.all(result.spring_effort_nm[:, 2:] < 0)
    # Geometry springs replace angular springs on their channels.
    p2 = plant(replace(p.cfg, spring_stiffness_nm_per_rad=100.0))
    p2.reset(q)
    torch.testing.assert_close(p2.evaluate(q, q).spring_effort_nm[:, 2:], result.spring_effort_nm[:, 2:])


def test_series_stop_caps_travel_not_force_and_preserves_length_energy():
    s = SeriesRoutingSpring(torch.tensor(8000.0), torch.tensor(1000.0), torch.tensor(0.003))
    extensions = torch.tensor([-0.001, 0.0, 0.001, 0.003375, 0.004, 0.006], dtype=torch.float64)
    tension, routing, material, locked, energy = s.evaluate(extensions)
    torch.testing.assert_close(routing + material, extensions)
    assert tension[0] == 0 and tension[1] == 0
    assert torch.all(routing <= 0.003 + 1e-9)
    assert routing[-1] == routing[-2]
    assert tension[-1] > tension[-2] > 3
    assert locked[-1] and not locked[2]
    for i in (2, 4, 5):
        e = extensions[i]
        derivative = (s.evaluate(e + 1e-7)[4] - s.evaluate(e - 1e-7)[4]) / 2e-7
        torch.testing.assert_close(derivative, tension[i])
    assert torch.all(energy >= 0)


def test_motor_state_is_independent_of_joint_and_includes_series_stretch():
    p = plant(replace(load_config(PRESET).tendon, motor_encoder_sign=-1))
    q = torch.full((1, 10), 0.25, dtype=torch.float64)
    motor = torch.full_like(q, 1.2)
    p.reset(q, initial_tension_n=7, motor_position_rad=motor)
    before = p.evaluate(q, torch.zeros_like(q))
    torch.testing.assert_close(before.motor_position_rad, motor)
    torch.testing.assert_close(before.proximal_tension_n, torch.full_like(q, 7))
    assert torch.all(before.routing_locked)
    p.advance(torch.full_like(q, 2), 0.05, q_rad=q, qd_rad_s=torch.zeros_like(q))
    after = p.evaluate(q, torch.zeros_like(q))
    torch.testing.assert_close(after.motor_position_rad, motor - 0.1)
    torch.testing.assert_close(after.motor_velocity_rad_s, torch.full_like(q, -2))
    torch.testing.assert_close(after.length_balance_error_m, torch.zeros_like(q))
    assert torch.all(after.proximal_tension_n > before.proximal_tension_n)


def test_partial_reset_preserves_other_motor_offsets_and_length_state():
    p = plant(n=3)
    q = torch.full((3, 10), 0.1, dtype=torch.float64)
    p.reset(q, motor_position_rad=torch.ones_like(q))
    p.advance(torch.ones_like(q), 0.1, q_rad=q, qd_rad_s=torch.zeros_like(q))
    before = p.evaluate(q, q * 0)
    p.reset(
        q[1:2],
        env_ids=torch.tensor([1]),
        initial_tension_n=3,
        motor_position_rad=torch.full((1, 10), 2.0, dtype=torch.float64),
    )
    after = p.evaluate(q, q * 0)
    torch.testing.assert_close(after.motor_position_rad[[0, 2]], before.motor_position_rad[[0, 2]])
    torch.testing.assert_close(after.motor_position_rad[1], torch.full((10,), 2.0, dtype=torch.float64))


def test_geometry_overrides_are_isolated_and_round_trip(tmp_path):
    cfg = load_config(PRESET)
    changed = override_parameters(
        cfg,
        [
            "tendon.geometry.index.tendons.first.5.point_m=[0.006, 0.082]",
            "tendon.routing_spring_free_length_m.FFJ2=0.030",
            "tendon.routing_spring_max_length_m.FFJ2=0.034",
        ],
    )
    assert changed.tendon.geometry["index"]["tendons"]["first"][-1]["point_m"] == [0.006, 0.082]
    assert changed.tendon.geometry["middle"] == cfg.tendon.geometry["middle"]
    out = tmp_path / "resolved.yaml"
    save_config(out, changed)
    assert load_config(out).to_dict() == changed.to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routing_spring_free_length_m": 0.02, "routing_spring_max_length_m": 0.01},
        {"routing_spring_stiffness_n_per_m": 0},
        {"routing_spring_max_length_m": 0.01, "cable_damping_n_s_per_m": 0.1},
        {"motor_encoder_sign": 0},
        {"geometry": {"invalid_finger": {}}},
    ],
)
def test_invalid_transmission_parameters_fail_before_launch(kwargs):
    with pytest.raises(ValueError):
        TendonModelConfig(**kwargs)
