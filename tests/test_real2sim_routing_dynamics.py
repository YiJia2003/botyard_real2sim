"""Independent dynamics, stiction, hard-stop and energy checks for routing."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from botyard_real2sim.config import TendonModelConfig
from botyard_real2sim.model import TendonPlant
from botyard_real2sim.routing_dynamics import DynamicRoutingTransmission


def config(**kwargs):
    return replace(TendonModelConfig(routing_model="dynamic", routing_spring_max_length_m=0.001), **kwargs)


def array(value=0, n=1):
    return torch.full((n, 10), float(value), dtype=torch.float64)


def run(dynamics, x, y, samples, dt=0.0001):
    history = []
    for value in samples:
        new_x = array(value, x.shape[0])
        dynamics.advance(x, new_x, y, y, dt)
        x = new_x
        result = dynamics.evaluate(x, y)
        history.append((x.clone(), {k: v.clone() for k, v in result.items()}))
    return x, history


def test_inertia_creates_distinct_end_tensions_without_friction_or_filters():
    c = config(routing_friction_mu_static=0, routing_friction_mu_dynamic=0, routing_viscous_friction_n_s_per_m=0)
    d = DynamicRoutingTransmission(c, 1, "cpu")
    x = y = array()
    d.reset(x, y, array())
    x, history = run(d, x, y, [0.0002] * 500)
    first = history[0][1]
    assert first["proximal_tension"].min() > 100 * first["distal_tension"].max()
    final = history[-1][1]
    equivalent_k = 1 / (1 / 8000 + 1 / 1000)
    torch.testing.assert_close(final["proximal_tension"], array(equivalent_k * 0.0002), atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(final["distal_tension"], final["proximal_tension"], atol=1e-8, rtol=1e-6)
    assert torch.count_nonzero(d.friction_dissipated) == 0


def test_static_bristles_support_a_persistent_tension_difference_after_motion():
    d = DynamicRoutingTransmission(config(), 1, "cpu")
    x = y = array()
    d.reset(x, y, array())
    values = [min(0.006, (i + 1) * 0.00003) for i in range(600)]
    x, history = run(d, x, y, values)
    final = history[-1][1]
    assert d.du.abs().max() < 1e-8 and d.dv.abs().max() < 1e-8
    assert (final["proximal_tension"] - final["distal_tension"]).min() > 1
    torch.testing.assert_close(
        final["proximal_tension"] - final["distal_tension"], d.friction_u + d.friction_v, atol=1e-6, rtol=1e-6
    )
    assert d.zu.min() > 0 and d.zv.min() > 0
    torch.testing.assert_close(final["routing_extension"], array(0.001), atol=1e-12, rtol=0)
    assert (final["proximal_tension"] > 30).all()  # Hard stop does not cap tension at k_r*travel = 1 N.


def test_reversal_slack_impact_and_passive_energy_accounting():
    d = DynamicRoutingTransmission(config(), 1, "cpu")
    x = y = array()
    d.reset(x, y, array())
    last_energy = array()
    last_work = array()
    for i in range(501):
        new_x = array(0.003 * (1 - math.cos(2 * math.pi * (i + 1) / 500)))
        before_diss = d.friction_dissipated.clone()
        d.advance(x, new_x, y, y, 0.0001)
        x = new_x
        result = d.evaluate(x, y)
        assert result["proximal_tension"].min() >= 0 and result["distal_tension"].min() >= 0
        assert result["routing_extension"].max() <= 0.001 + 1e-12
        assert (d.friction_dissipated - before_diss).min() >= -1e-12
        if result["locked"].any():
            assert (d.du - d.dv).abs().max() < 1e-10
        energy = result["elastic_energy"] + result["kinetic_energy"]
        assert (energy - last_energy - (d.port_work - last_work)).max() < 1e-10
        last_energy, last_work = energy, d.port_work.clone()
    assert d.friction_u.min() < 0  # Released direction differs from initial pulling direction.
    assert d.impact_dissipated.min() > 0
    assert d.energy_residual.abs().max() < 1e-10
    assert d.solver_residual.max() < 1e-7
    assert d.numerical_dissipated.min() >= 0
    torch.testing.assert_close(
        d.port_work,
        last_energy + d.friction_dissipated + d.numerical_dissipated + d.impact_dissipated,
        atol=1e-10,
        rtol=1e-8,
    )


def test_contact_state_tracks_load_extension_and_path_with_memory():
    d = DynamicRoutingTransmission(config(), 1, "cpu")
    target_zero = d.contact_target(array(), array(), array())
    target_load = d.contact_target(array(10), array(), array())
    target_extension = d.contact_target(array(10), array(0.001), array())
    target_path = d.contact_target(array(10), array(0.001), array(0.02))
    assert torch.all(target_zero < target_load)
    assert torch.all(target_load < target_extension)
    assert torch.all(target_extension < target_path)
    # Same current geometry, different prior contact gives different next state.
    a, b = DynamicRoutingTransmission(config(), 1, "cpu"), DynamicRoutingTransmission(config(), 1, "cpu")
    x, y = array(0.003), array()
    a.reset(x, y, array(16))
    b.load_state_dict(a.state_dict())
    a.contact.fill_(0.1)
    b.contact.fill_(0.9)
    a.advance(x, x, y, y, 0.0001)
    b.advance(x, x, y, y, 0.0001)
    assert torch.all(a.contact < b.contact)
    assert torch.all(a.normal_u < b.normal_u)


def test_evaluation_is_pure_and_checkpoint_restores_memory_exactly():
    a = DynamicRoutingTransmission(config(), 2, "cpu")
    x, y = array(n=2), array(n=2)
    a.reset(x, y, x)
    x, _ = run(a, x, y, [0.00001 * (i + 1) for i in range(50)])
    snapshot = a.state_dict()
    a.evaluate(x, y)
    a.evaluate(x, y)
    for k, value in snapshot.items():
        assert torch.equal(value, a.state_dict()[k])
    b = DynamicRoutingTransmission(config(), 2, "cpu")
    b.load_state_dict(snapshot)
    a.advance(x, x * 0.9, y, y, 0.0001)
    b.advance(x, x * 0.9, y, y, 0.0001)
    for k, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[k])
    snapshot["contact"].fill_(2)
    with pytest.raises(ValueError, match="contact"):
        b.load_state_dict(snapshot)


def test_plant_applies_joint_side_tension_and_preserves_length_with_internal_slack():
    p = TendonPlant(config(), 1, "cpu", torch.float64)
    q = array()
    p.reset(q)
    p.advance(array(2), 0.0001, q_rad=q, qd_rad_s=q)
    result = p.evaluate(q, q)
    assert not torch.allclose(result.proximal_tension_n, result.distal_tension_n)
    expected = torch.einsum("bi,bij->bj", result.distal_tension_n, result.moment_arm_m)
    torch.testing.assert_close(result.tendon_effort_nm, expected)
    for _ in range(50):
        p.advance(array(-2), 0.0001, q_rad=q, qd_rad_s=q)
    result = p.evaluate(q, q)
    torch.testing.assert_close(
        result.cable_extension_m,
        result.routing_extension_m + result.material_extension_m + result.routing_state["routing_internal_slack_m"],
        atol=1e-12,
        rtol=1e-8,
    )
    assert result.proximal_tension_n.min() >= 0 and result.distal_tension_n.min() >= 0
    with pytest.raises(ValueError, match="requires current"):
        p.advance(array(), 0.001)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routing_model": "unknown"},
        {"routing_mass_kg": 0},
        {"routing_bristle_stiffness_n_per_m": 0},
        {"routing_proximal_compliance_fraction": 1},
        {"routing_max_substep_s": 0},
        {"routing_friction_mu_dynamic": 0.4},
        {"routing_contact_load_tau_s": 0},
        {"routing_contact_baseline": 2},
        {"routing_contact_tension_scale_n": 0},
        {"routing_normal_preload_n": -1},
    ],
)
def test_invalid_dynamic_parameters_fail_before_simulation(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)
