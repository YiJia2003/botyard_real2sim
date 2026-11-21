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


def test_reset_accepts_measured_distal_tension_and_keeps_proximal():
    """Measured held state: proximal tension from the sensor, distal from the held joint.

    中文：夹持态初始化——近端张力来自传感器，远端张力独立给定；两者不再强制相等。
    (Tp, Td, x, y) 必须满足长度恒等式 x−y = Tp/kp + es + Td/kd；Tp−Td 之差进入
    走线弹簧/硬停止与 bristle 预载，旧调用（缺省 Td=Tp）行为不变。
    """
    d = DynamicRoutingTransmission(config(), 1, "cpu")
    # x−y = 0.008: 串联平衡张力 ~12.2 N，锁定差 1 N 由走线弹簧+硬停止+摩擦分担。
    x = array(0.008)
    y = array(0.0)
    proximal = array(11.66)
    distal = array(10.66)
    d.reset(x, y, proximal, distal_tension=distal)
    state = d.evaluate(x, y)
    torch.testing.assert_close(state["proximal_tension"], proximal, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(state["distal_tension"], distal, atol=1e-9, rtol=1e-9)
    # es = x − y − Tp/kp − Td/kd = 0.008 − 22.32/16000 = 0.006605 m (> travel 1 mm → stop).
    torch.testing.assert_close(state["routing_extension"], array(0.006605), atol=1e-12, rtol=1e-9)
    # With the stop saturated, spring_tension = mean(Tp, Td) = 11.16 N.
    torch.testing.assert_close(state["spring_tension"], array(11.16), atol=1e-9, rtol=1e-9)
    # The bristle preload carries the held difference: zero node acceleration.
    assert state["acceleration_u"].abs().max() < 1e-6
    assert state["acceleration_v"].abs().max() < 1e-6
    torch.testing.assert_close(d.friction_u, array(0.5), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(d.friction_v, array(0.5), atol=1e-9, rtol=1e-9)


def test_reset_bristle_preload_clamps_to_the_friction_cone():
    """A held difference beyond mu_s*N cannot be static; preload saturates at the cone.

    中文：超过摩擦锥的锁定差不可能静态成立——预载钳位到锥边界，
    剩余不平衡留给后续步弛豫（与真实系统“锁不住就滑”一致）。
    """
    d = DynamicRoutingTransmission(config(), 1, "cpu")
    x = array(0.002)
    y = array(0.0)
    d.reset(x, y, array(11.66), distal_tension=array(7.7))  # 3.96 N difference
    state = d.evaluate(x, y)
    torch.testing.assert_close(state["proximal_tension"], array(11.66), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(state["distal_tension"], array(7.7), atol=1e-9, rtol=1e-9)
    # Preload sits exactly at the static cone: friction = mu_s * normal.
    normal = 0.5 * (0.5 * 0.5 * (11.66 + 7.7) * (0.25 + 1.5 * d.contact[0, 0]).clamp_min(0))
    torch.testing.assert_close(d.friction_u, array(0.25 * normal), atol=1e-9, rtol=1e-6)
    torch.testing.assert_close(d.friction_v, array(0.25 * normal), atol=1e-9, rtol=1e-6)


def test_reset_without_distal_tension_keeps_legacy_equal_split():
    """Old callers keep the exact old behavior when distal_tension is omitted."""
    d = DynamicRoutingTransmission(config(), 1, "cpu")
    x = array(0.004)
    y = array(0.0)
    d.reset(x, y, array(11.66))
    state = d.evaluate(x, y)
    torch.testing.assert_close(state["proximal_tension"], array(11.66), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(state["distal_tension"], array(11.66), atol=1e-9, rtol=1e-9)


def test_slip_state_is_off_by_default_and_bit_identical() -> None:
    """With routing_slip_max_length_m = 0 the dynamics must be unchanged."""
    base = config()
    with_slip_api = config()
    a = DynamicRoutingTransmission(base, 1, "cpu")
    b = DynamicRoutingTransmission(with_slip_api, 1, "cpu")
    x = y = array()
    a.reset(x, y, array())
    b.reset(x, y, array(), initial_slip=array(0.0))
    for value in [min(0.006, (i + 1) * 0.00003) for i in range(300)]:
        new_x = array(value)
        a.advance(x, new_x, y, y, 0.0001)
        b.advance(x, new_x, y, y, 0.0001)
        x = new_x
    torch.testing.assert_close(a.u, b.u)
    torch.testing.assert_close(a.slip, b.slip)
    assert float(a.slip.abs().max()) == 0.0


def test_slip_pays_out_slack_on_release_and_takes_it_up_on_reload() -> None:
    """Routing slip accumulates while the path unloads and clears while it reloads.

    中文：滑动累积状态——卸载时让出绳长（走线张力走低分支），重新加载时收回。
    时间常数取实测量级：释放 0.1 s 内让出大部分，0.3 s 内到地板。
    """
    c = config(
        routing_slip_max_length_m=(0.008,) * 10,
        routing_slip_release_tau_s=(0.1,) * 10,
        routing_slip_load_tau_s=(0.02,) * 10,
        routing_spring_free_length_m=(0.0,) * 10,
        routing_spring_max_length_m=(0.01,) * 10,
    )
    d = DynamicRoutingTransmission(c, 1, "cpu")
    x = array(0.006)
    y = array()
    d.reset(x, y, array(11.62), distal_tension=array(13.6))
    tension_at_start = float(d.evaluate(x, y)["spring_tension"].max())
    # 卸载 0.4 s：电机侧放绳 6 mm。
    for _ in range(400):
        new_x = array(x[0, 0].item() - 0.000015)
        d.advance(x, new_x, y, y, 0.001)
        x = new_x
    released = float(d.slip.max())
    assert released > 0.006  # 0.4 s / τ=0.1 s 应让出 > 75% 的上限
    assert float(d.evaluate(x, y)["spring_tension"].max()) < tension_at_start
    # 重新加载 0.2 s：收绳量回到原处附近 → 滑移快速收回。
    for _ in range(200):
        new_x = array(x[0, 0].item() + 0.000025)
        d.advance(x, new_x, y, y, 0.001)
        x = new_x
    assert float(d.slip.max()) < 0.05 * released


def test_initial_slip_lowers_the_spring_tension_at_fixed_endpoints() -> None:
    """The slip state is exactly a lost routing extension: kr * slip less tension."""
    c = config(routing_slip_max_length_m=(0.012,) * 10)
    d = DynamicRoutingTransmission(c, 1, "cpu")
    x = array(0.012)
    y = array()
    d.reset(x, y, array(11.62), distal_tension=array(13.6), initial_slip=array(0.006))
    state = d.evaluate(x, y)
    torch.testing.assert_close(
        state["routing_extension"], torch.clamp_min(d.u - d.v - 0.006, 0.0), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(state["routing_slip"], array(0.006), atol=1e-12, rtol=0.0)


def test_creep_rate_adds_a_slow_component_on_top_of_the_relay() -> None:
    """Optional creep term: after the relay settles, the slack keeps growing slowly.

    中文：relay 之后的可选小速率项——实测 3° 保持 8.5 s 继续爬 +0.45 N（≈0.053 N/s），
    对应 θ 的爬升速率 5.3e-5 m/s（kr = 1000 N/m）；静止保持时 θ 不得被收回。
    """
    common = dict(
        routing_slip_max_length_m=(0.05,) * 10,
        routing_slip_release_tau_s=(0.15,) * 10,
        routing_slip_load_tau_s=(0.02,) * 10,
        routing_spring_free_length_m=(0.0,) * 10,
        routing_spring_max_length_m=(0.03,) * 10,
    )
    x = array(0.012)
    y = array()

    def run(cfg):
        d = DynamicRoutingTransmission(cfg, 1, "cpu")
        d.reset(x, y, array(11.62), distal_tension=array(13.6))
        cur = x
        for _ in range(60):  # 0.6 s 卸载
            nxt = array(cur[0, 0].item() - 0.00005)
            d.advance(cur, nxt, y, y, 0.01)
            cur = nxt
        relayed = float(d.slip.max())
        for _ in range(200):  # 再静止保持 2 s（边界不动）
            d.advance(cur, cur, y, y, 0.01)
        return relayed, float(d.slip.max())

    r0, h0 = run(config(**common))
    r1_, h1 = run(config(**common, routing_slip_creep_m_per_s=(5.3e-05,) * 10))
    # relay 段两者一致（同一时间常数），保持段只有带爬升项的继续增长。
    assert abs(r0 - r1_) < 5e-4
    assert h0 - r0 < 1e-6  # 无速率项时静止保持不漂移
    assert 0.7e-4 < h1 - r1_ < 1.4e-4  # 2 s × 5.3e-5 m/s ≈ 1.06e-4 m
