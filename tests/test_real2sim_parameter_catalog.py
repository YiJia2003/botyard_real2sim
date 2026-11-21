from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from botyard_real2sim.config import load_config
from botyard_real2sim.parameter_catalog import describe_parameters

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "configs/real2sim/s1.yaml"
DEFAULT = ROOT / "configs/real2sim/default.yaml"


def _by_path(catalog):
    return {item["path"]: item for item in catalog["parameters"]}


def test_s1_catalog_marks_geometry_overrides_and_inactive_legacy_terms():
    catalog = describe_parameters(load_config(S1))
    items = _by_path(catalog)

    assert items["tendon.joint_moment_arm_m.FFJ2"]["active_status"] == "inactive"
    assert items["tendon.joint_moment_arm_m.THJ1"]["active_status"] == "active"
    assert items["tendon.spring_stiffness_nm_per_rad.FFJ2"]["active_status"] == "inactive"
    assert items["tendon.spring_stiffness_nm_per_rad.THJ1"]["active_status"] == "active"
    assert items["tendon.cable_damping_n_s_per_m.FFJ2"]["active_status"] == "unsupported"
    assert items["tendon.cable_damping_n_s_per_m.FFJ2"]["tunability"] == "locked_to_zero_in_current_model_branch"
    assert items["tendon.crossing_guide_mu.thumb"]["active_status"] == "unsupported"


def test_catalog_exposes_uncertainty_without_inventing_bounds():
    catalog = describe_parameters(load_config(S1))
    items = _by_path(catalog)

    for path in (
        "tendon.cable_stiffness_n_per_m.FFJ2",
        "tendon.routing_mass_kg.FFJ2",
        "tendon.routing_friction_mu_static.FFJ2",
        "tendon.motor_zero_angle_rad.FFJ2",
        "controller.kp_deg_s_per_deg.THJ1",
        "simulation.armature_scale",
        "simulation.effort_limit_nm",
    ):
        assert items[path]["bounds"] is None
        assert items[path]["provenance"]
        assert "kind" in items[path]

    assert items["tendon.motor_zero_angle_rad.FFJ2"]["active_status"] == "conditional"
    assert items["simulation.armature_scale"]["kind"] == "numerical"
    assert items["simulation.effort_limit_nm"]["kind"] == "numerical_guard"
    assert items["tendon.routing_max_substep_s"]["current_value"] == 0.0001
    json.dumps(catalog, allow_nan=False)


def test_quasistatic_preset_reports_legacy_terms_and_motor_geometry_active():
    config = load_config(DEFAULT)
    config = replace(config, tendon=replace(config.tendon, cable_damping_n_s_per_m=(0.2,) * 10))
    catalog = describe_parameters(config)
    items = _by_path(catalog)

    assert catalog["config"]["routing_model"] == "quasistatic"
    assert items["tendon.cable_damping_n_s_per_m.FFJ2"]["active_status"] == "active"
    assert items["tendon.spring_stiffness_nm_per_rad.FFJ2"]["active_status"] == "active"
    assert items["tendon.joint_moment_arm_m.FFJ2"]["active_status"] == "active"


def test_parameters_cli_emits_json_without_engine_imports():
    env = {"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "botyard_real2sim.cli.parameters", "--config", str(S1)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    catalog = json.loads(result.stdout)
    assert catalog["schema_version"] == 1
    assert "torch" not in result.stderr
    assert "isaaclab" not in result.stderr


def test_driver_catalog_distinguishes_disabled_legacy_gate_and_firmware_settings():
    config = load_config(S1)
    items = _by_path(describe_parameters(config))
    for path in (
        "controller.min_angle_deg.MFJ3",
        "controller.max_angle_deg.MFJ3",
        "controller.motor_velocity_deadband_rad_s.MFJ3",
        "controller.motor_gear_ratio.MFJ3",
        "controller.motor_spool_speed_limit_rad_s.MFJ3",
    ):
        assert items[path]["kind"] == "controller"
        assert items[path]["tunability"] == "lock_to_verified_firmware"
        assert items[path]["active_status"] == "active"
    legacy = replace(
        config, controller=replace(config.controller, motor_velocity_deadband_rad_s=0, motor_spool_speed_limit_rad_s=0)
    )
    legacy_items = _by_path(describe_parameters(legacy))
    assert legacy_items["controller.motor_velocity_deadband_rad_s.MFJ3"]["active_status"] == "inactive"
    assert legacy_items["controller.motor_gear_ratio.MFJ3"]["active_status"] == "inactive"
    assert legacy_items["controller.motor_spool_speed_limit_rad_s.MFJ3"]["active_status"] == "inactive"
