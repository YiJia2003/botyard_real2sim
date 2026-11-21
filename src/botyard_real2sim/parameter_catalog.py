"""Machine-readable descriptions of parameters relevant to real2sim fitting.

This catalog reports runtime activity and known identifiability constraints.
It intentionally leaves search bounds unset unless the repository documents a
measured or otherwise justified range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FINGER_NAMES, JOINT_NAMES, Real2SimConfig, load_config
from .paths import default_config_path


@dataclass(frozen=True)
class _Definition:
    section: str
    field: str
    unit: str | None
    description: str
    kind: str = "physical"
    tunability: str = "fit_after_data_validation"
    correlation: str | None = None
    provenance: str = "configs/real2sim/s1.yaml; parameter meaning in docs/REAL2SIM_GEOMETRY.md"


_DEFINITIONS = (
    _Definition(
        "tendon",
        "cable_stiffness_n_per_m",
        "N/m",
        "Intrinsic cable tensile stiffness",
        correlation=(
            "Correlated with routing spring stiffness and routing travel when absolute spool length is unknown."
        ),
    ),
    _Definition(
        "tendon",
        "cable_damping_n_s_per_m",
        "N*s/m",
        "Intrinsic cable damping",
        tunability="unsupported_in_active_dynamic_model",
        correlation="Must be zero in dynamic routing; damping with an active routing spring is rejected.",
    ),
    _Definition(
        "tendon",
        "cable_slack_m",
        "m",
        "Initial cable slack",
        correlation="Correlated with initial preload and motor zero/reference.",
    ),
    _Definition(
        "tendon",
        "initial_tension_n",
        "N",
        "Configured initial cable tension",
        correlation="Conflicts with positive cable slack; measured initial feedback can override it.",
    ),
    _Definition(
        "tendon",
        "tendon_speed_lag_s",
        "s",
        "First-order lag at the ideal motor speed port",
        correlation=(
            "May correlate with sensor delay/filter and controller timing; it is not motor electrical dynamics."
        ),
    ),
    _Definition(
        "tendon",
        "motor_zero_angle_rad",
        "rad",
        "Unwrapped spool encoder reference",
        tunability="conditional_on_missing_motor_feedback",
        correlation="When initial motor angle is measured, reset estimates and replaces this configured offset.",
    ),
    _Definition(
        "tendon",
        "motor_encoder_sign",
        "1",
        "Encoder direction sign",
        correlation="Resolve from known motor motion; do not fit together with direction ambiguity.",
    ),
    _Definition(
        "tendon",
        "drum_radius_m",
        "m",
        "Winding spool radius",
        correlation="Correlated with encoder scale/gear ratio and cable displacement scale.",
    ),
    _Definition(
        "tendon",
        "joint_moment_arm_m",
        "m",
        "Fallback constant joint moment arm",
        tunability="active_only_without_configured_tendon_geometry",
        correlation="Configured link-attached tendon geometry replaces this matrix entry.",
    ),
    _Definition(
        "tendon",
        "crossing_moment_arm_m",
        "m",
        "Fallback coupled-joint moment arm",
        tunability="active_only_without_configured_tendon_geometry",
        correlation=(
            "Configured path geometry supplies the path Jacobian; confirm which geometry path represents "
            "the crossing route."
        ),
    ),
    _Definition(
        "tendon",
        "routing_spring_stiffness_n_per_m",
        "N/m",
        "Equivalent routing spring stiffness",
        correlation="Correlated with intrinsic cable stiffness and force scale.",
    ),
    _Definition(
        "tendon",
        "routing_spring_free_length_m",
        "m",
        "Equivalent routing spring free length",
        correlation="Only travel relative to maximum length is identifiable without an absolute spool/path reference.",
    ),
    _Definition(
        "tendon",
        "routing_spring_max_length_m",
        "m",
        "Equivalent routing spring maximum length",
        correlation="Only travel relative to free length is identifiable without an absolute spool/path reference.",
    ),
    _Definition("tendon", "routing_mass_kg", "kg", "Total lumped routing mass across two nodes"),
    _Definition(
        "tendon",
        "routing_proximal_compliance_fraction",
        "1",
        "Share of material compliance assigned to proximal segment",
        correlation=(
            "Only weakly observable from motor-side tension alone; distal tension or additional excitation helps."
        ),
    ),
    _Definition(
        "tendon",
        "routing_friction_mu_static",
        "1",
        "Static routing friction coefficient",
        correlation=(
            "Correlated with dynamic coefficient, contact/wrap parameters, joint friction, and tension calibration."
        ),
    ),
    _Definition(
        "tendon",
        "routing_friction_mu_dynamic",
        "1",
        "Dynamic routing friction coefficient",
        correlation=(
            "Correlated with static coefficient, contact/wrap parameters, joint friction, and tension calibration."
        ),
    ),
    _Definition(
        "tendon",
        "routing_bristle_stiffness_n_per_m",
        "N/m",
        "Pre-slip routing friction stiffness",
        correlation="Correlated with bristle state initialization and contact-dependent friction bounds.",
    ),
    _Definition("tendon", "routing_viscous_friction_n_s_per_m", "N*s/m", "Velocity-dependent routing friction"),
    _Definition("tendon", "routing_stribeck_speed_m_s", "m/s", "Routing friction transition speed"),
    _Definition(
        "tendon",
        "routing_normal_preload_n",
        "N",
        "Normal-force contribution independent of tendon tension",
        correlation="May be confounded with routing friction coefficients.",
    ),
    _Definition("tendon", "routing_wrap_base_rad", "rad", "Base equivalent routing contact wrap"),
    _Definition("tendon", "routing_wrap_contact_gain_rad", "rad", "Change in equivalent wrap with contact state"),
    _Definition(
        "tendon",
        "routing_contact_baseline",
        "1",
        "Baseline routing contact state",
        correlation="Correlated with wrap and friction coefficients.",
    ),
    _Definition("tendon", "routing_contact_tension_gain", "1", "Contact-state response to tension"),
    _Definition("tendon", "routing_contact_extension_gain", "1", "Contact-state response to routing spring extension"),
    _Definition("tendon", "routing_contact_path_gain", "1", "Contact-state response to geometric path change"),
    _Definition("tendon", "routing_contact_tension_scale_n", "N", "Tension scale for contact-state response"),
    _Definition("tendon", "routing_contact_path_scale_m", "m", "Path scale for contact-state response"),
    _Definition("tendon", "routing_contact_load_tau_s", "s", "Contact formation time constant"),
    _Definition("tendon", "routing_contact_release_tau_s", "s", "Contact release time constant"),
    _Definition(
        "tendon",
        "return_spring_stiffness_n_per_m",
        "N/m",
        "Geometric return spring stiffness",
        correlation="Correlated with return-spring free length, installed path, and preload.",
    ),
    _Definition(
        "tendon",
        "return_spring_free_length_m",
        "m",
        "Geometric return spring free length",
        correlation="Correlated with installed path and spring preload.",
    ),
    _Definition(
        "tendon",
        "return_spring_initial_tension_n",
        "N",
        "Geometric return spring initial tension",
        correlation="Correlated with return-spring stiffness and free length.",
    ),
    _Definition("tendon", "return_tether_length_m", "m", "Short tether length before geometric return spring"),
    _Definition(
        "tendon",
        "return_path_length_offset_m",
        "m",
        "Unmodelled hook/path length correction",
        tunability="fit_only_with_absolute_length_reference",
        correlation="Trades directly with return tether/free length; does not change the moment arm.",
    ),
    _Definition(
        "tendon",
        "spring_rest_angle_rad",
        "rad",
        "Legacy torsional spring rest angle",
        tunability="active_only_without_geometric_return_path",
        correlation="Replaced per channel by the geometric return spring when a return path is configured.",
    ),
    _Definition(
        "tendon",
        "spring_stiffness_nm_per_rad",
        "N*m/rad",
        "Legacy torsional spring stiffness",
        tunability="active_only_without_geometric_return_path",
        correlation="Replaced per channel by the geometric return spring when a return path is configured.",
    ),
    _Definition(
        "tendon",
        "spring_preload_nm",
        "N*m",
        "Legacy torsional spring preload",
        tunability="active_only_without_geometric_return_path",
        correlation="Replaced per channel by the geometric return spring when a return path is configured.",
    ),
    _Definition(
        "tendon",
        "viscous_friction_nm_s_per_rad",
        "N*m*s/rad",
        "Joint-axis viscous friction",
        correlation="Can trade with routing friction and sensor filtering.",
    ),
    _Definition(
        "tendon",
        "coulomb_friction_nm",
        "N*m",
        "Joint-axis Coulomb friction",
        correlation="Correlated with static friction and tendon force/moment-arm calibration.",
    ),
    _Definition(
        "tendon",
        "static_friction_nm",
        "N*m",
        "Joint-axis static friction hold band",
        correlation=(
            "Holds the joint while |tendon + spring| stays inside the band and blends into the "
            "Stribeck law above friction_smoothing_speed_rad_s; correlated with coulomb_friction_nm."
        ),
    ),
    _Definition("tendon", "stribeck_speed_rad_s", "rad/s", "Joint friction transition speed"),
    _Definition(
        "tendon",
        "friction_smoothing_speed_rad_s",
        "rad/s",
        "Numerical smoothing speed for joint friction",
        kind="numerical",
        tunability="hold_fixed_during_physical_fit",
    ),
    _Definition(
        "tendon",
        "crossing_guide_mu",
        "1",
        "Per-guide friction coefficient",
        tunability="unsupported_by_current_model",
        correlation="Nonzero values are rejected until segmented guide geometry is implemented and validated.",
    ),
    _Definition(
        "tendon",
        "routing_model",
        None,
        "Routing model structure: quasistatic or dynamic",
        kind="model_choice",
        tunability="compare_as_separate_model_candidate",
        correlation="Model choice changes state and equations; compare on held-out experiments.",
    ),
    _Definition(
        "controller",
        "kp_deg_s_per_deg",
        "(deg/s)/deg",
        "Fallback position controller proportional gain",
        kind="controller",
        tunability="lock_to_verified_firmware",
        correlation=(
            "Controller values must match the captured firmware; do not use them to absorb mechanical residuals."
        ),
    ),
    _Definition(
        "controller",
        "ki_deg_s_per_deg_step",
        "(deg/s)/(deg*step)",
        "Fallback position controller integral gain",
        kind="controller",
        tunability="lock_to_verified_firmware",
    ),
    _Definition(
        "controller",
        "kd_deg_s_per_deg_step",
        "(deg/s)/(deg*step)",
        "Fallback position controller derivative gain",
        kind="controller",
        tunability="lock_to_verified_firmware",
    ),
    _Definition(
        "controller",
        "control_period_s",
        "s",
        "Fallback controller update period",
        kind="controller",
        tunability="lock_to_measured_firmware_schedule",
    ),
    _Definition(
        "controller",
        "firmware_time_delta_s",
        "s",
        "Firmware estimator integer time delta",
        kind="controller",
        tunability="lock_to_verified_firmware",
    ),
    _Definition(
        "controller",
        "dead_zone_deg",
        "deg",
        "Fallback controller angle dead zone",
        kind="controller",
        tunability="lock_to_verified_firmware",
    ),
    _Definition(
        "controller",
        "speed_cap_deg_s",
        "deg/s",
        "Fallback controller speed cap",
        kind="controller",
        tunability="lock_to_verified_firmware",
    ),
    *(
        _Definition(
            "controller",
            field,
            unit,
            description,
            kind="controller",
            tunability="lock_to_verified_firmware",
            provenance="docs/REAL2SIM_CONTROLLER_SYNC.md; verify against the recording's firmware/configuration",
        )
        for field, unit, description in (
            ("min_angle_deg", "deg", "H7 target-angle floor (not a mechanical joint limit)"),
            ("max_angle_deg", "deg", "H7 target-angle ceiling (not a mechanical joint limit)"),
            ("motor_velocity_deadband_rad_s", "rad/s (rotor)", "G4 velocity-target deadband; zero disables it"),
            ("motor_gear_ratio", "1", "G4 rotor-to-spool speed ratio used for the command deadband"),
            ("motor_spool_speed_limit_rad_s", "rad/s (spool)", "G4 module velocity cap; zero disables it"),
        )
    ),
)


_NUMERICAL_SETTINGS = (
    {
        "path": "simulation.physics_dt_s",
        "unit": "s",
        "description": "PhysX integration timestep (CLI setting, not a YAML physical parameter)",
        "kind": "numerical",
        "current_value": None,
        "default_value": 0.0001,
        "active_status": "active",
        "active_reason": "Replay CLI default; the actual value can be overridden and is stored in run metadata.",
        "tunability": "convergence_study_only",
        "bounds": None,
        "correlation": "Do not fit as a physical parameter; compare curves under timestep refinement.",
        "provenance": "src/botyard_real2sim/cli/replay.py",
    },
    {
        "path": "tendon.routing_max_substep_s",
        "unit": "s",
        "description": "Maximum internal transmission solver substep",
        "kind": "numerical",
        "current_value": None,
        "active_status": "active",
        "active_reason": "Read from the supplied configuration.",
        "tunability": "convergence_study_only",
        "bounds": None,
        "correlation": "Do not fit as a physical parameter; verify sensitivity by refinement.",
        "provenance": "src/botyard_real2sim/config.py",
    },
    {
        "path": "simulation.armature_scale",
        "unit": "1",
        "description": "Scale for inherited numerical joint armature",
        "kind": "numerical",
        "current_value": None,
        "default_value": 1.0,
        "active_status": "active",
        "active_reason": "Replay CLI default; the actual value can be overridden and is stored in run metadata.",
        "tunability": "hold_fixed_during_physical_fit",
        "bounds": None,
        "correlation": "Not a measured motor rotor inertia.",
        "provenance": "src/botyard_real2sim/scene.py; src/botyard_real2sim/cli/replay.py",
    },
    {
        "path": "simulation.effort_limit_nm",
        "unit": "N*m",
        "description": "Numerical joint-effort safeguard",
        "kind": "numerical_guard",
        "current_value": None,
        "default_value": 2.0,
        "active_status": "active",
        "active_reason": "Replay CLI default; the actual value can be overridden and is stored in run metadata. "
        "Any clipping makes the run unsuitable for physical fitting.",
        "tunability": "guard_only",
        "bounds": None,
        "correlation": "Not a tendon tension limit or physical parameter.",
        "provenance": "src/botyard_real2sim/scene.py; src/botyard_real2sim/cli/replay.py",
    },
)


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _finger_for_joint(joint: str) -> str:
    return {"TH": "thumb", "FF": "index", "MF": "middle", "RF": "ring", "LF": "little"}[joint[:2]]


def _activity(spec: _Definition, joint: str | None, config: Real2SimConfig) -> tuple[str, str]:
    if spec.section == "controller" and spec.field in {
        "motor_velocity_deadband_rad_s",
        "motor_gear_ratio",
        "motor_spool_speed_limit_rad_s",
    }:
        field = "motor_velocity_deadband_rad_s" if spec.field == "motor_gear_ratio" else spec.field
        if getattr(config.controller, field)[JOINT_NAMES.index(joint)] == 0:
            return "inactive", "The corresponding G4 speed gate is disabled (legacy ideal speed port)."
        return "active", "Applied by Real2SimRuntime after H7 output quantization, before the tendon speed port."
    tendon = config.tendon
    has_geometry = bool(tendon.geometry.get(_finger_for_joint(joint))) if joint else False
    if spec.field in {"joint_moment_arm_m", "crossing_moment_arm_m"}:
        if spec.field == "crossing_moment_arm_m":
            return (
                "partially_active",
                "Used only by fallback constant coupling paths; configured link-attached geometry replaces "
                "covered tendon path Jacobians. Confirm crossing topology before fitting.",
            )
        if has_geometry:
            return "inactive", "The configured link-attached tendon path supplies this joint's Jacobian."
        return "active", "This channel has no configured link-attached tendon geometry."
    if spec.field in {"spring_rest_angle_rad", "spring_stiffness_nm_per_rad", "spring_preload_nm"}:
        if has_geometry:
            return (
                "inactive",
                "A configured geometric return path replaces the legacy torsional spring for this channel.",
            )
        return (
            "active",
            "No geometric tendon path is configured for this channel; the legacy torsional term remains available.",
        )
    if spec.field == "cable_damping_n_s_per_m" and (
        tendon.routing_model == "dynamic"
        or tendon.routing_spring_max_length_m[JOINT_NAMES.index(joint)]
        > tendon.routing_spring_free_length_m[JOINT_NAMES.index(joint)]
    ):
        return "unsupported", "Nonzero cable damping is rejected for dynamic routing or an active routing spring."
    if spec.field == "crossing_guide_mu":
        return "unsupported", "Any nonzero value is rejected until segmented guide losses are implemented."
    if spec.field == "motor_zero_angle_rad":
        return (
            "conditional",
            "Configured value is used without measured initial motor angle; reset estimates it from measured "
            "motor angle when present.",
        )
    if spec.field == "routing_model":
        return "active", f"Selects the currently active transmission branch ({tendon.routing_model})."
    return "active", "Read by the current configuration/model path."


def describe_parameters(config: Real2SimConfig | None = None) -> dict[str, Any]:
    """Return parameter descriptions and their effective status for a config."""
    config = config or load_config(default_config_path())
    output: list[dict[str, Any]] = []
    for spec in _DEFINITIONS:
        values = getattr(getattr(config, spec.section), spec.field)
        if isinstance(values, tuple):
            names = JOINT_NAMES if len(values) == len(JOINT_NAMES) else FINGER_NAMES
            for index, name in enumerate(names):
                path = f"{spec.section}.{spec.field}.{name}"
                joint = name if name in JOINT_NAMES else None
                status, reason = _activity(spec, joint, config)
                tunability = spec.tunability
                if spec.field == "cable_damping_n_s_per_m" and status == "active":
                    tunability = "fit_only_in_supported_quasistatic_combinations"
                elif spec.field == "cable_damping_n_s_per_m" and status == "unsupported":
                    tunability = "locked_to_zero_in_current_model_branch"
                output.append(
                    {
                        "path": path,
                        "group_path": f"{spec.section}.{spec.field}",
                        "unit": spec.unit,
                        "description": spec.description,
                        "kind": spec.kind,
                        "current_value": _plain(values[index]),
                        "active_status": status,
                        "active_reason": reason,
                        "tunability": tunability,
                        "bounds": None,
                        "correlation": spec.correlation,
                        "provenance": spec.provenance,
                    }
                )
        else:
            output.append(
                {
                    "path": f"{spec.section}.{spec.field}",
                    "unit": spec.unit,
                    "description": spec.description,
                    "kind": spec.kind,
                    "current_value": _plain(values),
                    "active_status": "active",
                    "active_reason": "Read by the current configuration/model path.",
                    "tunability": spec.tunability,
                    "bounds": None,
                    "correlation": spec.correlation,
                    "provenance": spec.provenance,
                }
            )
    # Geometry is structured and has mixed units; expose each finger group as a
    # single record instead of flattening nodes into an unsafe fit parameter list.
    for finger, geometry in config.tendon.geometry.items():
        output.append(
            {
                "path": f"tendon.geometry.{finger}",
                "unit": "mixed (coordinates m, angles rad, radii m)",
                "description": "Link-attached tendon/return path geometry and guide topology",
                "kind": "physical_geometry",
                "current_value": _plain(geometry),
                "active_status": "active",
                "active_reason": "Path lengths/Jacobians are used by the plant for this finger.",
                "tunability": "fit_with_CAD_prior_and_direction_checks",
                "bounds": None,
                "correlation": "Geometry affects both motor displacement and joint torque; verify topology and sign "
                "before fitting coordinates.",
                "provenance": "configs/real2sim/s1.yaml; docs/REAL2SIM_GEOMETRY.md",
            }
        )
    for item in _NUMERICAL_SETTINGS:
        record = dict(item)
        if record["path"] == "tendon.routing_max_substep_s":
            record["current_value"] = config.tendon.routing_max_substep_s
        output.append(record)
    output.append(
        {
            "path": "tendon.distal_mimic_ratio",
            "unit": "1",
            "description": "Configured distal mimic ratio",
            "kind": "physical_constraint",
            "current_value": _plain(
                {key: value.get("distal_mimic_ratio", 1.0) for key, value in config.tendon.geometry.items()}
            ),
            "active_status": "must_match_asset",
            "active_reason": "The bundled USD enforces J1=J2 and replay checks a 1:1 relation; changing this ratio "
            "alone makes geometry and rigid constraint inconsistent.",
            "tunability": "locked_to_USD_mimic_constraint",
            "bounds": None,
            "correlation": None,
            "provenance": "src/botyard_real2sim/simulation/hand_cfg.py; src/botyard_real2sim/cli/replay.py",
        }
    )
    return {
        "schema_version": 1,
        "config": {
            "routing_model": config.tendon.routing_model,
            "geometry_fingers": sorted(config.tendon.geometry),
            "parameter_bounds_policy": "null means no repository-sourced physical search bound is available; "
            "supply evidence-backed bounds before fitting",
        },
        "parameters": output,
    }


def describe_parameters_from_path(path: str | None = None) -> dict[str, Any]:
    """Load and describe a YAML config, defaulting to the packaged S1 preset."""
    config = load_config(path) if path else load_config(default_config_path())
    return describe_parameters(config)


__all__ = ["describe_parameters", "describe_parameters_from_path"]
