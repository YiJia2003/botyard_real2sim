# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local Boya DexHand V3.2 articulation and actuator tables.

Copied with the hand USD at extraction. The ten flexion joints are replaced
with direct-effort inputs by scene.py; the remaining motors hold their pose.
Four *J1 joints retain the asset's mechanical *J1=*J2 mimic constraints.

The small link inertias require the inherited numerical armature. Its fixed
5e-4 s authoring reference is independent of the runtime timestep; these values
are numerical assumptions, not measured motor rotor inertia. The package carries
an independent copy of the same USD, including the fixed base, self-collisions,
and the two authored structural collision exclusions.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from botyard_real2sim.assets import BOTYARD_REAL2SIM_ASSETS_DIR

##
# Asset and joint layout.
##

USD_PATH = str(BOTYARD_REAL2SIM_ASSETS_DIR / "hand" / "boya.usda")
"""Fixed-base Boya hand, converted from ``assets/hand/boya.urdf`` (TendonSpin)."""

CONTRACT_PHYSICS_DT: float = 0.0001
"""Physics step [s]. Control frequency is set separately in env_cfg.py."""

ARMATURE_REFERENCE_DT: float = 5.0e-4
"""Reference step used to author the asset's numerical armature; independent of runtime dt."""

JOINT_NAMES: tuple[str, ...] = (
    "FAJ1",
    "FAJ3",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
)
"""All 22 joints, in contract order."""

ACTION_JOINT_NAMES: tuple[str, ...] = (
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "LFJ4",
    "LFJ3",
    "LFJ2",
)
"""13 source actions in order, followed by three locally enabled little-finger motors."""

HELD_JOINT_NAMES: tuple[str, ...] = ("FAJ1", "FAJ3")
"""Wrist motors held at their default target (no policy action)."""

PASSIVE_JOINT_NAMES: tuple[str, ...] = ("FFJ1", "MFJ1", "RFJ1", "LFJ1")
"""Proximal couplings, mimic-driven; zero gains so they never fight their master."""

FINGERTIP_BODY_NAMES: tuple[str, ...] = ("thdistal", "ffdistal", "mfdistal", "rfdistal", "lfdistal")
"""Distal link of each finger; the ``*tip`` frames are fixed geometry merged into these bodies."""


JOINT_DEFAULT_POS: dict[str, float] = {
    "FAJ1": 0.0,
    "FAJ3": 0.0,
    "THJ4": 0.0,
    "THJ3": 0.0,
    "THJ2": 0.0,
    "THJ1": 0.0,
    "FFJ4": 0.0,
    "FFJ3": 0.0,
    "FFJ2": 0.0,
    "FFJ1": 0.0,
    "MFJ4": 0.0,
    "MFJ3": 0.0,
    "MFJ2": 0.0,
    "MFJ1": 0.0,
    "RFJ4": 0.0,
    "RFJ3": 0.0,
    "RFJ2": 0.0,
    "RFJ1": 0.0,
    "LFJ4": 0.0,
    "LFJ3": 0.0,
    "LFJ2": 0.0,
    "LFJ1": 0.0,
}
"""Nominal grasp configuration from ``boya_native_contract.json`` [rad]."""

ACTUATOR_STIFFNESS: dict[str, float] = {
    "FAJ1": 40.0,
    "FAJ3": 40.0,
    **{name: 10.0 for name in ACTION_JOINT_NAMES},
}
"""Position gain per motor, matching the contract's ``gain``."""

ACTUATOR_DAMPING: dict[str, float] = {
    "FAJ1": 0.5,
    "FAJ3": 0.5,
    **{name: 0.05 for name in ACTION_JOINT_NAMES},
    # A coupled slave's dissipation maps onto its master as D * ratio**2, so the four
    # ``*J2`` motors that drive a ``*J1`` carry 0.05 + 0.05. The thumb has no coupling.
    "FFJ2": 0.1,
    "MFJ2": 0.1,
    "RFJ2": 0.1,
    "LFJ2": 0.1,
}
"""Damping per motor [N*m*s/rad], matching the contract's ``damping``."""

ACTUATOR_EFFORT_LIMIT: dict[str, float] = {
    "FAJ1": 1.0,
    "FAJ3": 1.0,
    **{name: 0.3 for name in ACTION_JOINT_NAMES},
}
"""Motor effort limit [N*m], the magnitude of the contract's ``force_range``."""

JOINT_ARMATURE: dict[str, float] = {
    name: 4.0 * ARMATURE_REFERENCE_DT * damping for name, damping in ACTUATOR_DAMPING.items()
}
"""Inherited numerical armature per motor joint [kg*m^2].

``A = 4 * D * ARMATURE_REFERENCE_DT`` uses a fixed authoring reference step. It keeps the tiny URDF
inertias solvable; see the module docstring. The same numbers are already authored into the
copied USD; keep the asset and these values in sync when changing numerical inertia.
"""


def _subset(values: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    """Restrict a per-joint table to the joints of one actuator group."""
    return {name: values[name] for name in names}


##
# Articulation.
##

BOYA_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        # Contact parameters used by the TendonSpin runtime: a stiff, low-offset contact
        # pair and a single friction coefficient for hand and object.
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5, restitution=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1.0e-4, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The USD filters only fabase/pmbase and palm/thproximal to avoid
            # structural convex-hull interference. Other self contacts stay enabled.
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    # The USD was imported with ``fix_base=True`` and its root sits at the world origin.
    # Contract hand_root.quat is wxyz; Isaac Lab 3.0 InitialStateCfg.rot is xyzw.
    # The source quaternion is approximately (0.5, 0.5, -0.5, -0.5) in wxyz.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(-0.5, -0.5, -0.5, 0.5),
        joint_pos=JOINT_DEFAULT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=list(ACTION_JOINT_NAMES),
            stiffness=_subset(ACTUATOR_STIFFNESS, ACTION_JOINT_NAMES),
            damping=_subset(ACTUATOR_DAMPING, ACTION_JOINT_NAMES),
            joint_effort_limit=_subset(ACTUATOR_EFFORT_LIMIT, ACTION_JOINT_NAMES),
            armature=_subset(JOINT_ARMATURE, ACTION_JOINT_NAMES),
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=list(HELD_JOINT_NAMES),
            stiffness=_subset(ACTUATOR_STIFFNESS, HELD_JOINT_NAMES),
            damping=_subset(ACTUATOR_DAMPING, HELD_JOINT_NAMES),
            joint_effort_limit=_subset(ACTUATOR_EFFORT_LIMIT, HELD_JOINT_NAMES),
            armature=_subset(JOINT_ARMATURE, HELD_JOINT_NAMES),
        ),
        # Zero gains and zero armature: the couplings must not fight the master motor.
        # Their velocities are damped by the ``*J2`` motor damping above, not by a second
        # controller.
        "passive": ImplicitActuatorCfg(
            joint_names_expr=list(PASSIVE_JOINT_NAMES),
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
        ),
    },
)
