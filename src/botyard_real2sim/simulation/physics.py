"""The PhysX solver settings used by the source hand project, without RL imports."""

from isaaclab_physx.physics import PhysxCfg


def make_physx_cfg():
    """Return a fresh engine config, equivalent to the extracted PhysX preset."""
    return PhysxCfg(solver_type=1, min_position_iteration_count=16, min_velocity_iteration_count=4)
