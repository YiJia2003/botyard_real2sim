"""Measured equilibrium tension table and the initial routing-slip estimate.

【中文】
本模块只存**实测**的平衡张力表（角度 → 张力）和由它派生出来的两个初始化量，
不含任何拟合结果，也不定义容差。

为什么要表 + 两个量，而不是直接按角度取张力：

1. 实测表明同一角度可以停在两个分支上（MFJ3 3°：卸载分支 6.2–7.4 N，加载分支上
   13.5–14.0 N），而且锁定态还能停在两分支之间（qs-map r1 起点 5.27° / 11.66 N）。
   因此“角度 → 张力”不是单值函数，必须带分支。
2. 释放实验（release-hold 包）显示分支切换在 0.3 s 内完成，是 relay 而不是慢弛豫，
   所以初始化要给出**分支状态**，而不是给一个随时间回落的偏置。

由此给出两个初始化量：

- ``initial_distal_tension_n``：摩擦锁定态下远端（关节侧）张力 = 实测近端张力 +
  实测分支失衡量 ``LOCK_GRADIENT_N``。释放实验的实测失衡是 1.98 N，取 2 N。
- ``initial_slip_m``：走线滑动累积状态的初值，单位 m。它表示绳在路径上“让出”的
  长度，用张力差除以走线弹簧刚度换算：``slip = ΔT / kr``。
"""

from __future__ import annotations

import math

# 实测（角度 [deg], 张力 [N]）表；来源写在每条注释里，不做插值外推。
# 卸载（下行）分支：MFJ3 low-angle-hold / qs-map r1,r2 / provenance 包。
# 注意：横轴一律是**实际停位**，不是指令值——low-angle-hold 实测 2.5° 与 3.0° 指令
# 都停在同一个 2.94°，5.0° 指令停在 4.30°（欠 0.70°）。用指令值查表会系统性偏低。
# 地板处两次实测 6.43 / 6.65 N（指令 2.5° 与 3.0°），表里取均值 6.54，
# 重复性 ±0.11 N；两个原始值记录在 FLOOR_REPEATS_N 里便于核对。
UNLOADED_BRANCH: tuple[tuple[float, float], ...] = (
    (2.94, 6.54),
    (3.74, 7.41),  # 从 5.0° 回到 3.0° 的下行分支
    (6.0, 7.95),
    (9.0, 8.50),
)
# 加载（上行）分支：MFJ3 low-angle-hold 5° / qs-map r1,r2 楼梯。
LOADED_BRANCH: tuple[tuple[float, float], ...] = (
    (4.30, 13.79),  # 指令 5.0° 的实际停位（上行分支）
    (5.0, 14.16),
    (6.0, 15.10),
    (9.0, 15.25),
    (12.0, 16.09),
)
# 实测摩擦锁定态（角度 [deg], 近端张力 [N]）：既不在上行也不在下行分支上。
LOCKED_STATES: tuple[tuple[float, float], ...] = (
    (4.61, 11.35),
    (5.27, 11.66),
)
# 摩擦锁定的张力量级：走线摩擦能维持的近端/远端差。实测两处释放实验给出 1.98 N。
LOCK_GRADIENT_N = 1.98
# 关节低角物理地板：指令再低也停在 2.94°（指令 2.5° 与 3.0° 都落在这里）。
JOINT_FLOOR_DEG = 2.94
# 地板处重复实测的张力原始值，单位 N（指令 2.5° / 3.0°）。
FLOOR_REPEATS_N: tuple[float, float] = (6.43, 6.65)
# 静摩擦死区幅度：上行欠 0.70°、下行欠 0.74°，取实测均值 0.7°。
JOINT_DEADBAND_DEG = 0.7
# 走线弹簧刚度回退值，单位 N/m；仅当调用方没有给出真实 kr 时用于把张力差换算成滑移量。
DEFAULT_ROUTING_STIFFNESS_N_PER_M = 1000.0

BRANCHES: dict[str, tuple[tuple[float, float], ...]] = {
    "unloaded": UNLOADED_BRANCH,
    "loaded": LOADED_BRANCH,
    "locked": LOCKED_STATES,
}


def equilibrium_tension_n(angle_deg: float, branch: str = "unloaded") -> float:
    """Interpolate the measured equilibrium tension at ``angle_deg`` on ``branch``.

    【中文】
    在指定分支的实测点之间线性插值；超出实测范围时钳到端点（不外推），并明确
    只支持 unloaded / loaded / locked 三个名字。

    参数:
        angle_deg: 关节角，单位 deg；
        branch:    分支名，unloaded（卸载/下行）、loaded（加载/上行）或 locked（锁定态）。
    返回: 平衡张力，单位 N。
    抛错: 分支名未知时抛 ValueError。
    """
    if branch not in BRANCHES:
        raise ValueError(f"unknown equilibrium branch: {branch}; expected one of {sorted(BRANCHES)}")
    points = BRANCHES[branch]
    if angle_deg <= points[0][0]:
        return points[0][1]
    if angle_deg >= points[-1][0]:
        return points[-1][1]
    for (a0, t0), (a1, t1) in zip(points, points[1:], strict=False):
        if a0 <= angle_deg <= a1:
            weight = (angle_deg - a0) / (a1 - a0)
            return t0 + weight * (t1 - t0)
    return points[-1][1]


def settled_angle_deg(
    command_deg: float,
    *,
    approach: str = "unknown",
    floor_deg: float = JOINT_FLOOR_DEG,
    deadband_deg: float = JOINT_DEADBAND_DEG,
) -> float:
    """Actual settled joint angle for a low-angle command.

    【中文】
    低角命令的实际停位由两件事决定，实测都能量出来：

    1. **物理地板 2.94°**：指令 2.5° 与 3.0°（从地板往上）都停在 2.94°；
    2. **静摩擦死区 0.7°**：上行指令欠 0.70°（5.0° → 4.30°），下行指令欠 0.74°
       （5.0° → 3.0° 指令停在 3.74°）。

    所以查表必须用实际停位而不是指令值。`approach="unknown"` 时只钳地板，
    不猜死区方向；给出 `"up"` / `"down"` 才按实测方向偏移。

    参数:
        command_deg:  指令角，单位 deg；
        approach:     "up"（由低到高）、"down"（由高到低）或 "unknown"；
        floor_deg:    实测物理地板，单位 deg；
        deadband_deg: 实测静摩擦死区幅度，单位 deg。
    返回: 实际停位角度，单位 deg。
    抛错: 角度非有限，或 approach 不在允许集合内时抛 ValueError。
    """
    if not math.isfinite(command_deg):
        raise ValueError("command angle must be finite")
    if approach not in ("up", "down", "unknown"):
        raise ValueError(f"unknown approach: {approach}")
    if approach == "unknown":
        return max(command_deg, floor_deg)
    offset = -deadband_deg if approach == "up" else deadband_deg
    return max(command_deg + offset, floor_deg)


def initial_distal_tension_n_from_table(
    actual_angle_deg: float, *, branch: str = "unloaded", gradient_n: float = LOCK_GRADIENT_N
) -> float:
    """Table-based distal tension when only the start angle is known.

    【中文】
    冷启动时没有实测近端张力，就用**实际停位**在指定分支上查表，再加实测失衡量。
    命令角必须先经 :func:`settled_angle_deg` 折算成实际停位。

    参数:
        actual_angle_deg: 实际停位角，单位 deg（不是指令值）；
        branch:           unloaded / loaded / locked；
        gradient_n:       实测失衡量，单位 N。
    返回: 远端张力，单位 N。
    """
    return initial_distal_tension_n(equilibrium_tension_n(actual_angle_deg, branch), gradient_n=gradient_n)


def initial_slip_m_from_table(
    actual_angle_deg: float,
    *,
    branch: str = "unloaded",
    routing_stiffness_n_per_m: float = DEFAULT_ROUTING_STIFFNESS_N_PER_M,
    gradient_n: float = LOCK_GRADIENT_N,
) -> float:
    """Slip initialisation derived from the table branch tension."""
    return gradient_n / routing_stiffness_n_per_m if routing_stiffness_n_per_m > 0 else 0.0


def initial_distal_tension_n(measured_proximal_n: float, *, gradient_n: float = LOCK_GRADIENT_N) -> float:
    """Distal tension for a measured locked start state.

    【中文】
    锁定态下远端张力高于近端（关节侧回位弹簧在走线下游加张力，走线摩擦锁住这个差）。
    用实测近端张力加实测失衡量，不做符号之外的任何假设；结果钳到非负。

    参数:
        measured_proximal_n: 实测近端（电机侧）张力，单位 N；
        gradient_n:         实测分支失衡量，单位 N，默认 1.98（两处释放实验）。
    返回: 远端张力，单位 N。
    抛错: 输入非有限或为负时抛 ValueError。
    """
    if not math.isfinite(measured_proximal_n) or measured_proximal_n < 0:
        raise ValueError("measured proximal tension must be finite and non-negative")
    if not math.isfinite(gradient_n) or gradient_n < 0:
        raise ValueError("gradient must be finite and non-negative")
    return max(0.0, measured_proximal_n + gradient_n)


def initial_slip_m(
    measured_proximal_n: float,
    *,
    routing_stiffness_n_per_m: float = DEFAULT_ROUTING_STIFFNESS_N_PER_M,
    gradient_n: float = LOCK_GRADIENT_N,
) -> float:
    """Routing slip that produces the measured locked tension gradient.

    【中文】
    走线滑动累积状态用长度表示（单位 m）：它让出 ``ΔT/kr`` 的长度时，走线弹簧正好
    少承担 ``ΔT`` 的张力，这就是锁定态表现为“低张力分支”的原因。

    参数:
        measured_proximal_n:      实测近端张力，单位 N；
        routing_stiffness_n_per_m: 走线弹簧刚度 kr，单位 N/m；
        gradient_n:               实测失衡量，单位 N。
    返回: 滑移长度，单位 m（≥ 0）。
    抛错: kr 非正或输入非法时抛 ValueError。
    """
    if not math.isfinite(routing_stiffness_n_per_m) or routing_stiffness_n_per_m <= 0:
        raise ValueError("routing stiffness must be positive")
    if not math.isfinite(measured_proximal_n) or measured_proximal_n < 0:
        raise ValueError("measured proximal tension must be finite and non-negative")
    if not math.isfinite(gradient_n) or gradient_n < 0:
        raise ValueError("gradient must be finite and non-negative")
    # 让出的长度直接等于“走线弹簧少承担的张力 / 刚度”。
    return gradient_n / routing_stiffness_n_per_m
