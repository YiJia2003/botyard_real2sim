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
UNLOADED_BRANCH: tuple[tuple[float, float], ...] = (
    (2.5, 6.15),
    (2.94, 6.43),
    (3.29, 6.15),
    (3.74, 7.35),
    (6.0, 7.95),
    (9.0, 8.50),
)
# 加载（上行）分支：MFJ3 low-angle-hold 5° / qs-map r1,r2 楼梯。
LOADED_BRANCH: tuple[tuple[float, float], ...] = (
    (4.30, 13.95),
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
