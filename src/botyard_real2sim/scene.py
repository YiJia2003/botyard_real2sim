"""Isaac Lab scene configuration for unloaded tendon calibration.

This scene uses the extracted local hand USD and PhysX setup, while replacing
position drives on the ten tendon joints with an explicit effort input.  The four
``*J1`` mimic constraints remain in the USD: they represent the rigid J1=J2 linkage.

【中文说明】
文件职责：为 real2sim“空载腱绳标定”构造 Isaac Lab 场景配置。它不新建机器人
描述，而是使用本项目自带的 BOYA_HAND_CFG（独立 USD 副本、提取时的初始状态
与增益表），把十个屈曲关节的驱动方式换成“显式力矩输入”。

改造方式（关键是“不重叠”地拆分执行器组）：
- tendon_effort：10 个腱关节（JOINT_NAMES）用 IdealPDActuatorCfg，kp = kd = 0，
  完全由外部腱绳模型显式施加 N·m 力矩，PhysX 不再做位置保持。
- fingers：剩下的 6 个非腱指关节继续用 ImplicitActuatorCfg 做隐式位置保持
  （沿用原 stiffness/damping/effort_limit/armature 表）。
- 手腕与被动组不改动，沿用 BOYA_HAND_CFG 原配置。
- 4 条 ``*J1`` mimic 约束仍由 USD 提供（刚性 J1=J2 连杆），不在场景里重复建模。

单位与量级：力矩 N·m；对手指关节，``effort_limit_nm`` 默认 2.0 N·m 只是数值
保护（5 mm 绕线轮半径下，100 N 腱张力仅产生 0.5 N·m），不是真实腱绳张力上限。
``armature_scale`` 是暴露出来的数值惯量旋钮（无量纲倍率），用于稳定显式力矩
积分，不代表真实电机转子惯量。

输入/输出：输入为力矩上限（N·m）与惯量倍率；输出为 NoLoadSceneCfg（单环境、
无抓取物体/相机/奖励/自动复位的固定基座手部场景）。
"""

# ── 标准库与 Isaac Lab 依赖 ──
from __future__ import annotations

import math

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# ── 新项目自带的手部 USD 与增益表（提取时保留原物理参数）──
from botyard_real2sim.simulation.hand_cfg import (
    ACTION_JOINT_NAMES,
    ACTUATOR_DAMPING,
    ACTUATOR_EFFORT_LIMIT,
    ACTUATOR_STIFFNESS,
    BOYA_HAND_CFG,
    JOINT_ARMATURE,
)

# 腱绳模型驱动的 10 个屈曲关节名，作为执行器拆分的分界线。
from .config import JOINT_NAMES


# ── 场景配置类：固定基座、无抓取物体、无相机/奖励/自动复位 ──
@configclass
class NoLoadSceneCfg(InteractiveSceneCfg):
    """Fixed-base hand without grasp objects, camera, rewards, or automatic reset.

    【中文】
    空载标定场景：固定基座手部，不含抓取物体、相机、奖励项与自动复位逻辑。

    输入：InteractiveSceneCfg 的通用字段（num_envs、env_spacing 等）。
    输出/副作用：``robot`` 为 BOYA_HAND_CFG 的拷贝，因此沿用提取时的
    初始关节状态和 PhysX 设置，USD 路径指向本项目内的独立副本；后续 make_no_load_scene_cfg
    会覆盖其执行器分组。
    """

    robot: ArticulationCfg = BOYA_HAND_CFG.copy()


# ── 工厂函数：把 10 个腱关节从位置驱动改为纯显式力矩驱动 ──
def make_no_load_scene_cfg(*, effort_limit_nm: float = 2.0, armature_scale: float = 1.0) -> NoLoadSceneCfg:
    """Split tendon joints from the legacy position actuator without overlap.

    ``effort_limit_nm`` is a numerical joint-side safeguard, not the command's
    tendon tension limit.  At 5 mm, 100 N from one tendon already produces
    0.5 N m, so the grasp task's 0.3 N m drive cap must not be reused here.

    【中文】
    构造空载标定场景配置：把 10 个腱关节从原来的位置执行器组里“摘出来”，
    换成纯显式力矩执行器，且两组关节名集合互不重叠（避免同一关节被两个
    执行器同时驱动）。

    参数：
        effort_limit_nm: 关节侧力矩上限（N·m），默认 2.0。它只是数值保护，
            不是腱绳张力上限：5 mm 绕线轮半径下，单根 100 N 腱张力只产生
            约 0.5 N·m，因此不能沿用抓取任务 0.3 N·m 的驱动上限。
        armature_scale: 关节 armature（数值惯量）倍率，无量纲，默认 1.0。
            它是暴露出来的数值稳定性旋钮，不是电机转子惯量模型。

    返回：NoLoadSceneCfg（num_envs=1, env_spacing=1.0），其中
        actuators["tendon_effort"] 为 IdealPDActuatorCfg（kp=kd=0，纯显式力矩），
        actuators["fingers"] 为剩余 6 个非腱指关节的隐式位置保持，
        手腕与被动组沿用 BOYA_HAND_CFG 原配置。

    异常：effort_limit_nm 非有限或非正、armature_scale 非有限或为负时抛
        ValueError。
    """

    # ── 入参校验：力矩上限必须有限且为正，惯量倍率必须有限且非负 ──
    if not math.isfinite(effort_limit_nm) or effort_limit_nm <= 0:
        raise ValueError("effort_limit_nm must be finite and positive")
    if not math.isfinite(armature_scale) or armature_scale < 0:
        raise ValueError("armature_scale must be finite and nonnegative")
    # 单环境、1 m 环境间距：标定只需一只手，不需要并行场景间距。
    cfg = NoLoadSceneCfg(num_envs=1, env_spacing=1.0)
    hand = BOYA_HAND_CFG.copy()
    # ── 执行器拆分：腱关节集合与“其余动作关节”集合互补且无交集 ──
    tendon_names = set(JOINT_NAMES)
    remaining_names = tuple(name for name in ACTION_JOINT_NAMES if name not in tendon_names)
    # 组 1：剩余 6 个非腱指关节，保留隐式位置保持（沿用原增益表，单位 N·m/rad、N·m·s/rad、N·m）。
    hand.actuators["fingers"] = ImplicitActuatorCfg(
        joint_names_expr=list(remaining_names),
        stiffness={name: ACTUATOR_STIFFNESS[name] for name in remaining_names},
        damping={name: ACTUATOR_DAMPING[name] for name in remaining_names},
        joint_effort_limit={name: ACTUATOR_EFFORT_LIMIT[name] for name in remaining_names},
        armature={name: JOINT_ARMATURE[name] for name in remaining_names},
    )
    # 组 2：10 个腱关节，kp = kd = 0 的纯显式力矩执行器；
    # 力矩完全由腱绳模型（TendonPlant）经 joint_effort_nm 施加，PhysX 不做位置保持。
    # 力矩限幅（N·m）在执行器侧与关节侧各设一道，默认 2.0 仅为数值保护（见 docstring）。
    # This is an exposed numerical inertia assumption, not a motor model.
    # 中文：这是暴露出来的数值惯量假设（armature 乘倍率），不是电机模型。
    hand.actuators["tendon_effort"] = IdealPDActuatorCfg(
        joint_names_expr=list(JOINT_NAMES),
        stiffness=0.0,
        damping=0.0,
        actuator_effort_limit=effort_limit_nm,
        joint_effort_limit=effort_limit_nm,
        armature={name: JOINT_ARMATURE[name] * armature_scale for name in JOINT_NAMES},
    )
    # 用拆分后的本地手部配置替换场景默认机器人。
    cfg.robot = hand
    return cfg
