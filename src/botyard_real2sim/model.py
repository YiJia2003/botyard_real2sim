"""Unloaded, joint-side tendon mechanics for the ten flexion DOFs.

The motor winding angle is a kinematic state at the 5 mm radius spool, with
an optional speed-response lag; electrical and rotor dynamics are not assumed.
The near-side tension is what the firmware's motor-side strain gauge reports,
upstream of the routing friction and spring. Dynamic routing uses two inertial
nodes, contact memory and friction to compute independent end tensions. The
path Jacobian maps the distal tension into joint torques by virtual work.
Individual guide losses are not resolved by this aggregate transmission model.

【中文说明】
文件职责：本模块是 real2sim 腱绳传动的“总装层”，把几何、走线弹性、可选
动态走线与关节侧回位弹簧/摩擦组装成每次物理子步所需的关节力矩。它本身
不推进 PhysX、不做电机电气或减速箱建模，只做“收绳状态 → 张力 → 关节力矩”
的准静态/单边拉索计算。

主要状态变量（全部按 10 个屈曲关节并行，形状 [num_envs, 10]）：
- self.cable_retraction_m   收绳坐标 x（m），绕在半径 5 mm 的绕线轮上，是
  本类唯一长期积分的机械状态；x 增大表示继续收绳（拉紧）。
- self.cable_speed_m_s      收绳速度 ẋ（m/s），可经过 tendon_speed_lag_s 的
  一阶速度滞后，不是电气或齿轮箱动力学。
- self.motor_acceleration_rad_s2  电机角加速度（rad/s²），由子步内速度差分
  得到（区间平均），理想速度阶跃没有有限的瞬时加速度。
- self.motor_zero_angle_rad 电机编码器零点偏置（rad），每个环境一份，reset
  时可由实测电机角反解校准。
- self._initialized / self._all_initialized  哪些环境已完成 reset；后者用于
  在热路径上免做张量归约（只在 reset 时同步一次）。

输入：关节角 q（rad）、关节角速度 q̇（rad/s，均为 [num_envs,10]）、电机侧
收绳角速度指令（rad/s）与子步长 dt（s）。

输出：TendonStepResult，含关节总力矩（N·m）、近端/远端张力（N）、走线诊断
（走线伸长 m、锁死标志、材料伸长 m、弹性势能 J）、回位弹簧长度（m）与力
（N）、力臂矩阵（m/rad）、长度守恒误差（m）等。

与其它模块的对接：
- geometry.TendonGeometry：提供 h(q)=L(0)-L(q)（路径缩短量，m）、雅可比
  dh/dq（m/rad）与回位弹簧的走线长度/雅可比；配置了几何之后即用它替换常力臂。
- transmission.SeriesRoutingSpring：无质量串联走线弹簧，把总伸长
  x - h(q) - slack 分摊为“走线伸长 + 绳材料伸长”，并给出弹性势能与锁死
  标志（无阻尼、可逆，不含滞回）。
- routing_dynamics.DynamicRoutingTransmission：routing_model == "dynamic"
  时的两惯性节点走线模型，给出近端/远端各自独立的张力与能量/摩擦诊断；
  关闭时为 None，近端张力等于远端张力（准静态）。
- config.TendonModelConfig / JOINT_NAMES / ROUTING_DIAGNOSTIC_FIELDS：参数、
  关节顺序与诊断字段顺序的唯一来源。

单位约定：长度 m、速度 m/s、加速度 m/s²、角度 rad、角速度 rad/s、角加速度
rad/s²、力 N、力矩 N·m、刚度 N/m 或 N·m/rad、阻尼 N·s/m 或 N·m·s/rad、
能量 J。力矩一律为关节侧（N·m），张力一律为腱绳轴向（N）。
"""

# ── 标准库与第三方依赖 ──
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ── 本包内依赖：配置、几何、走线动力学、串联走线弹簧 ──
from .config import JOINT_NAMES, ROUTING_DIAGNOSTIC_FIELDS, TendonModelConfig
from .geometry import TendonGeometry
from .routing_dynamics import DynamicRoutingTransmission
from .transmission import SeriesRoutingSpring


def bilinear_return_force(stretch, initial, k_low, k_high, knee, zero):
    """Return-spring force with an optional knee in the force-extension curve.

    【中文】
    单边（只能拉、不能压）的两段线性力律：``stretch ≤ knee`` 时
    ``initial + k_low·stretch``；超过拐点后第二段改用 ``k_high``。
    ``knee = inf`` 或 ``k_high = k_low`` 时逐位退化为旧的单段公式。

    参数:
        stretch: 拉伸量张量，单位 m；
        initial: 预紧力，单位 N；
        k_low:   第一段刚度，单位 N/m；
        k_high:  超过拐点后的第二段刚度，单位 N/m；
        knee:    拐点拉伸量，单位 m（inf 表示无拐点）；
        zero:    与 stretch 同形的零张量，用于松弛侧返回 0。
    返回: 弹簧力张量，单位 N。
    """
    # knee = inf 时 min(stretch, knee) = stretch 且 stretch − knee < 0，
    # 于是第二段自然后退化为单段（不需要 inf 分支，也不需要主机同步）。
    capped = torch.minimum(stretch, torch.as_tensor(knee, dtype=stretch.dtype, device=stretch.device))
    beyond = torch.clamp_min(stretch - capped, 0.0)
    force = initial + k_low * capped + k_high * beyond
    return torch.where(stretch > 0, force, zero)


# ── 子步输出容器 ──
@dataclass(frozen=True)
class TendonStepResult:
    """Outputs of one physics substep, with shape ``[num_envs, 10]``.

    【中文】
    一次物理子步（substep）的输出快照，frozen dataclass（构造后只读）。
    除特别注明者外，所有张量形状均为 [num_envs, 10]，第二维按 JOINT_NAMES
    顺序对应 10 个屈曲关节（五指 × 每指 2 个独立关节）。单位见各字段注释：
    长度 m、速度 m/s、角度 rad、角速度 rad/s、角加速度 rad/s²、力 N、
    力矩 N·m、能量 J。返回前对内部状态做了 clone()，因此调用方持有结果时
    可以继续推进 plant 而不影响本次读数。
    """

    # ── 力矩类输出（N·m）──
    # 关节总力矩（N·m）= 腱绳虚功力矩 + 回位弹簧力矩 + 关节轴摩擦力矩。
    joint_effort_nm: torch.Tensor
    # ── 张力类输出（N）──
    # 近端（电机侧）张力（N）：固件应变片观测位置，位于走线摩擦与弹簧上游。
    proximal_tension_n: torch.Tensor
    # 远端（关节侧）张力（N）：真正进入关节力矩的那一支；准静态时等于近端。
    distal_tension_n: torch.Tensor
    # 腱绳虚功力矩（N·m）= J^T · Td，J 为当前路径梯度 dh/dq（含 J1=J2 归约）。
    tendon_effort_nm: torch.Tensor
    # 回位弹簧力矩（N·m）：已由“长度/方向”几何模型替换旧的角弹簧项。
    spring_effort_nm: torch.Tensor
    # 关节轴摩擦力矩（N·m）= -(粘滞 + 平滑 Coulomb/Stribeck)，正负与 q̇ 相反。
    friction_effort_nm: torch.Tensor
    # ── 收绳与路径状态（m、m/s）──
    # 绝对收绳坐标 x（m），即绕在半径 5 mm 绕线轮上的收绳量；本类积分状态。
    cable_retraction_m: torch.Tensor
    # 路径缩短量 h(q) = L(0) - L(q)（m），由几何模块按当前 q 给出。
    cable_path_shortening_m: torch.Tensor
    # 总伸长 x - y（m），其中 y = h(q) + slack；>0 张紧，<0 表示仍在松弛间隙内。
    cable_extension_m: torch.Tensor
    # 收绳速度 ẋ（m/s），含可选的一阶速度滞后（tendon_speed_lag_s）。
    cable_speed_m_s: torch.Tensor
    # ── 电机侧运动学（rad、rad/s、rad/s²）──
    # 电机角（rad）= motor_zero_angle_rad + sign * x / drum_radius（半径 5 mm）。
    motor_position_rad: torch.Tensor
    # 电机角速度（rad/s）= sign * ẋ / drum_radius。
    motor_velocity_rad_s: torch.Tensor
    # 电机角加速度（rad/s²）：子步内速度差分得到的区间平均值。
    motor_acceleration_rad_s2: torch.Tensor
    # ── 走线弹簧与绳材料（m、J）──
    # 走线弹簧伸长（m）：代表摩擦/走线布置富余，不是绳材料应变。
    routing_extension_m: torch.Tensor
    # 走线弹簧当前长度（m）= 自由长度 + 走线伸长。
    routing_length_m: torch.Tensor
    # 走线弹簧是否顶到上止档（不可再伸长）的布尔标志，形状 [num_envs, 10]。
    routing_locked: torch.Tensor
    # 绳材料自身伸长（m）：与走线弹簧串联的柔性分量（≥0）。
    material_extension_m: torch.Tensor
    # 走线弹簧 + 绳材料的弹性势能（J）。
    elastic_energy_j: torch.Tensor
    # ── 回位弹簧（m、N）──
    # 回位弹簧长度（m）：松弛时等于自由长度，不会被压缩。
    return_spring_length_m: torch.Tensor
    # 回位弹簧力（N）：单边，只能拉不能推（stretch ≤ 0 时为 0）。
    return_spring_force_n: torch.Tensor
    # ── 力臂矩阵与守恒诊断 ──
    # 力臂矩阵 dh/dq（m/rad），形状 [num_envs, 10, 10]（唯一三维字段，非 [num_envs,10]）。
    moment_arm_m: torch.Tensor
    # 长度守恒误差（m）= x - y - (走线伸长 + 材料伸长)，数值上应≈0。
    length_balance_error_m: torch.Tensor
    # 走线内部诊断字典：键见 ROUTING_DIAGNOSTIC_FIELDS，值为 [num_envs, 10]。
    routing_state: dict[str, torch.Tensor]


# ── 腱绳总装层 ──
class TendonPlant:
    """Batched unilateral cable, return-spring and friction model.

    ``routing_matrix_m`` has tendon rows and joint columns.  Each 2x2 finger
    block is ``[[r_primary_first, r_cross], [0, r_primary_second]]``.  For the
    thumb, its first row is the THJ1 tendon passing THJ2.  For the four other
    fingers, it is the J2 tendon passing J3.  Four-finger J1=J2 is a separate
    mechanical mimic already authored in the hand USD.

    These constant rows remain the legacy fallback (and provisional thumb).
    Configured geometry replaces them with ``h(q)=L(0)-L(q)`` and ``dh/dq``.
    Total extension ``x - h(q) - slack`` is shared by the routing spring and
    intrinsic cable. A single pulling tendon cannot push across a slack gap.

    【中文】
    批量化的单边拉索 + 回位弹簧 + 关节摩擦模型（“plant” 即被控物理对象）。

    力臂表示：routing_matrix_m 是 [10 腱绳行 × 10 关节列] 的常力臂回退矩阵
    （单位 m/rad），只在旧模型/拇指临时值下使用；每指 2×2 块形如
    [[r_primary_first, r_cross], [0, r_primary_second]]，上三角的 r_cross 表示
    近端腱绳跨过远端关节（拇指是 THJ1 跨 THJ2，其余四指是 J2 跨 J3），下三角
    为 0 表示该腱绳不反向跨关节。四指的 J1=J2 是 USD 里已建好的机械 mimic，
    不在此矩阵中重复计力。一旦配置了几何（geometry specs），h(q)=L(0)-L(q)
    与其梯度 dh/dq 就替换这些常力臂行。

    核心符号约定：
    - x：收绳坐标（m，绕在半径 5 mm 绕线轮上），x 增大=继续收绳=更紧。
    - y = h(q) + slack：关节侧路径边界（m），h 为路径缩短量，slack 为几何间隙。
    - 总伸长 x - y 由走线弹簧与绳材料柔度串联分担；单边拉索只能拉，负总伸长
      （松弛间隙内）不产生张力，也绝不产生压力（压缩）。
    """

    def __init__(
        self,
        cfg: TendonModelConfig,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """【中文】
        构造腱绳总装层：校验参数、把配置量纲化为 [1, 10] 广播张量，并建立
        常力臂回退矩阵、串联走线弹簧、可选走线动力学与全部内部状态。

        参数：
            cfg: TendonModelConfig，所有长度 m、刚度 N/m、阻尼 N·s/m、
                摩擦 N·m / N·m·s/rad、角 rad 均按配置单位解释。
            num_envs: 并行环境数，必须为正整数（bool 也视为非法）。
            device: torch 设备。
            dtype: 只允许 torch.float32 或 torch.float64（与 CUDA 上的
                异步有限性断言配合）。

        副作用：分配收绳状态 cable_retraction_m（m，初值 0，需 reset 才能用）、
        cable_speed_m_s（m/s）、motor_acceleration_rad_s2（rad/s²）与
        motor_zero_angle_rad（rad）。

        异常：num_envs 非正整数、dtype 不受支持时抛 ValueError。
        """
        # ── 入参校验 ──
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("plant dtype must be torch.float32 or torch.float64")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        # 所有关节侧张量的统一形状 [num_envs, 10]（10 = 屈曲关节数）。
        self._shape = (num_envs, len(JOINT_NAMES))
        # ── 常力臂回退矩阵：每指 2×2 块 [[r_primary_first, r_cross], [0, r_primary_second]] ──
        # 单位 m/rad；配置了几何之后由 geometry 的 h(q)/dh/dq 替换。
        self.routing_matrix_m = torch.zeros((len(JOINT_NAMES), len(JOINT_NAMES)), device=self.device, dtype=dtype)
        for finger in range(5):
            first = finger * 2
            second = first + 1
            # 主腱绳力臂：近端关节自身的力臂（m/rad）。
            self.routing_matrix_m[first, first] = cfg.joint_moment_arm_m[first]
            # 近端腱绳跨过远端关节的串扰力臂（m/rad），拇指为 THJ1→THJ2。
            self.routing_matrix_m[first, second] = cfg.crossing_moment_arm_m[finger]
            # 远端腱绳只驱动远端关节；下三角保持 0，避免重复计入 J1=J2 mimic。
            self.routing_matrix_m[second, second] = cfg.joint_moment_arm_m[second]

        # ── 配置量纲化：每个参数展开成形状 [1, 10]，与 [num_envs, 10] 广播 ──
        # 绕线轮半径（m），默认 5 mm；把收绳量 x 换算成电机角。
        self._drum_radius = self._vector(cfg.drum_radius_m)
        # 电机编码器符号（±1），决定 x 增大时电机角增或减。
        self._motor_sign = self._vector(cfg.motor_encoder_sign)
        # 绳材料轴向刚度（N/m）。
        self._stiffness = self._vector(cfg.cable_stiffness_n_per_m)
        # 绳材料轴向阻尼（N·s/m），仅在张紧支路上生效。
        self._damping = self._vector(cfg.cable_damping_n_s_per_m)
        # 几何松弛间隙（m）：总伸长要先填满该间隙才开始产生张力。
        self._slack = self._vector(cfg.cable_slack_m)
        # 初始预紧张力（N），reset 时据此反解收绳坐标 x。
        self._initial_tension = self._vector(cfg.initial_tension_n)
        # 收绳速度一阶滞后时间常数（s）；≤0 表示不滞后（瞬时跟随）。
        self._lag = self._vector(cfg.tendon_speed_lag_s)
        # 走线弹簧自由长度（m），输出 routing_length_m 的基准。
        self._routing_free = self._vector(cfg.routing_spring_free_length_m)
        # ── 无质量串联走线弹簧：把总伸长摊到“走线伸长 + 绳材料伸长” ──
        # 参数顺序：绳刚度（N/m）、走线刚度（N/m）、可用行程（m，= 最大长度 - 自由长度）。
        self.transmission = SeriesRoutingSpring(
            self._stiffness,
            self._vector(cfg.routing_spring_stiffness_n_per_m),
            self._vector(cfg.routing_spring_max_length_m) - self._routing_free,
        )
        # ── 可选动态走线（两惯性节点 + 接触记忆 + 摩擦）；非 "dynamic" 时为 None ──
        self.routing_dynamics = (
            DynamicRoutingTransmission(cfg, num_envs, device) if cfg.routing_model == "dynamic" else None
        )
        # 几何模型：提供 h(q)、dh/dq 与回位弹簧走线；缺省 specs 时退回常力臂矩阵。
        self.geometry = TendonGeometry(cfg.geometry, self.routing_matrix_m)
        # ── 回位弹簧（长度/方向模型）参数，全部单边只能拉 ──
        # 回位弹簧刚度（N/m）、自由长度（m）、初始张力（N）。
        self._return_k = self._vector(cfg.return_spring_stiffness_n_per_m)
        self._return_free = self._vector(cfg.return_spring_free_length_m)
        self._return_initial = self._vector(cfg.return_spring_initial_tension_n)
        self._return_k_high = self._vector(cfg.return_spring_stiffness_high_n_per_m)
        self._return_knee = self._vector(cfg.return_spring_knee_m)
        # 低角单向限位（deg → rad）与死区带；-inf 表示不启用。
        self._joint_stop_rad = self._vector(cfg.joint_lower_stop_deg) * (math.pi / 180.0)
        self._stop_k = self._vector(cfg.joint_stop_stiffness_nm_per_rad)
        self._stop_c = self._vector(cfg.joint_stop_damping_nm_s_per_rad)
        self._stop_band_rad = self._vector(cfg.joint_stop_band_deg) * (math.pi / 180.0)
        self._stop_band_friction = self._vector(cfg.joint_stop_band_friction_nm)
        # 系绳长度与路径长度偏置（m），用于由几何路径长度换算出弹簧拉伸量。
        self._return_tether = self._vector(cfg.return_tether_length_m)
        self._return_offset = self._vector(cfg.return_path_length_offset_m)
        # ── 旧的角弹簧（仅在几何缺失时作为回退）与关节轴摩擦参数 ──
        # 静息角（rad）、角刚度（N·m/rad）、预载力矩（N·m）。
        self._spring_rest = self._vector(cfg.spring_rest_angle_rad)
        self._spring_k = self._vector(cfg.spring_stiffness_nm_per_rad)
        self._spring_preload = self._vector(cfg.spring_preload_nm)
        # 粘滞系数（N·m·s/rad）、Coulomb（N·m）、静摩擦（N·m）。
        self._viscous = self._vector(cfg.viscous_friction_nm_s_per_rad)
        self._coulomb = self._vector(cfg.coulomb_friction_nm)
        self._static = self._vector(cfg.static_friction_nm)
        # Stribeck 速度（rad/s）与摩擦平滑速度（rad/s），避免 tanh 在 0 附近抖振。
        self._stribeck_speed = self._vector(cfg.stribeck_speed_rad_s)
        self._friction_smoothing = self._vector(cfg.friction_smoothing_speed_rad_s)
        # ── 可积分状态：收绳量 x（m）、收绳速度 ẋ（m/s）、电机角加速度（rad/s²）──
        # 注意：初值 0 只是占位，reset(q_rad) 会写入手臂当前姿态对应的值。
        self.cable_retraction_m = torch.zeros(self._shape, device=self.device, dtype=dtype)
        self.cable_speed_m_s = torch.zeros_like(self.cable_retraction_m)
        self.motor_acceleration_rad_s2 = torch.zeros_like(self.cable_retraction_m)
        # 电机零点偏置（rad）：每个环境一份，reset 可由实测电机角反解校准。
        self.motor_zero_angle_rad = self._vector(cfg.motor_zero_angle_rad).expand(self._shape).clone()
        # 初始化标记：advance/evaluate 在全部环境 reset 之前拒绝执行。
        self._initialized = torch.zeros((num_envs,), device=self.device, dtype=torch.bool)
        self._all_initialized = False

    # ── 参数张量化辅助 ──
    def _vector(self, values: tuple[float, ...], *, width: int = 10) -> torch.Tensor:
        """【中文】
        把 10 个标量配置项转成形状 [1, 10] 的常量张量，便于与 [num_envs, 10]
        状态广播；单位沿用配置单位（不在此处换算）。

        参数：
            values: 长度为 width 的标量元组。
            width: 期望长度，默认 10（屈曲关节数）。

        异常：长度不符时抛 ValueError。
        """
        if len(values) != width:
            raise ValueError(f"expected {width} parameter values")
        return torch.as_tensor(values, device=self.device, dtype=self.dtype).unsqueeze(0)

    # ── 校验辅助 ──
    def _validate_joint_tensor(self, name: str, values: torch.Tensor) -> None:
        """【中文】
        校验一个关节侧输入张量：形状必须为 [num_envs, 10]，设备/dtype 与 plant
        一致，且全部元素有限。

        参数：
            name: 出错信息中使用的参数名。
            values: 待校验张量（角度 rad、角速度 rad/s 或收绳角速度 rad/s）。

        异常：形状不符或含非有限值时抛 ValueError（CUDA 上为异步断言）。
        """
        if values.shape != self._shape:
            raise ValueError(f"{name} must have shape {self._shape}, got {tuple(values.shape)}")
        self._validate_values(name, values)

    def _validate_values(self, name: str, values: torch.Tensor) -> None:
        """【中文】
        校验张量的设备/dtype 与 plant 内部状态一致，并检查取值有限。

        参数：
            name: 出错信息中使用的参数名。
            values: 待校验张量（热路径上可能位于 CUDA）。

        副作用：无（不修改任何状态）。

        异常：设备或浮点 dtype 不一致时抛 ValueError；CPU 上出现非有限值时抛
        ValueError，CUDA 上使用 torch._assert_async 把检查留在设备端，避免每
        个子步同步。
        """
        reference = self.cable_retraction_m
        if values.device != reference.device or values.dtype != reference.dtype:
            raise ValueError(f"{name} must share the plant's device and floating dtype")
        finite = torch.isfinite(values).all()
        if values.is_cuda:
            # Keep finite validation on-device in the physics hot path.
            # 中文：物理热路径上把有限性检查留在设备端，避免主机同步。
            torch._assert_async(finite, f"{name} must contain finite values")
        elif not bool(finite):
            raise ValueError(f"{name} must contain finite values")

    # ── 复位：在不积分时间的前提下把状态摆到静止 ──
    def reset(
        self,
        q_rad: torch.Tensor,
        initial_tension_n: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        *,
        cable_retraction_m: torch.Tensor | None = None,
        motor_position_rad: torch.Tensor | None = None,
        motor_velocity_rad_s: torch.Tensor | None = None,
        joint_velocity_rad_s: torch.Tensor | None = None,
        routing_initial_distal_tension_n: torch.Tensor | None = None,
        routing_initial_slip_m: torch.Tensor | None = None,
    ) -> None:
        """Initialize cable state at rest without integrating time.

        Without overrides, zero tension starts at ``x=h(q)`` with the configured
        slack gap still present. Positive initial tension requires zero initial
        slack on that channel. Explicit ``cable_retraction_m`` instead sets the
        absolute take-up coordinate, retaining the configured geometric slack;
        it is mutually exclusive with an explicit tension override. Measured
        tension initialization assumes zero cable/joint velocity.

        【中文】
        在不推进时间的前提下把收绳状态摆到静止平衡点（不做时间积分）。

        参数（角度 rad、张力 N、长度 m、角速度 rad/s）：
            q_rad: 当前关节角，形状 [num_envs, 10] 或 [选中环境数, 10]。
            initial_tension_n: 期望的初始张力（N）。为 None 时用配置的
                initial_tension_n。正值要求该通道松弛间隙为 0。
            env_ids: 只复位指定环境（None 表示全部）。
            cable_retraction_m: 直接给定绝对收绳坐标 x（m），保留配置的几何
                松弛间隙；与 initial_tension_n 互斥。
            motor_position_rad: 实测电机角（rad），据此反解零点偏置
                motor_zero_angle_rad，不额外引入形变。
            motor_velocity_rad_s: 实测电机角速度（rad/s），反解收绳速度 ẋ。
            joint_velocity_rad_s: 当前关节角速度（rad/s），仅动态走线用，
                用于估计路径边界速度。

        副作用：写入 cable_retraction_m（x，m）、motor_zero_angle_rad（rad）、
        cable_speed_m_s（ẋ，m/s）、motor_acceleration_rad_s2（置 0）、
        _initialized/_all_initialized，并在动态走线下调用
        routing_dynamics.reset 初始化两节点状态。

        异常：形状不合法、含非有限值、同时给张力与收绳坐标、负初始张力、
        正初始张力与配置松弛冲突、非零阻尼下用非零速度初始化等，均抛
        ValueError。
        """

        # 选中要复位/校验的环境行。
        rows = slice(None) if env_ids is None else env_ids
        expected_rows = self.num_envs if env_ids is None else int(env_ids.numel())
        if q_rad.shape == self._shape:
            q_selected = q_rad[rows]
        elif q_rad.shape == (expected_rows, len(JOINT_NAMES)):
            q_selected = q_rad
        else:
            raise ValueError(f"q_rad must have shape {self._shape} or ({expected_rows}, {len(JOINT_NAMES)})")

        self._validate_values("q_rad", q_selected)
        if initial_tension_n is not None and cable_retraction_m is not None:
            raise ValueError("provide initial tension or cable take-up, not both")

        def selected_values(value, name):
            """【中文】
            把标量 / [10] / [num_envs,10] / [选中环境数,10] 的输入规范成
            可广播的 (1, 10) 或 (选中环境数, 10) 张量，并检查有限性。

            参数：value 为输入（张力 N、长度 m、角度 rad、角速度 rad/s 等）；
            name 仅用于错误信息。
            异常：形状不合法或含非有限值时抛 ValueError。
            """
            tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
            if tensor.ndim == 0:
                tensor = tensor.expand(1, len(JOINT_NAMES))
            if tensor.shape == (len(JOINT_NAMES),):
                tensor = tensor.unsqueeze(0)
            if tensor.shape == self._shape:
                tensor = tensor[rows]
            if tensor.shape not in ((1, len(JOINT_NAMES)), (expected_rows, len(JOINT_NAMES))):
                raise ValueError(f"{name} must have shape [10], [num_envs,10] or [selected_envs,10]")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain finite values")
            return tensor

        # ── 实测电机量规范化（用于校准零点偏置与初始速度）──
        measured_motor = (
            None if motor_position_rad is None else selected_values(motor_position_rad, "motor_position_rad")
        )
        measured_speed = (
            None if motor_velocity_rad_s is None else selected_values(motor_velocity_rad_s, "motor_velocity_rad_s")
        )
        # 有阻尼时无法从静止姿态反解粘滞张力，因此只允许静止初始化。
        if measured_speed is not None and torch.any((measured_speed != 0) & (self._damping > 0)):
            raise ValueError("nonzero cable damping requires rest initialization; initial viscous tension is unknown")
        # 当前姿态下的路径缩短量 h(q)（m）。
        path_shortening = self.geometry.evaluate(q_selected)[0]
        if cable_retraction_m is not None:
            # 直接给定绝对收绳坐标 x（m），保留配置的几何松弛间隙。
            self.cable_retraction_m[rows] = selected_values(cable_retraction_m, "cable_retraction_m")
        else:
            target = selected_values(
                self._initial_tension if initial_tension_n is None else initial_tension_n,
                "initial_tension_n",
            )
            # 单边拉索不能承受负张力（压力）。
            if torch.any(target < 0):
                raise ValueError("initial_tension_n cannot be negative for a one-sided tendon")
            if torch.any((target > 0) & (self._slack > 0)):
                raise ValueError("positive initial tension conflicts with configured initial slack")
            # ── 由目标张力反解收绳坐标：x = h(q) + T/k_cable + 走线伸长(T) ──
            # 张力为 0 时 x 恰为 h(q)，即仍保留配置的 slack 间隙。
            self.cable_retraction_m[rows] = (
                path_shortening + target / self._stiffness + self.transmission.extension_for_tension(target)
            )
        self.motor_zero_angle_rad[rows] = self._vector(self.cfg.motor_zero_angle_rad)
        if measured_motor is not None:
            # Align an encoder reference using the same initial sample as q/T.
            # This sets only the reference, not an independent extra deformation.
            # 中文：用与 q/T 同一初始采样对齐编码器参考；只改参考零点，不额外引入形变。
            self.motor_zero_angle_rad[rows] = (
                measured_motor - self._motor_sign * self.cable_retraction_m[rows] / self._drum_radius
            )
        # 电机角速度（rad/s）→ 收绳速度（m/s）：ẋ = sign * ω_motor * r_drum。
        self.cable_speed_m_s[rows] = (
            0.0 if measured_speed is None else measured_speed * self._drum_radius * self._motor_sign
        )
        self.motor_acceleration_rad_s2[rows] = 0.0
        # ── 动态走线：按当前边界与张力初始化两个惯性节点 ──
        if self.routing_dynamics is not None:
            y = path_shortening + self._slack
            total = self.cable_retraction_m[rows] - y
            initial_force = self.transmission.evaluate(total)[0]
            path_velocity = torch.zeros_like(q_selected)
            if joint_velocity_rad_s is not None:
                qd = selected_values(joint_velocity_rad_s, "joint_velocity_rad_s")
                # 路径边界速度 ẏ = (dh/dq) · q̇（m/s）。
                path_velocity = torch.einsum(
                    "bij,bj->bi", self.geometry.evaluate(q_selected)[1], qd.expand_as(q_selected)
                )
            # 摩擦锁定的夹持态：远端（关节侧）张力可与近端（实测）不同；
            # 缺省仍取同一张力（旧的两侧平分行为，逐位不变）。
            distal_force = None
            if routing_initial_distal_tension_n is not None:
                distal_force = selected_values(routing_initial_distal_tension_n, "routing_initial_distal_tension_n")
                if torch.any(distal_force < 0):
                    raise ValueError("routing_initial_distal_tension_n cannot be negative")
            # 走线滑动累积初值（m）：锁定态“让出的绳长”，缺省 0（关闭该状态）。
            initial_slip = None
            if routing_initial_slip_m is not None:
                initial_slip = selected_values(routing_initial_slip_m, "routing_initial_slip_m")
                if torch.any(initial_slip < 0):
                    raise ValueError("routing_initial_slip_m cannot be negative")
            # 用“收绳速度 + 边界速度”的平均值作为端口的初始相对速度（m/s）。
            self.routing_dynamics.reset(
                self.cable_retraction_m[rows],
                y,
                initial_force,
                env_ids=env_ids,
                velocity=0.5 * (self.cable_speed_m_s[rows] + path_velocity),
                distal_tension=distal_force,
                initial_slip=initial_slip,
            )
        self._initialized[rows] = True
        # Synchronize only on a reset, never on each (potentially CUDA) physics substep.
        # 中文：只在复位时做一次同步归约，绝不放进（可能位于 CUDA 的）物理子步。
        self._all_initialized = bool(torch.all(self._initialized))

    # ── 推进：只积分离散状态，不推进 PhysX ──
    def advance(
        self,
        tendon_speed_rad_s: torch.Tensor,
        dt_s: float,
        *,
        q_rad: torch.Tensor | None = None,
        qd_rad_s: torch.Tensor | None = None,
    ) -> None:
        """Advance motor take-up and, when enabled, the inertial routing states.

        Dynamic routing requires current joint position and velocity to predict
        the joint-side boundary over this interval. It does not advance PhysX.

        【中文】
        推进电机侧收绳量与（启用时的）走线惯性状态；不推进 PhysX 仿真。

        参数：
            tendon_speed_rad_s: 电机侧收绳角速度指令（rad/s），形状
                [num_envs, 10]；乘以绕线轮半径得到收绳线速度（m/s）。
            dt_s: 子步长（s），必须有限且为正。
            q_rad: 当前关节角（rad）；动态走线时必填，用于算路径边界 y。
            qd_rad_s: 当前关节角速度（rad/s）；动态走线时必填，用于算边界速度。

        副作用：推进 cable_speed_m_s（m/s，含可选一阶滞后）、
        motor_acceleration_rad_s2（rad/s²，区间差分）与 cable_retraction_m
        （x，m）；动态走线下调用 routing_dynamics.advance 推进两节点状态。

        异常：形状/取值非法、dt_s 非有限或非正、未先 reset、动态走线缺少
        q_rad/qd_rad_s 时抛 ValueError / RuntimeError。
        """
        self._validate_joint_tensor("tendon_speed_rad_s", tendon_speed_rad_s)
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if not self._all_initialized:
            raise RuntimeError("call TendonPlant.reset(q_rad) before advancing")
        if self.routing_dynamics is not None:
            if q_rad is None or qd_rad_s is None:
                raise ValueError("dynamic routing advance requires current q_rad and qd_rad_s")
            self._validate_joint_tensor("q_rad", q_rad)
            self._validate_joint_tensor("qd_rad_s", qd_rad_s)
            # 路径边界 y = h(q) 与雅可比 dh/dq（m/rad），用于预测本子步末端边界。
            path, jacobian = self.geometry.evaluate(q_rad)
            path_velocity = torch.einsum("bij,bj->bi", jacobian, qd_rad_s)
            # 子步起点处的收绳坐标 x0（m），供走线模型做区间积分。
            old_x = self.cable_retraction_m.clone()
        # 目标收绳线速度（m/s）= 电机角速度（rad/s）× 绕线轮半径（m）。
        target_cable_speed = tendon_speed_rad_s * self._drum_radius
        # Optional first-order port lag, not electrical or gearbox mechanics.
        # 中文：可选的一阶端口速度滞后；不代表电气或齿轮箱动力学。
        # alpha = 1 - exp(-dt/τ)，τ ≤ 0 时取 1（瞬时跟随）。
        alpha = torch.where(
            self._lag <= 0, torch.ones_like(self._lag), -torch.expm1(-dt_s / self._lag.clamp_min(1e-12))
        )
        previous_speed = self.cable_speed_m_s.clone()
        # 一阶滞后离散更新：ẋ ← ẋ + alpha * (目标速度 - ẋ)（m/s）。
        self.cable_speed_m_s += alpha * (target_cable_speed - previous_speed)
        # Interval-average acceleration, aligned with the endpoint of this step.
        # Ideal speed jumps have no finite instantaneous acceleration.
        # 中文：区间平均加速度，对齐到本子步末端；理想速度阶跃没有有限瞬时加速度。
        # a_motor = sign * Δẋ / (dt * r_drum)（rad/s²）。
        self.motor_acceleration_rad_s2.copy_(
            self._motor_sign * (self.cable_speed_m_s - previous_speed) / (dt_s * self._drum_radius)
        )
        # Exact integral for a constant speed request during this interval.
        # 中文：区间内速度指令恒定时的一阶滞后精确积分（单位 m）：
        # x += v_target*dt + (v_prev - v_target)*τ*alpha。
        self.cable_retraction_m += target_cable_speed * dt_s + (previous_speed - target_cable_speed) * self._lag * alpha
        if self.routing_dynamics is not None:
            # 把本子步的收绳端口 x0→x1 与边界 y0→y1（m）交给走线模型积分。
            self.routing_dynamics.advance(
                old_x, self.cable_retraction_m, path + self._slack, path + self._slack + path_velocity * dt_s, dt_s
            )

    # ── 兼容辅助：先推进再求力 ──
    def step(
        self,
        q_rad: torch.Tensor,
        qd_rad_s: torch.Tensor,
        tendon_speed_rad_s: torch.Tensor,
        dt_s: float,
    ) -> TendonStepResult:
        """Compatibility helper: advance cable state, then evaluate given joints.

        For aligned simulation logging, call evaluate before advancing the next
        interval and log joint/cable values at the same physical timestamp.

        【中文】
        兼容用辅助函数：先 advance 推进收绳状态，再用同一组关节量 evaluate。

        参数：q_rad（rad）、qd_rad_s（rad/s）、tendon_speed_rad_s（rad/s）、
        dt_s（s），形状均为 [num_envs, 10] 或标量子步长。

        返回：本子步末端的 TendonStepResult（力矩 N·m、张力 N 等）。

        注意：为使日志与物理时刻对齐，正式仿真应在推进下一区间之前调用
        evaluate，并把关节/腱绳读数记在同一时间戳上；本函数仅为兼容旧调用。

        异常：形状/取值非法或未先 reset 时抛 ValueError / RuntimeError。
        """
        self._validate_joint_tensor("q_rad", q_rad)
        self._validate_joint_tensor("qd_rad_s", qd_rad_s)
        self.advance(tendon_speed_rad_s, dt_s, q_rad=q_rad, qd_rad_s=qd_rad_s)
        return self.evaluate(q_rad, qd_rad_s)

    # ── 求力：只读当前状态，不推进任何积分 ──
    def evaluate(self, q_rad: torch.Tensor, qd_rad_s: torch.Tensor) -> TendonStepResult:
        """Compute forces from the current state without changing it.

        【中文】
        由当前状态（收绳 x、收绳速度 ẋ、关节 q/q̇）计算全部力与力矩，
        纯函数式：不修改任何内部状态。

        参数：
            q_rad: 关节角（rad），形状 [num_envs, 10]。
            qd_rad_s: 关节角速度（rad/s），形状 [num_envs, 10]。

        返回：TendonStepResult（力矩 N·m、近端/远端张力 N、走线/材料伸长 m、
        弹性势能 J、回位弹簧长度 m/力 N、力臂 m/rad、长度守恒误差 m）。

        计算流程：
        1) 几何给出 h(q) 与 dh/dq；总伸长 = x - h(q) - slack。
        2) 串联走线弹簧把总伸长分摊为走线伸长 + 材料伸长，并给出张力。
        3) 张紧支路加阻尼；松弛支路张力恒为 0（单边，不能推）。
        4) 远端张力经雅可比做虚功 J^T·Td 得到腱绳关节力矩。
        5) 叠加回位弹簧（长度/方向模型）与关节轴摩擦（静止保持 + 平滑 Coulomb/Stribeck）。

        异常：形状/取值非法或未先 reset 时抛 ValueError / RuntimeError。
        """
        self._validate_joint_tensor("q_rad", q_rad)
        self._validate_joint_tensor("qd_rad_s", qd_rad_s)
        if not self._all_initialized:
            raise RuntimeError("call TendonPlant.reset(q_rad) before evaluating")
        # ── 几何：路径缩短量 h(q)（m）与力臂矩阵 dh/dq（m/rad，[B,10,10]）──
        path_shortening, routing = self.geometry.evaluate(q_rad)
        # 路径边界速度 ẏ = (dh/dq)·q̇（m/s），张紧时提供阻尼相对速度。
        path_speed = torch.einsum("bij,bj->bi", routing, qd_rad_s)
        # 总伸长（m）= 收绳量 x - 路径缩短 h(q) - 几何松弛 slack。
        extension = self.cable_retraction_m - path_shortening - self._slack
        # 伸长速度（m/s）= ẋ - ẏ。
        extension_speed = self.cable_speed_m_s - path_speed
        # Damping cannot pull across a geometric slack gap. The tensile branch
        # is clipped on release so the cable never supplies compression.
        # 中文：阻尼不能跨过几何松弛间隙拉紧；张紧支路在卸载时被截断到 0，
        # 保证绳索永不输出压力（压缩）。
        # 串联弹簧返回：弹性张力（N）、走线伸长（m）、材料伸长（m）、锁死标志、势能（J）。
        elastic_tension, routing_extension, material_extension, locked, energy = self.transmission.evaluate(extension)
        proximal_tension = torch.where(
            extension >= 0,
            torch.clamp_min(elastic_tension + self._damping * extension_speed, 0.0),
            torch.zeros_like(extension),
        )
        # 准静态（无动态走线）时近端与远端张力相等。
        distal_tension = proximal_tension.clone()
        routing_state = {name: torch.zeros_like(q_rad) for name in ROUTING_DIAGNOSTIC_FIELDS}
        # 长度守恒误差（m）= 总伸长 - 走线伸长 - 材料伸长，静态模型下应≈0。
        length_balance = extension - routing_extension - material_extension
        # ── 动态走线：近端/远端各自独立张力、能量与摩擦诊断 ──
        if self.routing_dynamics is not None:
            dynamics = self.routing_dynamics
            dynamic = dynamics.evaluate(self.cable_retraction_m, path_shortening + self._slack)
            proximal_tension = dynamic["proximal_tension"].to(q_rad)
            distal_tension = dynamic["distal_tension"].to(q_rad)
            routing_extension = dynamic["routing_extension"].to(q_rad)
            material_extension = dynamic["material_extension"].to(q_rad)
            locked = dynamic["locked"]
            energy = dynamic["elastic_energy"].to(q_rad)
            length_balance = dynamic["length_residual"].to(q_rad)
            # 诊断量按 ROUTING_DIAGNOSTIC_FIELDS 的顺序排列（单位见 config 中字段名后缀）。
            diagnostics = (
                dynamics.u,
                dynamics.v,
                dynamics.du,
                dynamics.dv,
                dynamic["acceleration_u"],
                dynamic["acceleration_v"],
                dynamics.zu,
                dynamics.zv,
                dynamics.contact,
                dynamics.friction_u,
                dynamics.friction_v,
                dynamics.normal_u,
                dynamics.normal_v,
                dynamic["spring_tension"],
                dynamic["stop_reaction"],
                dynamic["proximal_tension"] - dynamic["distal_tension"],
                dynamic["routing_gap"],
                dynamic["material_proximal_extension"],
                dynamic["material_distal_extension"],
                dynamic["kinetic_energy"],
                dynamics.friction_dissipated,
                dynamics.numerical_dissipated,
                dynamics.port_work,
                dynamics.energy_residual,
                dynamics.solver_residual,
                dynamics.plastic_slip_u,
                dynamics.plastic_slip_v,
                dynamics.impact_dissipated,
                dynamics.stop_impulse,
            )
            routing_state = {
                key: value.to(q_rad).clone() for key, value in zip(ROUTING_DIAGNOSTIC_FIELDS, diagnostics, strict=True)
            }
        # Virtual work uses the current path gradient, including J1=J2 reduction.
        # 中文：虚功原理用当前路径梯度（含 J1=J2 归约），τ_tendon = J^T · Td（N·m）。
        tendon_effort = torch.einsum("bi,bij->bj", distal_tension, routing)

        # Smooth Coulomb/Stribeck regularization; no stationary holding interval.
        # 中文：平滑 Coulomb/Stribeck 正则化；不设静止保持区间，因此没有静摩擦锁定。
        # 旧角弹簧项：τ = -k(q - q_rest) - preload（N·m），有几何时逐关节被清零。
        spring_effort = -self._spring_k * (q_rad - self._spring_rest) - self._spring_preload
        # 回位弹簧走线长度（m）与其雅可比（m/rad，[B,10,10]）。
        return_path, return_jacobian = self.geometry.return_lengths(q_rad)
        return_length = torch.zeros_like(q_rad)
        return_force = torch.zeros_like(q_rad)
        for index in self.geometry.returns:
            # 拉伸量（m）= 几何路径长度 + 路径偏置 - 系绳长 - 自由长度。
            stretch = (
                return_path[:, index]
                + self._return_offset[:, index]
                - self._return_tether[:, index]
                - self._return_free[:, index]
            )
            # 单边：松弛时保持自由长度，不被压缩。
            return_length[:, index] = self._return_free[:, index] + stretch.clamp_min(0)
            # 单边力（N）：stretch > 0 时为 预紧 + k·stretch，否则为 0。
            return_force[:, index] = bilinear_return_force(
                stretch,
                self._return_initial[:, index],
                self._return_k[:, index],
                self._return_k_high[:, index],
                self._return_knee[:, index],
                torch.zeros_like(stretch),
            )
            # Geometry replaces, never adds to, the legacy angular spring.
            # 中文：几何模型替换（而不是叠加）旧角弹簧，因此该关节角弹簧项清零。
            spring_effort[:, index] = 0.0
        # 弹簧力经回位弹簧自身雅可比做虚功：Δτ = -J_return^T · F_return（N·m）。
        spring_effort = spring_effort - torch.einsum("bi,bij->bj", return_force, return_jacobian)
        # ── 关节轴摩擦：静止保持 + 粘滞/Stribeck 滑移 ──
        # 速度比 |q̇|/v_stribeck（无量纲），分母下限 1e-12 防除零。
        speed_ratio = qd_rad_s.abs() / self._stribeck_speed.clamp_min(1e-12)
        # 干摩擦幅值（N·m）：低速趋近静摩擦，高速趋近 Coulomb：
        # μ(|q̇|) = Coulomb + (Static - Coulomb)·exp(-(|q̇|/v_s)²)。
        dry_friction_magnitude = self._coulomb + (self._static - self._coulomb) * torch.exp(-(speed_ratio**2))
        # 滑移支路（原 Stribeck 律）：τ = -b·q̇ - μ·tanh(q̇/v_smooth)（N·m）。
        stribeck_effort = -self._viscous * qd_rad_s - dry_friction_magnitude * torch.tanh(
            qd_rad_s / self._friction_smoothing
        )
        # 保持支路（静摩擦带）：把腱绳与回位弹簧的合力矩在 ±Static 内完全平衡，
        # 超过该带才留下 τ_applied - sign(τ_applied)·Static 的滑移残差。
        applied_effort = tendon_effort + spring_effort
        hold_effort = -torch.clamp(applied_effort, -self._static, self._static)
        # 平滑权重：|q̇| = 0 时完全保持；|q̇| ≥ friction_smoothing 时完全回到 Stribeck。
        # smoothstep 保证权重在两端一阶连续，避免静止/滑动切换时的力矩跳变。
        normalized_speed = (qd_rad_s.abs() / self._friction_smoothing.clamp_min(1e-12)).clamp(0.0, 1.0)
        hold_weight = 1.0 - normalized_speed * normalized_speed * (3.0 - 2.0 * normalized_speed)
        friction_effort = hold_weight * hold_effort + (1.0 - hold_weight) * stribeck_effort
        # ── 低角单向限位 + 方向死区 ──
        # penetration > 0 表示越过限位，反力为单边弹簧-阻尼；限位下方不拉回。
        penetration = self._joint_stop_rad - q_rad
        stop_effort = torch.where(
            penetration > 0,
            self._stop_k * penetration - self._stop_c * qd_rad_s,
            torch.zeros_like(penetration),
        )
        # 死区带：限位上方 band 内附加库仑保持力矩，复现实测 0.7° 的方向死区。
        in_band = (penetration <= 0) & (penetration > -self._stop_band_rad)
        band_effort = -self._stop_band_friction * torch.tanh(qd_rad_s / self._friction_smoothing) * in_band
        # 关节总力矩（N·m）= 腱绳虚功 + 回位弹簧 + 关节摩擦 + 限位/死区。
        joint_effort = tendon_effort + spring_effort + friction_effort + stop_effort + band_effort

        # 组装输出快照；长度/速度/能量量均为 m、m/s、J，力矩为 N·m。
        # 其中 motor_position_rad = 零点偏置 + sign·x/r_drum（rad），
        # motor_velocity_rad_s = sign·ẋ/r_drum（rad/s），
        # routing_length_m = 走线自由长度 + 走线伸长（m）。
        return TendonStepResult(
            joint_effort_nm=joint_effort,
            proximal_tension_n=proximal_tension,
            distal_tension_n=distal_tension,
            tendon_effort_nm=tendon_effort,
            spring_effort_nm=spring_effort,
            friction_effort_nm=friction_effort,
            cable_retraction_m=self.cable_retraction_m.clone(),
            cable_path_shortening_m=path_shortening,
            cable_extension_m=extension,
            cable_speed_m_s=self.cable_speed_m_s.clone(),
            motor_position_rad=self.motor_zero_angle_rad
            + self._motor_sign * self.cable_retraction_m / self._drum_radius,
            motor_velocity_rad_s=self._motor_sign * self.cable_speed_m_s / self._drum_radius,
            motor_acceleration_rad_s2=self.motor_acceleration_rad_s2.clone(),
            routing_extension_m=routing_extension,
            routing_length_m=self._routing_free + routing_extension,
            routing_locked=locked,
            material_extension_m=material_extension,
            elastic_energy_j=energy,
            return_spring_length_m=return_length,
            return_spring_force_n=return_force,
            moment_arm_m=routing,
            length_balance_error_m=length_balance,
            routing_state=routing_state,
        )
