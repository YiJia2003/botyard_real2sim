"""Joint-side replica of the firmware's unloaded ``Fallback_pidControl`` path.

【中文说明】
文件职责：
    复刻 H7 固件 ``Fallback_pidControl`` 的**关节侧位置环**（空载标定用）。
    只做“角度误差 -> 腱端等效速度”的控制律；不建模电机电气动态、不建模
    减速箱，也不建模腱与关节的力学（后者属于力学模型）。

输入/输出：
    - 输入：实测关节角 q_rad、目标角 cmd_rad、速度上限 cmd_speed_rad_s、
      第三指令字段 cmd_tension_command_n（可选）、选中掩码 active_mask（可选）。
    - 输出：**腱端等效速度**，单位 rad/s，形状 (num_envs, 10)。

单位约定：
    配置里的增益与限幅以 deg 为基础（deg/s per deg、deg、deg/s）；
    对外接口一律用 SI 的 rad、rad/s。内部在 deg 域计算，因为固件本身
    就是在 deg 域求和、限幅与量化的。

关键假设：
    1. 一次调用 = 一个固件控制节拍（tick），不是可变步长的数值积分；
       该位置环分支自身不乘 ``time_delta``。
    2. 输出被解释为“腱端等效速度”，不区分电机转速与卷筒几何。
    3. 强制走 Fallback 分支：不复刻接触估计器，也不复刻低/高张力分支切换；
       真实标定必须先确认记录区间内 ``is_fallback`` 为真。
"""

from __future__ import annotations

import math

import torch

from .config import JOINT_NAMES, FallbackControllerConfig


# ── 输入校验辅助 ──
def _finite_tensor(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
    """校验张量与参考张量同设备同浮点类型，且元素全部有限。

    【中文】
    参数：
        name:      出错信息中使用的参数名。
        value:     待校验张量。
        reference: 参考张量（通常是控制器内部状态张量），提供设备与 dtype。

    返回：
        None；校验通过时不返回任何值。

    异常：
        ValueError: 设备或 dtype 不一致；不是浮点类型；含 NaN/±inf。
                   CUDA 张量上的有限性检查以异步断言方式上报。

    说明：
        CUDA 上做 ``.all().item()`` 会强制同步、打断流水线，
        因此改用 ``torch._assert_async`` 把检查交给设备侧；
        CPU 上直接取布尔值判断即可。
    """
    # 设备与 dtype 必须与控制器内部状态一致，否则后续运算会隐式提升或报错。
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must share the reference tensor's device and floating dtype")
    # 整数/布尔张量无法承载角度与速度，必须先拒绝。
    if not value.is_floating_point():
        raise ValueError(f"{name} must contain finite floating-point values")
    finite = torch.isfinite(value).all()
    # CUDA：异步断言，避免每个 tick 都同步主机。
    if value.is_cuda:
        torch._assert_async(finite, f"{name} must contain finite floating-point values")
    # CPU：直接判断布尔结果。
    elif not bool(finite):
        raise ValueError(f"{name} must contain finite floating-point values")


def _boolean_mask(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """把“布尔或 0/1”张量规整成 bool 张量。

    【中文】
    参数：
        name:      出错信息中使用的参数名。
        value:     掩码张量，取值只允许 0/1（或本身就是 bool）。
        reference: 参考张量，提供设备。

    返回：
        torch.bool 张量，形状与输入相同。

    异常：
        ValueError: 设备不一致；出现 0/1 之外的取值。

    说明：
        CUDA 路径同样使用 ``torch._assert_async`` 做异步断言，
        避免每步同步主机。
    """
    if value.device != reference.device:
        raise ValueError(f"{name} must share the reference tensor's device")
    # 只接受 0/1 或 bool；其它数值（如 2、0.5）一律视为非法掩码。
    valid = ((value == 0) | (value == 1)).all()
    if value.is_cuda:
        torch._assert_async(valid, f"{name} must contain only boolean or zero/one values")
    elif not bool(valid):
        raise ValueError(f"{name} must contain only boolean or zero/one values")
    return value.bool()


# ── 同指补齐（固件的 autopair） ──
def autopair_command(
    actual_rad: torch.Tensor,
    target_rad: torch.Tensor,
    speed_limit_rad_s: torch.Tensor,
    tension_limit_n: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the firmware's same-finger partner completion *once on command receipt*.

    Every missing partner holds its angle at the time the command arrives.  Its
    speed limit is copied from the selected joint (or set to 120 deg/s if that
    value is <= 1 deg/s), and its tension-limit field becomes zero.  The zero
    field does **not** disable its Fallback position controller.

    【中文】
    在**收到指令的那一刻**执行一次固件的“同指补齐”。

    背景：
        十个 DOF 按“指对”成组（(0,1)、(2,3)、…）。一次指令里如果只选中了
        指对中的一个，固件会立刻为另一个补出目标，避免同一根手指的两个
        关节目标互相矛盾。

    参数：
        actual_rad:        当前实测关节角，单位 rad，形状 (..., 10)。
        target_rad:        指令目标角，单位 rad，形状同上。
        speed_limit_rad_s: 速度上限，单位 rad/s，形状同上。
        tension_limit_n:   第三字段（张力指令），单位 N，形状同上。
        selected:          选中掩码（bool 或 0/1），形状同上。

    返回：
        (targets, speeds, limits, mask) 四元组，形状均与输入相同：
            targets: 补齐后的目标角，单位 rad；
            speeds:  补齐后的速度上限，单位 rad/s；
            limits:  补齐后的第三字段，单位 N；
            mask:    补齐后的选中掩码（被补齐的关节变成“选中”）。

    异常：
        ValueError: 形状不一致；最后一维不是 10；输入非有限；
                    掩码含 0/1 之外的取值。

    补齐规则（逐条）：
        1. 只选了指对中的“第一个”时，第二个保持**收到指令瞬间的实测角**；
           只选“第二个”时同理反向处理。
        2. 被补齐关节的速度上限抄伙伴的值；若伙伴速度 ≤ 1 deg/s，
           则改用默认 120 deg/s（否则极小的速度上限会把伙伴“冻住”）。
        3. 被补齐关节的第三字段清 0。
        4. 第三字段为 0 **不**会关闭它的 Fallback 位置环：位置环照常工作，
           只是不再携带张力指令信息。
        5. 补齐后该关节在掩码中变成“选中”，因此后续每个 tick 都会继续
           跟踪这个“冻结在接收瞬间”的目标。
    """

    # ── 形状校验 ──
    if actual_rad.shape != target_rad.shape or actual_rad.shape != speed_limit_rad_s.shape:
        raise ValueError("actual, target and speed tensors must have identical shape")
    if actual_rad.shape != tension_limit_n.shape or actual_rad.shape != selected.shape:
        raise ValueError("tension and selected tensors must match the joint shape")
    # 最后一维必须是十个屈曲 DOF，否则“指对”分组无意义。
    if actual_rad.ndim == 0 or actual_rad.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"last dimension must have {len(JOINT_NAMES)} flexion joints")
    # ── 数值校验 ──
    for name, value in (
        ("actual", actual_rad),
        ("target", target_rad),
        ("speed", speed_limit_rad_s),
        ("tension", tension_limit_n),
    ):
        _finite_tensor(name, value, actual_rad)

    # 复制输入，下面只在副本上修改，保持本次调用无副作用。
    targets = target_rad.clone()
    speeds = speed_limit_rad_s.clone()
    limits = tension_limit_n.clone()
    mask = _boolean_mask("selected", selected, actual_rad).clone()
    # ── 逐指对（每两个 DOF 一组）执行补齐 ──
    for first in range(0, len(JOINT_NAMES), 2):
        second = first + 1
        # 只选了第二个 -> 第一个缺失；只选了第一个 -> 第二个缺失。
        first_missing = ~mask[..., first] & mask[..., second]
        second_missing = mask[..., first] & ~mask[..., second]
        # 缺失的一方把目标角冻结为“接收瞬间的实测角”。
        targets[..., first] = torch.where(first_missing, actual_rad[..., first], targets[..., first])
        targets[..., second] = torch.where(second_missing, actual_rad[..., second], targets[..., second])
        # 速度抄伙伴；伙伴速度 ≤ 1 deg/s 时用默认 120 deg/s 兜底。
        copied_first_speed = torch.where(
            speeds[..., second] <= math.radians(1.0),
            torch.full_like(speeds[..., second], math.radians(120.0)),
            speeds[..., second],
        )
        copied_second_speed = torch.where(
            speeds[..., first] <= math.radians(1.0),
            torch.full_like(speeds[..., first], math.radians(120.0)),
            speeds[..., first],
        )
        speeds[..., first] = torch.where(first_missing, copied_first_speed, speeds[..., first])
        speeds[..., second] = torch.where(second_missing, copied_second_speed, speeds[..., second])
        # 第三字段清 0：只是不再携带张力指令信息，位置环仍然工作。
        limits[..., first] = torch.where(first_missing, torch.zeros_like(limits[..., first]), limits[..., first])
        limits[..., second] = torch.where(second_missing, torch.zeros_like(limits[..., second]), limits[..., second])
        # 补齐后的关节纳入选中集合，后续 tick 才会继续跟踪该目标。
        mask[..., first] |= first_missing
        mask[..., second] |= second_missing
    return targets, speeds, limits, mask


# ── 控制器 ──
class FallbackController:
    """Batched position-to-effective-tendon-speed controller.

    Input and returned speed are SI radians/s.  The firmware computes an
    angle-error velocity request in deg/s, converts it to mrad/s for the
    motor's built-in velocity loop, and clips it at +/-7000 mrad/s.  This
    model takes that final velocity request as an *equivalent tendon-port*
    speed; it does not model motor electrical or gearbox dynamics.

    ``cmd_tension_limit_n`` is accepted to keep the three-field command
    interface explicit.  In Fallback it has no direct effect on the output,
    matching the firmware.  The current firmware's cold-start global
    ``init_ctrl_count`` can be enabled with ``startup_pair_evaluations=50``;
    normal calibration replays should start warm.

    This deliberately forces the fallback branch: it does not reproduce the
    firmware's contact estimator or low/high tension branch switching. Real
    identification logs must verify ``is_fallback`` for the fitted intervals.
    Calls represent ticks, not variable-duration integration steps. Position
    feedback quantization is a configurable sensor approximation; the H7
    controller itself consumes its already-calibrated measured angle.

    【中文】
    批量化的“位置 -> 腱端等效速度”控制器。

    输入与返回的速度都是 SI 单位 rad/s。固件在 deg/s 域算出角度误差速度
    指令，再换算成 mrad/s 交给电机内置速度环，并在 ±7000 mrad/s 处截断。
    本模型把这个最终速度指令当作**腱端等效速度**，不建模电机电气动态与
    减速箱动态。

    参数说明（与构造参数/配置字段对应）：
        cmd_tension_limit_n: 仅为保持“三字段指令”接口显式而保留。在
            Fallback 分支中它对输出没有直接影响，与固件一致。
        startup_pair_evaluations: 当前固件的冷启动全局 ``init_ctrl_count``
            可以通过设为 50 来模拟；正常标定回放应当从“热态”开始。

    关键假设：
        1. 刻意强制走 Fallback 分支：不复刻固件的接触估计器，
           也不复刻低/高张力分支切换。真实标定日志必须确认拟合区间内
           ``is_fallback`` 为真。
        2. 每次调用代表一个 tick，而不是可变时长的积分步。
        3. 位置反馈量化是可配置的传感器近似；H7 控制器本身直接使用
           已经标定好的实测角。
    """

    def __init__(
        self,
        cfg: FallbackControllerConfig,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """构造控制器并分配内部状态张量。

        【中文】
        参数：
            cfg:      控制器配置（增益、限幅、量化开关等）。
            num_envs: 并行环境数，须为正整数。
            device:   计算设备（如 "cpu"、"cuda:0"）。
            dtype:    浮点类型，只允许 float32 或 float64。

        返回：
            None。

        异常：
            ValueError: num_envs 不是正整数（布尔值也不算）；
                        dtype 不是 float32/float64。

        说明：
            内部状态（积分项、上一步误差、指对评估计数）按
            (num_envs, 10) 或 (num_envs,) 分配，设备与 dtype 保持一致。
        """
        # num_envs 必须是真正的正整数：显式排除 bool（True 会被当成 1）。
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        # 只支持单/双精度浮点，避免半精度下的量化误差干扰复刻结果。
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("controller dtype must be torch.float32 or torch.float64")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        # 所有张量状态的统一形状：(环境数, 关节数)。
        self._shape = (num_envs, len(JOINT_NAMES))
        # 逐关节参数以 (1, 10) 形式缓存，便于与 (num_envs, 10) 广播。
        self._kp = self._vector(cfg.kp_deg_s_per_deg)
        self._ki = self._vector(cfg.ki_deg_s_per_deg_step)
        self._kd = self._vector(cfg.kd_deg_s_per_deg_step)
        self._min_angle = self._vector(cfg.min_angle_deg)
        self._max_angle = self._vector(cfg.max_angle_deg)
        self._dead_zone = self._vector(cfg.dead_zone_deg)
        # 积分项，单位 deg/s（固件在 deg 域累加）。
        self.integral_deg_s = torch.zeros(self._shape, device=self.device, dtype=self.dtype)
        # 上一步（死区处理后的）误差，单位 deg，用于微分项。
        self.previous_error_deg = torch.zeros_like(self.integral_deg_s)
        # 每个指对已评估的 tick 数（int32），用于冷启动预热期间抑制输出。
        self.pair_evaluations = torch.full(
            (num_envs,), cfg.startup_pair_evaluations, device=self.device, dtype=torch.int32
        )

    def _vector(self, values: tuple[float, ...]) -> torch.Tensor:
        """把 10 个标量参数变成 (1, 10) 的常量张量，便于广播。

        【中文】
        参数：
            values: 长度 10 的浮点元组（每个屈曲 DOF 一个值）。

        返回：
            shape 为 (1, 10) 的张量，设备与 dtype 与控制器一致。
        """
        return torch.as_tensor(values, device=self.device, dtype=self.dtype).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None, *, cold_start: bool = False) -> None:
        """清零所选环境的内部状态。

        【中文】
        参数：
            env_ids:    要重置的环境下标；None 表示重置全部环境。
            cold_start: True 时把指对评估计数也清零（模拟固件冷启动）；
                        False 则直接把计数置为“已完成预热”的值。

        返回：
            None。

        说明：
            积分项与上一步误差一律清零；预热计数由 cold_start 决定，
            因此正常标定回放（cold_start=False）从热态开始。
        """
        rows = slice(None) if env_ids is None else env_ids
        self.integral_deg_s[rows] = 0.0
        self.previous_error_deg[rows] = 0.0
        # cold_start=True 才能复现冷启动的前 N 次“空转”行为。
        self.pair_evaluations[rows] = 0 if cold_start else self.cfg.startup_pair_evaluations

    def step(
        self,
        q_rad: torch.Tensor,
        cmd_rad: torch.Tensor,
        cmd_speed_rad_s: torch.Tensor,
        cmd_tension_limit_n: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        *,
        cmd_tension_command_n: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one firmware control tick and return tendon-port speed [rad/s].

        ``active_mask`` must contain both joints of every active finger.  For
        sparse commands, call :func:`autopair_command` on the command event and
        retain the resulting targets and mask until the next event.

        ``cmd_tension_command_n`` is the canonical third-field name;
        ``cmd_tension_limit_n`` is a legacy alias, not an enforced limit.

        【中文】
        推进一个固件控制 tick，返回腱端等效速度，单位 rad/s。

        参数：
            q_rad:                 实测关节角，单位 rad，形状 (num_envs, 10)。
            cmd_rad:               目标关节角，单位 rad，形状同上。
            cmd_speed_rad_s:       速度上限，单位 rad/s，形状同上。
            cmd_tension_limit_n:   第三字段旧名，单位 N，形状同上；不执行张力上限。
            active_mask:           选中掩码，须成对包含同一根手指的两个关节；
                                   None 表示全部选中。
            cmd_tension_command_n: 第三字段的规范名（关键字参数），单位 N。

        返回：
            torch.Tensor，形状 (num_envs, 10)，腱端等效速度，单位 rad/s；
            未运行（非活动，或尚未预热完成）的关节返回 0。

        异常：
            ValueError: 形状不是 (num_envs, 10)；两个张力参数同时给出；
                        张量含非有限值；掩码含 0/1 之外的值；
                        掩码破坏了“整指成对”。

        说明：
            稀疏指令应在**指令事件**上调用 :func:`autopair_command`，
            并把补齐后的目标与掩码保留到下一次事件。
            ``cmd_tension_command_n`` 是第三字段的规范名；
            ``cmd_tension_limit_n`` 只是历史别名，不构成张力上限。

        内部流程（与固件顺序一致）：
            角度限幅 -> 误差 -> 死区 -> 条件积分（含积分限幅）
            -> P + I + D 求和 -> 速度限幅（≤1 deg/s 用 120 deg/s，上限 360 deg/s）
            -> 换算成 mrad/s 并截断到 ±7000 -> int16 量化（int() + 0.5）
            -> 换算回 rad/s 输出。
        """

        # ── 第三字段：规范名与旧名只允许给一个 ──
        if cmd_tension_command_n is not None:
            if cmd_tension_limit_n is not None:
                raise ValueError("provide only one tension command argument")
            cmd_tension_limit_n = cmd_tension_command_n

        # ── 形状与数值校验 ──
        for name, value in (("q_rad", q_rad), ("cmd_rad", cmd_rad), ("cmd_speed_rad_s", cmd_speed_rad_s)):
            if value.shape != self._shape:
                raise ValueError(f"{name} must have shape {self._shape}, got {tuple(value.shape)}")
            _finite_tensor(name, value, self.integral_deg_s)
        if cmd_tension_limit_n is not None and cmd_tension_limit_n.shape != self._shape:
            raise ValueError(f"cmd_tension_limit_n must have shape {self._shape}")
        if cmd_tension_limit_n is not None:
            _finite_tensor("cmd_tension_limit_n", cmd_tension_limit_n, self.integral_deg_s)
        # ── 活动掩码 ──
        if active_mask is None:
            # 缺省视为全部关节活动。
            active_mask = torch.ones(self._shape, device=self.device, dtype=torch.bool)
        elif active_mask.shape != self._shape:
            raise ValueError(f"active_mask must have shape {self._shape}")
        else:
            active_mask = _boolean_mask("active_mask", active_mask, q_rad)
        # 位置环按“指对”整体运行，因此同一根手指的两个关节必须同进同出。
        pairs_valid = (active_mask[:, ::2] == active_mask[:, 1::2]).all()
        if active_mask.is_cuda:
            torch._assert_async(pairs_valid, "active_mask must contain complete finger pairs")
        elif not bool(pairs_valid):
            raise ValueError("active_mask must contain complete finger pairs")

        # ── 冷启动预热：每个指对先累计若干次评估才开始输出 ──
        run_mask = torch.zeros_like(active_mask)
        for first in range(0, len(JOINT_NAMES), 2):
            pair_active = active_mask[:, first]
            pair_ready = self.pair_evaluations >= self.cfg.startup_pair_evaluations
            run_mask[:, first : first + 2] = (pair_active & pair_ready).unsqueeze(-1)
            self.pair_evaluations += pair_active.to(self.pair_evaluations.dtype)

        # ── 1) 实测角换算到 deg，并可选按传感器分辨率量化 ──
        q_deg = torch.rad2deg(q_rad)
        if self.cfg.quantize_angle_feedback:
            quantum = self.cfg.angle_feedback_resolution_deg
            q_deg = torch.round(q_deg / quantum) * quantum
        # ── 2) 目标角限幅到 [min_angle, max_angle]（单位 deg） ──
        target_deg = torch.rad2deg(cmd_rad).clamp(min=self._min_angle, max=self._max_angle)
        # ── 3) 误差与死区 ──
        # 误差 = 限幅后的目标角 - （可选量化后的）实测角，单位 deg。
        error_deg = target_deg - q_deg
        # 积分窗：只有 |e| 小于 integral_region_deg 时才允许累加积分，
        # 避免大误差阶段把积分器顶到饱和。
        within_integral_region = error_deg.abs() < self.cfg.integral_region_deg
        # 死区：|e| 小于 dead_zone 时视为无误差，输出与积分都不再动作。
        in_dead_zone = within_integral_region & (error_deg.abs() < self._dead_zone)
        effective_error_deg = torch.where(in_dead_zone, torch.zeros_like(error_deg), error_deg)
        # ── 4) 条件积分：只在积分窗内累加，并对积分项做限幅 ──
        proposed_integral = self.integral_deg_s + self._ki * effective_error_deg
        proposed_integral = proposed_integral.clamp(-self.cfg.integral_limit_deg_s, self.cfg.integral_limit_deg_s)
        # 分支含义：
        #   进入死区   -> 清掉积分，避免死区内残留积分把关节慢慢推走；
        #   在积分窗内 -> 累加（已限幅）；
        #   在积分窗外 -> 保持不变，即条件积分的“冻结”行为。
        next_integral = torch.where(
            in_dead_zone,
            torch.zeros_like(proposed_integral),
            torch.where(within_integral_region, proposed_integral, self.integral_deg_s),
        )
        # 未运行的指对保持原积分状态不变。
        self.integral_deg_s = torch.where(run_mask, next_integral, self.integral_deg_s)

        # ── 5) P + I + D 求和，单位 deg/s ──
        reference_speed_deg_s = self._kp * effective_error_deg
        # 积分项只在积分窗内且不在死区时参与求和。
        reference_speed_deg_s += torch.where(within_integral_region & ~in_dead_zone, self.integral_deg_s, 0.0)
        # 微分项用“上一步的有效误差”，保证死区切换时不会产生跳变。
        reference_speed_deg_s += self._kd * (effective_error_deg - self.previous_error_deg)
        # ── 6) 速度限幅 ──
        # 指令速度上限取绝对值（固件只看大小），先截到 [0, speed_cap]。
        speed_max_deg_s = torch.rad2deg(cmd_speed_rad_s).abs().clamp(0.0, self.cfg.speed_cap_deg_s)
        # 上限 ≤ 1 deg/s 时改用默认 120 deg/s：这是固件对“几乎为 0 的速度上限”的兜底。
        speed_max_deg_s = torch.where(
            speed_max_deg_s <= 1.0,
            torch.full_like(speed_max_deg_s, self.cfg.default_speed_deg_s),
            speed_max_deg_s,
        )
        reference_speed_deg_s = reference_speed_deg_s.clamp(-speed_max_deg_s, speed_max_deg_s)
        # 记录本步的有效误差，供下一步微分项使用；未运行的指对保持旧值。
        self.previous_error_deg = torch.where(run_mask, effective_error_deg, self.previous_error_deg)

        # ── 7) 换算成 mrad/s 并截断到 ±output_cap_mrad_s ──
        # 1 deg/s = 17.4532925 mrad/s，与固件的换算常数一致。
        output_mrad_s = (reference_speed_deg_s * 17.4532925).clamp(
            -self.cfg.output_cap_mrad_s, self.cfg.output_cap_mrad_s
        )
        # ── 8) int16 量化 ──
        if self.cfg.quantize_output:
            # Fallback_pidControl returns ``int(output)``; control.c then
            # stores ``(int16_t)(output_final + 0.5f)``.
            # 即：先向零截断取整，再加 0.5 后再次向零截断（复刻 C 的 int 转换）。
            output_mrad_s = torch.trunc(torch.trunc(output_mrad_s) + 0.5)
        # ── 9) 换回 rad/s；未运行的指对输出 0 ──
        return torch.where(run_mask, output_mrad_s / 1000.0, torch.zeros_like(output_mrad_s))
