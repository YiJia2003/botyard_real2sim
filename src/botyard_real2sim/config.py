"""Parameters and joint ordering for the unloaded tendon calibration model.

The defaults define a runnable starting point, not measured hand parameters.  The
two known radii are 5 mm; all compliance, spring and friction values are exposed
for fitting once measured curves are available.

【中文说明】
本文件是空载腱绳标定层的**参数与关节顺序唯一来源**：只声明数据结构、单位约定
与验证规则，不含任何力学求解或控制器实现。

- 输入：YAML 解析出的字典（字段可省略，省略项取本文件默认值），或直接构造
  dataclass；也接受 `controller` / `tendon` / `curves` 三段式快照。
- 输出：不可变（frozen dataclass）的 `Real2SimConfig`，由 `controller.py`
  （位置环，单位 deg、deg/s、mrad/s）、`model.py`（单边张力、回位弹簧、关节
  摩擦、虚功力矩）、`routing_dynamics.py`（有惯量走线传动）、`runtime.py`
  （时间调度）与 `scene.py`（把关节力矩提交给 PhysX）读取。
- 对外接口：`load_config` / `save_config` 读写 YAML 快照；`JOINT_NAMES`、
  `FINGER_PAIRS`、`ROUTING_DIAGNOSTIC_FIELDS` 固定 DOF 顺序与 CSV 列名，
  `protocol.py` 的 A0 载荷编码与回放脚本都依赖这些顺序。
- 向量参数三种写法：标量广播；10 元素（或 5 元素）有序数组；具名部分覆盖
  （关节/手指键名 + 可选 `default`）。解析规则见 `_resolve_vectors`。
- 关键假设：默认值只是**可运行初值**，不是实机辨识结果；5 mm 是绕线轮/关节轮
  **半径**而非直径；刚度、阻尼、摩擦、质量都要等实测曲线到位后再拟合。
- 限制：验证只在构造时执行一次（有限性、正性、跨字段约束），进入仿真前不会
  再做隐式裁剪或修正；非法组合直接抛 `ValueError`。本文件不读取硬件，也不
  判断参数是否已被实测证认。

单位约定（全文统一）：长度 m，角度 rad，角速度 rad/s，力 N，力矩 N·m，
质量 kg，时间 s；controller 块因直接照搬固件，改用 deg、deg/s、mrad/s，
单位写在各字段名后缀里。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
from pathlib import Path
from typing import Any

# ── 关节顺序与模型常量 ──
# 十个腱驱动屈曲关节，顺序等于固件 DOF 0–9：
#   THJ1 THJ2 | FFJ2 FFJ3 | MFJ2 MFJ3 | RFJ2 RFJ3 | LFJ2 LFJ3
# （“|”只是阅读分组，不属于键名。）协议编码、CSV 列名与所有 10 元素向量参数
# 都按这个顺序解释，任何重排都会让参数错位。
JOINT_NAMES: tuple[str, ...] = (
    "THJ1",
    "THJ2",
    "FFJ2",
    "FFJ3",
    "MFJ2",
    "MFJ3",
    "RFJ2",
    "RFJ3",
    "LFJ2",
    "LFJ3",
)
"""The ten flexion DOFs, in the firmware's order (DOF 0 through 9)."""

# 注：四指 J1 由 USD 中的机械 mimic 约束随 J2 同步运动（J1=J2），不单独占用
# 一根腱，所以不在这 10 个 DOF 里；拇指的 THJ1 绳跨 THJ2。

# 五个手指的键名，顺序为拇指到小指；用于 5 元素向量参数与 geometry 段。
FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")

# 动态走线模式（routing_model: dynamic）下，每条腱绳额外输出到 CSV 的内部诊断列。
ROUTING_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    # ── 两个节点的运动学（绳长坐标）──
    "routing_proximal_position_m",  # u：近端节点已收进的绳长，x−u 就是近端材料伸长
    "routing_distal_position_m",  # v：远端节点位置，v−y 就是远端材料伸长
    "routing_proximal_velocity_m_s",  # du：近端节点速度，正 = 继续收绳
    "routing_distal_velocity_m_s",  # dv：远端节点速度
    "routing_proximal_acceleration_m_s2",  # ü = (Tp−Ts−Fu)/(m/2)：由力平衡算出，非速度差分
    "routing_distal_acceleration_m_s2",  # v̈ = (Ts−Td−Fv)/(m/2)
    # ── 摩擦界面：bristle 弹性挠度与接触状态 ──
    "routing_bristle_proximal_m",  # zu：近端"刷毛"弹性变形（钳在 ±bound/kb 内），摩擦力 = kb·zu
    "routing_bristle_distal_m",  # zv：远端刷毛弹性变形，同上
    "routing_contact_state",  # c∈[0,1]：绳贴紧导槽的程度（一阶松弛：加载 20 ms、释放 80 ms）
    # ── 摩擦与法向力（集中到两个点，各占一半）──
    "routing_friction_proximal_N",  # Fu = kb·zu + c_u·du：近端集中摩擦力，带符号
    "routing_friction_distal_N",  # Fv = kb·zv + c_v·dv：远端集中摩擦力
    "routing_normal_proximal_N",  # N_u = ½(预紧 + ½(Tp+Ts)·包角)：决定摩擦上限 μ·N
    "routing_normal_distal_N",  # N_v = ½(预紧 + ½(Td+Ts)·包角)
    "routing_spring_tension_N",  # Ts = kr·max(u−v,0) + reaction：中间那根等效弹簧的张力
    "routing_stop_reaction_N",  # reaction ≥ 0：顶到行程上限后的单向约束反力（只推不拉）
    # ── 张力落差与长度账 ──
    "routing_tension_drop_N",  # Tp − Td：被摩擦与惯性吃掉的张力，稳态时 ≈ Fu+Fv
    "routing_internal_slack_m",  # min(u−v, 0) ≤ 0：走线段内部富余长度，取绝对值才是间隙
    "routing_material_proximal_extension_m",  # ep = x−u：近端材料伸长，负值 = 该段松弛
    "routing_material_distal_extension_m",  # ed = v−y：远端材料伸长，负值 = 该段松弛
    # ── 能量台账（除 kinetic 外均为累计值）──
    "routing_kinetic_energy_J",  # ½·(m/2)·(du²+dv²)：两个集中质量的瞬时动能
    "routing_friction_dissipated_J",  # 物理摩擦耗散：塑滑功 + 粘性耗散（真正变成热的部分）
    "routing_numerical_dissipated_J",  # 离散格式引入的耗散（bristle/惯性/跨折点/约束功）
    "routing_port_work_J",  # 端口功：电机侧输入 − 关节侧输出，传动实际传递的机械能
    "routing_energy_residual_J",  # 能量账没平上的余量，不做事后配平，供审计
    # ── 求解健康与滑移、冲击 ──
    "routing_solver_residual_N",  # 本步 active set 的最大力残差，应 < 1e-7 + 1e-8·张力
    "routing_plastic_slip_proximal_m",  # 近端累计塑性滑移：绳相对导槽真正滑过的长度
    "routing_plastic_slip_distal_m",  # 远端累计塑性滑移
    "routing_impact_dissipated_J",  # 硬停止碰撞的完全非弹性损失 ¼·(m/2)·closing²
    "routing_stop_impulse_Ns",  # 上一子步把朝外相对速度清零的冲量（恢复系数 0，恒 ≥ 0）
)
# 每指的 (J2, J3) 配对：JOINT_NAMES 偶数下标是近端 J2、奇数下标是远端 J3。
# 供“同指 partner 补齐”等必须成对处理两关节的逻辑使用
# （见 controller.autopair_command）。
FINGER_PAIRS: tuple[tuple[str, str], ...] = tuple(zip(JOINT_NAMES[::2], JOINT_NAMES[1::2], strict=True))


# ── 内部校验与解析辅助函数 ──
def _finite(name: str, value: Any, *, minimum: float | None = None, positive: bool = False) -> None:
    """检查一个标量是有限实数，并按需施加下界或正性约束。

    参数：
        name: 出错信息中显示的字段名（用于定位 YAML 里的错误项）。
        value: 待检查值；bool 会被明确拒绝（bool 是 int 的子类，否则
            `true` / `false` 会被当成 1 / 0 悄悄通过）。
        minimum: 允许的下界，含等于；None 表示不检查下界。
        positive: 为 True 时要求 value > 0（与 minimum 二选一使用）。
    返回：无。
    失败条件：非实数、非有限值（nan/inf）或越界时抛 ValueError。
    副作用：无，纯校验。
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else f'>= {minimum}'}")


# 允许 inf 的向量字段（inf 有明确语义：关闭该分段/限制）。
_INF_ALLOWED_FIELDS = frozenset({"return_spring_knee_m", "joint_lower_stop_deg"})


def _resolve_vectors(instance: Any) -> None:
    """Expand scalar, ordered sequence, or partial named overrides of defaults.

    【中文】
    把三种写法统一解析成定长 `tuple[float, ...]` 并原地写回实例。

    - 只处理类默认值是 tuple 的字段（即向量参数）；长度为 10 时用
      `JOINT_NAMES` 作名字表，长度为 5 时用 `FINGER_NAMES`（crossing 参数）；
    - 具名映射：未出现的分量先取映射里的 `default`，再取该字段的类默认值；
      出现未知键名直接报错，防止 YAML 拼写错误被静默忽略；
    - 标量（非 bool）广播到所有分量；list / tuple 原样转为 tuple；
    - 结果逐元素检查有限性，再通过 `object.__setattr__` 绕过 frozen 限制写回，
      因此构造成功后这些字段必定是定长 float 元组。

    参数：instance —— 正在初始化的 dataclass 实例（由 `__post_init__` 传入）。
    副作用：原地改写 instance；不合法时抛 ValueError（此前解析成功的字段
    可能已写回，但构造已失败，该对象不应再被使用）。
    """
    for item in fields(instance):
        if not isinstance(item.default, tuple):
            continue
        names = JOINT_NAMES if len(item.default) == 10 else FINGER_NAMES
        value = getattr(instance, item.name)
        if isinstance(value, dict):
            unknown = set(value) - set(names) - {"default"}
            if unknown:
                raise ValueError(f"{item.name}: unknown names {sorted(unknown)}")
            value = tuple(
                value.get(name, value.get("default", default))
                for name, default in zip(names, item.default, strict=True)
            )
        elif isinstance(value, Real) and not isinstance(value, bool):
            value = (value,) * len(names)
        elif isinstance(value, (list, tuple)):
            value = tuple(value)
        else:
            raise ValueError(f"{item.name} must be a scalar, sequence, or named mapping")
        if len(value) != len(names):
            raise ValueError(f"{item.name} must contain {len(names)} values")
        # 个别字段允许 inf（例如回位弹簧拐点，inf 表示无拐点）：只拒绝 nan。
        allow_inf = item.name in _INF_ALLOWED_FIELDS
        for number in value:
            if allow_inf:
                if not isinstance(number, Real) or isinstance(number, bool) or math.isnan(number):
                    raise ValueError(f"{item.name} must be a number (inf allowed)")
            else:
                _finite(item.name, number)
        object.__setattr__(instance, item.name, tuple(float(number) for number in value))


def _from_dict(cls, values: dict[str, Any]):
    """把（YAML 解析出的）字典转换成配置对象，并拒绝未知键名。

    参数：cls —— 目标 dataclass 类型；values —— 必须为 mapping。
    返回：`cls(**values)` 构造的实例，完整验证在其 `__post_init__` 中执行。
    失败条件：非 mapping 或含未知键名时抛 ValueError（未知键报错而不是忽略，
    是为了暴露 YAML 里的拼写错误）。
    """
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def _check_vector(name: str, values: tuple[float, ...], size: int, *, minimum: float | None = None) -> None:
    """检查向量长度，并逐元素检查有限性（可选下界）。

    参数：name 字段名（用于报错）；values 已解析的浮点元组；size 期望长度
    （10 对应 `JOINT_NAMES`、5 对应 `FINGER_NAMES`）；minimum 允许下界（含）。
    返回：无。失败条件：长度不符或任一元素越界/非有限时抛 ValueError。
    """
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values, got {len(values)}")
    for value in values:
        _finite(name, value, minimum=minimum)


# ── 控制器参数：固件 Fallback_pidControl 位置环 ──
@dataclass(frozen=True)
class FallbackControllerConfig:
    """The selected ``Fallback_pidControl`` path, with units in the field names.

    The controller runs once every ``control_period_s`` of physical time.  The
    current firmware passes an integer-divided 2 ms ``time_delta`` into parts
    of its state-estimation code despite its roughly 3 ms interrupt period.
    This position-loop branch itself does not multiply by that ``time_delta``.

    【中文】
    复刻固件 `Fallback_pidControl` 位置环的参数集合，字段名后缀即单位。

    输入：目标角（deg）、最大速度（deg/s）、实测关节角与开关掩码；输出：
    等效绳端速度指令，最终以 mrad/s 限幅（`output_cap_mrad_s`）后再换算回
    rad/s 交给腱绳模型。控制器每 `control_period_s` 秒物理时间执行一次；
    本类只承载参数，差分/积分/量化逻辑在 `controller.py` 的 `FallbackController`。

    关键约定与限制：
    - `control_period_s = 0.003` 是**调度周期**（定时器 ARR=2999 计数 3000 µs），
      调度器按绝对时间推进，不使用 `firmware_time_delta_s`；
    - `firmware_time_delta_s = 0.002` 只是固件状态估计代码里整数截断后的时间量，
      不能当作控制周期使用，当前位置环分支本身也不乘它；
    - 积分/微分项按“每个控制步”累加（字段名带 `_step`），因此与周期绑定；
    - 所有 10 元素字段按 `JOINT_NAMES` 顺序解释，支持标量/序列/具名覆盖三种写法；
    - 第三指令字段（张力）在 Fallback 分支不参与输出，所以这里没有对应参数；
    - 参数取自当前源码分支，采集时固件版本不同必须重新核对。

    单位提醒：`kp`、`ki`、`kd` 的内部量都是“等效速度 deg/s”对“角误差 deg”
    （分别按 1、1/step、1/step 归一）的增益，不是 SI 增益。
    """

    # 【物理】位置环的比例环节：把"还差多少度"换算成"绳端该走多快"。
    #   注意它不是力矩增益——固件这一段是速度指令环，P 输出一路限幅后变成电机速度
    #   请求，真正的力来自腱绳模型（张力→关节力矩）。1 deg 误差 → 15 deg/s 等效绳端速度。
    # 单位 (deg/s)/deg。本组取值是当前固件源码的复刻：J2 一律 20、J3 一律 10、
    #   THJ1 15、THJ2 10（未列出的关节走 `default:`），不是辨识结果；改它等于改闭环带宽。
    kp_deg_s_per_deg: tuple[float, ...] = (15.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0)
    # 【物理】积分环节：消除稳态角度误差。实机上"指令到位了但还差 1–2 deg"的来源是
    #   腱绳蠕变、绳在导槽里的微量滑动、重力矩——纯 P 环只能靠残余误差换力矩，I 环把这些
    #   静差积掉。每 1 个控制步（3 ms）把当前误差累加一次，所以增益天然带 1/step 量纲。
    # 单位 (deg/s)/(deg·step)；默认 1.0，允许 0（停用积分）。
    ki_deg_s_per_deg_step: tuple[float, ...] = (1.0,) * 10
    # 【物理】微分环节：按误差变化率提前刹车，抑制超调、增加阻尼。
    #   当前固件 Fallback 分支默认 0.0，即**没有**微分作用；保留字段是为了复刻源码中
    #   存在但未启用的这一项（改成非 0 会引入源码里不存在的阻尼）。
    # 单位 (deg/s)/(deg/step)。
    kd_deg_s_per_deg_step: tuple[float, ...] = (0.0,) * 10
    # 【物理】目标角下限 deg。这是"指令"的机械安全边界，不是关节真实限位：
    #   关节名义行程 0–90 deg，指令夹到 0.5–89.5 deg，避免把关节压到机械止点上造成
    #   堵转、过流与绳过载。上界见下一字段。
    min_angle_deg: tuple[float, ...] = (0.5,) * 10
    # 【物理】目标角上限 deg（默认 89.5），与 min_angle_deg 一起把指令夹在行程内侧半度。
    max_angle_deg: tuple[float, ...] = (89.5,) * 10
    # 【物理】死区 deg：小于 1 deg 的误差在实机上根本推不动（关节摩擦、绳的装配间隙、
    #   编码器量化），所以固件把它当 0 处理并**复位积分项**——否则积分器会在死区边缘
    #   反复充放，产生持续的嗡嗡抖动（极限环）。死区是 integral_region 的内层。
    dead_zone_deg: tuple[float, ...] = (1.0,) * 10
    # 【物理】积分分离区 deg（默认 2 deg）：只有误差落在这个窗口内才允许积分。
    #   大误差阶段积分会累计出远超需要的量，等接近目标时释放出来 → 超调；这是
    #   工程上常见的"积分分离"做法，也解释了为什么 ki 不小却没有明显超调。
    integral_region_deg: float = 2.0
    # 【物理】积分项限幅 deg/s（默认 10）：抗积分饱和（windup）。执行器已经到速度上限
    #   时误差仍存在，若积分继续增长，松开的瞬间会冲出目标。
    integral_limit_deg_s: float = 10.0
    # 【物理】默认速度 deg/s（120）：指令带的第三类信息里没有给速度、或给的速度 <= 1 deg/s
    #   （视为无效）时，用这个"兜底最大速度"。它决定阶跃指令的上升快慢。
    default_speed_deg_s: float = 120.0
    # 【物理】指令速度上限 deg/s（360 = 1 圈/s）：驱动与机械允许的最大关节速度，
    #   任何指令速度都会被夹到这里；也决定控制器最快的跟踪能力。
    speed_cap_deg_s: float = 360.0
    # 【物理】电机内环速度请求限幅 mrad/s（±7000 = ±7 rad/s）。这是**驱动器层面**的
    #   饱和：固件把算出的 mrad/s 请求截断成 int16 范围的量再下发。按 5 mm 半径折算，
    #   7 rad/s 约等于 35 mm/s 收绳速度——它是速度限制，不产生力矩上限。
    output_cap_mrad_s: float = 7000.0
    # Timer ARR=2999 counts 3000 microseconds (ARR+1), not 2999.
    # 【物理】控制周期 s（0.003 s = 3 ms ≈ 333 Hz）：这是**控制器真正的采样与执行周期**，
    #   决定指令被量化的时刻、积分/微分的离散步长，以及"同一 tick 内多条指令只认一次"。
    #   ARR=2999 计数 3000 个 µs，不是 2999 个；它不是物理步长（PhysX 用 0.1 ms）。
    control_period_s: float = 0.003
    # 【物理】固件状态估计里被整数截断成 2 ms 的时间量 s。它只是复刻固件内部的一个
    #   记录量（2 ms ≠ 3 ms 周期），**不能**当调度周期；当前 Fallback 位置环分支不乘它，
    #   所以把它改成别的值不会改变控制输出——这正是它与 control_period_s 的区别。
    firmware_time_delta_s: float = 0.002
    # 【物理】角度反馈量化步长 deg（0.01 = 编码器分辨率）：真实编码器只能读到 0.01 deg
    #   的整数倍，所以闭环看到的误差是台阶状的。它不会让误差消失，只改变误差的取值。
    angle_feedback_resolution_deg: float = 0.01
    # 【物理】是否模拟上面这个量化。关掉 = 假设反馈是连续理想量（便于和理想模型对照）；
    #   打开 = 复刻实机"误差被量化成 0.01 deg 台阶"的行为。
    quantize_angle_feedback: bool = True
    # 【物理】是否模拟输出量化 `trunc(trunc(x) + 0.5)`：固件把浮点速度写进整型变量前
    #   的取整规则。它让输出变成阶梯状，并在低速时产生"最小速度台阶"（不是死区）。
    quantize_output: bool = True
    # 【物理】冷启动时每个手指先执行的评估/控制次数（固件 init_ctrl_count）。上电后
    #   固件要先跑若干拍"同指配对评估"建立基准，这段时间的行为与稳态不同；0 = 热启动
    #   （回放默认），50 = 复刻固件冷启动等待。它不改变控制律，只改变起步的瞬态。
    startup_pair_evaluations: int = 0

    def __post_init__(self) -> None:
        """解析向量参数并执行构造期验证（frozen 对象的唯一校验点）。

        顺序：先把三种写法的向量字段解析为定长元组，再检查长度、有限性、
        非负/正性，最后检查跨字段约束（默认速度不超过速度上限、每个关节的
        角度下限小于上限、周期为正）。任一项不满足即抛 ValueError，不做裁剪。
        """
        _resolve_vectors(self)
        # 以下六个字段必须恰好 10 个分量，顺序按 JOINT_NAMES。
        for name in (
            "kp_deg_s_per_deg",
            "ki_deg_s_per_deg_step",
            "kd_deg_s_per_deg_step",
            "min_angle_deg",
            "max_angle_deg",
            "dead_zone_deg",
        ):
            _check_vector(name, getattr(self, name), 10)
        # 增益类字段允许为 0（例如停用积分或微分），但不允许为负。
        for name in ("kp_deg_s_per_deg", "ki_deg_s_per_deg_step", "kd_deg_s_per_deg_step", "dead_zone_deg"):
            _check_vector(name, getattr(self, name), 10, minimum=0.0)
        # 以下标量参数必须严格为正：任何 0 周期或 0 分辨率都会让控制器失效。
        for name in (
            "integral_region_deg",
            "default_speed_deg_s",
            "speed_cap_deg_s",
            "output_cap_mrad_s",
            "control_period_s",
            "firmware_time_delta_s",
            "angle_feedback_resolution_deg",
        ):
            _finite(name, getattr(self, name), positive=True)
        _finite("integral_limit_deg_s", self.integral_limit_deg_s, minimum=0.0)
        # 整数计数与布尔开关单独检查类型：0.0 之类的浮点不能冒充计数值。
        if type(self.startup_pair_evaluations) is not int or self.startup_pair_evaluations < 0:
            raise ValueError("startup_pair_evaluations must be a nonnegative integer")
        if type(self.quantize_angle_feedback) is not bool or type(self.quantize_output) is not bool:
            raise ValueError("quantization switches must be booleans")
        # 跨字段约束：默认速度不能高于速度上限，否则限幅会静默改写默认值。
        if self.default_speed_deg_s > self.speed_cap_deg_s:
            raise ValueError("default_speed_deg_s cannot exceed speed_cap_deg_s")
        # 每个关节的目标角区间必须非空（下限严格小于上限）。
        if any(lo >= hi for lo, hi in zip(self.min_angle_deg, self.max_angle_deg, strict=True)):
            raise ValueError("every min_angle_deg must be smaller than its max_angle_deg")
        # 两个周期量都必须是正的物理时间（此处重复检查以防未来放松上面的 _finite）。
        if self.control_period_s <= 0 or self.firmware_time_delta_s <= 0:
            raise ValueError("control periods must be positive")
        if self.angle_feedback_resolution_deg <= 0 or self.startup_pair_evaluations < 0:
            raise ValueError("feedback resolution must be positive and startup evaluations nonnegative")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FallbackControllerConfig:
        """从字典构造控制器配置：未知键名报错，省略项取类默认值。"""
        return _from_dict(cls, values)

    def to_dict(self) -> dict[str, Any]:
        """导出为可 YAML 序列化的字典（asdict，字段顺序即定义顺序）。"""
        return asdict(self)


# ── 腱绳/弹簧/摩擦参数：单边腱、回位拉簧与关节摩擦 ──
@dataclass(frozen=True)
class TendonModelConfig:
    """One-sided tendon, return spring and joint-friction parameters.

    Every ten-element field follows :data:`JOINT_NAMES`.  The five-element
    crossing fields follow :data:`FINGER_NAMES`: first tendon of each finger
    passes over the second joint.  The first tendon also drives its own joint.

    【中文】
    单边腱绳、走线等效弹簧、回位实体拉簧与关节摩擦的完整参数集合。

    映射关系与单位：
    - 10 元素字段按 `JOINT_NAMES`（THJ1 THJ2 | FFJ2 FFJ3 | ...）解释；
    - 5 元素字段（`crossing_*`、`crossing_guide_mu`、`crossing_wrap_angle_rad`）
      按 `FINGER_NAMES`（thumb index middle ring little）解释，表示每指第一根腱
      跨过该指第二关节；这根腱同时驱动它自己的关节，所以在 `model.py` 的
      routing_matrix_m 里既有自身力臂也有跨关节力臂；
    - 长度 m、角度 rad、速度 m/s 或 rad/s、刚度 N/m 或 N·m/rad、阻尼
      N·s/m 或 N·m·s/rad、摩擦 N·m、质量 kg、时间 s。

    两套互相独立的弹簧参数（不要混用、也不要叠加同一通道）：
    - `routing_spring_*`：电机与关节之间**走线等效弹簧**的有限行程，达到
      `routing_spring_max_length_m` 后停止伸长、张力仍可继续升高；
    - `return_spring_*`：几何路径上的**回位实体拉簧**，用长度/方向模型，
      只对 geometry 中存在的通道生效。
    geometry 中不存在对应通道的关节（例如未配置几何的拇指）继续使用
    `joint_moment_arm_m` / `crossing_moment_arm_m` 常量力臂与关节扭转弹簧。

    `geometry` 是四指可编辑的平面路径（附着在 base/proximal/middle/distal
    连杆上的端点、圆导向件、winding 绕行方向、可脱离导向件），由
    `geometry_config.validate_geometry` 校验；配置了几何时用 h(q)=L(0)-L(q)
    及其解析雅可比计算力臂与力矩，常量力臂字段只作回退/拇指参数。

    限制：没有逐导轮三维接触，没有电机电气或减速比模型；走线摩擦是集总等效；
    `crossing_guide_mu` 非零会被拒绝（避免与集总走线摩擦重复计入损失）；
    本类的默认值全部是可运行初值，**不是**辨识结果。
    """

    # ── 常量力臂与电机侧几何（无 geometry 时的回退路径，也用于拇指）──
    # 【物理】关节侧力臂 r：腱张力作用到关节转轴上的杠杆长度，即 τ = r·T 里的 r。
    #   1 N 腱张力 → 0.005 N·m 关节力矩。它是几何量（绳到轴心的垂直距离），不是可自由
    #   标定的增益；有 geometry 时由 h(q) 的解析雅可比给出，这里只在无几何（拇指）时生效。
    # 单位 m；典型 3–8 mm，指尖关节小于掌指关节。
    joint_moment_arm_m: tuple[float, ...] = (0.005,) * 10
    # 【物理】电机输出轴上的绕线轮半径 r_d：绳直接缠在它上面，电机转 θ 就收上 θ·r_d 的绳。
    #   它是"电机角 ↔ 收绳长度"的唯一换算系数，也是电机侧速度换算的杠杆。
    # 单位 m；**是半径不是直径**（5 mm 半径 = 10 mm 直径，这里写成 0.0025 会让所有
    #   收绳量、张力和速度统统差一倍）。
    drum_radius_m: tuple[float, ...] = (0.005,) * 10
    # Signed routing hypothesis, not a measured crossing geometry.
    # 【物理】每指第一根腱跨过**第二关节**（DIP）时对该关节的带符号力臂 m（5 元素，按手指名）。
    #   欠驱动手指的耦合来源：同一根腱既驱动自己的 J2，又绕过 J3 的轴心，于是拉一下会同时
    #   弯两个关节。符号 = 该腱产生的是同向还是反向力矩，决定"勾指"还是"伸指"。
    # 这是待验证的路径假设（不是实测 CAD），必须用实测运动方向确认符号。
    crossing_moment_arm_m: tuple[float, ...] = (0.005,) * 5
    # ── 材料绳（单边，只能受拉，不能受压）──
    # 【物理】绳本身的轴向弹性（Dyneema/PEEK 等编织绳的"绷紧程度"）。8000 N/m 表示
    #   1 m 长的绳被 8 N 拉长 1 mm。它决定"收绳量 → 张力"的斜率，也决定张力波传播的快慢；
    #   绳越硬，同样的收绳量产生的张力越大、力传递越"硬"。
    # 单位 N/m。只受拉：伸长取正部，绳不能被推（放松时张力归 0，改为 Slack）。
    cable_stiffness_n_per_m: tuple[float, ...] = (8000.0,) * 10
    # 【物理】绳的内部阻尼（材料粘弹性、纤维间摩擦）。它让张力的变化滞后于位移，
    #   在准静态模型里表现为张力-伸长曲线出现滞回。
    # 单位 N·s/m。dynamic 走线模式**要求为 0**：此时耗散已由走线摩擦/接触状态承担，
    #   两边同时给会让同一份损失被计两次。
    cable_damping_n_s_per_m: tuple[float, ...] = (0.0,) * 10
    # Initial gap at x=h(q); positive configured pretension is incompatible.
    # 【物理】装配预松弛：绳比"刚好绷直"的长度多出来的那段（m）。在这段间隙被拉完之前，
    #   收绳动作只是把绳拉直、不产生任何张力——实机上表现为"空行程/死区"，指令先走一段
    #   手指却不动。越过间隙后张力从 0 开始按 cable_stiffness 上升。
    # 与 initial_tension_n 互斥：同一根绳不可能既松着又有正预紧（校验会拒绝）。
    cable_slack_m: tuple[float, ...] = (0.0,) * 10
    # 【物理】初始预紧张力：装配时把绳预先拉紧到 N，于是伸长量从 0 起就带一个偏置张力
    #   （相当于在弹簧上预压一段）。它让关节在"零指令"时也有微小保持力矩，减少起动死区。
    # 单位 N；与正 cable_slack_m 互斥。
    initial_tension_n: tuple[float, ...] = (0.0,) * 10
    # 【物理】绳端速度的一阶跟踪滞后时间常数：电机/减速器/绳的柔顺使"命令速度"到
    #   "实际收绳速度"不是瞬间完成，用一阶惯性近似。0 = 理想速度端口（命令即到位）。
    # 单位 s；它会把阶跃指令抹圆，也是"指令与实测之间有延迟"的来源之一。
    tendon_speed_lag_s: tuple[float, ...] = (0.0,) * 10
    # Encoder zero is independent of the cable length reference. Radius, not diameter.
    # 【物理】电机编码器的参考零点：装配时"电机角 0 rad"对应的物理位置，与绳长参考点
    #   相互独立（初始化会按反馈行重新估计一次）。它决定同一收绳量 x 对应的电机转角。
    # 单位 rad；换算用半径不是直径。偏了会让所有张力-角度关系的起点整体平移。
    motor_zero_angle_rad: tuple[float, ...] = (0.0,) * 10
    # 【物理】编码器计数方向（±1）：电机正转到底对应"收绳"还是"放绳"。装配反了就必须
    #   取 -1，否则控制器会朝着让误差变大的方向推——正反馈、瞬间打到底。
    # 只允许 ±1（0 或 0.5 这类值会被拒绝，见 __post_init__）。
    motor_encoder_sign: tuple[float, ...] = (1.0,) * 10
    # ── 走线等效弹簧与动态走线传动（routing_*）──
    # 【物理图像】dynamic 模式下，电机与关节之间的整段绳被聚合成两个集中质量点：
    #
    #   motor x --[ kp ]-- u ==[ kr ]== v --[ kd ]-- y(q)
    #   (sensor)             |             |
    #                    bristle 摩擦   bristle 摩擦   (法向力 x 包角, 各占一半)
    #                        +------ 硬停止: u - v <= travel ------+
    #
    # 语义：u、v 是"绳长坐标"（已收进去多少米），不是三维位置。这一层负责解释
    #   "电机侧张力和关节侧张力为什么不相等"：中间那根 kr 弹簧、两个摩擦界面与硬停止。
    # 【物理】走线等效弹簧刚度 kr：把"绳在导向件之间可以拉长一点"集总成一根弹簧。
    #   它是张力传递的弹性环节：外力先拉长它，张力才升上去，因此它同时决定滞后与
    #   "两侧张力不同步"的程度。
    # 单位 N/m（1000 = 1 N/mm）。若 max_length <= free_length（零行程），这根弹簧被锁死
    #   不再伸长，但 dynamic 分支的质量与摩擦仍照常起作用。
    routing_spring_stiffness_n_per_m: tuple[float, ...] = (1000.0,) * 10
    # 【物理】上面那根等效弹簧的自由长度（未伸长时不产生力的状态）。它是长度的基准，
    #   本身不影响力，只决定"从哪一点开始算伸长"。
    # 单位 m。
    routing_spring_free_length_m: tuple[float, ...] = (0.0,) * 10
    # 【物理】等效弹簧的有限行程上限：伸长到 travel = max − free 后顶到**硬停止**。
    #   此时绳不能再被拉长（u−v 冻结），但两端张力仍可继续升高——硬停止是位移约束，
    #   不是力的上限，否则绳就没法传力了。
    # 单位 m；s1 里设 0.020 → 0.023，即 3 mm 行程。
    routing_spring_max_length_m: tuple[float, ...] = (0.0,) * 10
    # Legacy algebraic series spring or stateful inertial/frictional transmission.
    # 【物理】走线描述层级的选择，决定上图的右边部分是否被建模：
    #   "quasistatic" = 只保留代数串联弹簧（无质量、无历史，进出口张力近似相等，快但无滞后）；
    #   "dynamic"    = 两个有质量的节点 + 摩擦/接触状态（有惯性延迟、有张力落差、有记忆）。
    # 回放要解释实测的张力落差与相位滞后时必须用 dynamic；只想复现旧结果时用 quasistatic。
    routing_model: str = "quasistatic"
    # 【物理】参与运动的走线等效质量（绳 + 导向件附近随动部分），两个节点各占一半。
    #   它给张力传播引入惯性：电机侧先动、关节侧后动，于是瞬态出现张力差与"张力波"。
    #   它不是弹簧的等效质量，也不是电机转子惯量（本模型不含电机电气/减速比）。
    # 单位 kg；默认 2 g（每节点 1 g）。
    routing_mass_kg: tuple[float, ...] = (0.002,) * 10
    # 【物理】材料柔度在近端/远端的分配比例 f：把一根绳的总柔度拆成两段，使 u 与 v
    #   各自承担一部分材料伸长，从而能观测绳内的张力梯度。kp = k_c/f、kd = k_c/(1−f)，
    #   串联后 1/kp + 1/kd = 1/k_c，**总柔度不变**，所以它不改变力学，只改变"哪段更软"。
    # 严格 0 < f < 1；0.5 = 两侧均分。
    routing_proximal_compliance_fraction: tuple[float, ...] = (0.5,) * 10
    # 【物理】绳与导向件之间的静摩擦系数 μs：粘着状态下的切向承载上限 = μs·法向力。
    # 它决定"要靠多大张力差才能让绳开始滑动"，也决定摩擦落差的上限。
    # 无量纲；必须 >= 动摩擦系数 μd（否则 Stribeck 曲线非物理）。
    routing_friction_mu_static: tuple[float, ...] = (0.25,) * 10
    # 【物理】动摩擦系数 μd：已经稳定滑动时的摩擦水平。μs 与 μd 的差决定
    #   "粘着→滑移"过渡有多明显（Stribeck 效应），差值越大越容易出现爬行/抖动。
    # 无量纲；默认 0.15。
    routing_friction_mu_dynamic: tuple[float, ...] = (0.15,) * 10
    # 【物理】微滑（bristle）刚度 kb：把接触面看成一根刚度恒定的"刷毛弹簧"，粘着时靠它
    #   的弹性变形存切向力（摩擦力 = kb·z）。它不是摩擦系数、也不是绳刚度，而是
    #   "粘着状态下界面有多硬"——越大越接近理想粘着（位移小、力大）。
    # 单位 N/m；2000 N/m 表示 0.1 mm 弹性挠度对应 0.2 N 切向力。
    routing_bristle_stiffness_n_per_m: tuple[float, ...] = (2000.0,) * 10
    # 【物理】粘性摩擦系数：绳在导槽里被挤压时与速度成正比的剪切阻力（类阻尼器）。
    #   它提供高速段的耗散与平滑过渡，避免纯库仑摩擦在速度过零时的数值跳变。
    # 单位 N·s/m；实际粘性还随法向力饱和：c·N/(tension_scale + N)。
    routing_viscous_friction_n_s_per_m: tuple[float, ...] = (0.05,) * 10
    # 【物理】Stribeck 特征速度 vs：摩擦系数从 μs 降到 μd 的速度尺度：
    #   μ(w) = μd + (μs − μd)·exp(−(w/vs)²)。w ≪ vs 时接近静摩擦，w ≫ vs 时趋于动摩擦。
    # 单位 m/s；默认 2 mm/s，与手指的实际滑动速度同量级，所以过渡区正好落在工作范围内。
    routing_stribeck_speed_m_s: tuple[float, ...] = (0.002,) * 10
    # 【物理】法向力预紧 N：导槽对绳的预压偏置。即使绳里没有张力，绳也被槽壁夹着，
    #   于是仍有摩擦（对应"空载也有阻力"）。它为 0 时法向力完全由"张力 × 包角"产生。
    routing_normal_preload_n: tuple[float, ...] = (0.0,) * 10
    # 【物理】基础包角 rad：绳在导槽/销上贴合的角度（几何决定），乘当前张力得到法向力，
    #   再乘摩擦系数得到摩擦上限。包角越大，同样张力被吃掉得越多。
    routing_wrap_base_rad: tuple[float, ...] = (0.25,) * 10
    # 【物理】包角随接触状态的增益 rad：绳绷得越紧（接触状态越大），贴合越紧密、包角越大。
    #   可为负：某些路径绷紧时反而会脱离部分接触面。
    routing_wrap_contact_gain_rad: tuple[float, ...] = (1.5,) * 10
    # 【物理】接触状态基线 c0：完全不加载时绳与导槽的基础贴合程度（0–1 无量纲）。
    #   下面四个增益把"张力、走线伸长、路径弯曲"换算成接触状态的目标值：
    #   c* = clamp01(c0 + aT·T/T_scale + ae·伸长/行程 + ay·|y|/path_scale)
    routing_contact_baseline: tuple[float, ...] = (0.1,) * 10
    # 【物理】张力对接触状态的增益 aT：张力越大绳压得越紧（用 tension_scale 归一化）。
    # 无量纲；默认 0.5。
    routing_contact_tension_gain: tuple[float, ...] = (0.5,) * 10
    # 【物理】走线弹簧收紧量对接触状态的增益 ae：绳被拉长说明它正在被拽紧，接触更实。
    # 无量纲（伸长按行程归一化）。
    routing_contact_extension_gain: tuple[float, ...] = (0.3,) * 10
    # 【物理】几何收绳量 |y| 对接触状态的增益 ay：手指弯得越多，绳在关节处的折角越大、
    #   贴合越紧。无量纲（按 path_scale 归一化）。
    routing_contact_path_gain: tuple[float, ...] = (0.1,) * 10
    # 【物理】张力归一化尺度 N（5 N）：把"张力大小"换算成 0–1 的无量纲量再去影响接触
    #   状态。它本身不是阈值，只决定增益 aT 的"灵敏度中心"。
    routing_contact_tension_scale_n: tuple[float, ...] = (5.0,) * 10
    # 【物理】路径归一化尺度 m（20 mm）：把收绳量换算成无量纲量的参考长度，含义同上。
    routing_contact_path_scale_m: tuple[float, ...] = (0.02,) * 10
    # 【物理】接触建立（收紧）的时间常数 s：绳贴到导槽上不是瞬间完成，按一阶松弛逼近
    #   目标接触状态。默认 20 ms——比一次手指屈曲快，所以加载时几乎"立即贴合"。
    routing_contact_load_tau_s: tuple[float, ...] = (0.02,) * 10
    # 【物理】接触释放的时间常数 s（默认 80 ms，比加载慢 4 倍）：绳离开导槽时被摩擦与
    #   残余张力拖住，所以"松开"滞后于"拉紧"。这是实机张力曲线上的滞回/记忆来源。
    routing_contact_release_tau_s: tuple[float, ...] = (0.08,) * 10
    # 【物理】隐式积分的子步上限 s（0.1 ms）。它不是物理现象，而是数值要求：后向欧拉在
    #   刚性弹簧（8000 N/m / 2 g 的特征频率很高）下必须用小步长才稳定。与 PhysX 的
    #   物理步长相互独立，物理步再粗也不会让这里失稳。
    routing_max_substep_s: float = 0.0001
    # ── 回位实体拉簧（只对 geometry 中存在的通道生效）──
    # Length-law return springs; used only on paths present in geometry.
    # 【物理】手指背侧的实体拉伸弹簧：一端固定在指节上、一端拴在绳上，作用是把手指
    #   拉回伸展位，并在屈曲时提供被动回复力矩（这就是"松手会自己伸直"的来源）。
    # 刚度 N/m（1000 = 1 N/mm）。
    # 注意它与 routing_spring_stiffness_n_per_m 是**两套独立参数**（当前默认值恰好相同）：
    #   前者是手指上的实体弹簧，走长度/方向模型；后者是绳在导槽间的等效弹性。不可混用。
    return_spring_stiffness_n_per_m: tuple[float, ...] = (1000.0,) * 10
    # 分段回位刚度：拉伸超过 return_spring_knee_m 后改用第二刚度。
    # 缺省拐点 inf、第二刚度等于第一刚度 ⇒ 逐位等同旧的单段行为。
    return_spring_stiffness_high_n_per_m: tuple[float, ...] = (1000.0,) * 10
    return_spring_knee_m: tuple[float, ...] = (math.inf,) * 10
    # 关节低角单向限位：q 低于该角时产生单边反力。-inf 表示无限位（缺省，逐位不变）。
    joint_lower_stop_deg: tuple[float, ...] = (-math.inf,) * 10
    joint_stop_stiffness_nm_per_rad: tuple[float, ...] = (10.0,) * 10
    joint_stop_damping_nm_s_per_rad: tuple[float, ...] = (0.05,) * 10
    # 限位上方的方向死区：在 [stop, stop+band] 内附加库仑保持力矩，复现实测 0.7° 死区。
    joint_stop_band_deg: tuple[float, ...] = (0.7,) * 10
    joint_stop_band_friction_nm: tuple[float, ...] = (0.0,) * 10
    # 【物理】回位弹簧的自由长度（未拉伸时的长度，图纸标注 20 mm）。实际安装长度由几何
    #   决定，通常 > 自由长，所以弹簧常态处于拉伸状态、持续提供回复力。
    # 单位 m；它只影响"从哪一点开始算伸长"，不是力值。
    return_spring_free_length_m: tuple[float, ...] = (0.020,) * 10
    # 【物理】回位弹簧的初拉力 N：超过自由长之后额外叠加的常量力（预紧）。
    #   用来表达"弹簧在装配时已经被拉到某个预紧程度"，让低屈曲角也有回复力矩。
    # 单位 N；为 0 表示纯长度律（力 = k·max(0, 实际长 − 自由长)）。
    return_spring_initial_tension_n: tuple[float, ...] = (0.0,) * 10
    # 【物理】连接短绳（tether）长度 m：把回位弹簧末端连到指节上的不可伸长段。
    #   它把弹簧的安装位置平移一段固定距离，本身不产生力，只改变弹簧的瞬时长度。
    # 单位 m；s1.yaml 按图纸覆盖为 J2 = 17 mm、J3 = 19 mm（默认 0 = 无短绳）。
    return_tether_length_m: tuple[float, ...] = (0.0,) * 10
    # Constant unsampled path/hooks length; does not alter the moment arm.
    # 【物理】未建模的端钩、扎带、固定走线的常量长度补偿 m。实物上绳总比理想折线长一点，
    #   这段长度只平移弹簧的长度参考点，**不改变力臂**（因此不影响张力→力矩的换算）。
    # 单位 m；它修的是"长度账"，不是"力学"。
    return_path_length_offset_m: tuple[float, ...] = (0.0,) * 10
    # Editable planar, link-attached points and circular guide paths.
    # 【物理】四指在矢状面（x–z 平面）内的**腱路几何**：绳从哪里出、绕过哪些导向件、
    #   在每根连杆上的附着点。它决定绳长 h(q) 与"变力臂" J(q) = −∂h/∂q ——
    #   手指弯得越多，绳到关节轴的垂直距离越小，同样张力产生的力矩也越小。这正是
    #   实测"力矩随角度下降"的物理来源，也是几何段比常量力臂更可信的原因。
    # 键含义（四指共用一份，最容易被误当成实测 CAD 的部分）：
    #   centers_m          3 个 (x, z) 点，顺序 J3 / J2 / J1，单位 m（关节轴心位置）；
    #   distal_mimic_ratio J1=J2 的从动比例（默认 1.0，即远端由近端机械耦合带动）；
    #   tendons/returns    first / second 两条路径的节点列表（每路至少 2 个节点）；
    #   节点 frame         取 base/proximal/middle/distal，表示附着在哪根连杆上；
    #   节点 point_m       该帧下的 (x, z)，单位 m；
    #   节点 radius_m      > 0 时表示内部圆导向件半径 m；
    #   节点 winding       圆导向绕行方向，只能 -1 或 +1；
    #   节点 detachable    该内部圆导向能否在包角归零时脱离（每条路径最多一个）；
    #   provenance         几何来源说明（s1.yaml 里记录 STEP 的 sha256）。
    # 约定：正角从 +z 朝 −x 弯曲。空字典 = 不用几何，改走上面的常量力臂 + 关节扭转弹簧
    #   旧路径（拇指目前就是这种情况，其几何仍是临时值）。
    geometry: dict[str, Any] = field(default_factory=dict)
    # ── 关节侧扭转弹簧与关节摩擦（旧路径仍用；有几何时回位弹簧替代弹簧项）──
    # 【物理】关节自身的被动刚度：肌腱、皮肤、韧带的集总弹性，等效成一根绕关节轴的
    #   扭转弹簧。手指完全"断电"时被它和回位弹簧一起推回伸展位。
    # 弹簧静止角 rad：扭转弹簧的零应力角（这一项为 0 时张力矩也为 0 的角度）。
    spring_rest_angle_rad: tuple[float, ...] = (0.0,) * 10
    # 【物理】上面那根扭转弹簧的刚度 N·m/rad（0.01 = 1 deg 变形约 0.17 mN·m）。
    #   它给关节提供"越弯越顶"的被动回复力矩，与回位拉簧方向一致但机制不同
    #   （回位弹簧靠长度变化，本项直接对角度线性）。
    spring_stiffness_nm_per_rad: tuple[float, ...] = (0.01,) * 10
    # 【物理】关节弹簧的常值预紧力矩 N·m：装配时的预扭（手指被"预先掰弯"或被扎带
    #   拉住），按 −preload 叠加到弹簧力矩上，即偏向使关节伸展的方向。
    # 与初张力/预紧不同，它是力矩不是力，也不随角度变化。
    spring_preload_nm: tuple[float, ...] = (0.0,) * 10
    # 【物理】关节黏性摩擦 N·m·s/rad：轴承、密封、肌腱侧向挤压带来的"越转越快阻力越大"，
    #   与关节角速度成正比。它决定手指运动的平滑度与抖动衰减，不影响静态保持力。
    viscous_friction_nm_s_per_rad: tuple[float, ...] = (0.001,) * 10
    # 【物理】Coulomb 摩擦力矩幅值 N·m：关节静止/滑动时的恒定阻力，方向与运动相反。
    #   它决定了"多小的指令推不动关节"——也就是角度死区的物理来源。
    coulomb_friction_nm: tuple[float, ...] = (0.002,) * 10
    # Low-speed magnitude of a smooth law; does not implement stationary stiction.
    # 【物理】平滑 Stribeck 律的低速摩擦幅值 N·m：角速度趋于 0 时的摩擦水平（接近但
    #   不等于静摩擦）。**不是**理想 stiction——它不会真正"锁死"关节，只是低速时阻力变大；
    #   所以模型里不能指望它把关节停在任意角度（真实保持靠腱张力）。
    # 约束：该值必须 >= coulomb_friction_nm（否则 Stribeck 曲线非物理）。
    # 走线滑动累积状态（θ，单位 m）：绳在路径上让出的长度。
    # 全 0 表示关闭该状态（与历史版本逐位一致）；上限由实测分支差换算
    # （ΔT/kr ≈ 7.6 N / 1000 N/m ≈ 8 mm）。
    routing_slip_max_length_m: tuple[float, ...] = (0.0,) * 10
    routing_slip_load_tau_s: tuple[float, ...] = (0.02,) * 10
    routing_slip_release_tau_s: tuple[float, ...] = (0.15,) * 10
    # 可选的小速率项（m/s）：relay 到地板后继续缓慢让出绳长。
    # 实测 3° 保持 8.5 s 爬 +0.45 N ≈ 0.053 N/s，折算 5.3e-5 m/s（kr = 1000 N/m）。
    routing_slip_creep_m_per_s: tuple[float, ...] = (0.0,) * 10

    static_friction_nm: tuple[float, ...] = (0.002,) * 10
    # 【物理】Stribeck 特征角速度 rad/s（0.05 rad/s ≈ 2.9 deg/s）：摩擦从上表的低速值
    #   下降到 Coulomb 值的过渡尺度。手指的正常运动速度远大于它，所以工作中主要表现为
    #   Coulomb + 粘性；只有在"快到位、慢慢爬"时才会经过这段过渡区（表现为末段爬行）。
    stribeck_speed_rad_s: tuple[float, ...] = (0.05,) * 10
    # 【物理】低速平滑角速度 rad/s：纯粹为了避免速度过零时 tanh/sign 式的数值跳变，
    #   让摩擦在 0 附近连续。它不是物理参数，改小会让低速更"脆"，改大会软化保持力。
    friction_smoothing_speed_rad_s: tuple[float, ...] = (0.01,) * 10
    # 【物理】跨关节导向件（腱从一根手指的关节上绕过去时的销/槽）的摩擦系数，
    #   5 元素按手指名。当前只允许 0：该损耗尚未建模，非零会被拒绝，避免与下面的
    #   集总走线摩擦重复计入同一份损失。
    crossing_guide_mu: tuple[float, ...] = (0.0,) * 5
    # 【物理】跨关节导向件的包角 rad，与上面的 μ 配对（法向力 = 张力 × 包角相关项）。
    #   属于尚未实现的导向损失模型，当前只做校验、不参与计算。
    crossing_wrap_angle_rad: tuple[float, ...] = (0.0,) * 5
    # 【物理】导向件滑移的低速平滑速度 m/s，同样是数值用的预留字段（当前只校验正性）。
    guide_slip_smoothing_m_s: float = 0.0001

    def __post_init__(self) -> None:
        """解析向量参数并执行构造期验证（含跨字段与几何拓扑检查）。

        验证顺序：向量解析 → 长度/有限性 → 非负与正性 → 枚举与符号
        （`motor_encoder_sign` 只能 ±1，`routing_model` 只能是 quasistatic /
        dynamic）→ 跨字段约束（静摩擦系数 >= 动摩擦系数、柔度比例在 (0,1)、
        宏静摩擦 >= Coulomb、松弛与预紧互斥、dynamic 走线要求材料阻尼为 0 等）
        → 调用 `geometry_config.validate_geometry` 校验几何拓扑。

        任一项不满足即抛 ValueError，不做静默裁剪或自动修正；这里的检查是
        参数进入仿真前的唯一关口。
        """
        _resolve_vectors(self)
        # 这批 10 元素字段只允许非负（0 是合法占位值，例如 0 阻尼、0 松弛）。
        nonnegative_10 = (
            "joint_moment_arm_m",
            "drum_radius_m",
            "cable_stiffness_n_per_m",
            "cable_damping_n_s_per_m",
            "cable_slack_m",
            "initial_tension_n",
            "tendon_speed_lag_s",
            "routing_spring_stiffness_n_per_m",
            "routing_spring_free_length_m",
            "routing_spring_max_length_m",
            "return_spring_stiffness_n_per_m",
            "return_spring_stiffness_high_n_per_m",
            "return_spring_free_length_m",
            "joint_stop_stiffness_nm_per_rad",
            "joint_stop_damping_nm_s_per_rad",
            "joint_stop_band_deg",
            "joint_stop_band_friction_nm",
            "return_spring_initial_tension_n",
            "return_tether_length_m",
            "spring_stiffness_nm_per_rad",
            "spring_preload_nm",
            "viscous_friction_nm_s_per_rad",
            "coulomb_friction_nm",
            "static_friction_nm",
            "stribeck_speed_rad_s",
            "friction_smoothing_speed_rad_s",
        )
        for name in nonnegative_10:
            _check_vector(name, getattr(self, name), 10, minimum=0.0)
        _check_vector("crossing_moment_arm_m", self.crossing_moment_arm_m, 5)
        _check_vector("spring_rest_angle_rad", self.spring_rest_angle_rad, 10)
        _check_vector("motor_zero_angle_rad", self.motor_zero_angle_rad, 10)
        # 编码器符号只能是纯 ±1；0 或 0.5 之类的值会静默改变电机角方向。
        if any(x not in (-1.0, 1.0) for x in self.motor_encoder_sign):
            raise ValueError("motor_encoder_sign must be -1 or +1")
        _check_vector("return_path_length_offset_m", self.return_path_length_offset_m, 10)
        # 分段回位刚度：拐点允许 inf（表示无拐点），其余必须为正有限值。
        if len(self.return_spring_knee_m) != 10:
            raise ValueError("return_spring_knee_m must contain 10 values")
        if any(v <= 0 or math.isnan(v) for v in self.return_spring_knee_m):
            raise ValueError("return_spring_knee_m must be positive or inf")
        # 走线模型是枚举，不接受其他拼写（例如 "dynamic " 或 "quasi"）。
        if self.routing_model not in ("quasistatic", "dynamic"):
            raise ValueError("routing_model must be quasistatic or dynamic")
        for name in (
            "routing_mass_kg",
            "routing_bristle_stiffness_n_per_m",
            "routing_stribeck_speed_m_s",
            "routing_contact_tension_scale_n",
            "routing_contact_path_scale_m",
            "routing_contact_load_tau_s",
            "routing_contact_release_tau_s",
        ):
            if any(v <= 0 for v in getattr(self, name)):
                raise ValueError(f"{name} must be positive")
        for name in (
            "routing_slip_load_tau_s",
            "routing_slip_release_tau_s",
        ):
            if any(v <= 0 for v in getattr(self, name)):
                raise ValueError(f"{name} must be positive")
        _check_vector("routing_slip_max_length_m", self.routing_slip_max_length_m, 10, minimum=0.0)
        _check_vector("routing_slip_creep_m_per_s", self.routing_slip_creep_m_per_s, 10, minimum=0.0)
        for name in (
            "routing_friction_mu_static",
            "routing_friction_mu_dynamic",
            "routing_viscous_friction_n_s_per_m",
            "routing_normal_preload_n",
            "routing_wrap_base_rad",
        ):
            _check_vector(name, getattr(self, name), 10, minimum=0.0)
        # 跨字段：静摩擦系数不得小于动摩擦系数，否则摩擦律会自相矛盾。
        if any(s < d for s, d in zip(self.routing_friction_mu_static, self.routing_friction_mu_dynamic, strict=True)):
            raise ValueError("routing static friction coefficient must be >= dynamic coefficient")
        if any(not 0 < v < 1 for v in self.routing_proximal_compliance_fraction):
            raise ValueError("routing_proximal_compliance_fraction must lie strictly between 0 and 1")
        if any(not 0 <= v <= 1 for v in self.routing_contact_baseline):
            raise ValueError("routing_contact_baseline must lie between 0 and 1")
        _finite("routing_max_substep_s", self.routing_max_substep_s, positive=True)
        # dynamic 走线已用自身摩擦/阻尼建模耗散，材料阻尼必须为 0，避免重复计入。
        if self.routing_model == "dynamic" and any(self.cable_damping_n_s_per_m):
            raise ValueError("dynamic routing uses routing friction/damping; cable_damping_n_s_per_m must be zero")
        for k, free, maximum, damping in zip(
            self.routing_spring_stiffness_n_per_m,
            self.routing_spring_free_length_m,
            self.routing_spring_max_length_m,
            self.cable_damping_n_s_per_m,
            strict=True,
        ):
            if maximum < free or k <= 0:
                raise ValueError("routing spring requires positive stiffness and max_length >= free_length")
            if maximum > free and damping != 0:
                raise ValueError("cable damping with an active routing spring is unsupported; use zero damping")
        # 局部导入：几何校验模块只在真正需要校验 geometry 时才加载。
        from .geometry_config import validate_geometry

        validate_geometry(self.geometry)
        _check_vector("crossing_guide_mu", self.crossing_guide_mu, 5, minimum=0.0)
        _check_vector("crossing_wrap_angle_rad", self.crossing_wrap_angle_rad, 5, minimum=0.0)
        # 力臂与绕线轮半径必须严格为正：0 会让电机角或力矩变成 0 或除零。
        for name in ("joint_moment_arm_m", "drum_radius_m"):
            if any(value <= 0.0 for value in getattr(self, name)):
                raise ValueError(f"{name} must be strictly positive")
        if any(value <= 0.0 for value in self.cable_stiffness_n_per_m):
            raise ValueError("cable_stiffness_n_per_m must be strictly positive")
        # 跨字段：静摩擦幅值必须不小于 Coulomb 幅值，否则 Stribeck 曲线非单调。
        if any(
            static < coulomb for static, coulomb in zip(self.static_friction_nm, self.coulomb_friction_nm, strict=True)
        ):
            raise ValueError("static_friction_nm must be >= coulomb_friction_nm")
        if any(value <= 0.0 for value in self.friction_smoothing_speed_rad_s):
            raise ValueError("friction_smoothing_speed_rad_s must be strictly positive")
        # 非零导向摩擦会与集总走线摩擦重复计入损失，因此在实现逐导轮模型前拒绝。
        if any(self.crossing_guide_mu):
            raise ValueError("nonzero crossing_guide_mu is unsupported: guide loss requires validated segment geometry")
        # 跨字段：同一根绳不能同时有正松弛间隙和正初始张力（物理上互相矛盾）。
        if any(
            slack > 0 and tension > 0 for slack, tension in zip(self.cable_slack_m, self.initial_tension_n, strict=True)
        ):
            raise ValueError("initial slack and positive initial tension cannot coexist on the same tendon")
        if any(value <= 0 for value in self.stribeck_speed_rad_s):
            raise ValueError("stribeck_speed_rad_s must be positive")
        _finite("guide_slip_smoothing_m_s", self.guide_slip_smoothing_m_s, positive=True)
        if self.guide_slip_smoothing_m_s <= 0.0:
            raise ValueError("guide_slip_smoothing_m_s must be strictly positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TendonModelConfig:
        """从字典构造腱绳配置：未知键名报错，省略项取类默认值。"""
        return _from_dict(cls, values)

    def to_dict(self) -> dict[str, Any]:
        """导出为可 YAML 序列化的字典（向量字段为元组，dump 时写成列表）。"""
        return asdict(self)


# ── 顶层配置：控制器 + 腱绳模型 + 曲线选项 ──
@dataclass(frozen=True)
class Real2SimConfig:
    """Top-level configuration written by the calibration tools as YAML.

    【中文】
    顶层配置：`controller`（固件位置环参数）、`tendon`（腱绳/弹簧/摩擦/几何）、
    `curves`（只影响回放对齐与绘图的选项，不改变物理）。

    `curves` 段由回放 CLI 与 `scripts/real2sim_curves.py` 共享（张力刻度、指令沿
    对齐、稳态窗口等）；`save_config` 在新配置的 `curves` 为空时会保留文件里
    原有的 `curves`，避免一次写参数就把绘图选项丢掉。
    三段之外的键名一律报错；缺段时用各类的默认值。
    """

    controller: FallbackControllerConfig = field(default_factory=FallbackControllerConfig)
    tendon: TendonModelConfig = field(default_factory=TendonModelConfig)
    curves: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Real2SimConfig:
        """从三段字典构造顶层配置（各段可省略，省略即取该类默认值）。

        参数：values —— YAML 根节点解析出的 mapping。
        失败条件：非 mapping、出现未知段名、或某段不是 mapping 时抛 ValueError；
        各段内部字段的验证由对应 dataclass 的 `__post_init__` 负责。
        """
        # Curve overlays share the same YAML in the replay CLI, but do not
        # change the physics/controller model itself.
        # 中文：curves 段在回放 CLI 里与参数共用同一个 YAML，但它只影响绘图和
        # 时间对齐，不参与物理/控制器模型。
        if not isinstance(values, dict):
            raise ValueError("real2sim configuration must be a mapping")
        for section in ("controller", "tendon", "curves"):
            if section in values and not isinstance(values[section], dict):
                raise ValueError(f"{section} must be a mapping")
        unknown = set(values) - {"controller", "tendon", "curves"}
        if unknown:
            raise ValueError(f"unknown real2sim configuration sections: {sorted(unknown)}")
        return cls(
            controller=FallbackControllerConfig.from_dict(values.get("controller") or {}),
            tendon=TendonModelConfig.from_dict(values.get("tendon") or {}),
            curves=dict(values.get("curves", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """导出三段字典；`curves` 做浅拷贝，避免调用方改动影响本配置对象。"""
        return {"controller": self.controller.to_dict(), "tendon": self.tendon.to_dict(), "curves": dict(self.curves)}


# ── YAML 快照读写 ──
def load_config(path: str | Path) -> Real2SimConfig:
    """Read a calibration YAML; omitted fields use provisional defaults.

    【中文】
    读取标定 YAML 并返回已解析、已验证的 `Real2SimConfig`。

    参数：path —— YAML 路径。返回：配置对象（向量字段已是定长 float 元组）。
    行为：空文件视为空字典（全部取默认值）；根节点必须是 mapping；未知段或
    未知键名报错。这里 `import yaml` 放在函数内，使只需要数据结构（例如导出
    帮助或默认配置的帮助路径）的调用方不必安装 PyYAML。
    副作用：只读文件。失败条件：文件不存在（OSError）、YAML 语法错误
    （yaml.YAMLError）、根节点非 mapping 或字段非法（ValueError）。
    """

    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("real2sim YAML root must be a mapping")
    return Real2SimConfig.from_dict(values)


def save_config(path: str | Path, config: Real2SimConfig) -> None:
    """Write resolved model parameters, preserving any existing curve options.

    【中文】
    把配置写回 YAML 快照（写出的是**已解析**参数，向量字段为完整 10/5 元素
    列表，不再保留标量广播或具名覆盖的写法）。

    参数：path —— 输出路径；config —— 待保存的配置。
    行为：自动创建父目录；若目标文件已存在且新配置的 `curves` 为空，则沿用
    文件里原有的 `curves` 段（避免覆盖绘图选项）；`sort_keys=False` 保持字段
    定义顺序，`allow_unicode=True` 允许中文标题原样写出。
    副作用：覆盖写目标文件；写入不是原子的，中断会留下不完整文件。
    """

    import yaml

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = config.to_dict()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            previous = yaml.safe_load(stream) or {}
        if isinstance(previous, dict) and "curves" in previous and not config.curves:
            values["curves"] = previous["curves"]
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream, allow_unicode=True, sort_keys=False)
