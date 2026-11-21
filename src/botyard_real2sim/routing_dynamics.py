"""Inertial tendon transmission downstream of the motor-side tension sensor.

    motor x -- proximal material spring -- u -- routing spring -- v -- distal spring -- y(q)
                                           |                    |
                                     friction to guides   friction to guides

Half the effective routing mass is placed at each spring endpoint. Both material
segments and the routing spring are tension-only. The routing spring has an
inextensible upper stop, with an explicit reaction force and zero restitution.
Negative u-v is internal slack, not compression of a tensile spring.

Two elastic/plastic bristle states implement presliding, reversal memory and
static holding. A bounded, relaxing contact state modulates the normal load
using tension, routing extension and geometric take-up. Bristle stiffness stays
constant: changing contact bounds cannot manufacture stored tangential energy.

The backward-Euler step is solved by enumerating the 72 piecewise-linear active
sets (three tensile springs and two stick/slip elements), with an exact upper
stop constraint. It records port work, physical friction dissipation, numerical
dissipation and the uncorrected energy-balance residual. No tension filtering
or imposed motor-to-joint time delay is used.

【中文说明】
本模块实现“电机侧张力传感器下游”的腱绳（tendon）惯性传动：把从电机收绳量 x 到
关节侧等效收绳量 y(q) 之间的那段绳路建成两个集中质量的动力学。链路为

    x（电机收绳）— 近端材料弹簧 kp — u — 走线弹簧 kr — v — 远端材料弹簧 kd — y(q)

u、v 是走线弹簧两端的集中质量点位置；等效走线质量 routing_mass_kg 取一半
（m/2）放在每个端点上，故两个端点各自满足二阶牛顿方程：

    (m/2)·ü = Tp − Ts − Fu        （近端点：近端材料拉力向前拉，走线弹簧向后拉）
    (m/2)·v̈ =  Ts − Td − Fv        （远端点：走线弹簧向前拉，远端材料拉力向后拉）

符号约定：所有弹簧“拉力为正”；Tp、Td 分别把 u 拉向 x、把 v 拉向 y；Ts 连接 u 与
v，在 u 方程里是负号、在 v 方程里是正号；Fu、Fv 是集中摩擦合力（bristle + 粘性），
方向随相对运动自动取值，不做人工符号约定。走线伸长 es = u − v 为负时表示内部
松弛（slack），不是受拉弹簧被压缩——三个弹簧都只受拉，压缩段拉力恒为 0。

【一、内部状态】（共 21 个张量，形状统一为 [num_envs, 10]，第二维是 10 条腱/执行器
通道；全部用 float64 保存）
  u, v                  两个集中点位置，单位 m
  du, dv                两个集中点速度，单位 m/s
  zu, zv                两个弹塑性 bristle 内部位移状态，单位 m；承载预滑（presliding）
                        变形、反向记忆与静摩擦保持
  contact               归一化接触状态，无量纲，∈[0, 1]
  friction_u/v          上一子步的集中摩擦合力，单位 N
  normal_u/v            上一子步的法向压紧力，单位 N
  stop_reaction         走线弹簧撞上硬停止时的反力，单位 N
  friction_dissipated   物理摩擦耗散累计，单位 J
  numerical_dissipated  后向欧拉离散带来的数值耗散累计，单位 J
  port_work             端口做功累计（电机侧与关节侧），单位 J
  energy_residual       未做任何修正的能量平衡残差累计，单位 J（只记录、不配平）
  solver_residual       最近一次 active-set 求解的最大力残差，单位 N
  plastic_slip_u/v      bristle 累计塑性滑移，单位 m
  impact_dissipated     硬停止冲量耗散累计，单位 J
  stop_impulse          上一子步的停止冲量，单位 N·s

【二、本构与参数】三个弹簧都是 tension-only（只能受拉）：拉力取“刚度 × 伸长的
正部”，即 T = k·max(0, 伸长)。材料侧用“串联后总柔度不变”的方式拆成两段：
  kp = cable_stiffness / proximal_compliance_fraction
  kd = cable_stiffness / (1 − proximal_compliance_fraction)
串联后满足 1/kp + 1/kd = 1/cable_stiffness，与不拆分时的总材料柔度完全一致；
拆分的目的只是让 u 与 v 各自承担一部分材料伸长，从而能观测绳内的张力梯度。
走线弹簧刚度是 kr（routing_spring_stiffness_n_per_m）。走线可用行程
  travel = routing_spring_max_length_m − routing_spring_free_length_m   （m）
u − v 一旦达到 travel 就顶到不可伸长的硬停止，由显式反力 reaction ≥ 0 承担超额
载荷；硬停止不可压缩，越界不会发生。

【三、摩擦与接触】每个集中点有一个弹塑性 bristle 单元：刚度 kb 恒定，内部状态 z
每步先加上相对位移增量、再钳位到 ±bound/kb，超出的部分记为塑性滑移。因此摩擦
上限 bound 只改变“保持/滑移”的切换点，不会凭空制造切向储能（bristle 刚度始终
不变）。bound = 法向力 N × Stribeck 摩擦系数
  μ(w) = μd + (μs − μd)·exp(−(w / v_stribeck)²)
其中 w 是该点的速度，单位 m/s；μs、μd 无量纲。另有与法向力成正比的粘性项
  damping = viscous · N / (tension_scale + N)      （单位 N·s/m）
饱和形式保证法向力很大时粘性不会发散。法向力由预紧、包角与接触状态合成：
  wrap = max(0, wrap_base + wrap_gain · contact)   （单位 rad）
再乘当前张力项，且每个集中点只分摊一半。接触状态 contact 是一个一阶松弛量，其
目标值由“张力饱和项 + 走线伸长占比 + 几何路径项”相加后钳位到 [0, 1]；加载与
卸载使用不同时间常数（load_tau / release_tau，单位 s），因此接触的建立比释放快。

【四、求解方式：72 个分段线性 active set】一个外层步 dt 先按 routing_max_substep_s
细分成 count 个子步（子步上限保证后向欧拉在小步长下稳定），每个子步做一次后向
欧拉离散，然后精确枚举全部 72 个分段线性 active set 求隐式步的最小化解：
  3 个拉伸弹簧各自 on/off（是否处于受拉段）  → 2³ = 8 种；
  2 个粘滑单元各自三态（粘着 / 正向滑移 / 反向滑移） → 3² = 9 种；
  8 × 9 = 72 种组合。
每个组合把 KKT 条件化成一个 2×2 线性方程组（只在 b = −kr 处耦合，见 _solve），
解出候选 u、v；随后用解析回代把走线弹簧硬停止的约束反力 reaction 加回来
（reaction ≥ 0，只在“已到行程上限且相对速度朝外”时非零）；最后取力残差最小的
组合，并按容差校验残差。硬停止是刚性不可伸长约束：额外施加等大反向冲量把朝外的
相对速度清零，恢复系数为 0，该子步的碰撞动能损失如实记入 impact_dissipated。

【五、数值假设与约定】
  - 单位：位移 m、速度 m/s、力 N、刚度 N/m、粘性 N·s/m、能量 J、冲量 N·s、
    时间 s、角度 rad、质量 kg；
  - 内部一律 float64，避免 m/dt² 在小子步下与弹簧刚度相减时的相消误差；
  - 端点 x、y 由调用方给出“步首/步末”两点，子步内按线性插值，从而一个外层步内
    电机与关节同时运动时仍保持一致；
  - 不做张力滤波，也不人为引入电机到关节的时延：所有滞后都来自物理状态；
  - 分别记录端口功、物理摩擦耗散、数值耗散与未修正能量残差；能量账不平就如实
    留在 energy_residual 里，不做事后配平，便于审计。
"""

# ── 依赖导入 ──
from __future__ import annotations

import math
from itertools import product

import torch


# ── 传动模型主体 ──
class DynamicRoutingTransmission:
    """Two-mass inertial tendon routing model with tension-only springs and bristle friction.

    【中文】
    电机侧张力传感器下游的“两质量 + 三弹簧 + 双 bristle”腱绳走线动力学。

    它算什么：给定每个外层步的电机侧收绳量 x（步首/步末）与关节侧等效收绳量
    y(q)（步首/步末），推进 u、v 两个集中点的位置与速度，并输出两侧张力、走线
    弹簧张力、硬停止反力、摩擦/法向力、各项能量与能量残差。

    内部状态：见模块 docstring 的【一、内部状态】，21 个 [num_envs, 10] 的 float64
    张量，用 state_dict()/load_state_dict() 整体存取。

    求解方式：后向欧拉 + 枚举 72 个分段线性 active set 的精确最小化（详见
    _solve），硬停止作为解析约束反力回代，而不是靠迭代或罚函数。

    数值假设：弹簧只受拉、bristle 刚度恒定、接触状态是一阶松弛量；所有量用
    float64，靠 routing_max_substep_s 限制子步长以保证隐式步稳定。
    """

    # ── 参数绑定与状态分配 ──
    def __init__(self, cfg, num_envs, device):
        """Bind routing parameters from cfg and allocate the internal state tensors.

        【中文】
        构造时把 cfg 里每条绳子（共 10 条腱通道）的 routing_* 参数从 tuple 转成
        [1, 10] 的 float64 张量并缓存，再按 num_envs 分配全部内部状态并清零。

        参数:
            cfg:       real2sim 配置对象，必须提供下方逐行注释里列出的 routing_*
                       字段（长度 10 的 tuple，或标量）；
            num_envs:  并行环境数，即批量行数；
            device:    torch 设备（字符串或 torch.device），内部再规范化一次。

        返回: 无（就地初始化）。
        抛错: cfg 缺少某个 routing_* 字段时 getattr 抛 AttributeError；参数长度与
              10 个通道不一致时 torch.tensor 抛形状相关异常。
        """
        self.cfg = cfg
        self.device = torch.device(device)
        # Double precision avoids cancellation in m/dt^2 at small substeps.
        self.dtype = torch.float64
        # 批量形状：[环境数, 10 条腱通道]。
        self.shape = (num_envs, 10)

        # 把 cfg 中的一条参数取成 [1, 10] 的行向量，便于与批量状态广播。
        def vector(name):
            return torch.tensor(getattr(cfg, name), device=device, dtype=self.dtype).unsqueeze(0)

        # ── 质量与三个弹簧的刚度 ──
        # 等效走线质量一分为二，分别放在 u、v 两个端点上，故每点质量 m/2。
        self.mass = vector("routing_mass_kg") / 2
        fraction = vector("routing_proximal_compliance_fraction")
        # 材料柔度按比例拆成两段：串联后 1/kp + 1/kd = 1/cable_stiffness，总柔度不变。
        self.kp = vector("cable_stiffness_n_per_m") / fraction
        self.kd = vector("cable_stiffness_n_per_m") / (1 - fraction)
        # 走线弹簧（u、v 之间）的刚度，单位 N/m。
        self.kr = vector("routing_spring_stiffness_n_per_m")
        # 走线可用行程 = 最大长度 − 自由长度，单位 m；u−v 达到它即顶到硬停止。
        self.travel = vector("routing_spring_max_length_m") - vector("routing_spring_free_length_m")
        # bristle 切向刚度，单位 N/m；恒定不变，不随接触状态改变。
        self.kb = vector("routing_bristle_stiffness_n_per_m")
        # 走线滑动累积状态（θ，单位 m）：绳在路径上“让出”的长度，最大 r_slip_max。
        # 全部时间常数与上限都逐路径可配，缺省 0 表示关闭该状态（行为与历史版本逐位一致）。
        self.slip_max = vector("routing_slip_max_length_m")
        self.slip_load_tau = vector("routing_slip_load_tau_s")
        self.slip_release_tau = vector("routing_slip_release_tau_s")
        # 可选小速率项：relay 之后继续让出绳长的速度（m/s），缺省 0。
        self.slip_creep = vector("routing_slip_creep_m_per_s")
        # ── 摩擦参数 ──
        self.mu_s = vector("routing_friction_mu_static")  # 静摩擦系数（无量纲）
        self.mu_d = vector("routing_friction_mu_dynamic")  # 动摩擦系数（无量纲）
        self.viscous = vector("routing_viscous_friction_n_s_per_m")  # 粘性系数，N·s/m
        self.stribeck = vector("routing_stribeck_speed_m_s")  # Stribeck 特征速度，m/s
        # ── 法向力与包角参数 ──
        self.normal_preload = vector("routing_normal_preload_n")  # 预处理法向力，N
        self.wrap_base = vector("routing_wrap_base_rad")  # 基础包角，rad
        self.wrap_gain = vector("routing_wrap_contact_gain_rad")  # 接触状态对包角的增益，rad
        # ── 接触状态（一阶松弛）参数 ──
        self.contact_base = vector("routing_contact_baseline")  # 目标值的常数底，无量纲
        self.contact_tension_gain = vector("routing_contact_tension_gain")  # 张力项增益
        self.contact_extension_gain = vector("routing_contact_extension_gain")  # 走线伸长项增益
        self.contact_path_gain = vector("routing_contact_path_gain")  # 几何路径项增益
        self.tension_scale = vector("routing_contact_tension_scale_n")  # 张力饱和尺度，N
        self.path_scale = vector("routing_contact_path_scale_m")  # 路径饱和尺度，m
        self.load_tau = vector("routing_contact_load_tau_s")  # 加载时间常数，s
        self.release_tau = vector("routing_contact_release_tau_s")  # 卸载时间常数，s
        # ── 72 个 active set 的枚举表 ──
        # 前 3 列是 3 个拉伸弹簧的 on/off（0/1），后 2 列是 2 个粘滑单元的三态
        # （−1 反向滑移 / 0 粘着 / 1 正向滑移）。8 × 9 = 72 行，最后一维为组合轴。
        self._sets = torch.tensor(
            [(*a, *b) for a in product((0, 1), repeat=3) for b in product((-1, 0, 1), repeat=2)],
            device=device,
            dtype=self.dtype,
        )
        # 全部内部状态的字段名（顺序即 state_dict 的键顺序）。
        self.state_fields = (
            "u",
            "v",
            "du",
            "dv",
            "zu",
            "zv",
            "contact",
            "friction_u",
            "friction_v",
            "normal_u",
            "normal_v",
            "stop_reaction",
            "friction_dissipated",
            "numerical_dissipated",
            "port_work",
            "energy_residual",
            "solver_residual",
            "plastic_slip_u",
            "plastic_slip_v",
            "impact_dissipated",
            "stop_impulse",
            "slip",
            "slip_work",
        )
        # 逐字段分配零张量；之后所有更新都是就地 copy_/累加，不改变形状。
        for name in self.state_fields:
            setattr(self, name, torch.zeros(self.shape, device=device, dtype=self.dtype))

    # ── 状态存取 ──
    def state_dict(self):
        """Copy internal states for identification/checkpoints; boundaries remain caller-owned.

        【中文】
        返回全部 21 个内部状态的深拷贝字典（键为 state_fields，值为 [num_envs, 10]
        的 float64 张量），供参数辨识或断点续跑使用。

        参数: 无。
        返回: dict[str, Tensor]，克隆后的状态快照。
        抛错: 无。
        注意: 只导出内部状态；电机收绳量 x、关节侧 y(q) 等“边界量”由调用方自己
              保存与恢复。
        """
        return {name: getattr(self, name).clone() for name in self.state_fields}

    def load_state_dict(self, state):
        """Restore a complete snapshot under the same configuration, without advancing time.

        【中文】
        用完整快照覆盖当前内部状态；不推进时间、不触发求解，也不改变配置。

        参数:
            state: dict[str, Tensor]，必须与 state_fields 的键集合完全一致，值会先
                   用 torch.as_tensor 转成当前设备/float64。

        返回: 无。
        抛错:
            ValueError — 键集合不一致（必须包含每一个内部状态，缺一不可、不允许多）；
            ValueError — 任一值形状不是 [num_envs, 10]，或含 NaN/Inf；
            ValueError — contact 越出 [0, 1]；
            ValueError — u − v 超出 travel + 1e-10（快照违反硬停止约束）；
            ValueError — 耗散/冲量/反力等非负量出现负值（允许 −1e-10 的舍入容差）。
        """
        if set(state) != set(self.state_fields):
            raise ValueError("routing snapshot must contain every internal state")
        converted = {key: torch.as_tensor(value, device=self.device, dtype=self.dtype) for key, value in state.items()}
        if any(value.shape != self.shape or not bool(torch.isfinite(value).all()) for value in converted.values()):
            raise ValueError("routing snapshot must have finite [num_envs,10] tensors")
        # 接触状态是有界归一化量，越界说明快照不属于本模型。
        if not bool(((converted["contact"] >= 0) & (converted["contact"] <= 1)).all()):
            raise ValueError("routing contact state must lie in [0,1]")
        # u−v 不得超过硬停止行程（1e-10 只用于吸收浮点误差）。
        if bool((converted["u"] - converted["v"] > self.travel + 1e-10).any()):
            raise ValueError("routing snapshot exceeds the spring hard stop")
        # 反力、各耗散项与冲量在物理上都非负；允许 −1e-10 的舍入容差。
        for key in (
            "normal_u",
            "normal_v",
            "stop_reaction",
            "friction_dissipated",
            "numerical_dissipated",
            "impact_dissipated",
            "stop_impulse",
            "slip",
            "slip_work",
        ):
            if bool((converted[key] < -1e-10).any()):
                raise ValueError(f"routing snapshot {key} cannot be negative")
        for key, value in converted.items():
            getattr(self, key).copy_(value)

    # ── 接触状态的松弛目标 ──
    def contact_target(self, tension, extension, path):
        """Quasi-static target of the bounded contact state, before first-order relaxation.

        【中文】
        计算接触状态的“目标值”（尚未做一阶松弛），结果钳位到 [0, 1]。

        三项物理来源相加，每一项都用饱和函数以免发散：
          1) 常数底 contact_base；
          2) 张力项：contact_tension_gain · T / (tension_scale + T)，
             T 为张力（N）、tension_scale 为饱和尺度（N），T ≫ scale 时趋于增益值；
          3) 走线伸长项：contact_extension_gain · clamp(es / travel, 0, 1)，
             即已用行程占比；travel ≤ 0（无行程）时该分子/分母整体取 0，避免除零；
          4) 几何路径项：contact_path_gain · |path| / (path_scale + |path|)，
             描述绳路几何包绕随 q 变化带来的接触变化（path 单位 m）。

        参数:
            tension:   代表张力，单位 N，形状可广播到 [num_envs, 10]；
            extension: 走线伸长（u − v），单位 m，可为负（松弛时按 0 计）；
            path:      几何路径/收绳量，单位 m，取绝对值参与饱和。

        返回:
            Tensor，形状 [num_envs, 10]，取值 ∈ [0, 1]，无量纲。
        抛错: 无（travel ≈ 0 已用 clamp_min(1e-12) 保护）。
        """
        # 已用行程占比：travel ≤ 0 时置 0（不做除法，避免 0/0）。
        tightening = torch.where(self.travel > 0, extension.clamp_min(0) / self.travel.clamp_min(1e-12), 0.0)
        return (
            self.contact_base
            + self.contact_tension_gain * tension / (self.tension_scale + tension)
            + self.contact_extension_gain * tightening.clamp(0, 1)
            + self.contact_path_gain * path.abs() / (self.path_scale + path.abs())
        ).clamp(0, 1)

    # ── 初始化 / 重置 ──
    def reset(self, x, y, tension, *, env_ids=None, velocity=None, distal_tension=None, initial_slip=None):
        """Place the two masses on a static tension split and clear every accumulator.

        【中文】
        把模型重置到一个“与给定静态张力自洽”的状态，并清零所有累计量。

        初始位置由静态力平衡反解（不是随便给的）：
            u = x0 − Tp / kp         （近端材料弹簧承担近端张力 Tp 的伸长）
            v = y0 + Td / kd         （远端材料弹簧承担远端张力 Td 的伸长）
        于是 evaluate() 在该状态下给出 Tp、Td，两个端点受力平衡。

        ``distal_tension`` 用于“摩擦锁定的夹持态”：实测只能得到近端（电机侧）张力，
        而被走线摩擦锁住的远端张力可以不同。两者之差自动体现在走线弹簧伸长
        ``es = x − y − Tp/kp − Td/kd`` 里，即锁定的伸长量。缺省时 Td = Tp，
        与历史行为（两侧平分）完全一致。
        初始速度默认取 0，也可由 velocity 指定（两个端点取同一速度，即纯平移）。

        参数:
            x, y:      初始的电机侧收绳量与关节侧等效收绳量，单位 m，
                       形状 [num_envs, 10]（或可广播到它）；
            tension:   初始静态张力（近端/电机侧），单位 N，与 x/y 同形；
            env_ids:   可选行选择（切片或索引张量）；None 表示所有环境；
            velocity:  可选初始速度，单位 m/s；None 表示静止；
            distal_tension: 可选初始远端（关节侧）张力，单位 N，与 x/y 同形；
                       None 表示取近端张力（旧“两侧平分”行为）；
            velocity 之后的关键字参数必须用关键字传入（* 之后）。

        返回: 无（就地写入）。
        抛错: 输入无法广播到 self.shape 时由赋值本身抛形状异常。
        """
        rows = slice(None) if env_ids is None else env_ids
        x, y, tension = (value.to(device=self.device, dtype=self.dtype) for value in (x, y, tension))
        # 远端张力：显式给定（摩擦锁定的夹持态）或与近端相同（旧的两侧平分行为）。
        distal = tension if distal_tension is None else distal_tension.to(device=self.device, dtype=self.dtype)
        # 静态分解：近端/远端张力分别由两端材料弹簧承担；
        # 两者之差体现为走线弹簧的（锁定）伸长 es = x − y − Tp/kp − Td/kd。
        self.u[rows] = x - tension / self.kp
        self.v[rows] = y + distal / self.kd
        # 结构状态与所有计账量清零（逐项列出，顺序无关）。
        for name in (
            "zu",
            "zv",
            "friction_u",
            "friction_v",
            "normal_u",
            "normal_v",
            "friction_dissipated",
            "numerical_dissipated",
            "port_work",
            "energy_residual",
            "solver_residual",
            "plastic_slip_u",
            "plastic_slip_v",
            "impact_dissipated",
            "stop_impulse",
            "slip",
            "slip_work",
        ):
            getattr(self, name)[rows] = 0
        self.du[rows] = 0 if velocity is None else velocity.to(self.du)
        # 两个端点取同一速度：初始状态没有内部相对运动。
        self.dv[rows] = self.du[rows]
        # 若静态张力已超过“走线弹簧满行程能提供的拉力”，
        # 差额必须由硬停止反力承担（单向约束，取正部）。
        # evaluate 的判据用当前 es 与速度，这里只需给出与 evaluate 一致的初值。
        self.stop_reaction[rows] = (torch.maximum(tension, distal) - self.kr * self.travel).clamp_min(0)
        # 接触状态直接取当前条件下的松弛目标值（不做预热）。
        # 代表张力取两侧平均，与 _step 里的做法一致。
        self.contact[rows] = self.contact_target(0.5 * (tension + distal), self.u[rows] - self.v[rows], y)
        # 摩擦锁定的夹持态：bristle 预载到“恰好维持 Tp/Td 之差”的弹性变形，
        # 使两个节点在初始时刻加速度为零（真实系统里这就是“锁住”的含义）。
        # 平衡条件：Tp − Ts − Fu = 0 与 Ts − Td − Fv = 0，取 Ts = (Tp+Td)/2 得
        # Fu = Fv = (Tp − Td)/2，对应 zu = zv = (Tp − Td)/(2·kb)。
        # 包角按完整接触目标算（与 _step 的 wrap = base + gain·contact 一致）。
        wrap = (self.wrap_base + self.wrap_gain * self.contact[rows]).clamp_min(0)
        # 每个集中区的法向力与 _step 同式：0.5·(preload + 0.5·张力·wrap)，
        # 张力用两侧平均（锁定态下没有走线弹簧的独立读数）。
        normal = 0.5 * (self.normal_preload + 0.5 * (0.5 * (tension + distal)) * wrap)
        # 静摩擦上限：μs·N（单区）；预载钳位到摩擦锥——若差值超过锥，静态本来就不可能，
        # 多余部分会在后续步滑动（与真实系统“锁不住就弛豫”一致）。
        bound = self.mu_s * normal
        held = (0.5 * (tension - distal) / self.kb).clamp(min=-bound / self.kb, max=bound / self.kb)
        self.zu[rows] = held
        self.zv[rows] = held
        self.friction_u[rows] = self.kb * held
        # 走线滑动累积初值：给定锁定态的“让出绳长”，钳到 [0, slip_max]。
        if initial_slip is not None:
            start = initial_slip.to(device=self.device, dtype=self.dtype)
            self.slip[rows] = start.clamp_min(0).clamp_max(self.slip_max[rows])
        self.friction_v[rows] = self.kb * held

    # ── 无状态求值（不推进时间） ──
    def evaluate(self, x, y):
        """Report tensions, energies and accelerations for the current state, without integrating.

        【中文】
        纯查询：根据当前内部状态与给定的电机/关节侧收绳量，算出各处拉力、能量、
        加速度、硬停止反力与长度残差，不改变任何状态、也不推进时间。

        参数:
            x: 电机侧收绳量，单位 m，形状 [num_envs, 10]（或可广播）；
            y: 关节侧等效收绳量，单位 m，同上。

        返回:
            dict，键与含义如下（力 N、能量 J、加速度 m/s²、长度 m）：
              proximal_tension / distal_tension   近端/远端材料弹簧拉力 Tp、Td（只受拉，≥ 0）
              spring_tension                      走线弹簧总拉力 Ts = kr·max(es,0) + reaction
              routing_extension                   走线伸长 max(es, 0)
              routing_gap                         内部松弛量 max(−es, 0)（负伸长取正）
              material_extension                  材料总伸长 ep + ed
              material_proximal_extension         近端材料伸长 ep
              material_distal_extension           远端材料伸长 ed
              locked                              是否顶在硬停止上（reaction > 0）
              stop_reaction                       硬停止反力，N
              elastic_energy                      弹性储能，J
              kinetic_energy                      动能，J
              acceleration_u / acceleration_v     两端点加速度，m/s²
              length_residual                     长度恒等式残差（应恒为 0，用于自检）

        抛错: 输入形状无法广播时由张量运算抛形状异常。
        """
        x, y = x.to(self.u), y.to(self.u)
        # 三段伸长量，单位 m。es = u − v 为负表示内部松弛，而非受拉弹簧压缩。
        ep, ed, es_raw = x - self.u, self.v - y, self.u - self.v
        # 走线滑动累积：让出的绳长直接从走线弹簧的可用伸长里扣掉，
        # 因此 es 是“弹簧真正吃到的伸长”，es_raw 是两端点的几何差。
        es = es_raw - self.slip
        # 材料弹簧只受拉：取伸长正部再乘刚度，故 Tp、Td 恒 ≥ 0。
        tp, td = self.kp * ep.clamp_min(0), self.kd * ed.clamp_min(0)
        spring = self.kr * es.clamp_min(0)
        # 判据：已到行程上限（es ≥ travel）且相对速度不再朝内（du − dv ≥ 0），
        # 即走线弹簧还想继续伸长 → 硬停止介入。
        at_stop = (es >= self.travel - 1e-12) & (self.du - self.dv >= -1e-12)
        # 锁定后 u − v 恒定，故 ü = v̈；把两个端点方程相加消去加速度：
        #   (m/2)ü + (m/2)v̈ = Tp − Fu + Fv − Td = 0
        # 结合 Ts = kr·es + reaction 得 reaction = ½(Tp + Td − Fu + Fv) − kr·es。
        # 负值截断为 0（停止只能推不能拉，单向约束）。
        reaction = torch.where(
            at_stop, (0.5 * (tp + td - self.friction_u + self.friction_v) - spring).clamp_min(0), 0.0
        )
        ts = spring + reaction
        # 弹性储能：三段弹簧各 ½k·(伸长正部)²，加上两个 bristle 的 ½kb·z²（切向储能）。
        elastic = 0.5 * (
            self.kp * ep.clamp_min(0).square()
            + self.kd * ed.clamp_min(0).square()
            + self.kr * es.clamp_min(0).square()
            + self.kb * (self.zu.square() + self.zv.square())
        )
        # 动能：两个集中点各 ½·(m/2)·v²，质量已在 __init__ 中一分为二。
        kinetic = 0.5 * self.mass * (self.du.square() + self.dv.square())
        return dict(
            proximal_tension=tp,
            distal_tension=td,
            spring_tension=ts,
            routing_extension=es.clamp_min(0),
            routing_gap=es.clamp_max(0),
            routing_extension_raw=es_raw.clamp_min(0),
            routing_slip=self.slip,
            material_extension=ep + ed,
            material_proximal_extension=ep,
            material_distal_extension=ed,
            locked=(reaction > 0),
            stop_reaction=reaction,
            elastic_energy=elastic,
            kinetic_energy=kinetic,
            # 式 (m/2)·ü = Tp − Ts − Fu 与 (m/2)·v̈ = Ts − Td − Fv，各除以 m/2。
            acceleration_u=(tp - ts - self.friction_u) / self.mass,
            acceleration_v=(ts - td - self.friction_v) / self.mass,
            # 恒等式：x − y = ep + es + ed，应恒为 0；非零即状态自相矛盾。
            length_residual=x - y - ep - es_raw - ed,
        )

    # ── 外层时间推进 ──
    def advance(self, x0, x1, y0, y1, dt):
        """Advance one outer timestep by substepping the backward-Euler integration.

        【中文】
        推进一个外层步 dt：先把步长细分成若干子步，再在每个子步内把电机侧与关节侧
        的边界量按线性插值成一小段，逐段调用 _step。

        参数:
            x0, x1: 本步起、止的电机侧收绳量，单位 m，形状 [num_envs, 10]；
            y0, y1: 本步起、止的关节侧等效收绳量，单位 m，同形；
            dt:     本步步长，单位 s，必须有限且 > 0。

        返回: 无（就地推进状态与累计项）。
        抛错:
            ValueError — dt 非有限或 ≤ 0；
            其余异常由 _step / _solve 传播（例如 active-set 残差超容差）。
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("routing timestep must be finite and positive")
        # 子步数：保证每个子步不超过 routing_max_substep_s（−1e-12 吸收浮点误差，
        # 避免恰好整除时多算一个子步）。
        count = max(1, math.ceil(dt / self.cfg.routing_max_substep_s - 1e-12))
        step = dt / count
        x0, x1, y0, y1 = (t.to(self.u) for t in (x0, x1, y0, y1))
        for i in range(count):
            old_fraction, fraction = i / count, (i + 1) / count
            # 边界量在子步内线性插值：x(i) → x(i+1)、y(i) → y(i+1)。
            self._step(
                x0 + (x1 - x0) * old_fraction,
                x0 + (x1 - x0) * fraction,
                y0 + (y1 - y0) * old_fraction,
                y0 + (y1 - y0) * fraction,
                step,
            )

    # ── 单个后向欧拉子步 ──
    def _step(self, x0, x1, y0, y1, dt):
        """One backward-Euler substep: contact, normal load, active-set solve, bookkeeping.

        【中文】
        执行一个后向欧拉子步，顺序是：先更新接触状态与法向力，再由它们定出摩擦上限
        和粘性系数，然后解 72 个 active set 的隐式步，最后更新 bristle、施加硬停止
        冲量并记账能量。

        参数:
            x0, x1: 子步起、止的电机侧收绳量，单位 m；
            y0, y1: 子步起、止的关节侧等效收绳量，单位 m；
            dt:     子步长，单位 s（由 advance 保证 ≤ routing_max_substep_s）。

        返回: 无（就地更新 u、v、du、dv、zu、zv 与各累计量）。
        抛错: _solve 的残差超出容差时抛 RuntimeError（CUDA 上走设备端断言）。
        """
        # 步首的力/能量快照，供后面算能量差用。
        old = self.evaluate(x0, y0)
        # ── 接触状态：向目标值做一阶松弛 ──
        # 目标值里的张力取近端与远端的平均（代表走线段两侧的张力水平）。
        target = self.contact_target(0.5 * (old["proximal_tension"] + old["distal_tension"]), self.u - self.v, y0)
        # 目标高于当前则视为加载（load_tau），低于当前则视为卸载（release_tau）。
        tau = torch.where(target >= self.contact, self.load_tau, self.release_tau)
        # 后向欧拉的一阶松弛解：contact ← contact + (1 − exp(−dt/τ))·(target − contact)。
        # 用 expm1 在 dt/τ 很小时保持精度，且系数天然落在 [0, 1]。
        self.contact += -torch.expm1(-dt / tau) * (target - self.contact)
        # 包角随接触状态增大（接触越紧，绳在导槽上的包绕越多），并保证非负。
        wrap = (self.wrap_base + self.wrap_gain * self.contact).clamp_min(0)
        # Two lumped contact zones cover the route after the motor sensor.
        # 两个集中摩擦区覆盖传感器下游的整段绳路：每个区只分摊一半预紧与张力，
        # 张力项取该区两侧相邻弹簧拉力的平均（近端区用 Tp/Ts，远端区用 Td/Ts）。
        normal_u = 0.5 * (self.normal_preload + 0.5 * (old["proximal_tension"] + old["spring_tension"]) * wrap)
        normal_v = 0.5 * (self.normal_preload + 0.5 * (old["distal_tension"] + old["spring_tension"]) * wrap)
        # ── Stribeck 摩擦上限与粘性系数 ──
        # μ(w) = μd + (μs − μd)·exp(−(w/v_stribeck)²)：低速趋近 μs、高速趋近 μd。
        # 乘上法向力得到切向力上限 bound（单位 N）。
        bound_u = normal_u * (self.mu_d + (self.mu_s - self.mu_d) * torch.exp(-(self.du / self.stribeck).square()))
        bound_v = normal_v * (self.mu_d + (self.mu_s - self.mu_d) * torch.exp(-(self.dv / self.stribeck).square()))
        # 粘性系数随法向力饱和：viscous·N/(tension_scale + N)，单位 N·s/m。
        damping_u = self.viscous * normal_u / (self.tension_scale + normal_u)
        damping_v = self.viscous * normal_v / (self.tension_scale + normal_v)
        # ── 隐式求解（72 个 active set 精确枚举） ──
        new_u, new_v, reaction, residual = self._solve(x1, y1, dt, bound_u, bound_v, damping_u, damping_v)
        delta_u, delta_v = new_u - self.u, new_v - self.v
        # 后向欧拉的速度：用本子步的位移增量定义，而不是中心差分。
        new_du, new_dv = delta_u / dt, delta_v / dt
        # ── bristle 弹塑性更新 ──
        # 先按位移增量推进内部状态 z，再钳位到 ±bound/kb：钳位以外的部分就是塑性滑移。
        new_zu = (self.zu + delta_u).clamp(min=-bound_u / self.kb, max=bound_u / self.kb)
        new_zv = (self.zv + delta_v).clamp(min=-bound_v / self.kb, max=bound_v / self.kb)
        # 塑性滑移 = 总位移增量 − 弹性（bristle）部分；反向时会自然扣除已记录的变形。
        plastic_u, plastic_v = delta_u - (new_zu - self.zu), delta_v - (new_zv - self.zv)
        # 集中摩擦合力 = bristle 弹性力 + 粘性力，单位 N（符号随运动方向自动变化）。
        friction_u, friction_v = self.kb * new_zu + damping_u * new_du, self.kb * new_zv + damping_v * new_dv
        # ── 走线滑动累积（θ）：速率无关的 relay 更新 ──
        # 电机侧放绳时路径松弛，绳沿路径“让出”长度，θ 朝 slip_max 增长；
        # 重新张紧时 θ 朝 0 收回。
        # 两个方向都是一阶松弛，时间常数取实测量级（释放 ~0.15 s、张紧 ~0.02 s），
        # 因此分支切换在亚秒级完成——与 release-hold 包测到的 0.3 s 翻转一致，
        # 不需要显式速率积分项。
        # 判据取边界量的相对位移（电机侧 vs 关节侧），对硬停止锁定也稳健：
        # 电机放绳（Δx < Δy）即路径松弛，绳让出长度；反之重新张紧。
        # 静止保持（Δx = Δy）时 θ 冻结——否则停住不动会把已让出的绳长又收回去，
        # 与实测「3° 保持 8.5 s 仍在 7.35→7.80 N」矛盾。
        travel_boundary = (x1 - x0) - (y1 - y0)
        moving = travel_boundary.abs() > 1e-12
        releasing = travel_boundary < -1e-12
        target_slip = torch.where(releasing, self.slip_max, torch.zeros_like(self.slip))
        tau_slip = torch.where(releasing, self.slip_release_tau, self.slip_load_tau)
        fraction = -torch.expm1(-dt / tau_slip)
        slip_before = self.slip
        # relay 松弛 + 可选小速率项：卸载时额外按 slip_creep·dt 增长；静止保持时
        # 只允许爬升（θ 已 > 0 才爬），不会凭空把 θ 清零也不会自发产生松弛。
        creep_active = releasing | (~moving & (self.slip > 0))
        creep = torch.where(creep_active, self.slip_creep * dt, torch.zeros_like(self.slip))
        relaxed = (self.slip + fraction * (target_slip - self.slip) + creep).clamp_min(0.0).clamp_max(self.slip_max)
        new_slip = torch.where(moving, relaxed, (self.slip + creep).clamp_min(0.0).clamp_max(self.slip_max))
        # 让出/收回绳长改变走线弹簧的储能，这部分功记进数值项，保持能量账目自洽。
        slip_energy_change = (
            0.5
            * self.kr
            * (
                (self.u - self.v - new_slip).clamp_min(0).square()
                - (self.u - self.v - slip_before).clamp_min(0).square()
            )
        )
        self.slip = new_slip
        self.slip_work += slip_energy_change
        # ── 能量记账：物理耗散 ──
        # 摩擦耗散 = 滑移量 × 平均摩擦力 + 粘性项 dt·c·v²（两者都非负）。
        physical = self.kb * (new_zu * plastic_u + new_zv * plastic_v) + dt * (
            damping_u * new_du.square() + damping_v * new_dv.square()
        )
        # ── 能量记账：数值耗散 ──
        # 1) bristle 的隐式更新带来的 ½kb·Δz²（隐式阻尼效应）；
        bristle_numerical = 0.5 * self.kb * ((new_zu - self.zu).square() + (new_zv - self.zv).square())
        # 2) 惯性项 ½·(m/2)·Δv²（后向欧拉对动能的额外衰减）；
        inertial_numerical = 0.5 * self.mass * ((new_du - self.du).square() + (new_dv - self.dv).square())
        ep0, ed0, es0 = x0 - self.u, self.v - y0, self.u - self.v
        ep1, ed1, es1 = x1 - new_u, new_v - y1, new_u - new_v
        elastic_numerical = torch.zeros_like(self.u)
        # 3) 弹簧项：隐式力做的功 k·a₊·(a − b) 与真实储能变化 ½k(a₊² − b₊²) 之差。
        #    对线性段两者相等；跨过受拉/受压折点时才有非零差值，即数值耗散。
        for stiffness, before, after in ((self.kp, ep0, ep1), (self.kd, ed0, ed1), (self.kr, es0, es1)):
            elastic_numerical += stiffness * after.clamp_min(0) * (after - before) - 0.5 * stiffness * (
                after.clamp_min(0).square() - before.clamp_min(0).square()
            )
        # 4) 硬停止反力做的功 reaction·Δes 也计入数值项（刚性约束的离散功）。
        numerical = (
            bristle_numerical + inertial_numerical + elastic_numerical + reaction * (es1 - es0) - slip_energy_change
        )
        # ── 端口功 ──
        # 电机侧：近端弹簧力 × 电机收绳位移（输入为正）；关节侧：远端弹簧力 × 关节侧
        # 位移，取负号表示能量从传动流向关节。
        work = self.kp * ep1.clamp_min(0) * (x1 - x0) - self.kd * ed1.clamp_min(0) * (y1 - y0)
        # ── 硬停止冲量（刚性约束、恢复系数 0） ──
        # A taut stop must also eliminate outward relative endpoint velocity.
        # The equal/opposite impulse conserves total momentum, with restitution 0.
        # closing > 0 表示两端点仍在“朝外”分离；用等大反向冲量把它清零。
        closing = torch.where(reaction > 0, (new_du - new_dv).clamp_min(0), 0.0)
        # 冲量大小 ½·(m/2)·closing 使相对速度恰好归零（动量守恒）。
        impulse = 0.5 * self.mass * closing
        # 完全非弹性碰撞的动能损失 = ¼·(m/2)·closing²（恢复系数 0）。
        impact_loss = 0.25 * self.mass * closing.square()
        new_du, new_dv = new_du - impulse / self.mass, new_dv + impulse / self.mass
        # 冲量改变了速度，粘性摩擦力需按新速度重算（bristle 部分不变）。
        friction_u = self.kb * new_zu + damping_u * new_du
        friction_v = self.kb * new_zv + damping_v * new_dv
        # ── 写回端点与速度（冲量已并进去） ──
        self.u, self.v, self.du, self.dv = new_u, new_v, new_du, new_dv
        self.zu, self.zv = new_zu, new_zv
        self.stop_reaction = reaction
        self.friction_u, self.friction_v = friction_u, friction_v
        self.normal_u, self.normal_v = normal_u, normal_v
        self.stop_impulse = impulse
        # ── 累计量 ──
        self.impact_dissipated += impact_loss
        self.plastic_slip_u += plastic_u
        self.plastic_slip_v += plastic_v
        self.friction_dissipated += physical
        self.numerical_dissipated += numerical
        self.port_work += work
        # ── 能量平衡残差（不做任何修正，如实记录） ──
        # 端口功 − 储能增量 − 三类耗散 应趋近 0；剩余部分全部留在 residual 里。
        new = self.evaluate(x1, y1)
        self.energy_residual += (
            work
            - (new["elastic_energy"] + new["kinetic_energy"] - old["elastic_energy"] - old["kinetic_energy"])
            - physical
            - numerical
            - impact_loss
        )
        self.solver_residual = residual

    # ── 72 个 active set 的精确求解器 ──
    def _solve(self, x, y, dt, bound_u, bound_v, damping_u, damping_v):
        """Exact active-set minimizer of the frozen-coefficient implicit step.

        【中文】
        对“冻结系数”的隐式步做精确的 active-set 最小化：枚举全部 72 个分段线性
        组合，逐个把 KKT 条件化为 2×2 线性方程组求解，取力残差最小者。

        枚举的含义（组合轴 = 最后一维，长度 72）：
            3 个拉伸弹簧 on/off：ap（近端材料）、ar（走线）、ad（远端材料）取 0/1，
                0 表示该弹簧处于松弛段（拉力恒为 0）；
            2 个粘滑单元三态：pu、pv ∈ {−1, 0, 1}，0 表示粘着（bristle 未滑移、
                摩擦取内部弹性力），±1 表示该方向滑移（摩擦取 ±bound）。
            8 × 9 = 72。

        参数:
            x, y:       子步末的电机/关节侧收绳量，单位 m，[num_envs, 10]；
            dt:         子步长，单位 s；
            bound_u/v:  两个集中点的切向摩擦上限，单位 N；
            damping_u/v: 两个集中点的粘性系数，单位 N·s/m。

        返回:
            (u, v, reaction, residual) 四元组，形状均为 [num_envs, 10]：
            u、v 位置（m）、硬停止反力 reaction（N，≥ 0）、最大力残差 residual（N）。

        抛错:
            RuntimeError — 最小残差仍超出容差 1e-7 + 1e-8·(当前张力)，说明模型无解
                           （CUDA 上由 torch._assert_async 在设备端报错）。
        """

        # Final axis enumerates active sets, independently for each tendon/env.
        # e() 把 [num_envs, 10] 扩成 [num_envs, 10, 1]，与 72 个组合广播成最后一维。
        def e(t):
            return t.unsqueeze(-1)

        ap, ar, ad, pu, pv = self._sets.T
        # 每个组合下的有效刚度：off 的弹簧刚度按 0 计入（其拉力恒为 0）。
        kp, kr, kd = e(self.kp) * ap, e(self.kr) * ar, e(self.kd) * ad
        # 粘着态（p = 0）保留 bristle 刚度，滑移态把 bristle 刚度置 0（摩擦饱和）。
        bu, bv = e(self.kb) * (pu == 0), e(self.kb) * (pv == 0)
        # 摩擦力的常数项：粘着时取“让 bristle 继续弹性跟随”的 kb·(z − u)，
        # 滑移时取 ±bound（方向由 pu/pv 给出）。
        constant_u = torch.where(pu == 0, e(self.kb * (self.zu - self.u)), e(bound_u) * pu)
        constant_v = torch.where(pv == 0, e(self.kb * (self.zv - self.v)), e(bound_v) * pv)
        # 惯性项 m/dt²（质量 m/2 已含在 self.mass 里）；后向欧拉把加速度写成位移形式。
        mass_term = e(self.mass / dt**2)
        # 2×2 方程组的对角元：惯性 + 三段弹簧 + bristle + 粘性/dt。
        a = mass_term + kp + kr + bu + e(damping_u / dt)
        d = mass_term + kd + kr + bv + e(damping_v / dt)
        # 唯一的耦合项来自走线弹簧 kr（u、v 之间），故非对角元 b = −kr。
        b = -kr
        # 右端项：惯性历史项 m/dt²·(u_prev + dt·du_prev)、弹簧驱动项 k·x、
        # 摩擦常数项，以及粘性历史项 c·u_prev/dt。
        rhs_u = mass_term * e(self.u + dt * self.du) + kp * e(x) - constant_u + e(damping_u * self.u / dt)
        rhs_v = mass_term * e(self.v + dt * self.dv) + kd * e(y) - constant_v + e(damping_v * self.v / dt)
        # 克莱姆法则解 [[a, b], [b, d]]·[u, v] = [rhs_u, rhs_v]。
        determinant = a * d - b * b
        u = (d * rhs_u - b * rhs_v) / determinant
        v = (a * rhs_v - b * rhs_u) / determinant
        # wu、wv 是“单位约束反力”对 u、v 的灵敏度：u = u₀ − reaction·wu，
        # v = v₀ − reaction·wv（reaction 在 u 方程是 +、在 v 方程是 −，故取反号）。
        wu, wv = (d + b) / determinant, (-b - a) / determinant
        # 由 u − v 达到 travel 这一条件解析反解 reaction，并截断为非负（单向约束）。
        # kr > 0 时 wu − wv > 0，故该解唯一。
        reaction = ((u - v - e(self.travel)) / (wu - wv)).clamp_min(0)
        u, v = u - reaction * wu, v - reaction * wv
        # 回代出两个集中点的摩擦力：bristle 弹性力先钳到 ±bound，再加粘性力。
        fu = (e(self.kb) * (e(self.zu) + u - e(self.u))).clamp(min=-e(bound_u), max=e(bound_u)) + e(damping_u) * (
            u - e(self.u)
        ) / dt
        fv = (e(self.kb) * (e(self.zv) + v - e(self.v))).clamp(min=-e(bound_v), max=e(bound_v)) + e(damping_v) * (
            v - e(self.v)
        ) / dt
        # 各段拉力（只受拉）；走线弹簧总拉力还要加上硬停止反力。
        tp, td = e(self.kp) * (e(x) - u).clamp_min(0), e(self.kd) * (v - e(y)).clamp_min(0)
        ts = e(self.kr) * (u - v).clamp_min(0) + reaction
        # 残差即两个端点动量方程的余量（单位 N）：
        #   (m/2)/dt²·(u − u_prev − dt·du_prev) = Tp − Ts − Fu
        #   (m/2)/dt²·(v − v_prev − dt·dv_prev) = Ts − Td − Fv
        # 移项后写成下式，取两者绝对值较大者为该 active set 的残差。
        residual_u = mass_term * (u - e(self.u) - dt * e(self.du)) - tp + ts + fu
        residual_v = mass_term * (v - e(self.v) - dt * e(self.dv)) + td - ts + fv
        residual = torch.maximum(residual_u.abs(), residual_v.abs())
        # 在 72 个组合里选残差最小的那个（每个腱/环境独立选择）。
        choice = residual.argmin(-1, keepdim=True)
        values = [t.gather(-1, choice).squeeze(-1) for t in (u, v, reaction, residual)]
        # 收敛判据：最小残差 ≤ 1e-7 N + 1e-8 倍的当前张力水平（相对容差）。
        valid = values[3] <= 1e-7 + 1e-8 * (
            self.kp * (x - values[0]).clamp_min(0) + self.kd * (values[1] - y).clamp_min(0)
        )
        if self.device.type == "cuda":
            # CUDA 上不能同步取标量，用设备端断言；失败信息会带在断言文本里。
            torch._assert_async(valid.all(), "routing active-set solve did not converge")
        elif not bool(valid.all()):
            # CPU 上直接抛错，并给出最大残差（N）便于定位。
            raise RuntimeError(f"routing active-set residual {values[3].max().item():.3g} N")
        return values
