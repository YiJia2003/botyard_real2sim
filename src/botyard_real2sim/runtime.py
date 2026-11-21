"""Engine-independent orchestration of command events, controller ticks and tendons.

The backend owns the rigid-body state and advances physics. This module never
imports Isaac Lab; its contract is also usable by offline identification tools.
At time t: accept commands, tick the controller, evaluate/log forces, then advance
the cable boundary and the rigid-body solver together to t + dt.

【中文说明】
文件职责：
    与引擎无关的“编排层”。刚性体位姿/速度状态归后端（Isaac Lab / PhysX）所有，
    物理推进也由后端负责；本模块只做三件事——接收指令事件、按控制周期驱动控制器、
    推进腱绳（tendon）的收绳边界与走线内部状态。本模块绝不 import Isaac Lab，
    因此离线参数辨识工具可以复用同一套契约，而无需启动仿真器。

主循环时序（一个物理步 t → t + dt 内的顺序，与 real2sim_replay.py 对应）：
    1. 读引擎：从后端读出关节位置 q [rad] 与关节速度 q̇ [rad/s]；
    2. 指令生效：apply_command() 用新指令替换激活掩码（注意：掩码要到“下一个
       控制 tick”才真正生效，模拟固件在 tick 时刻才下发新的电机指令）；
    3. 控制 tick：update_controller() 到期才运行一次控制器，并记录迟到量 [s]；
    4. 算力矩：evaluate() 计算张力 [N] 与关节力矩 [N·m]（纯函数，只读不改状态）；
    5. 写引擎：由后端把 10 个腱关节的 effort [N·m] 写入仿真；
    6. 采样：到采样时刻则写一行 CSV（张力、指令、走线诊断量等）；
    7. 推进传动：advance() 只推进收绳边界与走线内部状态，刚体仍由引擎推进；
    8. sim.step：后端执行一个物理步（长度 dt [s]）。

输入/输出：
    本模块不读写任何文件。
    输入：Real2SimConfig（配置数据类）、CommandFrame（一条指令帧）、
          以及每个物理步的 q [rad] 与 q̇ [rad/s] 张量。
    输出：TendonStepResult（一步的张力/力矩/走线诊断量）与内部时钟、控制器状态。
    单位约定：长度 m、角度 rad、角速度 rad/s、力 N、力矩 N·m、时间 s（秒）。
"""

from __future__ import annotations

import math

import torch

from .commands import CommandFrame
from .config import JOINT_NAMES, Real2SimConfig
from .controller import FallbackController, autopair_command
from .model import TendonPlant, TendonStepResult


class PeriodicClock:
    """Schedule ticks against absolute simulation time without period rounding.

    A fixed-step backend executes a tick at the first step boundary at or after
    its scheduled time. Lateness is bounded by one physics step and does not
    accumulate. ``pop_due`` exposes missed ticks rather than silently dropping
    them; callers must choose a sufficiently small physics timestep.

    【中文】
    用途：控制 tick 的调度器。所有 tick 时刻都按“绝对时刻”计算，即第 k 个 tick
        的计划时刻为 k * period_s，而不是把周期反复累加到浮点数上，从而避免周期
        舍入误差随仿真时长累积（否则长时间回放后控制器相位会漂移）。
    约定：固定步长的后端会在“计划时刻当天或之后的第一个步边界”执行该 tick，
        因此迟到量（lateness）不超过一个物理步长且不累积。
    诊断：``pop_due`` 会把所有已到期 tick 一次性返回（而不是静默丢弃漏掉的 tick）；
        调用方若发现返回多于一个 tick，说明物理步长过大，必须调小物理步长。
    单位：period_s 与所有时刻均为 s（秒）；period_s 同时也是控制周期。
    """

    def __init__(self, period_s: float) -> None:
        """Create a clock with a fixed controller period.

        【中文】
        用途：构造固定周期的控制 tick 时钟。
        参数：period_s —— 控制周期，单位 s，要求有限且 > 0。
        异常：period_s 为 NaN/Inf 或 ≤ 0 时抛 ValueError（避免后面出现死循环或
              除零）；构造成功后自动调用 reset() 置零状态。
        """
        # ── 参数校验：周期必须有限且为正 ──
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("clock period must be finite and positive")
        self.period_s = period_s
        self.reset()

    def reset(self) -> None:
        """Reset the tick counter to the start of a new recording.

        【中文】
        用途：把时钟复位到回放起点，便于同一次运行内多次 reset（例如二次初始化）。
        副作用：tick_count 归零（下一个 tick 计划在 0 s），last_time_s 置为 -inf
              以便第一次 pop_due 的单调性检查一定通过。
        参数/返回：无。单位：last_time_s 为 s。
        """
        self.tick_count = 0
        self.last_time_s = -math.inf

    def pop_due(self, time_s: float) -> tuple[float, ...]:
        """Report every scheduled tick at or before ``time_s``.

        【中文】
        用途：查询当前仿真时刻已到期、且尚未返回过的全部控制 tick。
        参数：time_s —— 当前仿真时间，单位 s。
        返回：已到期的 tick 绝对计划时刻组成的有序元组，单位 s（升序）；
              没有到期 tick 时返回空元组 ()。
        异常：time_s 非有限、为负、或小于上次调用时间（时间非单调）时抛 ValueError。
        说明：比较时带 1e-12 的容差，使“恰好等于计划时刻”的步边界也能命中该 tick；
              每命中一次 tick_count 自增，因此同一个 tick 不会被返回两次。
        """
        # ── 输入校验：时间必须有限、非负且单调不减 ──
        if not math.isfinite(time_s) or time_s < 0 or time_s < self.last_time_s:
            raise ValueError("clock time must be finite, nonnegative and monotonic")
        self.last_time_s = time_s
        due: list[float] = []
        # ── 收集所有已到期的绝对时刻：k * period_s ≤ time_s（容差 1e-12 s）──
        while self.tick_count * self.period_s <= time_s + 1e-12:
            due.append(self.tick_count * self.period_s)
            self.tick_count += 1
        return tuple(due)


class Real2SimRuntime:
    """One recorded experiment, with a batched mechanical/controller state.

    The command adapter applies the same named command to each environment;
    missing partners capture each environment's own measured joint angle.
    The caller decides whether a sampled CSV row is a new command event. The
    third command field is retained for telemetry but is not a Fallback limit.

    【中文】
    用途：一次回放实验的运行时容器，把三块状态组合在一起——
        1) 控制器 FallbackController（固件侧指令整形：限速、限张力、同指补齐）；
        2) 被控对象 TendonPlant（腱绳/走线力学：张力、摩擦、力臂、关节力矩）；
        3) 时钟 PeriodicClock（决定控制器多久跑一次）。
        三者都按 num_envs 批量张量化，第 0 维是环境（并行实例）维度。

    指令语义：
        指令适配层把同一条“按名字给出的指令”施加到每个环境；某关节若在本条指令
        中缺失配对的“同指伙伴”，则用该环境自身实测关节角补齐（同指耦合件必须成对
        驱动，否则力臂/长度平衡会算错）。
        第三指令字段（张力指令 [N]）只作为遥测保留，Fallback 控制器不把它当作
        硬限值执行。采样 CSV 的某一行是否算“新的指令事件”由调用方决定。

    单位：角度 rad、角速度 rad/s、张力/力 N、力矩 N·m、时间 s。
    """

    def __init__(
        self,
        cfg: Real2SimConfig,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the controller, plant and clock from one configuration.

        【中文】
        用途：按配置构造控制器、被控对象与时钟，并分配全零的指令/状态张量。
        参数：
            cfg     —— Real2SimConfig，包含 controller / tendon 两个子配置；
            num_envs —— 并行环境数（批量第 0 维），默认 1；
            device  —— torch 设备字符串或对象，默认 "cpu"；
            dtype   —— 浮点精度，默认 torch.float32。
        返回：无（构造函数）。
        异常：由 FallbackController / TendonPlant 的构造校验抛出（如配置非法）。
        初始状态：目标角 [rad]、限速 [rad/s]、张力指令 [N]、控制器速度 [rad/s]
                  全为 0，激活掩码全为 False，指令计数 0，迟到量 0 s。
        """
        # ── 依赖组装：控制器 + 被控对象 + 控制周期时钟 ──
        self.cfg = cfg
        self.controller = FallbackController(cfg.controller, num_envs, device, dtype)
        self.plant = TendonPlant(cfg.tendon, num_envs, device, dtype)
        self.clock = PeriodicClock(cfg.controller.control_period_s)
        # ── 指令相关批量缓冲：形状均为 (num_envs, 关节数) ──
        # target_rad：本指令掩码下的目标关节角 [rad]
        # speed_limit_rad_s：本指令掩码下的角速度限幅 [rad/s]
        # tension_command_n：第三指令字段（电机侧张力指令）[N]，仅遥测
        # active：该关节是否被当前指令掩码激活（同指补齐后）
        self.target_rad = torch.zeros((num_envs, len(JOINT_NAMES)), device=device, dtype=dtype)
        self.speed_limit_rad_s = torch.zeros_like(self.target_rad)
        self.tension_command_n = torch.zeros_like(self.target_rad)
        self.active = torch.zeros_like(self.target_rad, dtype=torch.bool)
        # ── 控制器输出与运行统计 ──
        # controller_speed_rad_s：控制器本 tick 给出的收绳速度指令 [rad/s]
        self.controller_speed_rad_s = torch.zeros_like(self.target_rad)
        self.command_count = 0  # 已应用的指令事件个数
        self.controller_lateness_s = 0.0  # 最近一次控制 tick 的迟到量 [s]

    def reset(
        self,
        q_rad: torch.Tensor,
        initial_tension_n: torch.Tensor | None = None,
        *,
        cold_start: bool = False,
        motor_position_rad: torch.Tensor | None = None,
        motor_velocity_rad_s: torch.Tensor | None = None,
        joint_velocity_rad_s: torch.Tensor | None = None,
        routing_initial_distal_tension_n: torch.Tensor | None = None,
        routing_initial_slip_m: torch.Tensor | None = None,
    ) -> None:
        """Reset controller, plant, clock and every command buffer together.

        【中文】
        用途：把运行时恢复到回放起点——控制器、被控对象、时钟、指令缓冲全部复位。
        参数：
            q_rad               —— 初始关节角 [rad]，形状 (num_envs, 关节数)；
            initial_tension_n   —— 初始张力 [N]，同形状；None 表示用配置默认值；
            cold_start          —— True 表示冷启动（不继承真实固件的启动状态）；
            motor_position_rad  —— 初始电机（绕线筒）角 [rad]，None 表示由 q 反解；
            motor_velocity_rad_s —— 初始电机角速度 [rad/s]，None 表示静止；
            joint_velocity_rad_s —— 初始关节角速度 [rad/s]，用于走线内部状态初始化。
        返回：无。
        异常：由 controller.reset / plant.reset 内部校验抛出（例如张力为负、
              张力与初始松弛量冲突）。
        副作用：指令计数与迟到量清零，全部指令缓冲归零。
        """
        # ── 依次复位三个子系统 ──
        self.controller.reset(cold_start=cold_start)
        self.plant.reset(
            q_rad,
            initial_tension_n=initial_tension_n,
            motor_position_rad=motor_position_rad,
            motor_velocity_rad_s=motor_velocity_rad_s,
            joint_velocity_rad_s=joint_velocity_rad_s,
            routing_initial_distal_tension_n=routing_initial_distal_tension_n,
            routing_initial_slip_m=routing_initial_slip_m,
        )
        self.clock.reset()
        # ── 清空指令缓冲与运行统计 ──
        for value in (self.target_rad, self.speed_limit_rad_s, self.tension_command_n, self.controller_speed_rad_s):
            value.zero_()
        self.active.zero_()
        self.command_count = 0
        self.controller_lateness_s = 0.0

    def apply_command(self, frame: CommandFrame, q_rad: torch.Tensor) -> None:
        """Replace the active mask, just as a received H7 A0 command does.

        【中文】
        用途：接收一条指令帧（对应实机 H7 A0 指令），用它整体替换当前激活掩码。
        参数：
            frame —— CommandFrame，含时间戳 [s] 与 {关节名: JointCommand}；
                     JointCommand 内含目标角 [rad]、限速 [rad/s]、张力指令 [N]；
            q_rad —— 当前实测关节角 [rad]，用于给缺失的同指伙伴补值。
        返回：无（只更新内部缓冲，不立即改变控制器输出）。
        异常：
            ValueError —— 指令里出现未知关节名；
            ValueError —— 指令为空（空掩码不能解释为“硬件停机”，必须跳过该采样行）。
        时序要点：新掩码在“下一个控制 tick”才生效，而不是在收到指令的瞬间生效——
                  这样才与固件“在 tick 时刻统一下发新电机指令”的行为一致。
        其他：若配置开启 quantize_angle_feedback，会先把实测角按反馈分辨率
              [度] 量化再用于补齐，模拟编码器量化。
        """
        # ── 校验：关节名合法、掩码非空 ──
        unknown = set(frame.commands) - set(JOINT_NAMES)
        if unknown:
            raise ValueError(f"unsupported tendon joints: {sorted(unknown)}")
        if not frame.commands:
            raise ValueError("an empty A0 mask is not a verified hardware stop; omit missing sampled command rows")
        # ── 把本条指令摊平进批量缓冲（同一指令施加到所有环境）──
        target = torch.zeros_like(self.target_rad)
        speed = torch.zeros_like(target)
        tension = torch.zeros_like(target)
        active = torch.zeros_like(self.active)
        for name, command in frame.commands.items():
            index = JOINT_NAMES.index(name)
            target[:, index] = command.angle_rad
            speed[:, index] = command.speed_rad_s
            tension[:, index] = command.tension_command_n
            active[:, index] = True
        # ── 反馈角量化（可选）：按反馈分辨率 [度 → rad] 取整，模拟编码器分辨率 ──
        measured = q_rad
        if self.cfg.controller.quantize_angle_feedback:
            quantum = math.radians(self.cfg.controller.angle_feedback_resolution_deg)
            measured = torch.round(measured / quantum) * quantum
        # ── 同指补齐：缺失的配对关节用该环境自身实测角补上，避免单边驱动 ──
        paired = autopair_command(measured, target, speed, tension, active)
        for destination, source in zip(
            (self.target_rad, self.speed_limit_rad_s, self.tension_command_n, self.active), paired, strict=True
        ):
            destination.copy_(source)
        # The new mask takes effect at the next control tick, when the firmware
        # sends fresh motor commands, rather than on command receipt itself.
        # 【中文】新掩码要等到下一个控制 tick（固件下发新电机指令的时刻）才生效，
        #         不是收到指令的瞬间就生效；这里只累加指令事件计数。
        self.command_count += 1

    def update_controller(self, time_s: float, q_rad: torch.Tensor) -> bool:
        """Run one controller tick if the clock says it is due.

        【中文】
        用途：按控制周期驱动一次 Fallback 控制器（到期才跑，未到期直接跳过）。
        参数：
            time_s —— 当前仿真时间 [s]；
            q_rad  —— 当前关节角 [rad]，形状 (num_envs, 关节数)，作为控制器反馈。
        返回：bool —— True 表示本次确实执行了控制 tick；False 表示未到期、已跳过。
        异常：
            ValueError —— 一个物理步内到期了多个控制 tick（说明物理步长过大，
                          漏 tick 会破坏时序，因此报错而不是静默丢弃）；
            ValueError —— 由 PeriodicClock.pop_due 的时间单调性校验抛出。
        副作用：更新 controller_speed_rad_s（收绳速度指令 [rad/s]）与
                controller_lateness_s（本次 tick 的迟到量 [s]，被钳到 ≥ 0）。
        """
        # ── 查询到期的 tick；多于一个即视为物理步长过大 ──
        due = self.clock.pop_due(time_s)
        if len(due) > 1:
            raise ValueError("physics timestep skipped controller ticks; reduce --physics-dt")
        if not due:
            return False
        # ── 记录迟到量：实际执行时刻 − 计划时刻 [s]，迟到量不超过一个物理步 ──
        self.controller_lateness_s = max(0.0, time_s - due[0])
        # ── 跑一次控制器：输出每个关节的收绳速度指令 [rad/s] ──
        self.controller_speed_rad_s = self.controller.step(
            q_rad,
            self.target_rad,
            self.speed_limit_rad_s,
            cmd_tension_command_n=self.tension_command_n,
            active_mask=self.active,
        )
        return True

    def evaluate(self, q_rad: torch.Tensor, qd_rad_s: torch.Tensor) -> TendonStepResult:
        """Evaluate tendon forces and joint efforts for logging without mutating state.

        【中文】
        用途：在给定关节状态上“只算不改”地评估一次腱绳力学，得到张力 [N]、
              关节力矩 [N·m] 以及走线诊断量，供写回引擎与写 CSV 使用。
        参数：
            q_rad    —— 关节角 [rad]，形状 (num_envs, 关节数)；
            qd_rad_s —— 关节角速度 [rad/s]，同形状。
        返回：TendonStepResult，含 motor_* 电机侧量、proximal/distal 张力 [N]、
              tendon/spring/friction 力矩分量 [N·m]、joint_effort_nm（进力矩）、
              力臂 [m]、走线内部状态等。
        异常：形状/设备不匹配时由 Tensor 运算抛出。
        注意：本方法不修改控制器、被控对象或指令缓冲的任何状态，可多次调用。
        """
        return self.plant.evaluate(q_rad, qd_rad_s)

    def advance(self, dt_s: float, q_rad: torch.Tensor | None = None, qd_rad_s: torch.Tensor | None = None) -> None:
        """Advance only the tendon input boundary; backend advances joints.

        【中文】
        用途：推进一个物理步内的“传动侧”状态，即收绳边界与走线（routing）内部
              状态；刚体（关节角/角速度）仍然由后端引擎 sim.step 推进。
        参数：
            dt_s     —— 物理步长 [s]；
            q_rad    —— 当前关节角 [rad]，用于走线边界条件；None 表示由内部状态推断；
            qd_rad_s —— 当前关节角速度 [rad/s]，同上。
        返回：无。
        异常：dt_s 非正或非有限时由 TendonPlant.advance 抛出。
        时序：必须在“算力矩→写引擎→采样”之后、sim.step 之前调用，使收绳边界与
              下一步的关节状态保持同一时间基准。
        """
        self.plant.advance(self.controller_speed_rad_s, dt_s, q_rad=q_rad, qd_rad_s=qd_rad_s)
