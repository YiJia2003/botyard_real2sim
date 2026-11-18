"""Replay unloaded tendon commands using the project's existing Isaac Lab / PhysX.

CLI parsing, parameter export and input validation do not start Isaac Sim. The
runtime/controller/plant modules are independent of this rigid-body adapter.

【中文说明】
文件职责：
    真实数据回放（real2sim）入口脚本。把录制的“空载腱绳指令 + 反馈”CSV 在
    Isaac Lab / PhysX 里重放一遍，输出仿真轨迹与参数/元数据快照。本文件是唯一
    接触刚性体引擎的适配层；runtime/controller/plant 三个模块本身与引擎无关。

入口流程（main）：
    1. 解析参数（build_parser，纯 argparse，不启动仿真）；
    2. prepare_experiment —— 读配置 / CSV / 初态，并做全部输入校验（不覆盖输入文件、
       dt 之间的大小关系、四元数为单位四元数、Fallback 证据合法）；
    3. --write-default-config 时导出校验后的参数并直接退出（不启动 Isaac Sim）；
    4. 否则 import isaaclab 并进入 launch_simulation，在其上下文里：
       创建 SimulationContext（PhysX 后端，use_newton_actuators=False，
       走标准“显式执行器（direct effort）”路径）→ InteractiveScene →
       写初态位置/速度 → 锁定手腕限位 → 构造 Real2SimRuntime → 进入主循环。

主循环时序（每个物理步 t → t + dt，顺序即执行顺序）：
    读引擎（q [rad]、q̇ [rad/s]）→ 指令事件生效（apply_command，下一 tick 才下发）
    → 控制 tick（update_controller，到期才跑）→ 算力矩（evaluate 得到关节力矩 [N·m]）
    → 写引擎（10 个关节 effort [N·m]）→ write_data_to_sim → 读回 applied_effort
    判断是否饱和 → 采样写 CSV（到采样周期时）→ 推进传动（advance 收绳边界/走线）
    → sim.step（推进刚体）→ scene.update（刷新缓冲）。

输入文件：
    --commands 录制的指令/反馈 CSV；--initial-state 可选的初始反馈 CSV；
    --config 参数 YAML（默认 configs/real2sim/s1.yaml，可用 --set 覆盖）。

输出文件（三个，均由 --out 派生）：
    *.csv            每个采样点的完整状态/指令/张力/走线诊断量；
    *.params.yaml    本次实际生效的全部参数快照（save_config）；
    *.metadata.json  运行元数据：版本、修订号、初态、假设列表、状态统计。
    metadata 固定标记 calibrated: false、mode: forced_fallback、
    third_command_field_enforced: false，并在结束时写入 samples / physics_steps /
    saturated_force_evaluations / mimic_max_error_rad 等状态统计。

CSV 列结构（见 _header/_row）：
    每个腱关节一组列：requested_effort_Nm（请求力矩 [N·m]）、
    applied_effort_Nm（引擎实际施加力矩 [N·m]）、effort_saturated（是否被限幅）；
    张力分两处：tension_N 为电机侧（经两段走线摩擦与走线弹簧之前）张力 [N]，
    distal_tension_N 为关节侧张力 [N]（进关节力矩用的就是它）；
    另有 29 个走线（routing）诊断量、4 个手指 J1 角度 [rad] 与
    mimic_max_error_rad（J1/J2 耦合最大误差 [rad]）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from botyard_real2sim.commands import (
    CommandFrame,
    JointCommand,
    load_command_frames,
    load_initial_state,
    validate_fallback_capture,
)
from botyard_real2sim.config import JOINT_NAMES, ROUTING_DIAGNOSTIC_FIELDS, load_config, save_config
from botyard_real2sim.parameters import override_parameters
from botyard_real2sim.paths import checkout_root, default_config_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (no simulator is started here).

    【中文】
    用途：构造 argparse 解析器。此函数是纯 Python，不会启动 Isaac Sim，
          因此 --help 与参数导出路径都很快。
    参数：无。
    返回：argparse.ArgumentParser，其 description 直接复用模块 docstring。
    异常：无。
    说明：下面按“输入来源 / 指令整形 / 仿真步长 / 输出 / 场景位姿”几个小节
          组织参数，单位在注释中标注。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # ── 输入来源（二选一，互斥）：录制回放 或 生成单关节阶跃 ──
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--commands", type=Path, help="Recorded commands/feedback CSV")
    profile.add_argument("--joint", choices=JOINT_NAMES, help="Generate a single-joint step")
    # ── 初始状态：初始反馈 CSV 或 --initial-joint-deg 角度覆盖（单位 deg→rad）──
    parser.add_argument("--initial-state", type=Path, help="Initial feedback CSV; otherwise use feedback in --commands")
    parser.add_argument("--initial-joint-deg", action="append", default=[], metavar="JOINT=DEG")
    parser.add_argument("--command-frames", choices=("auto", "changes", "each-row"), default="auto")
    parser.add_argument("--strict-commands", action="store_true", help="Require all three recorded command fields")
    # ── 生成阶跃指令时用的整形参数：目标角 [deg]、阶跃时刻 [s]、限速 [deg/s] ──
    parser.add_argument("--target-deg", type=float, default=30.0)
    parser.add_argument("--step-at", type=float, default=0.3)
    parser.add_argument("--max-speed-deg-s", type=float, default=90.0)
    # ── 第三指令字段（张力指令 [N]）：仅遥测，Fallback 控制器不作硬限值 ──
    parser.add_argument(
        "--tension-command-n",
        "--tension-limit-n",
        dest="tension_command_n",
        type=float,
        default=20.0,
        help="Third input field [N]; Fallback does not enforce it",
    )
    parser.add_argument("--tension-raw-to-n", type=float, default=0.01)
    # ── 时长与配置：--duration 缺省时由最后一帧 + 0.5 s 推出 ──
    parser.add_argument("--duration", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD[.JOINT]=VALUE")
    # ── 输出：--out 生成 CSV，并派生 .params.yaml 与 .metadata.json 两个伴随文件 ──
    parser.add_argument("--write-default-config", type=Path, help="Export validated parameters and exit without Isaac")
    parser.add_argument("--out", type=Path, help="Output CSV, with resolved YAML and metadata JSON sidecars")
    # ── 仿真步长：physics_dt 为物理步长 [s]，sample_dt 为 CSV 采样周期 [s] ──
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physics-dt", type=float, default=0.0001)
    parser.add_argument("--sample-dt", type=float, default=0.003)
    # ── 数值保护：effort 上限 [N·m]（数值钳位，不是腱绳指令限值）与数值惯量缩放 ──
    parser.add_argument(
        "--effort-limit-nm", type=float, default=2.0, help="Numerical effort cap, not a tendon command cap"
    )
    parser.add_argument(
        "--armature-scale", type=float, default=1.0, help="Scale numerical inertia on the ten tendon DOFs"
    )
    # ── 场景位姿覆盖：根位置 [m]（3 个）与四元数 xyzw（4 个，必须是单位四元数）──
    parser.add_argument("--root-pos", type=float, nargs=3)
    parser.add_argument("--root-xyzw", type=float, nargs=4)
    # ── 启动方式与可视化：冷启动、是否打开 GUI ──
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--gui", action="store_true")
    return parser


def prepare_experiment(args):
    """Resolve all user inputs before allocating simulator resources.

    【中文】
    用途：在分配任何仿真资源之前，把全部用户输入解析、校验并整理成运行时数据。
          本函数不启动 Isaac Sim，因此配置错误会在耗时启动之前就暴露。
    参数：args —— build_parser() 解析出的 Namespace（本函数会写回
          args.fallback_evidence 字段）。
    返回：五元组 (cfg, frames, initial, overrides, duration)：
          cfg      —— 已应用 --set 覆盖的 Real2SimConfig；
          frames   —— list[CommandFrame]，指令帧序列（--write-default-config 时为空）；
          initial  —— InitialState 或 None（无可用反馈时为 None）；
          overrides —— {关节名: 角度 [rad]}，来自 --initial-joint-deg；
          duration —— 回放时长 [s]。
    异常（均为 ValueError，由 main 转成 parser.error）：
          输出路径与输入路径冲突；dt 关系不合法；四元数不是单位四元数；
          录制数据不是 Fallback 数据；初始张力为负或与初始松弛冲突；
          时长覆盖不到最后一条指令事件等。
    单位：duration 与 frame.time_s 为 s；角度内部统一为 rad（命令行输入为 deg）。
    """
    # ── 配置加载：默认配置 + --set 逐项覆盖；冷启动时把启动配对评估次数抬到 50 ──
    config_path = args.config or default_config_path()
    cfg = override_parameters(load_config(config_path), args.set)
    if args.cold_start and cfg.controller.startup_pair_evaluations == 0:
        cfg = replace(cfg, controller=replace(cfg.controller, startup_pair_evaluations=50))
    # ── 仅导出参数模式：写完 YAML 立刻返回，不启动 Isaac Sim ──
    if args.write_default_config:
        save_config(args.write_default_config, cfg)
        return cfg, [], None, {}, 0.0
    if args.out is None or (args.commands is None and args.joint is None):
        raise ValueError("provide --out and either --commands or --joint")
    # ── 输出路径保护：三个输出文件都不允许覆盖任何输入文件（含默认配置）──
    inputs = {path.resolve() for path in (args.commands, args.initial_state, args.config) if path is not None}
    inputs.add(config_path.resolve())
    outputs = {
        args.out.resolve(),
        args.out.with_suffix(".params.yaml").resolve(),
        args.out.with_suffix(".metadata.json").resolve(),
    }
    if inputs & outputs:
        raise ValueError("output paths must not overwrite command, initial-state or configuration inputs")
    # ── 数值校验一：物理步长/采样周期/力矩上限/张力换算系数必须有限且为正 ──
    for field in ("physics_dt", "sample_dt", "effort_limit_nm", "tension_raw_to_n"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    # ── dt 关系：物理步长 ≤ 控制周期（否则会漏控制 tick）；采样周期 ≥ 物理步长 ──
    if args.physics_dt > cfg.controller.control_period_s:
        raise ValueError("physics_dt must not exceed the controller period")
    if args.sample_dt < args.physics_dt:
        raise ValueError("sample_dt must be at least physics_dt")
    # ── 数值校验二：这些量必须有限且非负 ──
    for field in ("armature_scale", "step_at", "max_speed_deg_s", "tension_command_n"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    if not math.isfinite(args.target_deg):
        raise ValueError("target_deg must be finite")
    # ── 场景位姿校验：根位置 [m] 各分量有限；四元数 xyzw 需为单位四元数（容差 1e-5）──
    for field in ("root_pos", "root_xyzw"):
        value = getattr(args, field)
        if value is not None and not all(math.isfinite(x) for x in value):
            raise ValueError(f"{field} must contain finite values")
    if args.root_xyzw is not None and not math.isclose(sum(x * x for x in args.root_xyzw), 1.0, abs_tol=1e-5):
        raise ValueError("root_xyzw must be a unit quaternion in xyzw order")
    # ── 初始关节角覆盖：命令行给 deg，这里统一换算成 rad ──
    overrides = {}
    for assignment in args.initial_joint_deg:
        name, sep, value = assignment.partition("=")
        if not sep or name not in JOINT_NAMES:
            raise ValueError(f"invalid initial joint override: {assignment}")
        angle = float(value)
        if not math.isfinite(angle):
            raise ValueError("initial joint angle must be finite")
        overrides[name] = math.radians(angle)
    # ── 指令帧加载：先验证 Fallback 证据，再按 --command-frames 的事件语义切帧 ──
    frames = []
    initial = None
    # 生成阶跃（无录制数据）时，元数据里记为“生成的强制 Fallback”证据
    args.fallback_evidence = {"status": "generated_forced_fallback"}
    if args.commands:
        args.fallback_evidence = validate_fallback_capture(args.commands)
        mode = {"auto": "auto", "changes": "sampled", "each-row": "events"}[args.command_frames]
        frames = load_command_frames(
            args.commands, strict=args.strict_commands, event_mode=mode, tension_raw_to_n=args.tension_raw_to_n
        )
        # ── 录制内容校验：必须有指令、空事件不能被当成硬件停机、必须全是 Fallback 数据 ──
        if not any(frame.commands for frame in frames):
            raise ValueError("recording contains no selected tendon commands")
        if any(frame.is_event and not frame.commands for frame in frames):
            raise ValueError("empty command event cannot be interpreted as a hardware stop")
        if any(frame.is_fallback is False for frame in frames):
            raise ValueError("recording contains non-Fallback data; select a verified Fallback interval first")
        if any(frame.time_s < 0 for frame in frames):
            raise ValueError("command events predate recording start; include pre-command feedback in the capture")
    # ── 初始状态解析：优先 --initial-state，否则用 --commands 内首行的反馈 ──
    state_path = args.initial_state or args.commands
    if state_path:
        with state_path.open(newline="", encoding="utf-8-sig") as stream:
            fields = csv.DictReader(stream).fieldnames or []
        has_feedback = any(f"{name}/position_rad" in fields for name in JOINT_NAMES)
        if has_feedback or args.initial_state:
            initial = load_initial_state(state_path, tension_raw_to_n=args.tension_raw_to_n)
            # 初始张力 [N] 不能为负：负张力表示传感器零偏未校正
            if any(tension < 0 for tension in initial.tensions_n.values()):
                raise ValueError("negative initial tension; correct sensor bias before replay")
            # 若开启了缆绳阻尼 [N·s/m]，必须从静止状态起步，否则阻尼项会引入虚假冲击
            if any(cfg.tendon.cable_damping_n_s_per_m):
                speeds = list(initial.velocities_rad_s.values()) + list(initial.motor_velocities_rad_s.values())
                if any(speed != 0 for speed in speeds):
                    raise ValueError("nonzero cable damping requires rest initialization; select a stationary start")
            # 正初始张力与初始松弛量 [m] 互斥（松弛的缆绳不可能带张力）
            for name, tension in initial.tensions_n.items():
                if tension > 0 and cfg.tendon.cable_slack_m[JOINT_NAMES.index(name)] > 0:
                    raise ValueError(f"positive initial tension conflicts with initial slack for {name}")
            # 用 --commands 内的首行做初态时，要求该行就在录制起点（时间 ≈ 0 s）
            if args.initial_state is None and initial.time_s > 1e-9:
                raise ValueError("first feedback is after capture start; trim the capture or supply --initial-state")
            # 部分反馈缺失：不报错，用场景/配置默认值补齐，但要在 stderr 与元数据里留痕
            if initial.missing_positions or initial.missing_tensions:
                print(
                    "[WARN] partial initial feedback: missing values use scene/config defaults; see metadata",
                    file=sys.stderr,
                )
        else:
            # 只有指令、没有反馈列：使用配置中的初始位姿与张力
            print("[WARN] command-only replay uses configured initial pose/tension", file=sys.stderr)
    # ── 时长确定：显式 --duration，否则取“最后一帧 + 0.5 s”（无帧则 2.0 s）──
    duration = args.duration if args.duration is not None else (frames[-1].time_s + 0.5 if frames else 2.0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    # ── 时长必须覆盖到最后一个指令事件，否则那条指令永远不会被执行 ──
    last_command = frames[-1].time_s if frames else args.step_at
    if last_command > duration:
        raise ValueError("duration ends before the final command event")
    # ── 返回：已校验的配置 + 指令帧 + 初态 + 角度覆盖 + 时长 [s] ──
    return cfg, frames, initial, overrides, duration


def _revision() -> str | None:
    """Return the repository HEAD revision, or None when git is unavailable.

    【中文】
    用途：读取当前仓库的 git HEAD 提交号，写入 metadata 便于复现结果。
    参数：无。
    返回：提交号字符串；不是 git 仓库或没有 git 命令时返回 None（不报错）。
    异常：不向外抛异常（OSError 与 CalledProcessError 都被吞掉）。
    """
    try:
        root = checkout_root()
        if root is None:
            return None
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions():
    """Collect versions of the simulator/runtime packages for the metadata.

    【中文】
    用途：收集 isaaclab / isaacsim / torch / ovphysx 四个包的版本号，写入 metadata。
    参数：无。
    返回：dict，键为包名，值为版本字符串；未安装的包值为 None。
    异常：不抛出（PackageNotFoundError 被逐项捕获）。
    """
    result = {}
    for name in ("botyard_real2sim", "isaaclab", "isaacsim", "torch", "ovphysx"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _header():
    """Build the CSV header row.

    【中文】
    用途：生成 CSV 表头。前 6 列是整行级别的量（时间戳 [ns]、时间 [s]、
          指令序号、DOF 掩码、是否 Fallback、控制器迟到量 [s]）；
          其后每个腱关节一组列，列名形如 "关节名/字段名"。
    参数：无。
    返回：list[str]，表头各列名称。
    异常：无。
    单位：见下方每个字段名后缀——_rad、_rad_s、_rad_s2、_N、_Nm、_m、_J。
    """
    # ── 每关节字段顺序：与 _row() 里 arrays 的元组顺序严格一一对应，改动必须同步 ──
    fields = (
        "position_rad",
        "velocity_rad_s",
        "acceleration_rad_s2",
        "motor_position_rad",
        "motor_velocity_rad_s",
        "motor_acceleration_rad_s2",
        "tension_N",
        "distal_tension_N",
        "cmd_rad",
        "cmd_speed_rad_s",
        "cmd_tension_N",
        "active",
        "is_fallback",
        "controller_speed_rad_s",
        "cable_retraction_m",
        "cable_extension_m",
        "tendon_effort_Nm",
        "spring_effort_Nm",
        "friction_effort_Nm",
        "requested_effort_Nm",
        "applied_effort_Nm",
        "effort_saturated",
        "routing_extension_m",
        "routing_length_m",
        "routing_locked",
        "material_extension_m",
        "elastic_energy_J",
        "return_spring_length_m",
        "return_spring_force_N",
        "own_moment_arm_m",
        "paired_moment_arm_m",
        "length_balance_error_m",
    )
    # ── 行级前缀列 + 每关节字段 + 走线诊断量 + 4 个手指 J1 角度 + mimic 误差 ──
    return (
        ["timestamp_ns", "time_s", "command_seq", "dof_mask", "is_fallback", "controller_lateness_s"]
        + [f"{joint}/{field}" for joint in JOINT_NAMES for field in (*fields, *ROUTING_DIAGNOSTIC_FIELDS)]
        + [f"{finger}J1/position_rad" for finger in ("FF", "MF", "RF", "LF")]
        + ["mimic_max_error_rad"]
    )


def _row(time_s, q, dq, runtime, result, applied, slave_q, mimic_error, ddq=None):
    """Assemble one CSV row from the current step's state.

    【中文】
    用途：把一个物理步的状态拼成一行 CSV（列顺序必须与 _header() 完全一致）。
    参数：
        time_s       —— 当前仿真时间 [s]（时间戳列会换算成 ns 整数）；
        q            —— 关节角 [rad]，形状 (num_envs, 关节数)；
        dq           —— 关节角速度 [rad/s]，同形状；
        runtime      —— Real2SimRuntime，用于取指令侧量（目标角/限速/张力指令/掩码/
                        控制器速度/指令计数/迟到量）；
        result       —— TendonStepResult，用于取电机侧量、张力 [N]、力矩分量 [N·m]、
                        力臂 [m] 与走线诊断量；
        applied      —— 引擎实际施加的关节力矩 [N·m]，同形状；
        slave_q      —— 4 个从动指 J1 的角度 [rad]，形状 (num_envs, 4)；
        mimic_error  —— J1/J2 耦合最大误差 [rad]，标量；
        ddq          —— 关节角加速度 [rad/s^2]，None 时该列写 0（首个采样点无定义）。
    返回：list，可直接交给 csv.writer 的一行（数值多为 float/标量）。
    异常：形状不匹配时由 Tensor 运算抛出。
    说明：requested_effort_Nm 取 plant 算出的 joint_effort_nm，applied_effort_Nm 取引擎
          读回值；两者之差超过 1e-6 N·m 即视为引擎侧把力矩钳位（effort_saturated=1）。
    """
    active = runtime.active[0].int().tolist()
    # ── 每关节量按“字段顺序”排列：外层元组顺序 == _header() 中 fields 的顺序 ──
    arrays = [
        x[0].tolist()
        for x in (
            q,
            dq,
            dq * 0 if ddq is None else ddq,
            result.motor_position_rad,
            result.motor_velocity_rad_s,
            result.motor_acceleration_rad_s2 * float("nan") if time_s == 0 else result.motor_acceleration_rad_s2,
            result.proximal_tension_n,
            result.distal_tension_n,
            runtime.target_rad,
            runtime.speed_limit_rad_s,
            runtime.tension_command_n,
            runtime.active.int(),
            runtime.active.int(),
            runtime.controller_speed_rad_s,
            result.cable_retraction_m,
            result.cable_extension_m,
            result.tendon_effort_nm,
            result.spring_effort_nm,
            result.friction_effort_nm,
            result.joint_effort_nm,
            applied,
            (applied - result.joint_effort_nm).abs().gt(1e-6).int(),
            result.routing_extension_m,
            result.routing_length_m,
            result.routing_locked.int(),
            result.material_extension_m,
            result.elastic_energy_j,
            result.return_spring_length_m,
            result.return_spring_force_n,
            result.moment_arm_m[:, range(10), range(10)],
            result.moment_arm_m[:, range(10), [i ^ 1 for i in range(10)]],
            result.length_balance_error_m,
            *(result.routing_state[key] for key in ROUTING_DIAGNOSTIC_FIELDS),
        )
    ]
    # ── DOF 掩码：把激活掩码压成一个位图整数（第 i 位 = 第 i 个腱关节是否激活）──
    mask = sum((1 << i) for i, enabled in enumerate(active) if enabled)
    # ── 行级前缀 6 列：时间戳 [ns]、时间 [s]、指令序号、掩码、is_fallback=1、迟到量 [s] ──
    row = [round(time_s * 1e9), time_s, runtime.command_count, mask, 1, runtime.controller_lateness_s]
    # ── 按“关节优先”展开：先输出第 0 个关节的所有字段，再第 1 个关节，依此类推 ──
    for index in range(len(JOINT_NAMES)):
        row.extend(array[index] for array in arrays)
    # ── 行尾追加 4 个 J1 角度 [rad] 与 J1/J2 耦合最大误差 [rad] ──
    return row + slave_q[0].tolist() + [mimic_error]


def run_simulation(args, cfg, frames, initial, overrides, duration, app, sim_cfg):
    """Build the scene, replay every physics step, and write the three output files.

    【中文】
    用途：真正的仿真主体。创建 PhysX 上下文与交互场景、写入初态、构造
          Real2SimRuntime，然后进入“物理步级”主循环，边跑边写 CSV，
          循环结束后把状态统计写回 metadata。
    参数：
        args      —— 命令行 Namespace（步长 [s]、输出路径、设备、限幅等）；
        cfg       —— 已解析的 Real2SimConfig；
        frames    —— list[CommandFrame] 指令帧（为空时用 --joint 生成两帧阶跃）；
        initial   —— InitialState 或 None（真实反馈初态）；
        overrides —— {关节名: 角度 [rad]}，来自 --initial-joint-deg；
        duration  —— 回放总时长 [s]；
        app       —— omni.kit.app 的应用对象，用 is_running() 检测窗口是否被关闭；
        sim_cfg   —— SimulationCfg（物理步长 dt [s]、设备、PhysX 配置）。
    返回：无（成功时写出 CSV / params.yaml / metadata.json 三个文件）。
    异常：
        RuntimeError —— 手不是固定基座、仿真应用中途退出、状态或力矩出现非有限值、
                        J1/J2 耦合误差超过 0.01 rad；
        ValueError   —— 初始张力为负、空指令事件；
        其他引擎异常原样向上抛（metadata 会被标记 status: failed）。
    单位：时间 s、角度 rad、角速度 rad/s、力矩 N·m、张力 N、长度 m。
    """
    import torch

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    from botyard_real2sim.runtime import PeriodicClock, Real2SimRuntime
    from botyard_real2sim.scene import make_no_load_scene_cfg
    from botyard_real2sim.simulation.hand_cfg import (
        HELD_JOINT_NAMES,
        JOINT_ARMATURE,
    )

    # ── 一、创建引擎与场景：SimulationContext + 空载场景配置（可选覆盖根位姿）──
    sim = SimulationContext(sim_cfg)
    scene_cfg = make_no_load_scene_cfg(effort_limit_nm=args.effort_limit_nm, armature_scale=args.armature_scale)
    if args.root_pos is not None:
        scene_cfg.robot.init_state.pos = tuple(args.root_pos)
    if args.root_xyzw is not None:
        scene_cfg.robot.init_state.rot = tuple(args.root_xyzw)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    if not robot.is_fixed_base:
        raise RuntimeError("the hand must be recognized as a fixed-base articulation")
    # ── 二、建立关节索引映射：10 个腱关节 / 手腕关节 / 从动 J1 / 主动 J2 ──
    joint_ids = [robot.joint_names.index(name) for name in JOINT_NAMES]
    wrist_ids = [robot.joint_names.index(name) for name in HELD_JOINT_NAMES]
    slave_ids = [robot.joint_names.index(f"{finger}J1") for finger in ("FF", "MF", "RF", "LF")]
    master_ids = [robot.joint_names.index(f"{finger}J2") for finger in ("FF", "MF", "RF", "LF")]
    # ── 三、写初态：默认位姿 → 反馈初态 → 命令行覆盖；J1 跟随 J2，手腕置零 ──
    positions = robot.data.default_joint_pos.torch.clone()
    initial_angles = dict(initial.positions_rad) if initial else {}
    initial_angles.update(overrides)
    for name, angle in initial_angles.items():
        positions[:, robot.joint_names.index(name)] = angle
    positions[:, slave_ids] = positions[:, master_ids]
    positions[:, wrist_ids] = 0.0
    robot.write_joint_position_to_sim_index(position=positions)
    # ── 四、写初速度 [rad/s]：反馈有则用反馈，否则静止；J1 速度同样跟随 J2 ──
    initial_velocity = torch.zeros_like(positions)
    if initial:
        for name, velocity in initial.velocities_rad_s.items():
            initial_velocity[:, robot.joint_names.index(name)] = velocity
    initial_velocity[:, slave_ids] = initial_velocity[:, master_ids]
    robot.write_joint_velocity_to_sim_index(velocity=initial_velocity)
    scene.reset()
    # ── 五、锁定手腕：把持住手的手腕限位置零，使手腕不可动（相当于夹持固定）──
    robot.write_joint_position_limit_to_sim_index(
        limits=torch.zeros(1, len(wrist_ids), 2, device=args.device), joint_ids=wrist_ids
    )
    # ── 六、构造运行时并以默认张力先 reset 一次，用于反解电机（绕线筒）初始角 ──
    runtime = Real2SimRuntime(cfg, device=args.device)
    q = positions[:, joint_ids]
    initial_tension = torch.tensor([cfg.tendon.initial_tension_n], device=args.device)
    if initial:
        for name, tension in initial.tensions_n.items():
            if tension < 0:
                raise ValueError(f"negative initial tension for {name}; correct sensor bias before replay")
            initial_tension[0, JOINT_NAMES.index(name)] = tension
    runtime.reset(q, initial_tension, cold_start=args.cold_start)
    # ── 七、用反馈的电机角/角速度覆盖默认值，然后带完整初态再 reset 一次 ──
    motor_positions = runtime.evaluate(q, torch.zeros_like(q)).motor_position_rad.clone()
    motor_velocities = torch.zeros_like(q)
    if initial:
        for name, value in initial.motor_positions_rad.items():
            motor_positions[0, JOINT_NAMES.index(name)] = value
        for name, value in initial.motor_velocities_rad_s.items():
            motor_velocities[0, JOINT_NAMES.index(name)] = value
    runtime.reset(
        q,
        initial_tension,
        cold_start=args.cold_start,
        motor_position_rad=motor_positions,
        motor_velocity_rad_s=motor_velocities if initial and initial.motor_velocities_rad_s else None,
        joint_velocity_rad_s=initial_velocity[:, joint_ids],
    )
    # ── 八、没有录制帧时生成两帧阶跃：t=0 保持当前角，t=step_at 跳到目标角 [rad] ──
    if not frames:
        index = JOINT_NAMES.index(args.joint)
        speed = math.radians(args.max_speed_deg_s)
        frames = [
            CommandFrame(0.0, {args.joint: JointCommand(float(q[0, index]), speed, args.tension_command_n)}, True),
            CommandFrame(
                args.step_at,
                {args.joint: JointCommand(math.radians(args.target_deg), speed, args.tension_command_n)},
                True,
            ),
        ]
    # ── 九、准备输出：建目录、派生两个伴随文件名、先落盘参数快照 ──
    args.out.parent.mkdir(parents=True, exist_ok=True)
    params_path = args.out.with_suffix(".params.yaml")
    metadata_path = args.out.with_suffix(".metadata.json")
    save_config(params_path, cfg)
    # ── 十、先写一份 metadata（status=running）：即便随后崩溃也能留下现场 ──
    # 关键固定标记（本文件恒为强制 Fallback，不是标定结果）：
    #   calibrated: False、mode: "forced_fallback"、third_command_field_enforced: False
    # 说明：张力传感器装在电机侧（两段走线摩擦与走线弹簧之前），
    #       而进关节力矩用的是关节侧张力 distal_tension_N。
    metadata = {
        "schema_version": 3,
        "routing_model": cfg.tendon.routing_model,
        "routing_max_substep_s": cfg.tendon.routing_max_substep_s,
        "tension_sensor_location": "motor-side, before both routing friction zones and the routing spring",
        "joint_torque_tension_source": "distal_tension_N",
        "routing_initial_state_assumption": "equal initial end tensions, zero bristle memory, mean boundary velocity",
        "routing_energy_ledger_scope": "internal substeps, predicted joint boundary; excludes PhysX energy",
        "status": "running",
        "created_utc": datetime.now(UTC).isoformat(),
        "repository_revision": _revision(),
        "package_versions": _versions(),
        "engine": "isaacsim_physx",
        "actuator_execution": "torch_explicit_effort",
        "calibrated": False,
        "mode": "forced_fallback",
        "third_command_field_enforced": False,
        "command_source": str(args.commands) if args.commands else "generated_step",
        "command_frames": args.command_frames,
        "strict_commands": args.strict_commands,
        "physics_dt_s": args.physics_dt,
        "controller_period_s": cfg.controller.control_period_s,
        "firmware_estimator_dt_s": cfg.controller.firmware_time_delta_s,
        "sample_period_s": args.sample_dt,
        "duration_requested_s": duration,
        "effort_limit_nm": args.effort_limit_nm,
        "armature_scale": args.armature_scale,
        "numerical_armature_kg_m2": {name: JOINT_ARMATURE[name] * args.armature_scale for name in JOINT_NAMES},
        "root_position": list(scene_cfg.robot.init_state.pos),
        "root_quaternion_xyzw": list(scene_cfg.robot.init_state.rot),
        "initial_position_rad": dict(zip(JOINT_NAMES, q[0].tolist(), strict=True)),
        "initial_tension_n": dict(zip(JOINT_NAMES, initial_tension[0].tolist(), strict=True)),
        "initial_velocity_rad_s": dict(zip(JOINT_NAMES, initial_velocity[0, joint_ids].tolist(), strict=True)),
        "initial_velocity_source": "same_row_feedback_else_rest",
        "initial_motor_position_rad": dict(zip(JOINT_NAMES, motor_positions[0].tolist(), strict=True)),
        "initial_motor_velocity_rad_s": dict(zip(JOINT_NAMES, motor_velocities[0].tolist(), strict=True)),
        "motor_zero_angle_rad": dict(zip(JOINT_NAMES, runtime.plant.motor_zero_angle_rad[0].tolist(), strict=True)),
        "motor_angle_definition": "unwrapped winding-spool angle; convert geared rotor encoder angles before import",
        "motor_spool_radius_m": dict(zip(JOINT_NAMES, cfg.tendon.drum_radius_m, strict=True)),
        "acceleration_definition": "backward difference over physics_dt; first sample undefined (NaN)",
        "initial_missing_motor_positions": [
            n for n in JOINT_NAMES if not initial or n not in initial.motor_positions_rad
        ],
        "initial_missing_joint_velocities": [
            n for n in JOINT_NAMES if not initial or n not in initial.velocities_rad_s
        ],
        "initial_missing_motor_velocities": [
            n for n in JOINT_NAMES if not initial or n not in initial.motor_velocities_rad_s
        ],
        "geometry_fingers": list(cfg.tendon.geometry),
        "recorded_fallback_evidence": args.fallback_evidence,
        "initial_missing_positions": list(initial.missing_positions) if initial else list(JOINT_NAMES),
        "initial_missing_tensions": list(initial.missing_tensions) if initial else list(JOINT_NAMES),
        "params_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
        "assumptions": [
            "Mechanical parameters are provisional, not fitted to real data.",
            "CAD dimensions are measured; guide/attachment assignments and omitted route segments need verification.",
            "Routing spring models layout take-up separately from intrinsic cable stiffness.",
            "Dynamic routing mass, contact relaxation and friction hysteresis are provisional, not fitted.",
            "Motor angles are independent spool kinematics with optional lag; no electrical/rotor model is assumed.",
            "Numerical armature is not measured motor inertia.",
            "Joint-axis smooth friction has no stationary stiction; dynamic routing bristles do support holding.",
            "Sensor delay/filtering and distributed contact parameters are not yet calibrated.",
            "Command execution is quantized to the next physics boundary; controller phase does not drift.",
        ],
    }
    if args.commands:
        metadata["commands_sha256"] = hashlib.sha256(args.commands.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    # ── 主循环前的状态初始化 ──
    frame_index = 0  # 下一条待消费的指令帧下标（帧按时间升序）
    previous_commands = None  # 上一条已生效的指令字典，用于判断“指令是否真的变了”
    step_count = math.ceil(duration / args.physics_dt - 1e-10)  # 总物理步数（向上取整）
    sample_clock = PeriodicClock(args.sample_dt)  # 采样用时钟，周期 sample_dt [s]
    samples = 0  # 已写出的 CSV 行数
    saturated_steps = 0  # 被数值限幅的力评估次数（循环后统计）
    max_mimic_error = 0.0  # J1/J2 耦合最大误差 [rad]（循环后统计）
    max_command_lateness = 0.0  # 指令事件最大迟到量 [s]
    saturation_count = torch.zeros((), device=args.device, dtype=torch.int64)
    max_mimic = torch.zeros((), device=args.device)
    previous_dq = None  # 上一步关节角速度 [rad/s]，用于后向差分求角加速度
    try:
        with args.out.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(_header())
            output.flush()
            # ── 主循环：每个迭代是一个物理步 t → t + dt，顺序如下（不可交换）──
            for step in range(step_count + 1):
                # ── 0. 存活检查：GUI 窗口被关掉时立即中止，避免无限空转 ──
                if not app.is_running():
                    raise RuntimeError("simulation application stopped before replay completed")
                # ── 1. 读引擎：当前物理时刻 [s] + 关节角 q [rad] + 角速度 q̇ [rad/s] ──
                time_s = step * args.physics_dt
                q = robot.data.joint_pos.torch[:, joint_ids]
                dq = robot.data.joint_vel.torch[:, joint_ids]
                # 角加速度 [rad/s^2]：对物理步长做后向差分，首个采样点无定义故为 NaN
                ddq = torch.full_like(dq, float("nan")) if previous_dq is None else (dq - previous_dq) / args.physics_dt
                previous_dq = dq.clone()
                # ── 2. 指令事件生效：消费所有到期帧；apply_command 只换掩码，
                #       真正下发要等本步后面的控制 tick ──
                while frame_index < len(frames) and frames[frame_index].time_s <= time_s + 1e-12:
                    frame = frames[frame_index]
                    if frame.commands:
                        # 只有“事件帧”或“指令内容发生变化”才重新应用，避免重复刷新
                        if frame.is_event or frame.commands != previous_commands:
                            runtime.apply_command(frame, q)
                            # 记录指令迟到量：仿真时刻 − 录制时刻 [s]
                            max_command_lateness = max(max_command_lateness, time_s - frame.time_s)
                        previous_commands = frame.commands
                    elif frame.is_event:
                        raise ValueError("empty command event cannot be interpreted as a hardware stop")
                    frame_index += 1
                # ── 3. 控制 tick：到期才跑一次控制器，输出收绳速度指令 [rad/s] ──
                runtime.update_controller(time_s, q)
                # ── 4. 算力矩：由张力/力臂/摩擦算出 10 个关节所需力矩 [N·m]（只算不改）──
                result = runtime.evaluate(q, dq)
                # ── 5. 写引擎：把 10 个关节的 effort [N·m] 交给显式执行器，再冲刷到 PhysX ──
                robot.actuators.target_command.set_effort_index(value=result.joint_effort_nm, joint_ids=joint_ids)
                scene.write_data_to_sim()
                # ── 6. 读回 applied_effort [N·m] 并判断饱和：与请求值差 > 1e-6 N·m
                #       说明被引擎侧限幅钳位（数值保护，不是腱绳指令限值）──
                applied = robot.actuators.applied_effort.torch[:, joint_ids]
                saturation_count += (applied - result.joint_effort_nm).abs().gt(1e-6).any().long()
                # ── 7. 耦合诊断：4 个手指的 J1（从动）应等于 J2（主动），误差单位 rad ──
                slave_q = robot.data.joint_pos.torch[:, slave_ids]
                error = (slave_q - robot.data.joint_pos.torch[:, master_ids]).abs().max()
                max_mimic = torch.maximum(max_mimic, error)
                # ── 8. 采样：按 sample_dt [s] 到期才写一行；最后一步强制写一行保证收尾 ──
                due = sample_clock.pop_due(time_s)
                if due or step == step_count:
                    all_q = robot.data.joint_pos.torch
                    # 有限性检查：NaN/Inf 一旦写进 CSV 会污染后续参数拟合
                    if not torch.isfinite(all_q).all() or not torch.isfinite(applied).all():
                        raise RuntimeError(f"nonfinite rigid-body state or effort at t={time_s:.6f}")
                    mimic_error = float(error)
                    # J1/J2 耦合是硬约束：误差超过 0.01 rad 视为场景/执行器配置错误
                    if mimic_error > 0.01:
                        raise RuntimeError(f"J1=J2 constraint error {mimic_error:.6g} rad exceeds 0.01")
                    writer.writerow(_row(time_s, q, dq, runtime, result, applied, slave_q, mimic_error, ddq))
                    output.flush()
                    samples += 1
                # ── 最后一步只出收尾样本，不再推进（否则会多走一步物理）──
                if step == step_count:
                    break
                # ── 9. 推进传动：只推收绳边界与走线内部状态（单位 m、s），刚体不动 ──
                runtime.advance(args.physics_dt, q, dq)
                # ── 10. 推进刚体：sim.step 走一个物理步；GUI 模式下按约 60 Hz 抽帧渲染 ──
                sim.step(render=args.gui and step % max(1, round(1.0 / (60.0 * args.physics_dt))) == 0)
                # ── 11. 刷新场景缓冲（同步 data 视图与 PhysX 状态）──
                scene.update(args.physics_dt)
        # ── 循环正常结束：把标量统计从张量/计数器里取出 ──
        saturated_steps = int(saturation_count)
        max_mimic_error = float(max_mimic)
        # ── 收尾：把状态统计写进 metadata（这些是判断结果可否用于拟合的关键指标）──
        metadata.update(
            status="completed",
            samples=samples,
            physics_steps=step_count,
            duration_actual_s=step_count * args.physics_dt,
            command_events_applied=runtime.command_count,
            controller_ticks=runtime.clock.tick_count,
            saturated_force_evaluations=saturated_steps,
            mimic_max_error_rad=max_mimic_error,
            command_max_lateness_s=max_command_lateness,
        )
        # 出现饱和说明数值钳位介入过，结果不宜直接用于参数拟合
        if saturated_steps:
            print(
                f"[WARN] effort safeguard clipped {saturated_steps} force evaluations; inspect metadata before fitting",
                file=sys.stderr,
            )
    except BaseException as error:
        # ── 失败路径：metadata 记为 failed 并带上错误信息，异常继续上抛 ──
        metadata.update(status="failed", error=str(error), samples=samples)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        # launch_simulation owns SimulationContext/backend teardown and keeps
        # exception exit codes intact; no legacy callback cleanup API is used.
        # 【中文】这里只落盘 metadata：SimulationContext/后端销毁由 launch_simulation
        #         负责，并保留异常的退出码；不使用旧的 callback 清理接口。
    print(f"wrote {args.out}, {params_path}, {metadata_path}", flush=True)


def main(argv=None) -> int:
    """CLI entry point: parse, prepare, then (only if needed) start Isaac Sim.

    【中文】
    用途：命令行入口。顺序刻意安排为“先做全部纯 Python 校验，再引入 isaaclab”，
          这样 --help、参数导出和输入错误都不会付出启动仿真的代价。
    参数：argv —— 参数列表；None 表示使用 sys.argv[1:]。
    返回：进程退出码，0 表示成功。
    异常：参数/输入错误时通过 parser.error 抛 SystemExit(2)；仿真内部异常向上抛。
    单位：时间 s、角度 rad（命令行输入为 deg）、力矩 N·m。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # ── 参数准备与校验：任何 ValueError/OSError 都转成 argparse 的用法错误 ──
    try:
        cfg, frames, initial, overrides, duration = prepare_experiment(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    # ── 只导出参数的模式：已经写完 YAML，直接成功退出，不加载 Isaac Sim ──
    if args.write_default_config:
        print(f"wrote {args.write_default_config}")
        return 0
    # Runtime discovery stays after all
    # pure configuration, CSV validation and --help/parameter export paths.
    # 【中文】isaaclab 的 import 必须放在所有纯配置、CSV 校验与 --help/参数导出
    #         路径之后——导入它本身就会拉起仿真运行时。
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg

    from botyard_real2sim.simulation.physics import make_physx_cfg

    # Keep PhysX physics and use the standard direct-effort actuator path.
    # 【中文】保持 PhysX 物理后端，并令 use_newton_actuators=False 走标准
    #         “显式执行器（direct effort）”路径：每步由脚本直接写关节 effort [N·m]，
    #         而不是交给 Newton 执行器模型去积分。
    sim_cfg = SimulationCfg(
        dt=args.physics_dt, device=args.device, physics=make_physx_cfg(), use_newton_actuators=False
    )
    # ── 启动仿真上下文：require_kit 要求 Kit 可用；--gui 时挂 kit 可视化 ──
    with launch_simulation(
        sim_cfg,
        {
            "require_kit": True,
            "device": args.device,
            "visualizer": ["kit"] if args.gui else [],
        },
    ):
        import omni.kit.app
        import torch

        # ── 推理模式：关闭 autograd 记录，降低显存与开销 ──
        with torch.inference_mode():
            run_simulation(args, cfg, frames, initial, overrides, duration, omni.kit.app.get_app(), sim_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
