#!/usr/bin/env python3
"""Compare empty-hand joint angle and proximal tendon tension traces.

Input is a wide CSV with ``timestamp_ns`` (or ``time_s``) and columns named
``<joint>/position_rad``. Tendon tension is ``<joint>/tension_N`` or the
firmware's raw ``<joint>/effort_raw`` multiplied by ``tension_raw_to_N``.

The tool accepts ROS benchmark captures and real2sim replay CSVs. It never
changes the input data, and only compares samples in the shared time window.

【中文说明】
职责：
  本脚本把「实机采集 CSV」与「仿真/回放 CSV」放到同一时间基准上，逐关节比较三类曲线：
  关节角 q [rad]（绘图中换算为 deg）、电机绕线轮角 [rad]（绘图中为 deg）、
  近端张力（proximal tension）[N]；此外可选比较远端张力、速度、加速度、路由损耗等诊断通道。
  比较结果写成一份 JSON 误差报告，并输出 PNG 或自包含 HTML 图。

输入：
  --real 实机 CSV（ROS benchmark 或其它实机采集）；--sim 回放 CSV（由 real2sim_replay.py 产出）。
  两者都是「宽表」CSV：时间列为 timestamp_ns / time_s / t_s 之一，数值列命名为
  <joint>/position_rad、<joint>/motor_position_rad、<joint>/tension_N 或固件原始
  <joint>/effort_raw 等。关节顺序不作任何假设，全部按列名自动发现。

输出：
  --out 指定的图（.png 或 .html）以及 --report 指定的指标 JSON（默认与图同名的 .json）。
  报告中的误差约定为「sim minus real」；steady_* 只统计末尾固定长度窗口，并非自动检测到的稳态段。

与 replay 的关系：
  本脚本只读取回放已经落盘的 CSV，从不改动输入数据、不启动 Isaac、不参与模型参数求解。
  它把回放结果与实机对照，供人工判读，并为后续参数扫描/拟合（见 real2sim_sweep.py）提供指标。

单位约定：
  关节角与电机角 rad（报告中的角度指标以 deg 给出）、角速度 rad/s、角加速度 rad/s²、
  力与张力 N、力矩 N·m、时间 s、路由/材料伸长 mm。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ── 常量：默认关节与通道定义 ──
# 默认比较的 10 个目标肌腱关节：拇指 THJ1/THJ2、食指 FFJ2/FFJ3、中指 MFJ2/MFJ3、
# 无名指 RFJ2/RFJ3、小指 LFJ2/LFJ3。命令行 --joints 可覆盖该默认集合。
TARGET_JOINTS = ("THJ1", "THJ2", "FFJ2", "FFJ3", "MFJ2", "MFJ3", "RFJ2", "RFJ3", "LFJ2", "LFJ3")
# 通道表：键 = --channels 可选名；值 = (CSV 列后缀, 显示换算系数, 坐标轴单位标签)。
# 换算系数只改变绘图/报告里的显示单位（rad→deg 乘 180/π，m→mm 乘 1000，N、J、状态量保持 1.0），
# 不改变 CSV 中的原始采样值，也不代表任何传感器分辨率或容差。
CHANNELS = {
    "angle": ("position_rad", 180 / math.pi, "joint angle [deg]"),
    "motor": ("motor_position_rad", 180 / math.pi, "motor angle [deg]"),
    "tension": ("tension_N", 1.0, "proximal tension [N]"),
    "distal_tension": ("distal_tension_N", 1.0, "joint-side tension [N]"),
    "tension_drop": ("routing_tension_drop_N", 1.0, "motor minus joint tension [N]"),
    "contact": ("routing_contact_state", 1.0, "routing contact state [0..1]"),
    "routing_friction_proximal": ("routing_friction_proximal_N", 1.0, "proximal routing friction [N]"),
    "routing_friction_distal": ("routing_friction_distal_N", 1.0, "distal routing friction [N]"),
    "routing_bristle_proximal": ("routing_bristle_proximal_m", 1000.0, "proximal bristle deflection [mm]"),
    "routing_bristle_distal": ("routing_bristle_distal_m", 1000.0, "distal bristle deflection [mm]"),
    "routing_kinetic": ("routing_kinetic_energy_J", 1.0, "routing kinetic energy [J]"),
    "routing_dissipation": ("routing_friction_dissipated_J", 1.0, "routing friction dissipation [J]"),
    "routing_numerical_dissipation": ("routing_numerical_dissipated_J", 1.0, "integration dissipation [J]"),
    "velocity": ("velocity_rad_s", 1.0, "joint velocity [rad/s]"),
    "motor_velocity": ("motor_velocity_rad_s", 1.0, "motor velocity [rad/s]"),
    "acceleration": ("acceleration_rad_s2", 1.0, "joint acceleration [rad/s²]"),
    "motor_acceleration": ("motor_acceleration_rad_s2", 1.0, "motor acceleration [rad/s²]"),
    "routing": ("routing_extension_m", 1000.0, "routing spring extension [mm]"),
    "material": ("material_extension_m", 1000.0, "intrinsic cable extension / gap [mm]"),
    "return_spring": ("return_spring_length_m", 1000.0, "return spring length [mm]"),
}


@dataclass(frozen=True)
class CurveOptions:
    """Plot and comparison settings, independent of physical model parameters.

    【中文】绘图与比较选项，与物理模型参数无关。这里全部是「解释与对齐」层面的设置：
    单位换算、时间对齐、平移补偿、统计窗口以及要画哪些通道；本类不引入任何动力学假设。
    """

    # ── 单位换算与张力标定 ──
    # effort_raw 原始刻度 → 牛顿的换算系数 [N/计数]；最新固件约为 0.01。
    tension_raw_to_N: float = 0.01
    # ── 时间对齐 ──
    # 对齐模式："command_edge" 按指令跳变沿对齐（默认），"start" 直接按各自首行对齐。
    alignment: str = "command_edge"
    # 判定「指令发生跳变」的角度阈值 [rad]（默认 0.5° = math.radians(0.5)）。
    alignment_threshold_rad: float = math.radians(0.5)
    # 只用一个关节的指令沿作为两条曲线的共同锚点；None 表示用全部选中关节中最早的一次跳变。
    alignment_joint: str | None = None
    # 仅追加到仿真时间轴上的平移量 [s]，用于补偿回放启动延迟；实机时间轴不平移。
    sim_shift_s: float = 0.0
    # ── 误差统计窗口 ──
    # 尾部稳态统计窗口长度 [s]：取对齐后最后一个配对样本往前这么长的一段，不做稳态检测。
    steady_window_s: float = 0.5
    # 仿真插值允许跨过的最大样本间隙 [s]；超过则视为缺口并放弃插值，避免凭空造数据。
    max_interp_gap_s: float = 0.1
    # 图标题，仅用于显示。
    title: str = "Empty-hand real2sim"
    # 实机张力传感器零偏 [N]：只从实机反馈中减去，仿真值与指令字段都不校正。
    real_tension_bias_N: float = 0.0
    # 只保留遥测中 is_fallback == 1 的样本，其余样本置 NaN 后被跳过。
    fallback_only: bool = False
    # 是否额外画出第三指令字段（张力指令）：仅作参考，不代表 Fallback 实际执行的张力目标。
    show_tension_command: bool = False
    # 要绘制/比较的通道集合，取自 CHANNELS 的键，不允许为空且不允许重复。
    channels: tuple[str, ...] = ("angle", "motor", "tension", "distal_tension")
    # 是否在缺少速度/加速度列时用后向差分显式推导（差分放大噪声，故默认关闭）。
    derive_kinematics: bool = False

    def validate(self) -> None:
        """在读任何 CSV 之前校验选项组合是否可用。

        校验内容：
          - channels 必须是 CHANNELS 键的非空、无重复子集；
          - derive_kinematics / show_tension_command 必须是 bool；
          - tension_raw_to_N 必须有限且为正 [N/计数]；
          - alignment 只能是 command_edge 或 start；
          - alignment_threshold_rad [rad]、steady_window_s [s]、max_interp_gap_s [s]
            必须有限且为正，sim_shift_s [s] 与 real_tension_bias_N [N] 必须有限。
        返回：无（校验通过时什么也不做）。
        异常：任一条不满足即抛 ValueError，消息中给出对应字段名。
        """
        if (
            not isinstance(self.channels, (list, tuple))
            or not self.channels
            or any(c not in CHANNELS for c in self.channels)
            or len(set(self.channels)) != len(self.channels)
        ):
            raise ValueError(f"channels must be a nonempty unique selection from {tuple(CHANNELS)}")
        if type(self.derive_kinematics) is not bool:
            raise ValueError("derive_kinematics must be boolean")
        if not isinstance(self.show_tension_command, bool):
            raise ValueError("show_tension_command must be boolean")
        if not math.isfinite(self.tension_raw_to_N) or self.tension_raw_to_N <= 0:
            raise ValueError("tension_raw_to_N must be positive and finite")
        if self.alignment not in {"command_edge", "start"}:
            raise ValueError("alignment must be 'command_edge' or 'start'")
        if not math.isfinite(self.alignment_threshold_rad) or self.alignment_threshold_rad <= 0:
            raise ValueError("alignment_threshold_rad must be positive and finite")
        if not math.isfinite(self.real_tension_bias_N):
            raise ValueError("real_tension_bias_N must be finite")
        if not math.isfinite(self.sim_shift_s):
            raise ValueError("sim_shift_s must be finite")
        if not math.isfinite(self.steady_window_s) or self.steady_window_s <= 0:
            raise ValueError("steady_window_s must be positive and finite")
        if not math.isfinite(self.max_interp_gap_s) or self.max_interp_gap_s <= 0:
            raise ValueError("max_interp_gap_s must be positive and finite")


@dataclass(frozen=True)
class CurveTable:
    """一张已解析的宽表 CSV：共享的相对时间轴 + 原始数值列。

    字段：
      path    —— 源文件路径，仅用于报错与报告中的溯源；
      time_s  —— 相对首行的时间轴 [s]，由 timestamp_ns 或 time_s 归一化得到，严格递增；
      columns —— 列名 → 原始数值列表，单位保持 CSV 原样（rad、N、m、原始计数等），
                 缺失或非数值的单元格记为 math.nan。
    本类只承载数据，不做单位换算、不做零偏校正、不改变采样点。
    """

    path: Path
    time_s: list[float]
    columns: dict[str, list[float]]

    @property
    def joints(self) -> tuple[str, ...]:
        """按列名推断本表包含哪些关节。

        做法：筛选以 ``/position_rad`` 结尾的列，去掉该后缀即为关节名，
        因此不依赖 CSV 的列顺序，也不依赖硬编码的关节清单。
        返回：关节名组成的元组（保持列出现顺序）。
        """
        suffix = "/position_rad"
        return tuple(name[: -len(suffix)] for name in self.columns if name.endswith(suffix))

    def values(self, joint: str, suffix: str, *, raw_to_N: float = 0.01) -> list[float] | None:
        """取出某个关节某一物理量在整条时间轴上的数值。

        参数：
          joint    —— 关节名，例如 "FFJ2"；
          suffix   —— 列后缀，例如 "position_rad"、"tension_N"、"cmd_rad"；
          raw_to_N —— effort_raw / cmd_tension_raw 原始刻度到牛顿的换算系数 [N/计数]。

        返回：与该表 time_s 等长的数值列表；若该列（及可替代的原始列）不存在则返回 None。
             张力列的优先级为「显式 N 优先、缺样本时回退到同一行的固件原始值」。
        """
        if suffix == "tension_N":
            direct = self.columns.get(f"{joint}/tension_N")
            raw = self.columns.get(f"{joint}/effort_raw")
            if direct is None and raw is None:
                return None
            if direct is None:
                return [v * raw_to_N for v in raw]
            if raw is None:
                return direct
            # Explicit N takes priority, while a missing direct sample can still
            # use the raw firmware feedback from the same row.
            return [v if math.isfinite(v) else raw[i] * raw_to_N for i, v in enumerate(direct)]
        if suffix == "cmd_tension_N":
            # 张力指令字段在不同固件/记录版本里命名不一致（N 或 n、command 或 limit），
            # 因此按别名列表依次尝试，先收集所有可用的候选列。
            aliases = (
                "cmd_tension_N",
                "cmd_tension_command_N",
                "cmd_tension_command_n",
                "cmd_tension_limit_N",
                "cmd_tension_limit_n",
            )
            found = [self.columns[f"{joint}/{key}"] for key in aliases if f"{joint}/{key}" in self.columns]
            # 原始刻度形式的指令列同样接受，并按 raw_to_N 换算成 N [N/计数]。
            for key in ("cmd_tension_raw", "cmd_tension_command_raw", "cmd_tension_limit_raw"):
                if f"{joint}/{key}" in self.columns:
                    found.append([v * raw_to_N for v in self.columns[f"{joint}/{key}"]])
            if not found:
                return None
            # 逐样本取第一个有限值：同一行里高优先级列缺失时仍能退回其它别名列。
            return [
                next((values[i] for values in found if math.isfinite(values[i])), math.nan)
                for i in range(len(self.time_s))
            ]
        return self.columns.get(f"{joint}/{suffix}")


def _number(text: str | None, *, column: str, line: int) -> float:
    """把一个 CSV 单元格解析为 float：空单元格记为 NaN，真坏值仍然报错以便定位。

    参数：
      text   —— 单元格原文（可能为 None）；
      column —— 列名，仅用于报错定位；
      line   —— 1 基的 CSV 行号，仅用于报错定位。
    返回：
      数值；空单元格返回 math.nan；"true"/"false"（大小写不敏感）解析为 1.0/0.0，
      以兼容布尔型遥测列（如 is_fallback）。
    异常：
      单元格既非空、非布尔、也非有限数值（含 inf）时抛 ValueError，消息带列名与行号。
    """
    if text is None or not text.strip():
        return math.nan
    try:
        if text.strip().lower() in {"true", "false"}:
            return float(text.strip().lower() == "true")
        value = float(text)
        if math.isinf(value):
            raise ValueError("infinite value")
        return value
    except ValueError as exc:
        raise ValueError(f"{column} at CSV line {line} is not numeric: {text!r}") from exc


def read_curve(path: str | Path) -> CurveTable:
    """Read a ROS benchmark or simulator wide CSV without assuming joint order.

    【中文】读取一张宽表 CSV（实机或仿真），不假设关节顺序，也不要求列齐全。

    参数：path —— CSV 路径（UTF-8，允许 BOM）。
    返回：CurveTable，时间轴已归一化为「相对首样本的秒数」，严格递增。
    读取规则：
      - 时间列取 timestamp_ns / time_s / t_s 中第一个存在的列；两者单位不同但都被换算成 s；
      - 数值列按列名原样保存，空单元格与非数值记为 NaN，不插值、不填补；
      - 至少要有两行样本，且至少要有一个 <joint>/position_rad 列。
    异常：缺少时间列、时间戳非整数/非数值、时间戳非严格递增、样本不足或缺角度列时抛 ValueError。
    """

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        time_name = next((name for name in ("timestamp_ns", "time_s", "t_s") if name in names), None)
        if time_name is None:
            raise ValueError(f"{path}: expected timestamp_ns, time_s or t_s")
        value_names = [name for name in names if name != time_name]
        columns = {name: [] for name in value_names}
        times: list[float] = []
        epoch_ns: int | None = None
        epoch_s: float | None = None
        # ── 逐行读取：时间列归一化为相对秒，数值列容错解析为 float 或 NaN ──
        # line 从 2 开始计数，与人类可读的 CSV 行号对齐（第 2 行是首条数据）。
        for line, row in enumerate(reader, start=2):
            stamp = row[time_name]
            if stamp is None or not stamp.strip():
                raise ValueError(f"{path}: missing {time_name} at CSV line {line}")
            # 纳秒整数时间戳：以首样本为纪元零点，换算成相对秒 [s]。
            if time_name == "timestamp_ns":
                try:
                    stamp_ns = int(stamp)
                except ValueError as exc:
                    raise ValueError(f"{path}: {time_name} at CSV line {line} is not an integer") from exc
                if epoch_ns is None:
                    epoch_ns = stamp_ns
                t = (stamp_ns - epoch_ns) * 1e-9
            # 以秒为单位的浮点时间戳：同样减去首样本时刻，得到相对时间 [s]。
            else:
                stamp_s = _number(stamp, column=time_name, line=line)
                if epoch_s is None:
                    epoch_s = stamp_s
                t = stamp_s - epoch_s
            # 时间轴必须有限且严格递增，否则后续的对齐与插值都没有定义。
            if not math.isfinite(t) or (times and t <= times[-1]):
                raise ValueError(f"{path}: timestamps must be finite and strictly increasing (line {line})")
            times.append(t)
            for name in value_names:
                columns[name].append(_number(row.get(name), column=name, line=line))
    if len(times) < 2:
        raise ValueError(f"{path}: at least two samples are required")
    # 至少要有一列关节角，否则这张表无法参与本工具的任何比较。
    if not any(name.endswith("/position_rad") for name in columns):
        raise ValueError(f"{path}: no <joint>/position_rad columns found")
    return CurveTable(path, times, columns)


def load_options(path: str | Path | None = None, **overrides: Any) -> CurveOptions:
    """Load the optional ``curves:`` YAML section and apply explicit CLI overrides.

    【中文】读取可选的 YAML 配置（``curves:`` 段）并叠加命令行显式覆盖，构造 CurveOptions。

    参数：
      path      —— YAML 路径；为 None 时完全依赖 overrides 与 dataclass 默认值；
      **overrides —— 命令行给出的字段；值为 None 的项被忽略，从而保留 YAML/默认值。
    返回：已通过 validate() 校验的 CurveOptions。
    异常：
      使用 --config 但未安装 PyYAML 时抛 RuntimeError；
      YAML 顶层或 curves 段不是映射、出现未知键、或校验失败时抛 ValueError。
    """

    data: dict[str, Any] = {}
    if path is not None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required when --config is used") from exc
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError("curve config must be a YAML mapping")
        # Full model files may omit curves altogether. Standalone curve files
        # remain supported, and unknown keys in an explicit curves block fail.
        # 模型配置文件可能整体没有 curves 段，此时退回空字典（全部用默认值）；
        # 也可以直接传一份「只有曲线设置」的 YAML，此时整份文件就是 curves 内容。
        # 但显式写出的 curves 段里若含未知键，则视为错误而不是静默忽略。
        model_keys = {"controller", "tendon", "model", "schema_version", "initial_state", "replay", "scene"}
        data = loaded.get("curves", {} if set(loaded) & model_keys else loaded)
        if not isinstance(data, dict):
            raise ValueError("curves must be a YAML mapping")
    allowed = set(CurveOptions.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown curve setting(s): {', '.join(sorted(unknown))}")
    # 只有非 None 的命令行参数才覆盖配置文件，从而让 CLI 与 YAML 可以叠加使用。
    data.update({key: value for key, value in overrides.items() if value is not None})
    options = CurveOptions(**data)
    options.validate()
    return options


def command_edge_time(
    table: CurveTable,
    joints: tuple[str, ...] | list[str],
    threshold_rad: float,
) -> float | None:
    """Return the earliest finite target angle change exceeding ``threshold_rad``.

    【中文】在指令序列中找「指令跳变沿」的最早时刻，作为两条曲线的共同时间锚点。

    参数：
      table       —— 待搜索的曲线表（实机或仿真）；
      joints      —— 参与搜索的关节名；
      threshold_rad —— 跳变判定阈值 [rad]，相邻两个有限目标值之差达到该值即算一次跳变。
    返回：最早跳变时刻 [s]（相对该表首样本）；所有关节都没有可用跳变时返回 None。
    说明：非有限目标值（缺采样/NaN）被跳过，且不会重置上一个有效目标值；
          多个关节都有跳变时取其中最早的一次。
    """

    found: list[float] = []
    for joint in joints:
        # 指令列缺失（没有 cmd_rad）的关节直接跳过，不视为错误。
        command = table.values(joint, "cmd_rad")
        if command is None:
            continue
        previous = math.nan
        # 顺序扫描该关节指令，记录第一处超过阈值的相邻差分时刻。
        for t, target in zip(table.time_s, command, strict=True):
            if not math.isfinite(target):
                continue
            # 跳变判定：|target - previous| >= threshold_rad [rad]。
            if math.isfinite(previous) and abs(target - previous) >= threshold_rad:
                found.append(t)
                break
            previous = target
    # 多个候选关节时，用最早出现的沿保证两条曲线锚到同一个物理事件。
    return min(found) if found else None


def aligned_time(
    table: CurveTable,
    options: CurveOptions,
    joints: tuple[str, ...] | list[str],
    *,
    is_sim: bool,
) -> tuple[list[float], float]:
    """Align each trace to the same command edge, or explicitly to its first row.

    【中文】把一张曲线表的时间轴对齐到共同锚点。

    参数：
      table    —— 待对齐的曲线表；
      options  —— 提供 alignment / alignment_joint / alignment_threshold_rad / sim_shift_s；
      joints   —— 参与搜索的关节名（指定 alignment_joint 时只用该关节）；
      is_sim   —— 是否为仿真曲线：只有仿真才追加 sim_shift_s [s] 平移，实机不平移。
    返回：(对齐后的时间轴 [s], 锚点在本表原始时间轴上的位置 [s])。
          返回的时间轴一律从 0 附近开始（锚点处为 0，或平移后为 sim_shift_s）。
    异常：
      alignment == "command_edge" 但在这些关节上找不到超过阈值的指令跳变时抛 ValueError，
      提示改用 --align start（适用于静止段采集）。
    """

    origin = 0.0
    if options.alignment == "command_edge":
        # 指定了 alignment_joint 就只用它；否则在全部选中关节里找最早的跳变。
        anchor_joints = (options.alignment_joint,) if options.alignment_joint else joints
        edge = command_edge_time(table, anchor_joints, options.alignment_threshold_rad)
        if edge is None:
            raise ValueError(
                f"{table.path}: no command edge found for {', '.join(anchor_joints)}; "
                "use --align start for a stationary capture"
            )
        origin = edge
    # 平移只作用于仿真时间轴：实机采样是对齐基准，不应被移动。
    shift = options.sim_shift_s if is_sim else 0.0
    return [t - origin + shift for t in table.time_s], origin


def _interpolate(times: list[float], values: list[float], query: float, max_gap_s: float) -> float | None:
    """在仿真时间轴上对任意查询时刻做线性插值。

    参数：
      times    —— 严格递增的仿真时间轴 [s]；
      values   —— 与 times 等长的数值（单位随通道，例如 rad、N）；
      query    —— 查询时刻 [s]；
      max_gap_s —— 允许跨过的最大样本间隙 [s]，超过则拒绝插值。
    返回：
      插值结果；查询点恰好落在已有样本上时直接返回该样本值；
      查询点落在时间轴之外、跨越过大间隙、或相邻样本非有限时返回 None。
      None 表示「此处不给预测」，而不是 0，也不做外推。
    """
    index = bisect.bisect_left(times, query)
    # 精确命中已有样本：直接取值，避免插值引入无谓的舍入误差。
    if index < len(times) and abs(times[index] - query) <= 1e-12:
        value = values[index]
        return value if math.isfinite(value) else None
    # 落在时间轴之外（首样本之前/末样本之后）：不外推。
    if index == 0 or index == len(times):
        return None
    left, right = index - 1, index
    # ── 插值间隙保护：间距超过 max_gap_s 的缺口一律不插 ──
    if times[right] - times[left] > max_gap_s + 1e-12:
        return None
    low, high = values[left], values[right]
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    # 线性插值：alpha 为该查询点在左/右样本之间的归一化位置，取值 [0, 1)。
    alpha = (query - times[left]) / (times[right] - times[left])
    return low + alpha * (high - low)


def _hold_previous(times: list[float], values: list[float], query: float, max_gap_s: float) -> float | None:
    """Sample a piecewise-constant command without inventing intermediate targets.

    【中文】对阶梯式（零阶保持）信号取值，不做插值。

    参数：
      times     —— 严格递增的时间轴 [s]；
      values    —— 与 times 等长的指令值（例如张力指令 [N]）；
      query     —— 查询时刻 [s]；
      max_gap_s —— 允许的最大保持时长 [s]。
    返回：查询时刻之前（含）最近一个样本的值；查询超出末样本、或距最近样本超过
          max_gap_s、或该值非有限时返回 None。
    为什么必须这样：张力指令是按「保持」语义生效的台阶信号，若改用线性插值，就相当于
    凭空造出固件从未下发过的中间目标值。
    """

    index = bisect.bisect_right(times, query + 1e-12) - 1
    if index < 0 or query > times[-1] + 1e-12 or query - times[index] > max_gap_s + 1e-12:
        return None
    value = values[index]
    return value if math.isfinite(value) else None


def compare_series(
    real_time: list[float],
    real_values: list[float] | None,
    sim_time: list[float],
    sim_values: list[float] | None,
    *,
    scale: float,
    options: CurveOptions,
    interpolation: str = "linear",
) -> dict[str, float | int] | None:
    """Compare at real sample times inside the simulator's valid interpolation range.

    【中文】在实机样本时刻上取仿真预测，逐点比较，并汇总误差指标。

    参数：
      real_time / real_values —— 实机对齐后时间轴 [s] 与实测值（单位随通道）；
      sim_time / sim_values   —— 仿真对齐后时间轴 [s] 与预测值；
      scale    —— 显示单位换算系数（rad→deg 为 180/π，N 为 1.0），只影响误差数值的报出单位；
      options  —— 提供 steady_window_s [s] 与 max_interp_gap_s [s]；
      interpolation —— "linear"（连续量）或 "previous"（指令类台阶量，零阶保持）。
    返回：含 samples/overlap_s/rmse/mae/bias/max_abs/steady_bias/steady_rmse 的字典；
          两侧任一通道缺失、或没有任何可配对的点时返回 None。
    误差符号约定：err = (sim - real) * scale，即「仿真减实机」。
    """

    if interpolation not in {"linear", "previous"}:
        raise ValueError("interpolation must be 'linear' or 'previous'")
    # 通道缺失（例如仿真是空手、没有远端张力列）时不做比较，而不是当作零误差。
    if real_values is None or sim_values is None:
        return None
    paired: list[tuple[float, float]] = []
    # ── 时间对齐取样：以实机样本时刻为准，向仿真曲线查询同一时刻的预测 ──
    for t, measurement in zip(real_time, real_values, strict=True):
        if not math.isfinite(measurement):
            continue
        prediction = (
            _hold_previous(sim_time, sim_values, t, options.max_interp_gap_s)
            if interpolation == "previous"
            else _interpolate(sim_time, sim_values, t, options.max_interp_gap_s)
        )
        if prediction is not None:
            paired.append((t, (prediction - measurement) * scale))
    if not paired:
        return None
    # ── 误差指标（单位由 scale 决定：deg、N、rad/s 等）──
    errors = [err for _, err in paired]
    # 稳态窗口：以最后一个成功配对的样本为基准向前取 steady_window_s 秒 [s]；
    # 这只是固定尾部窗口，并不检测曲线是否真的进入稳态。
    cutoff = paired[-1][0] - options.steady_window_s
    tail = [err for t, err in paired if t >= cutoff]
    # rmse/mae/bias/max_abs 覆盖全部配对样本；steady_bias/steady_rmse 只覆盖尾部窗口。
    return {
        "samples": len(errors),
        "overlap_s": paired[-1][0] - paired[0][0],
        "rmse": math.sqrt(sum(err * err for err in errors) / len(errors)),
        "mae": sum(abs(err) for err in errors) / len(errors),
        "bias": sum(errors) / len(errors),
        "max_abs": max(abs(err) for err in errors),
        "steady_bias": sum(tail) / len(tail),
        "steady_rmse": math.sqrt(sum(err * err for err in tail) / len(tail)),
    }


def channel_values(table: CurveTable, joint: str, suffix: str, options: CurveOptions, *, is_real=False):
    """One SI conversion/calibration path for metrics and both plot formats.

    Bias is an explicitly supplied sensor offset subtracted only from real
    feedback; the command third field is never corrected or treated as a limit.

    【中文】指标计算与两种绘图格式（PNG/HTML）共用的唯一一条「换算 + 标定」路径，
    保证图和报告用的是同一套数。

    参数：
      table    —— 曲线表；
      joint    —— 关节名；
      suffix   —— 列后缀，如 position_rad、tension_N、velocity_rad_s；
      options  —— 提供 tension_raw_to_N [N/计数]、real_tension_bias_N [N]、
                  fallback_only 与 max_interp_gap_s [s]；
      is_real  —— 是否为实机数据：只有实机张力才减去传感器零偏。
    返回：与 time_s 等长的数值列表；列不存在且无法推导时返回 None。
    要点：
      - 张力零偏是人工给定的传感器偏置 [N]，只从实机反馈中扣除；
        第三指令字段既不校正，也不被当作限值使用；
      - derive_kinematics 打开时，速度/加速度由后向差分显式推导（不滤波，缺口处给 NaN）；
      - fallback_only 打开时，非 Fallback 样本被置为 NaN（需要 is_fallback 遥测，否则抛 ValueError）。
    """
    values = table.values(joint, suffix, raw_to_N=options.tension_raw_to_N)
    # 运动学推导链：位置 → 速度 → 加速度（电机侧同理）。
    derivative_sources = {
        "velocity_rad_s": "position_rad",
        "motor_velocity_rad_s": "motor_position_rad",
        "acceleration_rad_s2": "velocity_rad_s",
        "motor_acceleration_rad_s2": "motor_velocity_rad_s",
    }
    if values is None and options.derive_kinematics and suffix in derivative_sources:
        # 后向差分：v[i] ≈ (x[i] - x[i-1]) / dt，单位与原列一致（rad→rad/s，rad/s→rad/s²）。
        source = channel_values(table, joint, derivative_sources[suffix], options, is_real=is_real)
        if source is not None:
            # 首样本没有前一点，只能给 NaN。
            values = [math.nan]
            for i in range(1, len(source)):
                dt = table.time_s[i] - table.time_s[i - 1]
                # 只在 dt 不超过插值间隙上限、且两端样本都有限时才计算，避免跨缺口造数。
                valid = dt <= options.max_interp_gap_s and math.isfinite(source[i]) and math.isfinite(source[i - 1])
                values.append((source[i] - source[i - 1]) / dt if valid else math.nan)
    if values is None:
        return None
    values = list(values)
    # ── 实机张力零偏扣除：只作用于实机张力，不碰仿真与指令字段 ──
    if suffix == "tension_N" and is_real:
        values = [value - options.real_tension_bias_N for value in values]
    # ── Fallback 段过滤：逐样本按 is_fallback 掩码稀疏化（非 Fallback 记为 NaN）──
    if options.fallback_only:
        mask = table.columns.get(f"{joint}/is_fallback", table.columns.get("is_fallback"))
        if mask is None:
            raise ValueError(f"{table.path}: fallback_only requires is_fallback telemetry")
        values = [value if flag == 1 else math.nan for value, flag in zip(values, mask, strict=True)]
    return values


def residual_series(real, sim, joint, suffix, aligned, options, scale=1.0):
    """构造某一通道的残差曲线（仿真减实机），供绘图使用。

    参数：
      real / sim —— 实机与仿真 CurveTable（可缺省，缺则无法比较）；
      joint      —— 关节名；
      suffix     —— 列后缀（如 position_rad、tension_N）；
      aligned    —— build_report 给出的对齐时间轴字典，含 "real" 与 "sim" 两个键 [s]；
      options    —— 提供零偏、换算与插值间隙等设置；
      scale      —— 显示换算系数（如 rad→deg 用 180/π，N 用 1.0）。
    返回：(时间轴 [s], 残差序列)，两者等长；任一侧数据缺失时返回两个空列表。
    说明：残差在实机对齐时间轴上逐点计算，仿真值按 max_interp_gap_s [s] 做间隙保护插值。
    """
    if real is None or sim is None:
        return [], []
    measured = channel_values(real, joint, suffix, options, is_real=True)
    predicted = channel_values(sim, joint, suffix, options)
    if measured is None or predicted is None:
        return [], []
    times, errors = [], []
    # 逐实机样本查询仿真预测；非有限实测值或缺口处不出点，因此残差序列可能稀疏。
    for t, value in zip(aligned["real"], measured, strict=True):
        pred = _interpolate(aligned["sim"], predicted, t, options.max_interp_gap_s)
        if math.isfinite(value) and pred is not None:
            times.append(t)
            errors.append((pred - value) * scale)
    return times, errors


def build_report(
    real: CurveTable | None,
    sim: CurveTable | None,
    joints: tuple[str, ...] | list[str],
    options: CurveOptions,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Build metrics and aligned time arrays used by both plot backends.

    【中文】组装误差报告与对齐后的时间轴，PNG 与 HTML 两种绘图后端共用这份结果。

    参数：
      real / sim —— 实机与仿真曲线表；至少给出一个，只给一侧时只做对齐、不算误差；
      joints     —— 要比较的关节名（顺序即绘图行序）；
      options    —— 对齐、换算、窗口等设置。
    返回：(report 字典, aligned 字典)；aligned 形如 {"real": [...], "sim": [...]}，单位 s。
    异常：
      两侧都为 None、关节列表为空、或某侧缺少所选关节的 position_rad 列时抛 ValueError。
    报告要点：误差约定 sim - real；tension_command_semantics 说明第三指令字段
              只是指令、并非 Fallback 实际执行的张力目标；anchor_s_from_first_row
              记录每一侧相对首行的锚点位置 [s]，便于人工复核对齐是否合理。
    """

    if real is None and sim is None:
        raise ValueError("at least one of real and sim must be provided")
    joints = tuple(joints)
    if not joints:
        raise ValueError("no joints selected")
    # 关节是否有角度列是能否比较的前提，先统一检查，避免后面逐通道静默缺席。
    for table in (real, sim):
        if table is not None:
            missing = set(joints) - set(table.joints)
            if missing:
                raise ValueError(f"{table.path}: missing position columns for {', '.join(sorted(missing))}")
    # ── 时间对齐：两侧各自锚到同一条指令跳变沿；仿真额外追加 sim_shift_s [s] ──
    aligned: dict[str, list[float]] = {}
    anchors: dict[str, float] = {}
    for name, table in (("real", real), ("sim", sim)):
        if table is not None:
            aligned[name], anchors[name] = aligned_time(table, options, joints, is_sim=name == "sim")
    # ── 报告骨架：来源文件、选项快照、锚点、误差约定与语义说明 ──
    report: dict[str, Any] = {
        "real_csv": str(real.path) if real is not None else None,
        "sim_csv": str(sim.path) if sim is not None else None,
        "options": asdict(options),
        "error_convention": "sim minus real; steady_* measures the final configured window, not detected settling",
        "tension_command_semantics": "third command field; not enforced by Fallback",
        "anchor_s_from_first_row": anchors,
        "joints": {},
        "motor_angle_semantics": "unwrapped winding-spool angle; no automatic zero subtraction or joint substitution",
        "derived_kinematics": "explicit backward differences; no smoothing; gaps excluded"
        if options.derive_kinematics
        else None,
        "missing_motor_columns": {
            name: [j for j in joints if table.values(j, "motor_position_rad") is None]
            for name, table in (("real", real), ("sim", sim))
            if table is not None
        },
    }
    # ── 逐关节 × 逐通道计算误差指标：仅当实机与仿真同时存在时才有意义 ──
    if real is not None and sim is not None:
        for joint in joints:
            # 每个通道 = (CSV 列后缀, 绘图/误差换算系数)：角度类乘 180/π 换算成 deg，
            # 张力类系数为 1.0（保持 N），速度/加速度保持 rad/s、rad/s² 等 SI 原始单位。
            fields = {
                "angle_deg": ("position_rad", 180.0 / math.pi),
                "tension_N": ("tension_N", 1.0),
                "distal_tension_N": ("distal_tension_N", 1.0),
                "routing_tension_drop_N": ("routing_tension_drop_N", 1.0),
                "command_angle_deg": ("cmd_rad", 180.0 / math.pi),
                "velocity_rad_s": ("velocity_rad_s", 1.0),
                "motor_angle_deg": ("motor_position_rad", 180.0 / math.pi),
                "motor_velocity_rad_s": ("motor_velocity_rad_s", 1.0),
                "acceleration_rad_s2": ("acceleration_rad_s2", 1.0),
                "motor_acceleration_rad_s2": ("motor_acceleration_rad_s2", 1.0),
                "command_speed_rad_s": ("cmd_speed_rad_s", 1.0),
                "command_tension_N": ("cmd_tension_N", 1.0),
            }
            # 逐通道计算结果：指令类通道（command_*）走零阶保持取样，连续量走线性插值，
            # 否则会给台阶指令造出从未下发过的中间目标。
            report["joints"][joint] = {
                name: compare_series(
                    aligned["real"],
                    channel_values(real, joint, suffix, options, is_real=True),
                    aligned["sim"],
                    channel_values(sim, joint, suffix, options),
                    scale=scale,
                    options=options,
                    interpolation="previous" if name.startswith("command_") else "linear",
                )
                for name, (suffix, scale) in fields.items()
            }
    return report, aligned


def _minimum_display_range(values, minimum_span=0.01):
    """Prevent numerical noise from filling an axis; never change samples.

    Angle plots use degrees and tension plots use N, so a 0.01 minimum span
    is a display convention in those units, not a sensor resolution or tolerance.
    Ordinary varying curves keep the plotting backend's autoscaling.

    【中文】为避免「几乎恒定的曲线」被数值噪声占满整个坐标轴，给出一个最小显示跨度。

    参数：
      values       —— 该坐标轴上所有曲线的纵坐标样本（单位随通道，如 deg 或 N）；
      minimum_span —— 期望的最小显示跨度（deg 或 N，默认 0.01）。
    返回：
      需要强制纵轴范围时返回 (下限, 上限)；曲线跨度已达到 minimum_span、或没有任何
      有限值时返回 None，表示交给绘图后端自动缩放。
    说明：本函数只影响显示范围，绝不修改、裁剪或重采样任何样本；0.01 这一数值是
          「角度用 deg、张力用 N」时的显示约定，不是传感器分辨率或容差。
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite or max(finite) - min(finite) >= minimum_span:
        return None
    center = (max(finite) + min(finite)) / 2
    return center - minimum_span / 2, center + minimum_span / 2


def _channel_display_range(key, values):
    """按通道类型决定是否套用最小显示跨度。

    参数：key —— CHANNELS 的通道键；values —— 该轴上的纵坐标样本。
    返回：(下限, 上限) 或 None（None 表示用自动缩放）。
    说明：只有角度与张力类通道使用 0.01 的显示下限（单位分别为 deg 与 N）；
          能量、接触状态等诊断量保持自然尺度，因为 0.01 对它们没有物理含义。
    """
    # Keep tiny energy/contact diagnostics on their natural scale. The 0.01
    # display floor only has a defined meaning for degrees and newtons.
    # 这里刻意不换算单位，只挑通道，0.01 的语义由上面 docstring 说明。
    span = 0.01 if key in ("angle", "motor", "tension", "distal_tension", "tension_drop") else 0.0
    return _minimum_display_range(values, minimum_span=span)


def _plot_channels(real, sim, joints, options):
    """筛出「至少在某一侧真实存在」的通道，避免画出一整列空白轴。

    参数：real / sim —— 曲线表（可为 None）；joints —— 关节名；options —— 提供 channels。
    返回：按 options.channels 原顺序保留的通道键列表。
    异常：所有选中的通道在输入里都不存在时抛 ValueError（此时画图没有意义）。
    说明：只做存在性判断，不修改数据、不改变通道顺序。
    """
    selected = [
        key
        for key in options.channels
        if any(
            channel_values(table, joint, CHANNELS[key][0], options, is_real=table is real) is not None
            for table in (real, sim)
            if table is not None
            for joint in joints
        )
    ]
    if not selected:
        raise ValueError("none of the selected channels exists in the CSV inputs")
    return selected


def _display_traces(real, sim, joint, key, aligned, options):
    """按固定顺序生成某个子图要画的曲线元组，供 PNG 与 HTML 后端共用。

    参数：
      real / sim —— 实机与仿真曲线表（可为 None）；
      joint      —— 当前关节；
      key        —— 当前通道键（CHANNELS 的键）；
      aligned    —— 对齐时间轴字典（"real"/"sim"，单位 s）；
      options    —— 提供 show_tension_command 等显示开关。
    逐条产出 (时间轴 [s], 纵坐标序列, 图例名, 颜色, 是否虚线)：
      1) 实机实测曲线（实线，#1f77b4）；
      2) 仿真预测曲线（实线，#ff7f0e）；
      3) 仅在「首选那一侧」画参考曲线：角度通道画指令目标角 [deg]，
         张力通道在 show_tension_command 打开时画第三指令字段。
    说明：第三指令字段只是被画出来作对照，它并不代表 Fallback 实际执行的张力目标；
          参考曲线统一用虚线，以免与实测曲线混淆。
    """
    suffix, scale, _ = CHANNELS[key]
    for label, table, color in (("real", real, "#1f77b4"), ("sim", sim, "#ff7f0e")):
        if table is None:
            continue
        values = channel_values(table, joint, suffix, options, is_real=label == "real")
        if values is not None:
            yield aligned[label], [v * scale for v in values], f"{label} {key}", color, False
        if label != ("real" if real is not None else "sim"):
            continue
        if key == "angle":
            command = table.values(joint, "cmd_rad")
            if command is not None:
                yield aligned[label], [math.degrees(v) for v in command], "target angle", "#555555", True
        if key == "tension" and options.show_tension_command:
            command = channel_values(table, joint, "cmd_tension_N", options, is_real=label == "real")
            if command is not None:
                yield aligned[label], command, "third command field (not enforced by Fallback)", "#d62728", True


def _plot_png(output, real, sim, joints, aligned, options):
    """用 matplotlib 输出静态 PNG 图。

    参数：output —— 输出路径；real / sim —— 曲线表；joints —— 关节名；
          aligned —— 对齐时间轴（单位 s）；options —— 提供 title 与 channels。
    布局：每行一个关节；只给一侧数据时每通道一列，两侧都有时每通道再追加一列残差图
          （残差 = sim − real，紫色曲线并带 0 参考线）。
    返回：无（图写入 output，使用 Agg 后端，不弹窗）。
    异常：未安装 matplotlib 时抛 RuntimeError。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PNG output") from exc
    keys = _plot_channels(real, sim, joints, options)
    comparison = real is not None and sim is not None
    columns = len(keys) * (2 if comparison else 1)
    fig, axes = plt.subplots(len(joints), columns, figsize=(5 * columns, max(3.5, 3.2 * len(joints))), squeeze=False)
    fig.suptitle(options.title)
    for row, joint in enumerate(joints):
        for col, key in enumerate(keys):
            suffix, scale, label = CHANNELS[key]
            ax = axes[row][col]
            for t, values, name, color, dashed in _display_traces(real, sim, joint, key, aligned, options):
                ax.plot(t, values, "--" if dashed else "-", color=color, label=name)
            ax.set(ylabel=f"{joint} {label}", xlabel="aligned time [s]")
            if ax.lines:
                ax.legend(loc="best")
            if comparison:
                residual = axes[row][col + len(keys)]
                rt, rv = residual_series(real, sim, joint, suffix, aligned, options, scale)
                residual.plot(rt, rv, color="tab:purple")
                residual.axhline(0, color="0.5", linewidth=0.5)
                residual.set(ylabel=f"sim − real {label}", xlabel="aligned time [s]")
        for col, ax in enumerate(axes[row]):
            ax.grid(alpha=0.25)
            display_range = _channel_display_range(
                keys[col % len(keys)], (value for line in ax.lines for value in line.get_ydata())
            )
            if display_range is not None:
                ax.set_ylim(display_range)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_html(output, real, sim, joints, aligned, options):
    """用 plotly 输出单文件、可交互、可分享的 HTML 图。

    参数：output —— 输出路径；real / sim —— 曲线表；joints —— 关节名；
          aligned —— 对齐时间轴（单位 s）；options —— 提供 title 与 channels。
    布局与 PNG 版本保持一致：每行一个关节、每通道一列，比较模式下再加一列残差
          （残差 = sim − real）；同名列共用一个图例组，只在第一行显示图例。
    返回：无（HTML 写入 output，内嵌 plotly.js 以便离线打开）。
    异常：未安装 plotly 时抛 RuntimeError。
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("plotly is required for HTML output") from exc
    keys = _plot_channels(real, sim, joints, options)
    comparison = real is not None and sim is not None
    columns = len(keys) * (2 if comparison else 1)
    titles = [CHANNELS[key][2] for key in keys]
    if comparison:
        titles += [f"{CHANNELS[key][2]} residual" for key in keys]
    fig = make_subplots(
        rows=len(joints),
        cols=columns,
        subplot_titles=tuple(f"{joint} {title}" for joint in joints for title in titles),
        shared_xaxes=True,
        horizontal_spacing=min(0.08, 0.5 / columns),
        vertical_spacing=min(0.08, 0.35 / len(joints)),
    )
    for row, joint in enumerate(joints, start=1):
        for col, key in enumerate(keys, start=1):
            suffix, scale, label = CHANNELS[key]
            for t, values, name, color, dashed in _display_traces(real, sim, joint, key, aligned, options):
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=values,
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=row == 1,
                        line=dict(color=color, dash="dash" if dashed else "solid"),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_yaxes(title_text=label, row=row, col=col)
            if comparison:
                rt, rv = residual_series(real, sim, joint, suffix, aligned, options, scale)
                fig.add_trace(
                    go.Scatter(x=rt, y=rv, mode="lines", name=f"residual {key}", showlegend=row == 1),
                    row=row,
                    col=col + len(keys),
                )
                fig.update_yaxes(title_text=f"sim − real {label}", row=row, col=col + len(keys))
        for col in range(1, columns + 1):
            fig.update_xaxes(title_text="aligned time [s]", row=row, col=col)
            display_range = _channel_display_range(
                keys[(col - 1) % len(keys)],
                (value for trace in fig.select_traces(row=row, col=col) for value in trace.y),
            )
            if display_range is not None:
                fig.update_yaxes(range=list(display_range), row=row, col=col)
    fig.update_layout(title=options.title, height=max(430, 310 * len(joints)), hovermode="x unified")
    fig.write_html(str(output), include_plotlyjs=True, full_html=True)


def plot_curves(
    output: str | Path,
    real: CurveTable | None,
    sim: CurveTable | None,
    joints: tuple[str, ...] | list[str],
    aligned: dict[str, list[float]],
    options: CurveOptions,
) -> None:
    """Write a shareable PNG or self-contained interactive HTML figure.

    【中文】按输出文件后缀选择绘图后端，写出可直接分享的图。

    参数：output —— 输出路径，必须以 .png 或 .html 结尾；real / sim —— 曲线表（可为 None）；
          joints —— 关节名；aligned —— build_report 产出的对齐时间轴 [s]；options —— 显示设置。
    返回：无。输出目录不存在时会先创建（父目录一并创建）。
    异常：后缀不是 .png/.html 时抛 ValueError。
    """

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joints = tuple(joints)
    if output.suffix.lower() == ".png":
        _plot_png(output, real, sim, joints, aligned, options)
    elif output.suffix.lower() == ".html":
        _plot_html(output, real, sim, joints, aligned, options)
    else:
        raise ValueError("--out must end in .png or .html")


def _parser() -> argparse.ArgumentParser:
    """构建命令行解析器（模块 docstring 直接作为 description 使用）。

    返回：已注册全部参数的 ArgumentParser。
    参数语义：
      --real / --sim  两侧 CSV（至少给一个）；
      --joints        要比较的关节，默认取 TARGET_JOINTS 中两侧共有的部分；
      --channels      要画/比较的通道，取值来自 CHANNELS；
      --config        可选 YAML（含 curves: 段），命令行显式值优先；
      --out / --report 图与指标 JSON 的输出路径；
      --align / --alignment-joint / --sim-shift-s [s] / --steady-window-s [s] /
      --max-interp-gap-s [s]  时间对齐与统计窗口；
      --tension-raw-to-N [N/计数] / --real-tension-bias-N [N]  张力换算与零偏；
      --fallback-only / --show-tension-command  样本过滤与张力指令显示。
    注意：布尔型可选参数默认值为 None，以便区分「命令行没给」与「显式给 False」，
          从而让 YAML 配置能被命令行动作正确覆盖。
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── 参数注册：帮助字符串保持原文，便于与 replay 参数对照 ──
    parser.add_argument("--real", type=Path, help="ROS benchmark or another real-hand CSV")
    parser.add_argument("--sim", type=Path, help="real2sim replay CSV")
    parser.add_argument("--joints", nargs="+", help="joint names to plot; default: the 10 target tendon joints")
    parser.add_argument(
        "--channels", nargs="+", choices=tuple(CHANNELS), help="default: angle motor tension distal_tension"
    )
    parser.add_argument(
        "--derive-kinematics",
        action="store_true",
        default=None,
        help="explicitly derive missing velocities/accelerations; finite differences amplify noise",
    )
    parser.add_argument("--config", type=Path, help="YAML file with a curves: section")
    parser.add_argument("--out", type=Path, required=True, help="output .png or self-contained .html")
    parser.add_argument("--report", type=Path, help="metrics JSON; default: same stem as --out")
    parser.add_argument("--align", choices=("command_edge", "start"), help="alignment mode")
    parser.add_argument("--alignment-joint", help="joint whose command edge anchors both traces")
    parser.add_argument("--sim-shift-s", type=float, help="additional simulator time shift [s]")
    # ── 张力换算：本脚本用大写 N（--tension-raw-to-N），回放脚本用 --tension-raw-to-n；
    #    两者拼写不同但含义相同（原始刻度 → N 的系数 [N/计数]），命令行走不同入口时不要混用。
    parser.add_argument(
        "--tension-raw-to-N", type=float, help="conversion for <joint>/effort_raw; latest firmware uses 0.01"
    )
    parser.add_argument(
        "--real-tension-bias-N", type=float, help="known sensor offset to subtract from real feedback [N]"
    )
    parser.add_argument(
        "--fallback-only", action="store_true", default=None, help="exclude samples outside recorded Fallback mode"
    )
    parser.add_argument(
        "--show-tension-command",
        action="store_true",
        default=None,
        help="plot the third command field (not an enforced Fallback setpoint)",
    )
    parser.add_argument("--steady-window-s", type=float, help="tail window for steady error [s]")
    parser.add_argument("--max-interp-gap-s", type=float, help="do not interpolate across a longer simulator gap")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析参数 → 读取两侧 CSV → 对齐并算指标 → 画图 → 写 JSON 报告。

    参数：argv —— 参数列表；None 表示取 sys.argv[1:]，便于在测试中直接传入。
    返回：0 表示成功。
    异常/退出码：参数缺失（--real 与 --sim 都为 None）由 argparse 报错；
          文件/取值/依赖类错误（OSError、ValueError、RuntimeError）统一转成退出码 2，
          并在 stderr 打印 ``real2sim_curves: <原因>``。
    顺序说明：先建报告（含对齐）再画图，二者共用同一份对齐时间轴，避免图与报告不一致；
          报告默认写到与 --out 同名的 .json，可用 --report 改写。
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.real is None and args.sim is None:
        parser.error("provide --real, --sim, or both")
    try:
        # ── 选项合并：YAML 提供基线，命令行显式值覆盖（None 项不参与） ──
        options = load_options(
            args.config,
            alignment=args.align,
            alignment_joint=args.alignment_joint,
            sim_shift_s=args.sim_shift_s,
            tension_raw_to_N=args.tension_raw_to_N,
            steady_window_s=args.steady_window_s,
            real_tension_bias_N=args.real_tension_bias_N,
            fallback_only=args.fallback_only,
            show_tension_command=args.show_tension_command,
            max_interp_gap_s=args.max_interp_gap_s,
            channels=args.channels,
            derive_kinematics=args.derive_kinematics,
        )
        real = read_curve(args.real) if args.real else None
        sim = read_curve(args.sim) if args.sim else None
        # ── 默认关节：优先用 10 个目标关节中两侧共有者，否则退化为两侧共有的全部关节 ──
        available = (set(real.joints) if real else set(sim.joints)) & (set(sim.joints) if sim else set(real.joints))
        default_joints = tuple(joint for joint in TARGET_JOINTS if joint in available) or tuple(sorted(available))
        joints = tuple(args.joints or default_joints)
        report, aligned = build_report(real, sim, joints, options)
        plot_curves(args.out, real, sim, joints, aligned, options)
        # ── 报告落盘：默认与图同名，allow_nan=False 保证 JSON 严格合法（NaN 会直接报错）──
        report_path = args.report or args.out.with_suffix(".json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"real2sim_curves: {exc}\n")
    print(f"plot: {args.out}\nmetrics: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
