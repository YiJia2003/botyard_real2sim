"""Read sampled joint commands for unloaded real-to-simulation replay.

ROS benchmark captures contain a command *state* sampled alongside telemetry,
not a log of individual command receipt events.  A consumer should detect
changes before treating a row as a new firmware command, particularly when
the firmware's same-finger autopair operation is enabled.

【中文说明】
文件职责：
    读取 ROS 采集 CSV 中的关节指令与初态反馈，供“空载（unloaded）
    real-to-sim 回放”使用。本模块只做解析、单位归一与合法性校验，
    不执行控制、不做物理仿真、不访问硬件。

输入/输出：
    - 输入：CSV 文件路径（由 ROS benchmark 采集导出），时间列可以是
      ``timestamp_ns`` 或 ``time_s``，指令列形如 ``<关节>/cmd_rad``。
    - 输出：:class:`JointCommand` / :class:`CommandFrame` /
      :class:`RecordedInitialState` 等 Python 对象；时间统一以
      **采集首帧时间戳**为时间基准（零点）。

关键假设（务必注意）：
    1. 采集里记录的是“指令状态”（与遥测一起被采样），**不是**逐条指令的
       接收日志；同一目标可能在连续多行重复出现，因此消费者必须先判断
       变化，再把某一行当成一次新的固件指令（见 ``is_event``）。
    2. 固件的“同指补齐”（same-finger autopair）启用时，这一区分尤其重要：
       一次指令里只选了指对中的一个，固件会立刻为另一个补上目标。
"""

from __future__ import annotations

import csv
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .config import JOINT_NAMES


# ── 数据结构：单个关节的指令三元组 ──
@dataclass(frozen=True, slots=True)
class JointCommand:
    """Angle [rad], speed bound [rad/s], third command field [N].

    Fallback records the third field but does not enforce a tension bound.

    【中文】
    单个关节的三字段指令。

    参数（字段）：
        angle_rad:         目标关节角，单位 rad。
        speed_rad_s:       速度上限，单位 rad/s（须 ≥ 0）。
        tension_command_n: 第三字段张力指令值，单位 N（须 ≥ 0）。

    说明：
        Fallback 固件会记录第三字段，但**不**用它执行张力上限；
        取绝对值与限幅发生在 H7 侧，本类只负责携带与校验。
    """

    # 目标关节角，单位 rad。
    angle_rad: float
    # 速度上限，单位 rad/s（非负）。
    speed_rad_s: float
    # 第三字段张力指令值，单位 N（非负；不是实测张力）。
    tension_command_n: float

    @property
    def tension_limit_n(self) -> float:
        """Legacy name; the Fallback position loop does not enforce this field.

        【中文】
        历史命名别名，仅为兼容旧调用点保留。

        返回：
            与 :attr:`tension_command_n` 相同的值，单位 N。

        说明：
            Fallback 位置环**不**执行该字段，它不构成张力上限。
        """
        return self.tension_command_n

    def __post_init__(self) -> None:
        """校验三个字段有限，且速度上限/第三字段非负。

        【中文】
        参数：无（读取已赋值的三个字段）。
        返回：None。
        异常：
            ValueError: 任一字段非有限，或速度上限、第三字段为负。
        说明：
            负的速度上限与负的张力指令在本模型中无物理意义，
            因此直接拒绝，而不是取绝对值后继续。
        """

        if not all(math.isfinite(value) for value in (self.angle_rad, self.speed_rad_s, self.tension_command_n)):
            raise ValueError("joint command values must be finite")
        if self.speed_rad_s < 0 or self.tension_command_n < 0:
            raise ValueError("speed and tension command must be nonnegative")


# ── 数据结构：一帧指令（事件或采样状态） ──
@dataclass(frozen=True, slots=True)
class CommandFrame:
    """Command event or sampled state, timed from the capture's first timestamp.

    is_event=True preserves repeated sends, including identical target triples.
    is_event=False must not be mistaken for a receipt event at every sample.

    An empty mapping means that none of the selected joints has a valid target
    in this row.  A missing joint in a nonempty mapping likewise means it is
    not selected by this row; the caller decides how to retain prior targets.

    【中文】
    一帧指令：既可能是“一次真实的指令接收事件”，也可能只是“某一采样时刻的
    指令状态”，由 ``is_event`` 区分。时间以**采集首帧时间戳**为基准，单位 s。

    字段：
        time_s:      相对采集首帧的时间，单位 s。
        commands:    关节名 -> :class:`JointCommand` 的映射。
        is_event:    True 表示一次真实的指令接收事件，会保留重复下发
                     （包括完全相同的目标三元组）；False 表示只是采样到的
                     状态，**不可**理解为每个采样点都收到了一条指令。
        sequence:    采集到的指令序号（若有），用于识别重复行。
        dof_mask:    该行的 DOF 掩码（若有），位序与协议顺序一致。
        is_fallback: 该行记录的固件是否处于 Fallback 模式；None 表示未知。

    说明：
        空映射表示本行没有任何被选中关节的有效目标；非空映射中缺少某个关节，
        表示本行没有选中它（而不是选中且目标为 0）。是否沿用上一次的目标，
        由调用方决定。
    """

    # 相对采集首帧的时间，单位 s。
    time_s: float
    # 本帧携带的关节目标；空字典表示本行没有有效目标。
    commands: dict[str, JointCommand]
    # 是否为真实的指令接收事件（False = 仅采样到的指令状态）。
    is_event: bool = False
    # 指令序号，None 表示采集未提供。
    sequence: int | None = None
    # DOF 掩码，None 表示采集未提供（即未知，不能当成“全部选中”）。
    dof_mask: int | None = None
    # 固件是否处于 Fallback 模式，None 表示未知。
    is_fallback: bool | None = None


# ── 数据结构：初态反馈 ──
@dataclass(frozen=True, slots=True)
class RecordedInitialState:
    """Feedback from one sample; missing channels are explicit, never filled later.

    【中文】
    某一采样时刻的反馈（初态）：关节角、张力、电机量与速度。

    字段：
        time_s:                 相对采集首帧的时间，单位 s。
        positions_rad:          关节角，单位 rad。
        tensions_n:             张力，单位 N。
        missing_positions:      缺失的关节角通道名（**显式记录**，不当 0）。
        missing_tensions:       缺失的张力通道名（**显式记录**，不当 0）。
        motor_positions_rad:    电机侧位置，单位 rad。
        velocities_rad_s:       关节速度，单位 rad/s。
        motor_velocities_rad_s: 电机侧速度，单位 rad/s。

    说明：
        缺失通道必须保持“缺失”状态，绝不填 0、也不事后插值补齐：
        把未知当成 0 会伪造出物理上并不存在的初态。
    """

    # 相对采集首帧的时间，单位 s。
    time_s: float
    # 关节角，单位 rad；缺失的关节记录在 missing_positions。
    positions_rad: dict[str, float]
    # 张力，单位 N；缺失的关节记录在 missing_tensions。
    tensions_n: dict[str, float]
    # 缺失的关节角通道名，显式保留以免被误当成 0 rad。
    missing_positions: tuple[str, ...]
    # 缺失的张力通道名，显式保留以免被误当成 0 N。
    missing_tensions: tuple[str, ...]
    # 电机侧位置，单位 rad（缺失项不出现在字典里）。
    motor_positions_rad: dict[str, float] = field(default_factory=dict)
    # 关节速度，单位 rad/s（缺失项不出现在字典里）。
    velocities_rad_s: dict[str, float] = field(default_factory=dict)
    # 电机侧速度，单位 rad/s（缺失项不出现在字典里）。
    motor_velocities_rad_s: dict[str, float] = field(default_factory=dict)


# ── 通用取值辅助 ──
def _first_number(row, columns):
    """按候选列名顺序取第一个可解析的数值。

    【中文】
    参数：
        row:     CSV 的一行（dict，值可能为 None 或字符串）。
        columns: 候选列名序列，按优先级从高到低。

    返回：
        float，或 None（所有候选列都缺失/空白/NaN 时）。

    说明：
        用生成器配合海象运算符，命中第一个有效值即停止，避免重复解析；
        "SI 列优先于 raw 列"就是靠这里的列顺序实现的。
    """
    return next((value for column in columns if (value := _optional_number(row, column)) is not None), None)


def _boolean(row, columns):
    """按候选列名顺序取第一个可解析的布尔值。

    【中文】
    参数：
        row:     CSV 的一行。
        columns: 候选列名序列，按优先级从高到低。

    返回：
        True / False / None（None 表示所有候选列都缺失或为空）。

    异常：
        ValueError: 某列有值但既不是真值字面量也不是假值字面量。

    说明：
        空白与 "nan" 视为“未提供”，而不是 False；
        非法字样必须报错，避免把未知模式静默当成 Fallback。
    """
    for key in columns:
        raw = row.get(key)
        # 缺失、空白或 "nan" 都表示该列没有可用信息，继续看下一个候选列。
        if raw is None or str(raw).strip().lower() in ("", "nan"):
            continue
        token = str(raw).strip().lower()
        # 真值字面量：true / 1 / 1.0。
        if token in ("true", "1", "1.0"):
            return True
        # 假值字面量：false / 0 / 0.0。
        if token in ("false", "0", "0.0"):
            return False
        raise ValueError(f"invalid boolean {raw!r} for {key}")
    return None


def validate_fallback_capture(path: str | Path, joint_names: Sequence[str] = JOINT_NAMES) -> dict:
    """Check every capture row before command-event deduplication.

    Explicit global false always rejects. A joint flag applies only while that
    joint is active: explicit <joint>/active (or /is_active) takes precedence
    over dof_mask/command_mask. Without activity evidence selected joints are
    relevant. Missing/blank/NaN branch flags are unavailable, not assumed true.
    This verifies recorded flags only; it cannot certify unrecorded firmware mode.

    【中文】
    在“指令事件去重”之前，先逐行确认这段采集确实是 Fallback 模式。
    任何一处明确为 false 都直接抛错拒绝；只有“未记录”才降级为部分可用。

    参数：
        path:        CSV 文件路径。
        joint_names: 需要检查的关节名序列，默认本模型十个屈曲 DOF。

    返回：
        dict，包含：
            status:                      validated / partial / unavailable
            rows_checked:                已检查的 CSV 数据行数
            eligible_flags_checked:      实际生效并被检查的标志数
            rows_without_eligible_flags: 没有任何可用标志的行数
            inactive_flags_ignored:      因关节非活动而被忽略的标志数

    异常：
        ValueError: joint_names 为空或重复；任一行显式声明 is_fallback=false；
                    某行的布尔/整数列内容非法。
        OSError:    文件无法打开。

    判定规则（逐条）：
        1. 全局 is_fallback=false 一律拒绝，与掩码无关。
        2. 关节级 <joint>/is_fallback 只在该关节“活动”时生效；显式的
           <joint>/active（或 /is_active）优先于 dof_mask/command_mask 的推断。
        3. 没有活动性证据时，假定“被选中的关节都相关”。
        4. 缺失/空白/NaN 的分支标志属于“不可用”，**不**假定为 true。

    说明：
        这里只校验**记录下来的**标志；没有记录下来的固件模式无法被证明。
    """
    names = tuple(joint_names)
    # 关节名必须非空且不重复，否则“名字 <-> 掩码位”的对应关系不成立。
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    # 计数器：检查行数 / 生效标志数 / 无有效标志的行数 / 被忽略的非活动标志数。
    rows_checked = eligible_flags = unavailable_rows = ignored_flags = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        # start=2 让行号与文本编辑器里的物理行号一致（第 1 行是表头）。
        for line, row in enumerate(csv.DictReader(stream), start=2):
            rows_checked += 1
            checked_in_row = 0
            # ── 全局模式标志：显式 false 一律拒绝 ──
            global_flag = _boolean(row, ("is_fallback",))
            if global_flag is False:
                raise ValueError(f"non-Fallback firmware mode at CSV row {line}: is_fallback=false")
            if global_flag is True:
                checked_in_row += 1
            # Missing masks often occur on telemetry-only rows in mixed logs.
            # 混合日志里“只有遥测”的行常常缺掩码；先把空白/NaN 过滤掉再解析，
            # 否则空白会被当成 0 而误判为“没有选中任何关节”。
            mask_row = {
                key: value
                for key, value in row.items()
                if value is not None and str(value).strip().lower() not in ("", "nan")
            }
            mask = _integer(mask_row, ("dof_mask", "command_mask"))
            # ── 逐关节标志：只有该关节“活动”时才算证据 ──
            for name in names:
                flag = _boolean(row, (f"{name}/is_fallback",))
                active = _boolean(row, (f"{name}/active", f"{name}/is_active"))
                # 显式 <joint>/active 优先；没有时用 dof_mask 推断该关节是否被选中。
                if active is None and mask is not None and name in JOINT_NAMES:
                    active = bool(mask & (1 << JOINT_NAMES.index(name)))
                # 未记录的标志属于“不可用”，不能假定为 true，直接跳过。
                if flag is None:
                    continue
                # 非活动关节上的 false 不构成证据：记一次忽略后跳过。
                if active is False:
                    ignored_flags += 1
                    continue
                # 活动关节上的显式 false 说明这段记录不是 Fallback，必须拒绝。
                if flag is False:
                    raise ValueError(f"non-Fallback firmware mode at CSV row {line}: {name}/is_fallback=false")
                checked_in_row += 1
            eligible_flags += checked_in_row
            # 本行没有任何可用标志 -> 计入“无有效标志行”。
            unavailable_rows += checked_in_row == 0
    # 结论分级：所有行都有可用标志 -> validated；部分可用 -> partial；
    # 完全没有可用标志 -> unavailable。
    return {
        "status": "validated"
        if rows_checked and not unavailable_rows
        else ("partial" if eligible_flags else "unavailable"),
        "rows_checked": rows_checked,
        "eligible_flags_checked": eligible_flags,
        "rows_without_eligible_flags": unavailable_rows,
        "inactive_flags_ignored": ignored_flags,
    }


def _integer(row, columns):
    """按候选列名顺序取第一个可解析的非负整数。

    【中文】
    参数：
        row:     CSV 的一行。
        columns: 候选列名序列（如 dof_mask、command_mask），按优先级排列。

    返回：
        int，或 None（所有候选列都缺失/为空时）。

    异常：
        ValueError: 内容无法解析，或解析结果为负数。

    说明：
        支持 "0x" 前缀的十六进制写法（如掩码 "0x3ff"）；
        负数没有意义（掩码位与序号都不会为负），因此直接拒绝。
    """
    # 先按列名优先级取出第一个非空的原始字符串。
    raw = next((row.get(key) for key in columns if row.get(key) not in (None, "")), None)
    if raw is None:
        return None
    try:
        # 带 0x 前缀时按十六进制解析，其余按十进制。
        value = int(raw, 0) if str(raw).lower().startswith("0x") else int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid integer {raw!r} for {columns}") from exc
    if value < 0:
        raise ValueError(f"negative integer for {columns}")
    return value


def _tension(row, name, *, command, raw_to_n):
    """读取张力字段：优先 SI 单位列，其次 raw 列并按比例换算成 N。

    【中文】
    参数：
        row:      CSV 的一行。
        name:     关节名，用于拼出列名。
        command:  True 读“指令”相关列（cmd_* 前缀），False 读反馈相关列。
        raw_to_n: raw 量纲 -> N 的换算比例，默认 0.01 N/count。

    返回：
        float（单位 N），或 None（SI 列与 raw 列都缺失/为空时）。

    说明：
        SI 列优先于 raw 列：只要存在 N 单位的列就不再使用 raw 换算，
        以免两种来源被混用。
        对指令而言，第三字段只是“指令信息”，不是实测张力，
        因此这里特意不接受反馈侧的 effort_raw/tension_raw 作为指令来源。
    """
    prefix = "cmd_" if command else ""  # 指令列带 cmd_ 前缀，反馈列没有
    # ── 第一优先级：已经是 N 单位的列 ──
    si = [
        f"{name}/{prefix}{suffix}"
        for suffix in (
            ("tension_N", "tension_command_N", "tension_command_n", "tension_limit_N", "tension_limit_n")
            if command
            else ("tension_N", "tension_n")
        )
    ]
    value = _first_number(row, si)
    if value is not None:
        return value
    # ── 第二优先级：raw 计数列，需要乘换算比例 ──
    raw = _first_number(
        row,
        [
            f"{name}/{suffix}"
            for suffix in (
                ("cmd_tension_raw", "cmd_tension_command_raw", "cmd_tension_limit_raw")
                if command
                else ("effort_raw", "tension_raw")
            )
        ],
    )
    # raw 缺失时如实返回 None（表示未知），而不是返回 0。
    return None if raw is None else raw * raw_to_n


def load_initial_state(path, *, tension_raw_to_n=0.01, joint_names=JOINT_NAMES):
    """Return the earliest valid position sample, normalized to capture start.

    Tensions and positions always come from that same row. A partial sample is
    returned with missing channel names; no valid position sample raises ValueError.

    【中文】
    返回**最早一条含有效位置的行**作为初态，时间归一化到采集起点。

    参数：
        path:            CSV 文件路径。
        tension_raw_to_n: raw 张力计数 -> N 的换算比例，须有限且 > 0，默认 0.01 N。
        joint_names:     需要读取的关节名序列，默认本模型十个屈曲 DOF。

    返回：
        :class:`RecordedInitialState`。角度/张力/电机量都取自**同一行**，
        以保证初态在时间上自洽；不完整的通道以 missing_* 元组显式记录。

    异常：
        ValueError: 换算比例非法；joint_names 为空或重复；整份 CSV 都没有
                    任何有效位置样本。
        OSError:    文件无法打开。

    说明：
        行按 (时间戳, 原始行号) 稳定排序，因此同一时刻的多行不会被随机打乱。
        先扫全表排序、再挑第一行有位置的行：即使 CSV 行序不是时间序也正确。
    """
    if not math.isfinite(tension_raw_to_n) or tension_raw_to_n <= 0:
        raise ValueError("tension_raw_to_n must be finite and positive")
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        # 时间列优先用纳秒整数 timestamp_ns，否则退回浮点秒 time_s。
        key = "timestamp_ns" if "timestamp_ns" in (reader.fieldnames or ()) else "time_s"
        # 展开成 (时间戳, 行号, 行内容) 后按时间稳定排序，行号作为同刻次序。
        rows = sorted(((_timestamp(row, key), i, row) for i, row in enumerate(reader)), key=lambda x: (x[0], x[1]))
    for stamp, _, row in rows:
        # 该行的有效关节角（单位 rad）；缺失的关节不进字典。
        positions = {
            name: value for name in names if (value := _optional_number(row, f"{name}/position_rad")) is not None
        }
        # 整行没有任何有效位置 -> 不是可用的初态行，继续看下一行。
        if not positions:
            continue
        # 张力必须与位置取自同一行，避免把不同时刻的状态拼在一起。
        tensions = {
            name: value
            for name in names
            if (value := _tension(row, name, command=False, raw_to_n=tension_raw_to_n)) is not None
        }
        # 组装初态：位置、张力、缺失通道名、以及可选的电机量/速度。
        # 时间零点 = 排序后第一行的时间戳（ns 列换算成 s）；
        # 缺失项显式记录为通道名，绝不当 0 处理。
        return RecordedInitialState(
            (stamp - rows[0][0]) * (1e-9 if key == "timestamp_ns" else 1),
            positions,
            tensions,
            tuple(name for name in names if name not in positions),
            tuple(name for name in names if name not in tensions),
            {name: v for name in names if (v := _optional_number(row, f"{name}/motor_position_rad")) is not None},
            {name: v for name in names if (v := _optional_number(row, f"{name}/velocity_rad_s")) is not None},
            {name: v for name in names if (v := _optional_number(row, f"{name}/motor_velocity_rad_s")) is not None},
        )
    # 全表都没有一条含有效位置的行：无法给出初态，直接报错。
    raise ValueError("CSV has no valid selected initial position sample")


def _optional_number(row: dict[str, str | None], column: str) -> float | None:
    """解析可为空的数值单元格。

    【中文】
    参数：
        row:    CSV 的一行。
        column: 列名。

    返回：
        float，或 None（单元格缺失/空白/NaN）。

    异常：
        ValueError: 内容不是数字，或为 ±inf 之类的非有限值。

    说明：
        空值与 NaN 都表示“没有测到”，因此返回 None；
        但 ±inf 是明确的异常数据，必须报错而不是当成缺失。
    """
    raw = row.get(column)
    # 列不存在或内容全为空白 -> 视为未提供。
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value in {column}: {raw!r}") from exc
    # NaN 表示“没有数值”，按缺失处理。
    if math.isnan(value):
        return None
    # ±inf 是非法数据，不能静默当作缺失。
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric value in {column}: {raw!r}")
    return value


def _timestamp(row: dict[str, str | None], column: str) -> int | float:
    """解析时间戳单元格。

    【中文】
    参数：
        row:    CSV 的一行。
        column: 时间列名；以 ``_ns`` 结尾时按整数纳秒解析。

    返回：
        int（纳秒列）或 float（秒列）。调用方负责把纳秒换算成 s。

    异常：
        ValueError: 时间戳缺失、无法解析，或为 ±inf。

    说明：
        返回值保留原始量纲（ns 或 s），单位换算交给使用处显式完成，
        这样代码里能直接看出在哪一步做了 1e-9 的缩放。
    """
    raw = row.get(column)
    if raw is None or not raw.strip():
        raise ValueError(f"missing timestamp in {column}")
    try:
        # 纳秒列必须是整数计数；秒列允许浮点。
        value = int(raw) if column.endswith("_ns") else float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp in {column}: {raw!r}") from exc
    # ±inf 会让后续排序与归一化失效，必须拒绝。
    if not math.isfinite(value):
        raise ValueError(f"non-finite timestamp in {column}: {raw!r}")
    return value


def load_command_frames(
    path: str | Path,
    *,
    joint_names: Sequence[str] = JOINT_NAMES,
    default_speed_rad_s: float | None = 1.5708,
    default_tension_limit_n: float | None = 20.0,
    default_tension_command_n: float | None = None,
    strict: bool = False,
    event_mode: str = "auto",
    tension_raw_to_n: float = 0.01,
) -> list[CommandFrame]:
    """Load sampled states or explicit send events, retaining equal-time order.

    Canonical triple columns: cmd_rad, cmd_speed_rad_s, cmd_tension_N (or
    cmd_tension_command_N). Legacy cmd_tension_limit_n/N/raw remain accepted.
    SI takes precedence over raw; feedback effort_raw is never a command.
    strict=True rejects incomplete triples. Otherwise configured defaults are
    used with a warning; None defaults also reject missing fields.

    Auto mode uses command_event/is_event or command_seq/sequence as event
    evidence. A mask alone does not imply a send. False event rows are skipped;
    repeated sampled sequence IDs collapse to one event unless the row explicitly
    says it is an event. Different sequences preserve identical triple sends.
    command_timestamp_ns/command_time_s can supply actual send times; otherwise
    row timestamps are the available time evidence. Mask bits use protocol order,
    and filter cached unselected joint targets. A missing mask remains unknown.

    【中文】
    读取“采样到的指令状态”或“显式的下发事件”，并保持同一时刻的原始行序。

    参数：
        path:                     CSV 文件路径。
        joint_names:              需要读取的关节名序列，默认十个屈曲 DOF。
        default_speed_rad_s:      速度列缺失时使用的默认速度上限，单位 rad/s
                                  （默认 1.5708 rad/s ≈ 90 deg/s）；None 表示不给默认值。
        default_tension_limit_n:  旧名，第三字段缺失时的默认值，单位 N（默认 20 N）。
        default_tension_command_n: 第三字段缺失时的默认值；给出时优先于上面那个旧名。
        strict:                   True 时缺少任一字段就报错，不使用默认值。
        event_mode:               auto / events / sampled 三种事件判定方式。
        tension_raw_to_n:         raw 张力计数 -> N 的换算比例，须有限且 > 0。

    返回：
        list[CommandFrame]，时间为相对**采集首帧时间戳**的秒数；无数据时返回空列表。

    异常：
        ValueError: 参数非法；CSV 缺少时间列或 ``<关节>/cmd_rad`` 列；
                    strict 模式下缺字段；事件时间早于采集起点。

    列名约定：
        规范三元组为 cmd_rad、cmd_speed_rad_s、cmd_tension_N（或
        cmd_tension_command_N）；旧的 cmd_tension_limit_n/N/raw 仍被接受。
        SI 单位列优先于 raw 列；反馈侧的 effort_raw **绝不**当作指令。

    事件判定（event_mode）：
        auto     —— 行里出现 command_event/is_event 或 command_seq/sequence
                    就按事件处理，否则按采样状态处理。
        events   —— 一律按事件处理。
        sampled  —— 一律按采样状态处理（每个采样点都可能重复同一目标）。
        仅有掩码**不**意味着发生过一次下发。

    去重规则：
        is_event 为 False 的行被跳过；采样行里重复出现的 sequence 归并成一次事件，
        除非该行显式声明自己是一次事件；不同 sequence 即使三元组完全相同也保留。

    时间来源：
        command_timestamp_ns / command_time_s 可给出真实下发时刻，
        缺失时只能用行时间戳作为时间证据。

    掩码：
        位序与协议一致（DOF 0 为最低位）；掩码会过滤掉“未选中关节的缓存目标”；
        掩码缺失时保持“未知”，不做任何推断。
    """

    # ── 参数校验 ──
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    if event_mode not in {"auto", "events", "sampled"}:
        raise ValueError("event_mode must be auto, events or sampled")
    # 第三字段默认值：新名优先，旧名兜底（二者都是 N）。
    tension_default = default_tension_command_n if default_tension_command_n is not None else default_tension_limit_n
    for value in (default_speed_rad_s, tension_default):
        # 默认值只允许 None（表示“无默认，缺字段即报错”）或有限非负数。
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("defaults must be finite and nonnegative")
    # 收集“用了默认值”的字段名，最后统一发一次告警，避免逐行刷屏。
    missing_fields = set()
    if not math.isfinite(tension_raw_to_n) or tension_raw_to_n <= 0:
        raise ValueError("tension_raw_to_n must be finite and positive")

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = set(reader.fieldnames or ())
        # ── 时间列与活动关节列 ──
        # 纳秒列优先；两者都没有则无法建立统一时间轴，直接报错。
        time_column = "timestamp_ns" if "timestamp_ns" in header else "time_s"
        if time_column not in header:
            raise ValueError("CSV needs timestamp_ns or time_s")
        # 只有存在 <关节>/cmd_rad 列的关节才被认为“本段采集里活动”。
        active_names = tuple(name for name in names if f"{name}/cmd_rad" in header)
        if not active_names:
            raise ValueError("CSV has no selected <joint>/cmd_rad columns")

        samples = []
        capture_start = None  # 采集首帧时间戳（所有事件时间的公共零点）
        seen_sequences = set()  # 已处理过的采样序号，用于把重复采样归并为一次事件
        for row_index, row in enumerate(reader):
            stamp = _timestamp(row, time_column)
            # 逐行取最小值：即使行序不是时间序，也能得到真正的采集起点。
            capture_start = stamp if capture_start is None else min(capture_start, stamp)
            commands: dict[str, JointCommand] = {}
            event_flag = _boolean(row, ("command_event", "is_event"))
            sequence = _integer(row, ("command_seq", "sequence"))
            mask = _integer(row, ("dof_mask", "command_mask"))
            # ── 事件判定 ──
            # events 模式强行按事件；auto 模式只在出现事件证据（标志或序号）时才算事件；
            # sampled 模式一律按采样状态，仅有掩码不算“发生过一次下发”。
            is_event = event_mode == "events" or (
                event_mode == "auto" and (event_flag is not None or sequence is not None)
            )
            # 显式标记为非事件的行直接跳过，不产生任何帧。
            if is_event and event_flag is False:
                continue
            # 采样行里同一个序号会被反复记录：第一次出现才当作事件，之后折叠掉。
            # 若该行已显式声明是事件（event_flag 非 None），则保留重复下发。
            if is_event and sequence is not None and event_flag is None:
                if sequence in seen_sequences:
                    continue
                seen_sequences.add(sequence)
            # 事件行若有真实下发时间列，就用它覆盖行时间戳。
            event_column = "command_timestamp_ns" if time_column == "timestamp_ns" else "command_time_s"
            if is_event and row.get(event_column) not in (None, ""):
                stamp = _timestamp(row, event_column)
            # ── 逐关节收集三元组 ──
            for name in active_names:
                # 掩码未选中该关节 -> 跳过（过滤掉采集缓存里残留的旧目标）。
                # 掩码缺失（None）时保持“未知”，不做过滤。
                if mask is not None and name in JOINT_NAMES and not (mask & (1 << JOINT_NAMES.index(name))):
                    continue
                angle = _optional_number(row, f"{name}/cmd_rad")
                # 没有目标角就等于本行没有该关节的指令。
                if angle is None:
                    continue
                speed = _optional_number(row, f"{name}/cmd_speed_rad_s")
                tension = _tension(row, name, command=True, raw_to_n=tension_raw_to_n)
                for label, value, default in (
                    ("speed", speed, default_speed_rad_s),
                    ("tension", tension, tension_default),
                ):
                    # 缺字段：strict 或“没有默认值”都必须报错，不允许悄悄补 0。
                    if value is None:
                        if strict or default is None:
                            raise ValueError(f"missing command {label} for {name} at CSV row {row_index + 2}")
                        missing_fields.add(label)
                # 允许用默认值时的兜底赋值（会触发一次 UserWarning 提示非实测值）。
                speed = default_speed_rad_s if speed is None else speed
                tension = tension_default if tension is None else tension
                commands[name] = JointCommand(angle, speed, tension)
            # 先按原始顺序累积，稍后统一排序，避免在流式读取时反复插入。
            samples.append((stamp, row_index, commands, is_event, sequence, mask, _boolean(row, ("is_fallback",))))

    # ── 收尾：告警、排序、时间归一化 ──
    if missing_fields:
        # 用默认值补出的字段不是实测值，必须显式告警，避免被当成真实采集结果。
        warnings.warn(
            "Missing captured command "
            + ", ".join(sorted(missing_fields))
            + "; using configured defaults, not measured values",
            UserWarning,
            stacklevel=2,
        )
    if not samples:
        return []
    # 按 (时间戳, 原始行号) 稳定排序：同一时刻保持采集顺序。
    samples.sort(key=lambda sample: (sample[0], sample[1]))
    # 事件时间早于采集起点说明初态与指令不共享同一时间原点，回放会失真。
    if samples[0][0] < capture_start:
        raise ValueError(
            "command event timestamp precedes capture start; include pre-command feedback "
            "in the capture so initial state and command events share one time origin"
        )
    # 时间零点取采集起点（不是第一条事件），这样初态时间与事件时间同源。
    first_stamp = capture_start
    scale = 1e-9 if time_column.endswith("_ns") else 1.0  # ns -> s 的换算比例
    return [
        CommandFrame((stamp - first_stamp) * scale, commands, event, seq, mask, fallback)
        for stamp, _, commands, event, seq, mask, fallback in samples
    ]
