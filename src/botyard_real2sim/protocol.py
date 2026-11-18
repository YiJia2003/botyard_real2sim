"""Offline codec for the H7 A0 ``CMD_CTL_DOFS`` payload.

There is no transport, framing, checksum or hardware access here. The payload
is a three-byte little-endian mask followed by signed little-endian int16
angle, maximum-speed and third-field words, ordered by increasing DOF index.
Angle units are 0.01 degree, speed units 0.1 degree/s. For flexion DOFs the
third field has a 0.01 N command convention; it is neither observed tension
nor an enforced fallback tension target/ceiling. Other DOFs interpret the
same third word differently, so this codec intentionally supports only the
ten tendon flexion DOFs used by this model.

Unlike a controller command validator, decoding preserves signed wire values,
including negative speeds and tension commands. H7 subsequently takes the
absolute speed and applies its own defaults/limits. Sparse commands remain
sparse; autopair requires measured joint angles at receipt time and belongs
to the controller layer. Empty masks are valid command payloads.

【中文说明】
文件职责：
    本模块是 H7 固件 ``A0``（``CMD_CTL_DOFS``）指令载荷的**纯编解码器**：
    只负责把 Python 侧的 SI 单位命令（rad、rad/s、N）与线上 int16 原始字
    相互转换。这里没有传输层（无串口、无 UDP、无 socket）、没有帧头帧尾、
    没有校验和，也不访问任何硬件。

输入/输出：
    - 编码输入：``Mapping[str, DofCommand]``，键是关节名（本模型十个腱驱动屈曲 DOF）。
    - 编码输出：``bytes``，3 字节小端掩码 + 每个被选中关节的 3 个 int16 字。
    - 解码输入：``bytes`` / ``bytearray`` / ``memoryview`` 形式的 A0 载荷。
    - 解码输出：``dict[str, DofCommand]``，按 DOF 索引升序排列。

线格式（wire format）：
    [0:3]  3 字节小端掩码：bit i 置 1 表示 DOF i 被选中；随后按 DOF 升序，
           每个被选中的关节各占 6 字节（3 个有符号小端 int16）：
             角度      单位 0.01 deg
             最大速度  单位 0.1 deg/s
             第三字段  单位 0.01 N（仅屈曲 DOF 采用这一“指令值”约定）

关键假设：
    1. 只支持本模型使用的十个腱驱动屈曲 DOF。其他 DOF 对同一个第三字段
       有各自的解释，因此不纳入本编解码器的支持范围。
    2. 解码**保留符号**：负速度、负第三字段都会原样返回；H7 后续自行取
       绝对值并套用固件默认值与限幅，那属于固件层行为，不在本模块内。
    3. 稀疏指令保持稀疏：未被选中的关节不会出现在返回字典中。“同指补齐”
       （autopair）需要接收瞬间的实测关节角，属于控制器层的职责。
    4. 空掩码（mask = 0）是合法的指令载荷，此时返回空字典。
    5. 严格性：未知位、多余或缺失字节、超出 int16 范围一律直接抛错，
       绝不静默裁剪或补零。
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from .config import JOINT_NAMES

# ── 命令字与 DOF 掩码常量 ──
CMD_CTL_DOFS = 0xA0  # H7 指令号 0xA0：设置各 DOF 的目标角/速度/第三字段
SUPPORTED_DOF_NAMES = JOINT_NAMES  # 本编解码器支持的 DOF 名称，顺序即协议中的 DOF 顺序
# 十个 DOF 对应的合法掩码全集：低 10 位全 1（0x3FF）。超出此范围的位一律视为未知位。
_SUPPORTED_MASK = (1 << len(SUPPORTED_DOF_NAMES)) - 1


# ── 数据结构：SI 单位的三字命令 ──
@dataclass(frozen=True, slots=True)
class DofCommand:
    """SI representation of three signed command words, not measured feedback.

    【中文】
    单个 DOF 三个有符号指令字的 SI 表示。这是**指令**，不是实测反馈：
    角度是目标角、速度是速度上限、第三字段是张力指令值。

    参数（字段）：
        angle_rad:        目标关节角，单位 rad。
        speed_rad_s:      最大速度（速度上限指令），单位 rad/s。
        tension_command_n: 第三字段张力指令值，单位 N。

    说明：
        字段保留符号（可能为负），因为编解码层忠实还原线上字；
        取绝对值与限幅由 H7 固件在后续处理时完成。
    """

    # 目标关节角，单位 rad（来自线上 0.01 deg 字）。
    angle_rad: float
    # 最大速度指令，单位 rad/s（来自线上 0.1 deg/s 字，保留符号）。
    speed_rad_s: float
    # 第三字段：屈曲 DOF 的张力指令值，单位 N（来自线上 0.01 N 字）。
    tension_command_n: float

    def __post_init__(self) -> None:
        """校验三个指令字段都是有限数。

        【中文】
        参数：无（读取已赋值的三个字段）。
        返回：None。
        异常：
            ValueError: 任一字段为 NaN 或 ±inf 时抛出。
        说明：
            dataclass 在构造完成后自动调用本方法，非法命令不会被静默接受。
        """

        if not all(math.isfinite(v) for v in (self.angle_rad, self.speed_rad_s, self.tension_command_n)):
            raise ValueError("command fields must be finite")

    @property
    def tension_limit_n(self) -> float:
        """Legacy alias; this value does not enforce a fallback tension limit.

        【中文】
        历史命名别名，仅为兼容旧调用点而保留。

        返回：
            与 :attr:`tension_command_n` 完全相同的值，单位 N。

        说明：
            该值**不会**在 Fallback 位置环中充当张力上限；
            它只是随指令一起下发的第三字段信息。
        """
        return self.tension_command_n


# ── 编码辅助 ──
def _word(value: float, name: str) -> int:
    """把已换算的浮点字值校验并取整为线上 int16 原始字。

    【中文】
    参数：
        value: 已除以量化步长的浮点字值（例如 角度[deg] * 100）。
        name:  字段名，仅用于错误信息定位。

    返回：
        int，位于 [-32768, 32767] 区间内的有符号 int16 原始值。

    异常：
        ValueError: value 非有限，或四舍五入后超出 int16 表示范围。

    说明：
        round() 使用 Python 的 ties-to-even（四舍六入五取偶）规则；
        四舍五入规则由 H7 的解码方式决定，主机侧不另立规则。
    """
    # 非有限值无法编码到线上，直接拒绝而不是写入哨兵值。
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    # 量化到最近的整数原始字。
    raw = round(value)
    # 超出 int16 表示范围就报错；不做静默裁剪（firmware 注释中的 360 deg
    # 名义值本身不可表示：int16 最大正角只有 327.67 deg）。
    if not -32768 <= raw <= 32767:
        raise ValueError(f"{name} exceeds signed int16 wire range")
    return raw


# ── 编码：SI 命令 -> A0 载荷 ──
def encode_a0_payload(commands: Mapping[str, DofCommand]) -> bytes:
    """Encode SI commands in H7 order, rounding to nearest wire quantum.

    Half-quantum ties use Python's ties-to-even rule; H7 defines decoding,
    not host rounding. Quantized words must fit signed int16: in particular
    the nominal 360-degree comment in firmware is not representable (the
    largest positive wire angle is 327.67 degrees). No clipping is hidden.
    The returned bytes exclude the A0 command ID and outer packet framing.

    【中文】
    按 H7 的 DOF 顺序把 SI 指令编码成 A0 载荷。

    参数：
        commands: 关节名 -> :class:`DofCommand` 的映射。可以只包含部分关节
                  （稀疏指令）；不在映射中的关节不会被写入载荷。

    返回：
        bytes：3 字节小端掩码 + 每个被选中关节的 3 个 int16（小端）。
        返回值**不含**命令字 0xA0，也**不含**外层打包/帧结构。

    异常：
        ValueError: 存在不支持的关节名；某字段非有限；量化后超出 int16 范围。

    说明：
        量化步长：角度 0.01 deg、速度 0.1 deg/s、第三字段 0.01 N。
        量化后不做裁剪，任何越界都会显式报错，避免把饱和值当成真实指令。
    """
    # 先做名称白名单校验：其它 DOF 的第三字段语义不同，不能混进来。
    unknown = set(commands) - set(SUPPORTED_DOF_NAMES)
    if unknown:
        raise ValueError(f"unsupported flexion DOF names: {sorted(unknown)}")
    mask = 0  # 3 字节小端掩码的整数形式
    words = bytearray()  # 按 DOF 升序拼接的 int16 字串
    for index, name in enumerate(SUPPORTED_DOF_NAMES):
        # 稀疏指令：未提供的关节直接跳过，掩码对应位保持 0。
        if name not in commands:
            continue
        cmd = commands[name]
        mask |= 1 << index  # 在掩码中标记该 DOF 被选中
        # 三个字依次为：角度(0.01 deg)、速度(0.1 deg/s)、第三字段(0.01 N)。
        words.extend(
            struct.pack(
                "<hhh",
                _word(math.degrees(cmd.angle_rad) * 100.0, f"{name} angle"),
                _word(math.degrees(cmd.speed_rad_s) * 10.0, f"{name} speed"),
                _word(cmd.tension_command_n * 100.0, f"{name} tension command"),
            )
        )
    # 固定 3 字节小端掩码 + 全部选中关节的字串。
    return mask.to_bytes(3, "little") + bytes(words)


# ── 解码：A0 载荷 -> SI 命令 ──
def decode_a0_payload(payload: bytes | bytearray | memoryview) -> dict[str, DofCommand]:
    """Strictly decode an A0 payload; reject trailing bytes and unknown bits.

    【中文】
    严格解码一个 A0 载荷：拒绝多余尾部字节与未知掩码位。

    参数：
        payload: 形如 bytes 的载荷（不含命令字 0xA0 与外层帧）。

    返回：
        dict[str, DofCommand]，键为关节名，按 DOF 索引升序排列；
        未被掩码选中的关节不出现在结果中。

    异常：
        TypeError:   payload 不是 bytes/bytearray/memoryview。
        ValueError:  长度不足 3 字节；掩码含未知位；总长度与掩码位数不匹配。

    说明：
        长度必须**精确**等于 3 + 6 * popcount(mask)，多一个字节都视为错误，
        以此防止把相邻帧或填充字节误当成指令内容。
        解码保留符号：负速度、负第三字段原样返回，交由 H7 决定如何取绝对值。
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    data = bytes(payload)
    # 至少要有 3 字节掩码。
    if len(data) < 3:
        raise ValueError("A0 payload requires a three-byte mask")
    mask = int.from_bytes(data[:3], "little")
    # 未知位（本模型不支持的 DOF）直接拒绝，不忽略。
    if mask & ~_SUPPORTED_MASK:
        raise ValueError("A0 payload selects unsupported flexion DOF bits")
    # 期望长度由掩码中置位的个数唯一确定。
    expected = 3 + 6 * mask.bit_count()
    if len(data) != expected:
        raise ValueError(f"A0 payload length must be exactly {expected}, got {len(data)}")
    commands: dict[str, DofCommand] = {}
    offset = 3  # 从掩码之后的第一个字开始读取
    for index, name in enumerate(SUPPORTED_DOF_NAMES):
        # 只读取被选中的 DOF，按 DOF 升序依次消耗 6 字节。
        if mask & (1 << index):
            angle, speed, tension = struct.unpack_from("<hhh", data, offset)
            # 反量化：0.01 deg -> rad、0.1 deg/s -> rad/s、0.01 N -> N。
            commands[name] = DofCommand(math.radians(angle * 0.01), math.radians(speed * 0.1), tension * 0.01)
            offset += 6
    return commands
