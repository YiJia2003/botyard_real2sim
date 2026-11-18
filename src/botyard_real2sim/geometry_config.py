"""Validate editable planar routing without importing a simulation backend.

【中文说明】不依赖任何仿真引擎/后端的平面走线配置校验器。

职责：对用户可编辑的 geometry 配置做结构 + 数值校验，把错误尽早暴露为 ValueError，
避免非法配置流进几何/仿真代码（也就不需要为了校验而 import torch 或仿真依赖）。

输入：geometry，形如 {手指名: spec} 的字典；spec 允许的键只有
"centers_m"、"distal_mimic_ratio"、"tendons"、"returns"、"provenance"。
输出：无返回值；全部合法时静默通过，否则抛 ValueError。

校验规则（逐条对应下面的检查）：
- 顶层必须是字典，且只能出现 thumb/index/middle/ring/little 五个手指；
- centers_m 必须是 3 个 (x, z) 轴心对，单位 m；
- distal_mimic_ratio 必须有限（无量纲，J1/J2 耦合比）；
- tendons / returns 各自只能把 "first"/"second" 映射到节点列表，每列表至少 2 个节点；
- 节点 frame 必须是 base/proximal/middle/distal，point_m 为 (x, z)，单位 m；
- radius_m 有限且非负（单位 m）；半径 > 0 的圆导向件只能出现在内部节点，
  且必须给出绕行方向 winding ∈ {−1, +1}；
- detachable 必须是严格的 bool，且只有内部圆导向件才允许脱离；
- 每条路径最多 1 个可脱离导向件。

限制：只做静态结构校验，不验证几何上是否真能构成合法公切线（那由几何模块在
求值时报错），也不做任何单位换算或补默认值。
"""

from __future__ import annotations

import math
from numbers import Real

FRAMES = ("base", "proximal", "middle", "distal")
FINGERS = ("thumb", "index", "middle", "ring", "little")


def validate_geometry(geometry):
    """Raise ValueError unless `geometry` is a legal planar routing configuration.

    【中文】校验平面走线配置，任何不合法之处直接抛 ValueError。

    参数：geometry —— 见模块 docstring 的 {手指名: spec} 结构。
    返回：None。只做校验，不返回新结构、不修改入参。

    内部辅助：
    - number(value)：有限实数判定；显式排除 bool（bool 是 int 的子类）。
    - point(value)：长度为 2 的 (x, z) 数值对判定，单位 m。

    抛错：结构错误、未知键、类型错误、非有限数值、半径/绕行方向/可脱离位置不合法，
    统一以 ValueError 报出，消息中尽量带上手指与路径定位信息。
    """
    # 中文：顶层必须是字典；出现五个手指以外的键即视为拼写错误或非法手指。
    if not isinstance(geometry, dict) or set(geometry) - set(FINGERS):
        raise ValueError("geometry must map thumb/index/middle/ring/little to planar geometry")

    def number(value):
        # 中文：有限实数判定；bool 是 int 的子类，必须显式排除，否则 True 会当 1 用。
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    def point(value):
        # 中文：平面点判定：必须是长度为 2 的数值对 (x, z)，单位 m。
        return isinstance(value, (list, tuple)) and len(value) == 2 and all(number(x) for x in value)

    # ── 逐指校验 ──
    for finger, spec in geometry.items():
        # 中文：每指允许的键集合；多出的键一律报错，避免拼写错误被静默忽略。
        allowed = {"centers_m", "distal_mimic_ratio", "tendons", "returns", "provenance"}
        if not isinstance(spec, dict) or set(spec) - allowed:
            raise ValueError(f"invalid geometry settings for {finger}")
        # 中文：必须给出 3 个轴心 (x, z)，依次为 J3 / J2 / J1 的转轴，单位 m。
        centers = spec.get("centers_m")
        if not isinstance(centers, (list, tuple)) or len(centers) != 3 or not all(point(c) for c in centers):
            raise ValueError(f"{finger}.centers_m must contain J3/J2/J1 (x,z) pairs in metres")
        # 中文：末端耦合比可缺省为 1.0，但必须是有限实数（无量纲）。
        if not number(spec.get("distal_mimic_ratio", 1.0)):
            raise ValueError("distal_mimic_ratio must be finite")
        # ── 逐条路径校验（腱路径与回程腱路径规则相同） ──
        for kind in ("tendons", "returns"):
            # 中文：tendons/returns 可缺省为空；只允许 first/second 两条路径。
            paths = spec.get(kind, {})
            if not isinstance(paths, dict) or set(paths) - {"first", "second"}:
                raise ValueError(f"{finger}.{kind} must map first/second to node lists")
            for name, nodes in paths.items():
                # 中文：一条路径至少两个节点，否则无法形成任何线段。
                if not isinstance(nodes, (list, tuple)) or len(nodes) < 2:
                    raise ValueError(f"{finger}.{kind}.{name} needs at least two nodes")
                # ── 逐个节点校验 ──
                for i, node in enumerate(nodes):
                    # 中文：节点必须是字典，且只能含已知键（未知键即报错）。
                    if not isinstance(node, dict) or set(node) - {
                        "frame",
                        "point_m",
                        "radius_m",
                        "winding",
                        "detachable",
                    }:
                        raise ValueError("unknown route node settings")
                    # 中文：frame 决定节点挂在哪根连杆上，point_m 为其参考姿态坐标（m）。
                    if node.get("frame") not in FRAMES or not point(node.get("point_m")):
                        raise ValueError("route nodes require a valid frame and point_m (x,z)")
                    # 中文：radius_m 缺省 0，表示点导向件；圆导向件半径单位 m。
                    radius = node.get("radius_m", 0.0)
                    if not number(radius) or radius < 0:
                        raise ValueError("route radius_m must be finite and nonnegative")
                    # 中文：圆导向件必须是内部节点（首尾节点无法生成前后两段切线），
                    # 且 winding 必须是 −1 或 +1（绕行方向）。
                    if radius > 0 and (i in (0, len(nodes) - 1) or node.get("winding") not in (-1, 1)):
                        raise ValueError("circular guides must be internal nodes with winding -1 or +1")
                    # 中文：detachable 用 type() 严格判定，拒绝 0/1 之类的整数。
                    if type(node.get("detachable", False)) is not bool:
                        raise ValueError("detachable must be boolean")
                    # 中文：只有内部的圆导向件才有“包角归零后脱离”的语义。
                    if node.get("detachable") and (radius <= 0 or i in (0, len(nodes) - 1)):
                        raise ValueError("only an internal circular guide can detach")
                # 中文：每条路径最多一个可脱离导向件，几何与求导实现都基于此假设。
                if sum(bool(n.get("detachable")) for n in nodes) > 1:
                    raise ValueError("at most one detachable guide is supported per path")
