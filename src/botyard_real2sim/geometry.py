"""Planar moving tendon paths with analytic length gradients.

Coordinates are reference-pose (x,z) in metres. Positive rotation bends +z
towards -x. A point is attached to its link, including a tied tendon endpoint.
Circular guides use common tangents plus signed arcs, not polygonal chords.
The guide topology is explicit: +1 means counterclockwise traversal from the
motor to the attachment. Guide centers are physical geometry, not diagram labels.

All gradients are with respect to the two *independent* joint coordinates.
The distal link incorporates J1 = mimic_ratio * J2 before differentiating.
Thus virtual work includes distal-link forces without double-applying a J1 torque.

【中文】平面运动腱路径与解析长度梯度。

职责：给定每根手指的轴心平面坐标与每条腱/回程腱的节点列表，计算腱路径长度
L(q) 及其对两个独立关节角的偏导数，并组装成整只手的 shortening（缩短量）
与 jacobian（雅可比）矩阵。

输入/输出：
- 输入：spec（centers_m、distal_mimic_ratio、tendons/returns）、节点列表、
  关节向量 q（形状 [..., 10]，单位 rad，每根手指 2 个独立关节角）。
- 输出：shortening = h(q) = L(0) − L(q)，单位 m，正值表示该腱被拉短（收线）；
  jacobian = −∂L/∂q，10×10，单位 m/rad，每根手指只有 2 列非零。

坐标系与符号约定：
- 平面坐标取参考姿态下的 (x, z)，单位 m。
- 正角把 +z 转向 −x，即屈曲为正（与实机屈曲方向一致）。

关键假设与限制：
- 圆导向件（半径 > 0）用“公切线 + 带符号圆弧”精确求解，而不是折线近似。
- winding = ±1 表示绕行方向；+1 表示从电机到连接点按逆时针绕行。
- 每条路径最多一个 detachable 导向件；包角缩到 0 时靠旁通切线的连续性判据
  平滑退化为直线，避免角度回绕凭空多出 2πr。
- distal 上的点先代入 J1 = distal_mimic_ratio * J2（即 J1 从属于 J2）再求导，
  于是虚功包含末端连杆上的力，但不会重复施加 J1 力矩。
- 多候选拓扑中取“合法且最短”的一条；全部非法时抛 ValueError（CUDA 上改用
  torch._assert_async 触发表内断言）。
"""

from __future__ import annotations

import json
from itertools import product

import torch

from .config import FINGER_NAMES


def _rotate(vector, angle):
    """Rotate planar (x,z) vectors by `angle` radians.

    【中文】把平面向量绕原点旋转 angle（单位 rad）；向量的最后一维为 (x, z)，
    分量单位为 m（也可用于无量纲方向向量）。正角把 +z 转向 −x，即屈曲为正。
    angle 与 vector 需满足广播规则（常见情形：同一角度旋转一批点）。
    返回旋转后的新张量，不修改入参。
    """
    # 中文：标准二维旋转矩阵作用于 (x, z)，两者形状按 torch 广播对齐。
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * vector[..., 0] - s * vector[..., 1], s * vector[..., 0] + c * vector[..., 1]), -1)


def _perp(vector):
    """Return planar vectors rotated by +90 degrees.

    【中文】把平面向量旋转 +90°：(x, z) → (−z, x)。用来由切向/方向向量得到法向，
    例如旋转半径的方向 ∂(旋转后向量)/∂angle。单位与输入一致（通常为 m）。
    返回新张量，不修改入参。
    """
    # 中文：逆时针 90° 只需交换分量并取负，不必调用三角函数。
    return torch.stack((-vector[..., 1], vector[..., 0]), -1)


class PlanarPath:
    """A frictionless point/circle path; returns length and d(length)/d(q).

    【中文】一条无摩擦的“点导向件 / 圆导向件”混合平面路径。

    职责：由节点链计算路径总长 L(q) 与 ∂L/∂q（解析梯度，单位 m/rad）。
    节点按 frame 挂在 base / proximal / middle / distal 四个连杆上，随关节角一起运动；
    内部圆导向件（radius_m > 0）用公切线 + 带符号圆弧求解，而非折线近似。

    限制：路径只含短接触弧，不做整圈绕行；每条路径最多一个 detachable 导向件。
    """

    def __init__(self, spec, nodes, *, device, dtype):
        """Build one planar tendon path from its spec and node list.

        【中文】构造一条平面腱路径。

        参数：
        - spec：手指几何配置，读取 "centers_m"（三个轴心 (x,z)，单位 m，依次为
          基座 J3 轴心、中间 J2 轴心、末端 J1 轴心）与 "distal_mimic_ratio"
          （J1/J2 耦合比，无量纲，缺省 1.0）。
        - nodes：节点列表（应由 geometry_config.validate_geometry 校验过）。每个节点含
          "frame"（base/proximal/middle/distal，决定挂在哪个连杆）、
          "point_m"（参考姿态下 (x, z)，单位 m）、"radius_m"（圆导向件半径，单位 m，
          0 表示点导向件）、"winding"（±1，绕行方向）、
          "detachable"（是否允许包角归零后脱离）。
        - device / dtype：张量所在的 torch 设备与浮点类型。

        副作用：把节点按原顺序整理成 (frame, 坐标, 半径, winding) 元组；
        对 detachable 导向件枚举“保留/脱离”的全部组合，存入 self.variants，
        供 evaluate 逐候选求长度后择优。本函数不计算几何量、不会抛错。
        """
        # 中文：三个轴心 (x, z)，单位 m；同时作为 base/proximal 等连杆的旋转中心。
        self.centers = torch.tensor(spec["centers_m"], device=device, dtype=dtype)
        # 中文：末端耦合比 J1 = mimic * J2，无量纲；缺省 1.0 表示刚性耦合。
        self.mimic = spec.get("distal_mimic_ratio", 1.0)
        # 中文：节点缓存为 (frame, point_m, radius_m, winding)；半径 0 即点导向件。
        self.nodes = [
            (
                node["frame"],
                torch.tensor(node["point_m"], device=device, dtype=dtype),
                node.get("radius_m", 0.0),
                node.get("winding", 1),
            )
            for node in nodes
        ]
        # 中文：可脱离导向件的下标；每条路径校验阶段已保证最多一个。
        optional = [i for i, node in enumerate(nodes) if node.get("detachable", False)]
        self.variants = []
        # 中文：枚举 2^n 个保留/脱离组合（n 为可脱离导向件数，实际 ≤ 1）；
        # 每个候选记录 (保留的节点下标, 被移除的节点下标)。
        for switches in product((False, True), repeat=len(optional)):
            removed = {i for i, remove in zip(optional, switches, strict=True) if remove}
            self.variants.append(([i for i in range(len(nodes)) if i not in removed], removed))

    def points(self, q):
        """Map joint angles to planar positions and their derivatives.

        【中文】由关节角 q 求每个节点在当前姿态下的平面位置及其对 q 的偏导数。

        参数：q 形状 [..., 2]，单位 rad。q[..., 0] = a 为中间（J2）关节角，
        q[..., 1] = b 为近端（J3）关节角；末端（J1）角由耦合关系给出。
        返回：(positions, gradients)
        - positions：长度等于节点数的列表，每个元素形状 [..., 2]，单位 m。
        - gradients：同长度列表，每个元素形状 [..., 2, 2]，即 ∂p/∂(a, b)，
          单位 m/rad，最后一维分别对应 a、b 两个独立自变量。

        说明：base 上的点随基座固定，导数恒为 0；proximal / middle / distal 上的点
        先按各自连杆的绝对转角旋转，再叠加对应轴心的平移及其导数。
        纯前向计算，不会抛错。
        """
        # 中文：拆出两个独立关节角（rad）：a = 中间关节，b = 近端关节。
        a, b = q.unbind(-1)
        c0, c1, c2 = self.centers
        # 中文：中间轴心 c1 随近端关节角 b 绕基座轴心 c0 转动。
        p1_offset = _rotate(c1 - c0, b)
        p1 = c0 + p1_offset
        # 中文：末端轴心 c2 相对中间轴心的向量随 (a + b) 转动（累积角度）。
        p2_offset = _rotate(c2 - c1, a + b)
        p2 = p1 + p2_offset
        # 中文：∂(旋转 v)/∂θ = perp(旋转 v)，故各轴心导数由 perp 直接给出。
        zero = torch.zeros_like(p1)
        # 中文：p1 只随 b 变 → 对 a 的导数为 0，对 b 的导数为 perp(p1_offset)。
        dp1 = torch.stack((zero, _perp(p1_offset)), -1)
        # 中文：p2 = p1 + 旋转项，对 a、b 都各多一份 perp(p2_offset)。
        dp2 = dp1 + torch.stack((_perp(p2_offset), _perp(p2_offset)), -1)
        positions, gradients = [], []
        for frame, point, _, _ in self.nodes:
            if frame == "base":
                # 中文：挂在基座上的点不动（含系在基座上的腱端点），导数为零。
                p, dp = point.expand_as(p1), torch.zeros_like(dp1)
            else:
                # 中文：origin/ref 为该连杆的转轴与参考点，angle 为连杆绝对转角，
                # da 为该转角对 a 的偏导（对 b 的偏导恒为 1）。
                if frame == "proximal":
                    origin, d_origin, ref, angle, da = c0, torch.zeros_like(dp1), c0, b, 0.0
                elif frame == "middle":
                    origin, d_origin, ref, angle, da = p1, dp1, c1, a + b, 1.0
                else:
                    # 中文：distal 先代入 J1 = mimic * J2，故转角为 b + (1 + mimic) * a，
                    # 对 a 的偏导为 1 + mimic；这样虚功含末端力但不重复施加 J1 力矩。
                    origin, d_origin, ref, angle, da = p2, dp2, c2, b + (1 + self.mimic) * a, 1 + self.mimic
                # 中文：节点相对转轴的参考向量旋转到当前连杆姿态，再平移到转轴位置。
                offset = _rotate(point - ref, angle)
                p = origin + offset
                # 中文：链式法则：位置对 a 的导数为 da*perp(offset)，对 b 的为 perp(offset)。
                dp = d_origin + torch.stack((da * _perp(offset), _perp(offset)), -1)
            positions.append(p)
            gradients.append(dp)
        return positions, gradients

    def evaluate(self, q):
        """Length and gradient of the selected (shortest valid) guide variant.

        【中文】在所有 detachable 保留/脱离组合中选出“合法且最短”的一条，
        返回其路径长度 L(q) 与长度梯度 ∂L/∂q。

        参数：q 形状 [..., 2]，单位 rad（a = 中间关节角，b = 近端关节角）。
        返回：(length, gradient)
        - length：形状 [...]，单位 m，腱路径总长 L(q)。
        - gradient：形状 [..., 2]，单位 m/rad，解析梯度 ∂L/∂q。

        选择规则：每个候选先按自身几何合法性过滤；对“移除某导向件”的候选，还要求
        该导向件在原始拓扑下的包角已归零（arcs <= 1e-7），以保持指定的同伦——
        不会在接触还没抬起前跳到另一侧更短的路线上。
        抛错：全部候选非法时抛 ValueError（CUDA 张量走 torch._assert_async 触发断言），
        提示检查导向件配置与标定角度范围。
        """
        points, derivatives = self.points(q)
        candidates = [self._candidate(q, points, derivatives, kept, removed) for kept, removed in self.variants]
        # 中文：按候选堆叠长度（最后一维为候选）与梯度（倒数第二维为候选）。
        lengths = torch.stack([item[0] for item in candidates], -1)
        gradients = torch.stack([item[1] for item in candidates], -2)
        validity = [item[2] for item in candidates]
        # 中文：可脱离导向件必须已“抬起”（原拓扑包角归零）才允许使用移除它的候选。
        for k, (_, removed) in enumerate(self.variants):
            for index in removed:
                # Preserve the specified homotopy: do not jump to a shorter
                # route on the other side before the original contact lifts.
                validity[k] = validity[k] & (candidates[0][3][index] <= 1e-7)
        valid = torch.stack(validity, -1)
        # 中文：合法性检查分走：CUDA 用异步断言（避免强制同步），CPU 直接判并抛错。
        if q.is_cuda:
            torch._assert_async(valid.any(-1).all(), "invalid tendon geometry/contact topology")
        elif not bool(valid.any(-1).all()):
            raise ValueError("invalid tendon geometry/contact topology: check guides and calibrated angle range")
        # 中文：把非法候选长度置为 +inf 后取 argmin，等价于“合法候选里最短的一条”。
        choice = torch.where(valid, lengths, torch.inf).argmin(-1)
        # 中文：按 choice 收集长度与梯度，梯度取 [..., 候选, 2] 的倒数第二维。
        return (
            lengths.gather(-1, choice.unsqueeze(-1)).squeeze(-1),
            gradients.gather(-2, choice[..., None, None].expand(*choice.shape, 1, 2)).squeeze(-2),
        )

    def _candidate(self, q, all_points, all_derivatives, kept, removed):
        """Length, gradient and validity of one kept-node subset.

        【中文】计算单个候选拓扑的路径长度、长度梯度与合法性。

        参数：
        - q：形状 [..., 2] 的关节角，单位 rad（仅用于生成同形状的零张量）。
        - all_points / all_derivatives：points() 的完整输出（全部节点，未过滤）。
        - kept：保留下来的原始节点下标（升序），路径按此顺序连接。
        - removed：被移除的（可脱离）导向件原始下标，用于脱离判据。

        返回：(length, gradient, valid, arcs)
        - length：形状 [...]，单位 m；= Σ 公切线段长 + Σ 半径 × 带符号包角。
        - gradient：形状 [..., 2]，单位 m/rad，即 ∂length/∂q。
        - valid：布尔张量，形状 [...]；任一几何约束不满足即为 False。
        - arcs：以“原始节点下标”为键的带符号包角（单位 rad）；evaluate 用它判断
          可脱离导向件是否已抬起。

        几何要点：相邻两个节点之间的圆导向件用公切线加圆弧精确求解；winding = ±1 决定
        圆弧取正向还是反向；本类路径只含短接触弧、不做整圈绕行，故要求 arc >= -1e-7。
        """
        points = [all_points[i] for i in kept]
        derivatives = [all_derivatives[i] for i in kept]
        nodes = [self.nodes[i] for i in kept]
        length = torch.zeros_like(q[..., 0])
        gradient = torch.zeros_like(q)
        tangent_angles, tangent_derivatives = [], []
        arcs = {}
        valid = torch.ones_like(length, dtype=torch.bool)
        # ── 逐段公切线（含公切线长度与其导数） ──
        for i in range(len(points) - 1):
            # 中文：相邻节点的相对位置及其对 q 的偏导 ∂(p_{i+1} − p_i)/∂q。
            delta = points[i + 1] - points[i]
            derivative = derivatives[i + 1] - derivatives[i]
            r0, w0 = nodes[i][2:]
            r1, w1 = nodes[i + 1][2:]
            # 中文：带符号半径差（winding 决定取外公切线还是内公切线）。
            offset = w1 * r1 - w0 * r0
            square = (delta * delta).sum(-1)
            # 中文：|delta| 必须大于 |offset|，否则两圆相交/内含，不存在公切线。
            valid = valid & (square > offset**2 + 1e-16)
            # 中文：公切线长度 sqrt(|delta|² − offset²)，单位 m；加下限避免除零。
            tangent = (square - offset**2).clamp_min(1e-20).sqrt()
            # 中文：dot = delta · ∂delta/∂q = ½·∂(|delta|²)/∂q，供切线长度与角度求导。
            dot = (delta.unsqueeze(-1) * derivative).sum(-2)
            # 中文：二维叉积 delta × ∂delta/∂q，供 atan2 部分的角度导数使用。
            cross = delta[..., 0, None] * derivative[..., 1, :] - delta[..., 1, None] * derivative[..., 0, :]
            # 中文：公切线方向角 = 连线方位角 atan2(dy, dx) − asin(offset/|delta|)。
            angle = torch.atan2(delta[..., 1], delta[..., 0]) - torch.asin(
                (offset / square.clamp_min(1e-20).sqrt()).clamp(-1, 1)
            )
            # A detached guide must clear the actual bypass tangent segment.
            # This keeps lift-off continuous instead of adding a spurious 2*pi*r
            # when a shrinking wrap passes through zero.
            # 中文：脱离判据用“实际旁通切线段”而非整圆：要求被移除导向件的圆心到
            # 该切线段的最短距离 ≥ 其半径。这样包角缩到 0 时抬起是连续的，不会在
            # 缩小的包角穿过零时凭空多出 2πr。
            normal = torch.stack((-torch.sin(angle), torch.cos(angle)), -1)
            # 中文：切线的两个切点（位于各自圆上，沿法向由圆心偏移带符号半径）。
            start_point = points[i] - w0 * r0 * normal
            end_point = points[i + 1] - w1 * r1 * normal
            segment = end_point - start_point
            for omitted in removed:
                if kept[i] < omitted < kept[i + 1]:
                    # 中文：把“被跳过的节点”投影到该切线段上（参数 along 截断到 [0,1]）。
                    offset_point = all_points[omitted] - start_point
                    along = (offset_point * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-20)
                    closest = start_point + along.clamp(0, 1).unsqueeze(-1) * segment
                    separation = (all_points[omitted] - closest).square().sum(-1)
                    # 中文：圆心到切线段的距离平方必须 ≥ 半径平方（容差 1e-9 m），
                    # 否则说明导向件仍会挡住绳，不允许使用该脱离候选。
                    valid = valid & (separation >= (self.nodes[omitted][2] - 1e-9) ** 2)
            # 中文：∂angle/∂q = (delta × ∂delta)/|delta|²
            #      + offset·(delta·∂delta)/(|delta|²·tangent)（asin 项的链式导数）。
            d_angle = cross / square.clamp_min(1e-20).unsqueeze(-1)
            d_angle += offset * dot / (square * tangent).clamp_min(1e-30).unsqueeze(-1)
            tangent_angles.append(angle)
            tangent_derivatives.append(d_angle)
            # 中文：长度累加公切线段（m），梯度累加 ∂tangent/∂q = (delta·∂delta)/tangent。
            length = length + tangent
            gradient = gradient + dot / tangent.unsqueeze(-1)
        # ── 逐内部圆导向件累加带符号包角（弧长与弧长导数） ──
        for i in range(1, len(points) - 1):
            radius, winding = nodes[i][2:]
            if radius == 0:
                continue
            # 中文：包角 = winding × (后一段切线方向角 − 前一段切线方向角)，单位 rad。
            angle = winding * (tangent_angles[i] - tangent_angles[i - 1])
            # 中文：折回 (−π, π] 取主值，避免绕行累计造成 2π 跳变。
            arc = torch.atan2(torch.sin(angle), torch.cos(angle))
            arcs[kept[i]] = arc
            # These routes contain short contact arcs, no intentional full turns.
            # 中文：本类路径只有短接触弧、不作整圈绕行，故负包角视为拓扑非法。
            valid = valid & (arc >= -1e-7)
            # 中文：弧长 = 半径 × 包角（m）；梯度按同一权重累加 ∂angle/∂q。
            length = length + radius * arc
            gradient = gradient + radius * winding * (tangent_derivatives[i] - tangent_derivatives[i - 1])
        return length, gradient, valid, arcs


class TendonGeometry:
    """Replace selected constant-arm rows with measured/configured moving paths.

    【中文】把走线矩阵中被测量/配置成“运动路径”的那些行替换成真实路径结果。

    用途：常数臂模型给出 h = M q 的线性关系；本类对配置了 tendons/returns 的腱，
    改用 PlanarPath 求出的 L(q) 及其解析导数，其余行保留常数臂矩阵的值。

    约定：10 个腱行按手指顺序排列，每指两行（first、second）；关节向量 q 共 10 维，
    每指 2 个独立关节角，与行号一样按 FINGER_NAMES 顺序分组。
    """

    def __init__(self, specs, routing_matrix):
        """Register moving paths that override rows of the routing matrix.

        【中文】按各手指几何配置建立走线路径，并记录它们替换哪些腱行。

        参数：
        - specs：{手指名: spec}，手指名必须属于 FINGER_NAMES
          （thumb/index/middle/ring/little）。
        - routing_matrix：形状 [10, 10] 的常数臂走线矩阵（行 = 腱，列 = 关节角），
          同时提供目标 device 与 dtype。

        行号约定：手指 f 的两行是 2·index(f) 与 2·index(f)+1，"first" 用前者、
        "second" 用后者；每行只与该手指的 2 个关节角相关（对应 2 列非零）。
        同一 (centers_m, mimic_ratio, nodes) 几何会被归并到同一分组，以便批量求值。
        副作用：tendons 在 q = 0（参考姿态）处的长度存入 self.reference_lengths，
        用作缩短量 h = L(0) − L(q) 的基准。构造阶段不抛错。
        """
        self.matrix = routing_matrix
        self.paths, self.returns = {}, {}
        self.reference_lengths = {}
        self.groups = {"tendons": {}, "returns": {}}
        for finger, spec in specs.items():
            # 中文：该手指 2 个独立关节角在 q 中的起始列，也是两行腱的起始行号。
            start = 2 * FINGER_NAMES.index(finger)
            for kind, target in (("tendons", self.paths), ("returns", self.returns)):
                for name, nodes in spec.get(kind, {}).items():
                    # 中文：first 取第 start 行，second 取第 start+1 行（bool 当 0/1 用）。
                    index = start + (name == "second")
                    path = PlanarPath(spec, nodes, device=routing_matrix.device, dtype=routing_matrix.dtype)
                    target[index] = (start, path)
                    # 中文：用排序后的 JSON 作为几何指纹，几何相同的成员共享一次求值。
                    key = json.dumps([spec["centers_m"], spec.get("distal_mimic_ratio", 1.0), nodes], sort_keys=True)
                    self.groups[kind].setdefault(key, (path, []))[1].append((index, start))
                    if kind == "tendons":
                        # 中文：参考姿态 q = 0 下的绝对路径长度 L(0)，单位 m。
                        self.reference_lengths[index] = path.evaluate(routing_matrix.new_zeros((1, 2)))[0][0]

    def evaluate(self, q):
        """Shortening and Jacobian for every tendon row.

        【中文】计算 10 条腱的缩短量与雅可比。

        参数：q 形状 [..., 10]，单位 rad（10 个独立关节角，每指 2 个）。
        返回：(shortening, jacobian)
        - shortening：形状 [..., 10]，单位 m，= L(0) − L(q)，正值表示该腱被收短。
        - jacobian：形状 [..., 10, 10]，单位 m/rad，= −∂L/∂q。先用常数臂矩阵打底，
          被移动路径覆盖的行先清零、再填入解析梯度 −∂L/∂q。

        实现说明：同一几何分组内的多条路径拼成批量一次求值；每条路径只取本指对应的
        2 个关节角列，因此每指只有 2 列非零。不满足几何合法性时由 PlanarPath.evaluate
        抛 ValueError。
        """
        # 中文：常数臂基线：h = M q，对应的雅可比即 M 本身。
        shortening = q @ self.matrix.T
        jacobian = self.matrix.expand(*q.shape[:-1], 10, 10).clone()
        # ── 用运动路径覆盖被配置过的腱行 ──
        for path, members in self.groups["tendons"].values():
            # 中文：把同组每条路径对应的 2 个关节角取出来，拼成 [N, members, 2] 批量。
            pair_q = torch.stack([q[:, start : start + 2] for _, start in members], 1)
            length, gradient = path.evaluate(pair_q.reshape(-1, 2))
            length = length.reshape(q.shape[0], len(members))
            gradient = gradient.reshape(q.shape[0], len(members), 2)
            for member, (index, start) in enumerate(members):
                # 中文：缩短量 h = L(0) − L(q)，单位 m（正值为收线）。
                shortening[:, index] = self.reference_lengths[index] - length[:, member]
                # 中文：该行改用本指 2 列的运动路径导数，其余列清零。
                jacobian[:, index, :] = 0
                # 中文：雅可比 = −∂L/∂q（缩短量对关节角的导数，符号取负）。
                jacobian[:, index, start : start + 2] = -gradient[:, member]
        return shortening, jacobian

    def return_lengths(self, q):
        """Absolute return-tendon lengths and their gradients.

        【中文】计算回程腱（returns）的绝对长度及其对关节角的梯度。

        参数：q 形状 [..., 10]，单位 rad。
        返回：(length, jacobian)
        - length：形状 [..., 10]，单位 m；没有配置运动路径的行保持 0。
        - jacobian：形状 [..., 10, 10]，单位 m/rad，= +∂L/∂q。注意符号约定与
          evaluate 的缩短量相反（回程腱长度随关节角增长即为正）。

        说明：回程腱不做 L(0) 基准平移，直接返回绝对长度。
        """
        length = torch.zeros_like(q)
        jacobian = q.new_zeros((*q.shape[:-1], 10, 10))
        # ── 回程腱：只写被配置过的行，其余保持 0 ──
        for path, members in self.groups["returns"].values():
            pair_q = torch.stack([q[:, start : start + 2] for _, start in members], 1)
            lengths, gradients = path.evaluate(pair_q.reshape(-1, 2))
            lengths = lengths.reshape(q.shape[0], len(members))
            gradients = gradients.reshape(q.shape[0], len(members), 2)
            for member, (index, start) in enumerate(members):
                # 中文：绝对长度（m）与 ∂L/∂q（m/rad），只填本指的 2 列。
                length[:, index] = lengths[:, member]
                jacobian[:, index, start : start + 2] = gradients[:, member]
        return length, jacobian
