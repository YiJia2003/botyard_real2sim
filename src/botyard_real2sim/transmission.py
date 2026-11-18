"""Massless elastic routing take-up in series with intrinsic cable compliance.

Routing extension represents frictional/layout surplus, not material strain.
The present constitutive law is reversible; joint friction and motor-port lag
remain separate. Hysteresis must be identified from loading/unloading data.
At its upper stop the routing spring is inextensible, while reaction tension
can grow without bound through the cable stiffness (a numerical regularizer).

【中文说明】准静态“走线等效弹簧 + 绳材料柔度”串联模型。

职责：把一段总的腱位移（收线/总伸长）分配到两个串联的柔度来源：
1) 走线伸长 routing —— 代表布线/摩擦盈余（绳在导向件之间被拉直、走线几何余量），
   用刚度为 routing_stiffness 的等效弹簧描述，最大不超过行程上限 travel；
2) 绳材料伸长 cable —— 绳本身的弹性拉伸，刚度为 cable_stiffness。
两者串联意味着承受同一张力：总伸长 = 走线伸长 + 材料伸长。

输入/输出：
- 构造参数：cable_stiffness、routing_stiffness，单位 N/m；travel，单位 m。
- evaluate 输入：total_extension，总伸长，单位 m，形状任意；负值表示松弛。
- evaluate 输出：(tension, routing, cable, locked, energy)
  tension 单位 N；routing、cable 单位 m；locked 为 bool 张量；
  energy 为弹性能，单位 J（焦耳）。

关键假设与限制：
- 准静态、无质量模型，不含动力学、不含迟滞；关节摩擦与电机端口滞后另行建模。
- 张力只受拉：tension = k_cable * max(cable, 0)，松弛段不产生压力。
- 走线伸长达到 travel 后不再增长（locked），此后的额外总伸长全部由材料柔度承担，
  张力仍可继续上升。
- 材料刚度在此充当数值正则项，使张力在上限之后仍能无界增长而不会出现奇点。
- 走线伸长代表布线/摩擦盈余，不是材料应变，不要把它当作绳的弹性变形。
"""

from __future__ import annotations

import torch


class SeriesRoutingSpring:
    """Series spring: capped routing take-up plus cable compliance.

    【中文】串联弹簧模型：受限的走线等效弹簧 与 绳材料柔度 串联。

    构造参数：
    - cable_stiffness：绳材料刚度，单位 N/m。
    - routing_stiffness：走线等效刚度，单位 N/m。
    - travel：走线最大可伸长度（行程上限），单位 m；0 表示走线完全不可伸长。

    物理含义：两根弹簧串联、承受同一张力，故刚度越小分到的伸长越多；
    走线弹簧有行程止挡，顶到止挡后就由材料弹簧单独承担后续伸长。
    """

    def __init__(self, cable_stiffness, routing_stiffness, travel):
        # 中文：三个标量参数都按张量使用（形状任意，靠广播与输入对齐）。
        self.cable_stiffness = cable_stiffness
        self.routing_stiffness = routing_stiffness
        self.travel = travel

    def extension_for_tension(self, tension):
        """Routing extension at a given tension.

        【中文】由张力求走线伸长：routing = min(tension / routing_stiffness, travel)。

        参数：tension，单位 N，形状任意。
        返回：走线伸长，单位 m；超过止挡处饱和为 travel。
        说明：这是反向（先定张力）的估算式，只看走线弹簧本身、不分配材料柔度；
        实际用途是在初始化时由目标初始张力反解收绳坐标
        （x = h(q) + T/k_cable + 走线伸长(T)）。正向求解（由总伸长得张力）请用 evaluate。
        """
        # 中文：线性弹簧 F = k·x 求 x，再以行程上限截断（达到止挡即锁死）。
        return torch.minimum(tension / self.routing_stiffness, self.travel)

    def evaluate(self, total_extension):
        """Split a total extension into routing and cable parts.

        【中文】把总伸长分配到走线伸长与材料伸长，并给出张力、锁死标志与弹性能。

        参数：total_extension，单位 m，形状任意；负值表示松弛（不会被压缩）。
        返回：(tension, routing, cable, locked, energy)
        - tension：单位 N，= cable_stiffness * max(cable, 0)，只受拉。
        - routing：单位 m，走线伸长，被 travel 封顶。
        - cable：单位 m，带符号的材料伸长；三者严格满足
          total_extension = routing + cable（松弛时 cable 为负）。
        - locked：bool，走线已顶到行程上限（且 travel > 0，即确有行程）。
        - energy：单位 J，= ½·k_r·routing² + ½·k_c·max(cable,0)²，
          松弛的材料段不储能。
        本函数不抛错；travel 为 0 时走线伸长恒为 0、直接全部记到材料柔度上。
        """
        # 中文：只取拉伸部分参与走线分配；负的总伸长视为纯松弛。
        stretch = total_extension.clamp_min(0)
        # 中文：串联刚度分配：相同张力下 x = F/k，故走线分到 k_c/(k_c+k_r)·stretch；
        # 再按行程上限 travel 截断（上限之后走线不再伸长）。
        routing = torch.minimum(
            stretch * self.cable_stiffness / (self.cable_stiffness + self.routing_stiffness), self.travel
        )
        # Keep signed cable gap: total_extension = routing + cable exactly,
        # including slack, and never generate compression.
        # 中文：保留带符号的 cable 缺口，使 total_extension = routing + cable 恒成立
        # （松弛时 cable 为负）；后面的 clamp_min(0) 保证不产生压缩力。
        cable = total_extension - routing
        # 中文：张力只受拉：材料伸长截断到 0 以上；走线锁死后张力仍可由此项继续增长。
        tension = self.cable_stiffness * cable.clamp_min(0)
        # 中文：travel > 0 且走线伸长已达上限 → 走线锁死（不可再伸长）。
        locked = (self.travel > 0) & (routing >= self.travel)
        # 中文：弹性能（J）：走线弹簧 + 材料弹簧各 ½kx²；松弛的材料段不储能。
        energy = 0.5 * (self.routing_stiffness * routing.square() + self.cable_stiffness * cable.clamp_min(0).square())
        return tension, routing, cable, locked, energy
