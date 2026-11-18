"""Named configuration overrides shared by replay and parameter experiments.

【中文说明】
本文件实现命令行上的具名参数覆盖，回放脚本（`--set`）与参数扫描脚本共用同一
套语法，保证"扫参"和"单次回放"改的是同一条路径、走同一套校验。

语法（三种写法）：
    section.field=value                     标量字段，例如 controller.control_period_s=0.004
    section.vector.JOINT=value              10 元素向量按关节名覆盖，例如 tendon.cable_stiffness_n_per_m.FFJ2=9000
    section.vector.finger=value             5 元素向量按手指名覆盖，例如 tendon.crossing_moment_arm_m.thumb=-0.005
    section.vector.INDEX=value              也可用下标，例如：
                                           tendon.geometry.index.tendons.first.5.point_m=[0.006, 0.082]

输入：一个 `Real2SimConfig` 与若干条 `KEY=VALUE` 字符串（VALUE 按 YAML 语法解析，
因此支持列表、布尔、`null`，也支持 `.nan` / `.inf` 这种会立刻被验证拦下的写法）。
输出：**新的** `Real2SimConfig`；传入的配置对象不会被修改（先 `to_dict()` 复制）。

关键假设：
- `curves` 段是自由字典（绘图脚本的选项），允许新增键；`controller` / `tendon`
  两段是封闭 schema，未知字段一律报错，避免"打错一个字导致整轮实验静默用默认值"。
- 具名覆盖只替换目标元素，其余元素保留原值；内部把元组转成列表只是为了可写，
  最后统一由 `Real2SimConfig.from_dict` 重新校验并转回元组。
- 所有覆盖在进入仿真前完成验证；本函数不做隐式裁剪或类型转换。
"""

from __future__ import annotations

from collections.abc import Sequence

from .config import FINGER_NAMES, JOINT_NAMES, Real2SimConfig


def override_parameters(config: Real2SimConfig, assignments: Sequence[str]) -> Real2SimConfig:
    """Apply section.field=value or section.vector.JOINT=value overrides.

    Values use YAML syntax. Every resolved configuration is validated before it
    can reach the simulator. Unmentioned values retain their configured values.

    【中文】
    参数：
        config: 基线配置（只读，不会被就地修改）。
        assignments: `KEY=VALUE` 字符串序列；空序列表示"原样返回一份校验后的配置"。

    返回：覆盖并重新验证后的 `Real2SimConfig`。

    异常：`ValueError`，覆盖格式错误（缺 `=`、段名/字段名未知、关节/手指名未知、
    下标越界、试图往标量里继续下钻）时抛出；YAML 语法错误会由 `yaml.safe_load`
    抛出 `yaml.YAMLError`（不是 ValueError）。字段取值本身的合法性由
    `Real2SimConfig.from_dict` 负责。

    单位：不涉及单位换算——数值按字段自身单位原样写入（见 `config.py` 字段名后缀）。
    """
    import yaml

    # 深拷贝成纯字典：后续在字典上原地改写，最后再统一构造不可变配置。
    values = config.to_dict()
    for assignment in assignments:
        # 只在第一个 "=" 处切分：右侧允许出现 "="（例如字符串值）。
        key, separator, raw = assignment.partition("=")
        path = key.split(".")
        if not separator or len(path) < 2 or not all(path):
            raise ValueError(f"expected section.field[.joint]=value, got {assignment!r}")
        section, field = path[:2]
        # 段必须存在且是映射（controller / tendon / curves）。
        if section not in values or not isinstance(values[section], dict):
            raise ValueError(f"unknown parameter section: {section}")
        # 封闭段不允许新增字段；curves 是自由字典，允许新增（供绘图脚本使用）。
        if field not in values[section] and section != "curves":
            raise ValueError(f"unknown parameter: {key}")
        # 值走 YAML 解析：支持列表/布尔/null，也让 .nan/.inf 能被后续验证抓住。
        value = yaml.safe_load(raw)
        current = values
        for index, part in enumerate(path):
            last = index == len(path) - 1
            if isinstance(current, dict):
                # curves 段允许出现新键，其余段必须先存在才能继续下钻。
                if part not in current and not (section == "curves" and last):
                    raise ValueError(f"unknown parameter path: {key}")
                if last:
                    current[part] = value
                    break
                child = current[part]
                # 元组不可写：只在构造副本里转成列表，最后 from_dict 会再转回元组。
                if isinstance(child, tuple):
                    current[part] = child = list(child)
                current = child
            elif isinstance(current, list):
                # 第 3 层（index == 2）允许用关节名/手指名代替下标；
                # 5 元素向量用 FINGER_NAMES，10 元素向量用 JOINT_NAMES。
                if index == 2 and len(current) in (5, 10) and not part.isdigit():
                    names = JOINT_NAMES if len(current) == 10 else FINGER_NAMES
                    if part not in names:
                        raise ValueError(f"unknown joint/finger {part!r}")
                    position = names.index(part)
                elif part.isdigit() and int(part) < len(current):
                    # 允许用下标直接定位（几何节点列表这类没有名字的位置）。
                    position = int(part)
                else:
                    raise ValueError(f"invalid parameter index: {key}")
                if last:
                    current[position] = value
                else:
                    current = current[position]
            else:
                # 例如 controller.control_period_s.THJ1=...：标量后面不能再加一层。
                raise ValueError(f"cannot descend into scalar parameter: {key}")
    # 由 from_dict 做最终校验（有限性、正性、跨字段约束）并转回 frozen dataclass。
    return Real2SimConfig.from_dict(values)
