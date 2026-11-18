"""Unloaded hand calibration, with optional simulation dependencies loaded lazily.

CSV, configuration and protocol utilities can be used on a recording computer
without importing torch or starting Isaac Sim.

【中文说明】
本包是"空载腱绳 real2sim 标定"的公共入口，最关键的约束是**导入分层**：

- 顶层只立即导入 `config.py`（纯 dataclass + 验证，不依赖 torch）与本模块；
  CSV、协议、参数覆盖等工具因此可以在只有 numpy/pyyaml 的采集电脑上直接用。
- 类 `TendonPlant`、`Real2SimRuntime`、`PlanarPath` 等重依赖对象通过
  `__getattr__` **惰性**导入：`from botyard_real2sim import
  TendonPlant` 第一次访问时才 import 对应子模块（进而拉起 torch）。
- 这个分层是可测试的：`tests/test_real2sim_replay.py` 里断言
  `--write-default-config` 的整个进程中没有出现 isaaclab/isaacsim/torch。

典型用法：
    from botyard_real2sim import load_config
    cfg = load_config("configs/real2sim/s1.yaml")        # 只碰 YAML
    from botyard_real2sim import TendonPlant  # 这里才需要 torch
"""

from importlib import import_module

from .config import (
    FINGER_NAMES,
    FINGER_PAIRS,
    JOINT_NAMES,
    FallbackControllerConfig,
    Real2SimConfig,
    TendonModelConfig,
    load_config,
    save_config,
)

# 惰性导出表：公开名字 → 提供它的子模块（相对本包）。
# 注意 `PlanarPath` / `TendonGeometry` 都在 geometry.py，两个名字指向同一模块。
_LAZY_EXPORTS = {
    "FallbackController": ".controller",
    "autopair_command": ".controller",
    "TendonPlant": ".model",
    "TendonStepResult": ".model",
    "Real2SimRuntime": ".runtime",
    "PlanarPath": ".geometry",
    "TendonGeometry": ".geometry",
    "SeriesRoutingSpring": ".transmission",
    "DynamicRoutingTransmission": ".routing_dynamics",
}


def __getattr__(name: str):
    """按需导入重依赖对象，并缓存进模块全局命名空间。

    【中文】
    只有在访问 `pkg.<name>` 且 `<name>` 属于 `_LAZY_EXPORTS` 时才 import 子模块；
    取到对象后写回 `globals()`，后续访问不再走 `__getattr__`（每次只解析一次）。
    未知名字抛出标准的 `AttributeError`，保持与普通模块一致的行为——
    否则 `hasattr` / `getattr(..., default)` 这类兼容探测会失效。
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


# 公开接口清单：同时包含"立即导入"与"惰性导入"两类名字，便于 `from ... import *`
# 与静态检查工具识别导出面。
__all__ = [
    "FINGER_NAMES",
    "FINGER_PAIRS",
    "JOINT_NAMES",
    "FallbackController",
    "FallbackControllerConfig",
    "Real2SimConfig",
    "Real2SimRuntime",
    "PlanarPath",
    "TendonGeometry",
    "SeriesRoutingSpring",
    "DynamicRoutingTransmission",
    "TendonModelConfig",
    "TendonPlant",
    "TendonStepResult",
    "autopair_command",
    "load_config",
    "save_config",
]
