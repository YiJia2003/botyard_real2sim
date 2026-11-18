# 独立项目提取记录

2026-11-18（Asia/Shanghai）。源为 `/home/cyj/dexterous_hand_manipulation` 当前工作区，目标为 `/home/cyj/botyard_real2sim`。
提取包含已暂存/未提交的最新文件，并非只复制旧 Git HEAD。原文件 SHA-256、源提交和新位置见根目录 `EXTRACTION_MANIFEST.json`。

## 边界

- 原 `real2sim` 模块改名为顶层 Python 包 `botyard_real2sim`，保留原力学与控制行为。
- 三个 CLI 实现移动至包内 `cli/`，根目录 `scripts/` 保留便捷启动器；参数扫描以 `python -m botyard_real2sim.cli.*` 启动子进程，支持 wheel 安装后的运行。
- 本地 `simulation/hand_cfg.py` 和 `simulation/physics.py` 替代对原 RL 任务目录的导入。PhysX 仍为 solver type 1、位置迭代至少 16、速度迭代至少 4。
- 本地 USD 是完整复制，包含网格、固定基座、mimic、自碰撞设置和两对结构碰撞排除。不是原文件的符号链接或硬链接。
- YAML、图片、STEP、CAD 测量证据、real2sim 测试和历史参考曲线一起复制。CAD JSON 中的旧 URI 与历史回放元数据原样保留，作为来源记录，不用于运行时寻找文件。
- 未提取训练/抓握任务、奖励/观测、相机、RL checkpoint、node_modules 和原 Git 历史；不需要导入原 Python 包。
- 仿真依赖仍使用已配置的 Isaac Lab 环境。这是新的独立业务项目，不是复制整套 Isaac Sim 安装。

## 实际验证

1. `./run.sh test`：181 项通过，两个旧采集缺少指令字段的预期提示；测试覆盖控制、协议、回放、几何、惯性/摩擦、曲线与扫描，以及新增独立性检查。
2. `./run.sh check` 和 `bash -n run.sh`：通过。
3. 禁止导入 `dexterous_hand_manipulation` 后，直接在新项目运行 PhysX：FFJ2 20°、最大速度 120°/s、1.2 s、物理步 0.5 ms、走线内部步 0.1 ms。
4. 401 行、2400 物理步、无力矩裁剪；mimic 最大误差 `3.799796e-6 rad`。十通道检查张力非负、状态有限、接触范围、弹簧行程和长度平衡。
5. 与独立复制的历史同条件 `dynamic/j2.csv` 比较，FFJ2 的关节角、电机角、电机张力、关节张力四列最大逐点绝对差均为零。
6. `uv build` 成功生成 sdist 和 wheel；wheel 内置两份预设和完整手部 USD。在临时干净 venv 中仅安装本包和 PyYAML，三个 console 入口均可运行帮助/配置导出/扫描计划。安装后在仓库外解析内置配置与资产，未导入 Torch 或 Isaac。
7. 提取前后核对源项目 148 个文件内容（不含 Git 内部、依赖和缓存目录），哈希完全一致；Git 暂存区文件哈希也一致。

本次新运行结果位于 `logs/extraction_validation/`：`j2.csv`、参数和元数据、`j2.validation.json`、`j2_curves.html/.png`、`j2_routing.html`、`package_install.json`。
提取前的历史结果位于 `examples/reference_runs/`，两者明确区分。这些均为仿真验证，不是实机标定完成的证明。

## 以后在哪里修改

- 模型：`src/botyard_real2sim/`。
- 引擎和关节资产参数：`src/botyard_real2sim/simulation/`。
- 默认标定参数：`configs/real2sim/s1.yaml`。
- 实验数据：建议存放在本项目内自己的数据目录，输出放 `logs/`。
- 使用新项目根目录的 `./run.sh` 运行；不必回到原仓库启动。
