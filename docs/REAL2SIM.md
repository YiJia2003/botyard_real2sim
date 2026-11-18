# 空载 tendon real2sim：运行、记录与标定

目标是复现实机空载的**关节角、电机绕线轮角和电机侧应力片张力曲线**。当前提供可运行、可修改参数的标定框架；默认参数不是实机辨识结果，也不表示已经实现曲线一致。先用单指双关节建立可解释模型，后续用真实记录逐步改进。

2026-11-18 更新：S1 STEP/图片驱动的变力臂、电机角观测，以及电机传感器之后的**走线摩擦、内部接触/微滑状态和有惯量的有限行程弹簧**已接入。具体方程、参数来源与命令示例见 [几何与传动说明](REAL2SIM_GEOMETRY.md)。

## 范围与架构

使用 Isaac Lab / Isaac Sim **PhysX** 后端和本项目自带的 `boya.usda` 刚体资产，不切换 MuJoCo，不建立电机电气、齿轮箱或转子模型，不使用未经验证的逆动力学控制分支。

十个独立绳驱动关节按以下顺序排列：

```text
THJ1 THJ2 | FFJ2 FFJ3 | MFJ2 MFJ3 | RFJ2 RFJ3 | LFJ2 LFJ3
```

独立运行器显式解析 `simulation.physics.make_physx_cfg()`，由 `launch_simulation(...)` 上下文负责启动和清理，保留失败退出码。`SimulationCfg.use_newton_actuators=False` 选择标准 Torch 显式执行器路径：十个绳关节使用零增益 `IdealPD` 直接提交力矩。这只改变执行器适配路径，物理引擎仍为 PhysX；避免当前环境原生 Newton actuator adapter 首次 Warp 编译阻塞。

四指 `J1=J2` 保留 USD 中的机械 mimic 约束；`J1` 不增加独立绳控制器。拇指 THJ1 绳跨 THJ2，四指 J2 绳跨 J3。其余手指轴保持初始位置，腕部固定；空载场景无抓握物体。重力仍由 PhysX 计算，应设置与实机实验一致的手根朝向。

| 层 | 文件 | 职责 |
|---|---|---|
| 配置 | `src/botyard_real2sim/config.py` | 单位、关节顺序、参数解析和验证、配置快照 |
| 离线协议 | `src/botyard_real2sim/protocol.py` | 当前 H7 A0 指令载荷的纯编码/解码，不连接硬件 |
| 数据输入 | `src/botyard_real2sim/commands.py` | 指令事件/采样状态、初始角度和张力 |
| 控制器 | `src/botyard_real2sim/controller.py` | 当前源码 Fallback 位置环、速度限制、量化、同指补齐 |
| 绳与关节力学 | `src/botyard_real2sim/model.py` | 单边拉索、回位弹簧、关节摩擦、力矩映射 |
| 动态走线传动 | `src/botyard_real2sim/routing_dynamics.py` | 两节点惯量、接触演化、弹塑性摩擦、硬停止及能量诊断 |
| 时间调度 | `src/botyard_real2sim/runtime.py` | 指令、控制 tick、物理步和采样时刻的协调 |
| PhysX 场景 | `src/botyard_real2sim/scene.py` | 使用本地资产，对十关节施加显式力矩 |
| 运行/绘图 | `scripts/real2sim_replay.py` / `real2sim_curves.py` | 回放、元数据、对比图与误差报告 |

包路径前缀为 `src/botyard_real2sim/`。标定层与原 RL 抓握任务分离，原任务的 50 Hz policy 周期不是固件控制周期。当前模型的名义控制周期为 **0.003 s**，固件估计器使用的整数截断时间量为 **0.002 s**；后者不能替代真实调度周期。调度器按绝对时刻推进，事件在下一物理边界执行，记录执行迟到量而不累积相位舍入误差。

## 指令和张力的含义

外部三元组是目标角、最大速度和张力指令字段。当前 `Fallback_pidControl` 分支按角误差输出等效绳端速度，**第三字段不直接设置张力，也不执行张力上限保护**。CLI 使用 `--tension-command-n`；`--tension-limit-n` 和旧 CSV `cmd_tension_limit_*` 仅作为兼容别名。图中应把第三字段作为指令信息，不作为实测张力的目标曲线。

当前源码反馈中 `rope_tension × 100` 写入第三个 `int16`，ROS 接口原样发布到 `/joint_states.effort`，对应默认 `effort_raw × 0.01 = N`。它是**近端绳张力**，不是 Nm 关节力矩。回放原始刻度可用 `--tension-raw-to-n` 指定；绘图对应 `--tension-raw-to-N`（参数名大小写不同）。若采集时固件/驱动的刻度、DOF 顺序、Fallback 状态或载荷布局不同，须先核对并修正数据，不能直接套用当前源码解释。

`protocol.py` 仅支持这十个 flexion DOF 的 A0 载荷：三字节小端 mask，随后按 DOF 顺序排列角度/最大速度/第三字段的有符号 int16；对应 0.01°、0.1°/s、0.01 N。它不提供外层包、校验、串口、CAN、ROS bridge 或真实硬件收发，不证明运行中固件已采用该协议。重复发送相同指令可能重新执行同指 partner 补齐，不能无条件去重真实发送事件。

## 当前 S1 模型与限制

详细几何、CAD 测量证据、回位弹簧、有限行程走线等效弹簧、独立电机观测和调参示例见 [REAL2SIM_GEOMETRY.md](REAL2SIM_GEOMETRY.md)。

当前默认预设为 [`configs/real2sim/s1.yaml`](../configs/real2sim/s1.yaml)。四指以 link-attached 路径的长度变化 `h(q)=L(0)-L(q)` 和同一雅可比计算力矩，包含 J1=J2 的约束运动；回位拉簧用长度/方向模型。拇指仍保留待确认的常力臂参数。

电机收绳量与关节路径之间的变形分成走线等效弹簧和材料绳柔度，前者达到最大长度后行程停止、张力仍可增加。电机角是收绳状态除以 **5 mm 半径**再加编码器参考，不从关节角复制。

动态分支使用两个惯性节点，分别放在有限行程走线弹簧两端，各承担一半等效质量并具有走线摩擦。摩擦起点位于电机张力观测点之后、走线弹簧之前，等效覆盖至关节入口。接触状态随张力、收紧量和几何路径变化；微滑状态保留静止保持、滑移和反向记忆。

`tension_N` 是电机侧观测，`distal_tension_N` 是关节侧张力；关节力矩使用后者。动态时两者受摩擦和内部加速度影响，静止时也可由静摩擦维持差值。不会用滤波或人为时移制造张力差，也不会借第三指令字段强行约束 Fallback 张力。

`evaluate(q, qd)` 不推进状态；动态模式的 `advance(speed, dt, q_rad=q, qd_rad_s=qd)` 同时推进电机和走线内部状态。初态 q/T/电机角/速度从同一反馈行取值，缺失项和推定电机零位写入元数据。初始化暂假设两端张力相等、微滑历史为零；无法从单行反馈唯一恢复未知摩擦前史。角加速度是物理步上的后向差分，第一点未定义。

集总走线模型尚未拟合；没有逐个求解三维导向接触或分布式波传播。原关节轴摩擦仍为平滑 Coulomb/Stribeck，和新的走线静摩擦状态是两个独立机制。非零 `crossing_guide_mu` 仍拒绝，它表示尚未实现的逐导轮损失，不能与新的集总走线摩擦重复计算。

内部后向欧拉采用子步并记录数值耗散。能量残差针对预测关节边界下的内部传动求解，不包含 PhysX 实际终点与预测值的差异；不能据此宣称整个耦合系统步长收敛。拟合前应检查物理步和内部子步减半时整段曲线的敏感性。

**数值与物理参数须分开。** 共享 USD 中数值 armature 不是实测转子惯量，已有网格/碰撞资产与 CAD 也存在小尺寸差异。`--effort-limit-nm` 是仿真保护，不是张力上限。发生裁剪需先处理运行条件，再拟合。

需要复现旧模型时显式使用 [`configs/real2sim/default.yaml`](../configs/real2sim/default.yaml)。程序库 `TendonModelConfig()` 仍保持旧配置的默认值；只有 CLI 默认切换到了 S1 详细预设。

## 快速开始

**与引擎的连接已实现，使用独立标定入口启动。** `scripts/real2sim_replay.py` 创建 PhysX 场景，加载本项目独立复制的手部 USD。
每个物理步读取引擎关节角/速度，更新 Fallback 与传动状态，将腱绳、回位弹簧、摩擦合成的关节力矩写入执行器，再调用 `sim.step()` 推进刚体。
关节角来自 PhysX，电机绕线轮角和两侧张力来自传动状态。绳与弹簧通过外部力学层参与引擎计算，USD 中未新增可见绳/弹簧实体；单独打开 USD 不会运行这层代码。

本项目只保留空载标定入口，读取 CSV 指令；没有训练/抓握任务或实时 ROS2 bridge。
空载 real2sim 的流程是：真机记录同一时间基准的实际发送指令和反馈 → 回放到 PhysX → 对比三类曲线 → 修改参数后重放相同指令。
不需要用实测关节角逐帧驱动仿真；实测反馈用于初态和误差评价，之后关节运动由引擎自行求解。

在仓库根目录执行，使用项目对应 Isaac Lab 环境。帮助和配置导出不应启动 Isaac Sim：

```bash
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py --help
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py \
  --write-default-config /tmp/hand_real2sim.yaml
```

当前可编辑预设为 [`configs/real2sim/s1.yaml`](../configs/real2sim/s1.yaml)。CAD 尺寸和图片提供的长度有明确来源，弹簧刚度、走线行程及摩擦数值仍为待拟合初值。controller 数值来自当前源码分支，仍需与采集时固件核对。

单指阶跃：

```bash
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py \
  --config configs/real2sim/s1.yaml \
  --joint FFJ2 --initial-joint-deg FFJ2=10 --target-deg 30 \
  --max-speed-deg-s 90 --tension-command-n 20 \
  --physics-dt 0.0001 --sample-dt 0.003 \
  --out /tmp/ffj2_sim.csv
```

可将关节换为 `THJ1`；`--initial-joint-deg NAME=DEG` 可重复。`--gui` 打开显示，`--root-pos X Y Z` 和 `--root-xyzw X Y Z W` 设置实验姿态。默认暖启动；`--cold-start` 才复现初始控制等待。

重放完整实测指令：

```bash
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py \
  --commands /path/to/recording.csv --strict-commands \
  --command-frames auto --config configs/real2sim/s1.yaml \
  --out /tmp/replay_sim.csv
```

默认尝试从指令记录的反馈列读取初态；可用 `--initial-state /path/to/initial.csv` 单独指定。缺失反馈不能当成已测零值，应检查启动提示及元数据中的来源和缺项。如果第一条有效反馈晚于记录起点，运行器要求先裁剪记录或明确提供 `--initial-state`，避免将未来反馈冒充初态。记录明确包含 `is_fallback=false` 时拒绝回放，应先选取已确认的 Fallback 区间。旧记录缺少完整速度/张力指令时，省略 `--strict-commands` 可采用有提示的配置默认值；这类回放只能用于探索，不能据此验证实际发送指令。

每次输出包括 `*.csv`、`*.params.yaml`（解析后的参数）、`*.metadata.json`（来源、时序、后端/数值条件及运行检查）。保留三者和原始记录一起归档，避免只保存一张拟合图。

元数据固定标记 `calibrated: false`、`mode: forced_fallback`、`third_command_field_enforced: false`。schema 3 记录走线模式、内部子步、张力观测位置、关节力矩采用远端张力、内部初态约定和能量台账范围。它也保存实际初始角度/张力与缺失通道、参数和输入文件哈希、包版本、armature、手根姿态及时间配置；结束时写入 `status`、`controller_ticks`、`command_events_applied`、`command_max_lateness_s`、`mimic_max_error_rad` 和 `saturated_force_evaluations`。失败运行保留 `status: failed` 和错误原因。

仿真 CSV 额外区分 `requested_effort_Nm` 与 `applied_effort_Nm`，记录 `effort_saturated`、`controller_lateness_s`、四个 J1 角度和 mimic 误差。应先检查裁剪与耦合误差，再使用角度/张力指标；运行完成本身不等于参数已标定。

## 参数修改

向量字段支持标量广播、完整有序数组和具名覆盖：

```yaml
tendon:
  cable_stiffness_n_per_m: {default: 8000, FFJ2: 10000}
  crossing_moment_arm_m: {thumb: -0.005}
```

上例仅演示语法，不代表建议的实机参数。省略名称沿用该字段默认值；五指字段名称为 `thumb/index/middle/ring/little`。未知字段、未知名称、非有限数值和不合法范围会报错。命令行单项调整：

```bash
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py \
  --joint FFJ2 --config configs/real2sim/s1.yaml \
  --set tendon.cable_stiffness_n_per_m.FFJ2=8000 \
  --out /tmp/ffj2_trial.csv
```

## 一次一个参数的可审查扫描

`real2sim_sweep.py` 依次启动相同解释器的回放进程，保留每组参数和日志；它不是自动辨识器，不生成最优参数结论。先查看命令计划：

```bash
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/real2sim_sweep.py \
  --parameter tendon.cable_stiffness_n_per_m.FFJ2 \
  --values 4000 8000 12000 --out-dir /tmp/tendon-sweep-plan --dry-run -- \
  --joint FFJ2 --target-deg 30 --duration 0.5 \
  --config configs/real2sim/s1.yaml
```

上述数值仅用于探索参数影响，未被认定为实机范围。`--dry-run` 保存并打印完整 argv，不启动 Isaac 或子进程。实际运行去掉该选项并换用空输出目录。加入 `--real /path/to/real.csv` 可在每个成功回放后生成 `comparison.html/json`；该选项放在 `--` 分隔符前。比较默认 `--comparison-align start`，也可明确选 `command_edge`，不做自动时间扭曲、加权排名或优化。

每个 `run_###` 保存 `sim.csv` 及其配置/元数据、`replay.log`，有实测对比时还保存 `comparison.log`。顶层 `manifest.json` 记录计划及各次状态；某次失败仍继续其余参数。非空目录默认拒绝覆盖，只有显式 `--overwrite` 才复用；不要在分隔符后的回放参数中传 `--out`，它由扫描工具统一分配。

## CSV 契约：反馈、采样指令与发送事件

时间使用 `timestamp_ns` 整数纳秒或 `time_s` 秒。角度使用 rad，速度 rad/s，张力 N。不要将 Unix 纳秒先转浮点后相减。事件记录允许同时间的多条有序发送；绘图输入是采样曲线，应按采样时刻整理。

| 字段 | 含义 |
|---|---|
| `<joint>/position_rad` | 反馈关节角；曲线对比必需 |
| `<joint>/velocity_rad_s` | 可选反馈角速度 |
| `<joint>/motor_position_rad` | 同步电机绕线轮输出轴角；比较电机响应必需，连续不取模 |
| `<joint>/motor_velocity_rad_s` | 可选绕线轮角速度 |
| `<joint>/tension_N` | 反馈近端张力，优先于 raw |
| `<joint>/effort_raw` | 原始反馈；按已核对刻度换算，绝不当指令 |
| `<joint>/cmd_rad` | 目标角；该关节被选中时必需 |
| `<joint>/cmd_speed_rad_s` | 实际发送的速度字段 |
| `<joint>/cmd_tension_N` 或 `cmd_tension_command_N` | 实际发送的第三字段 |
| `command_event` / `is_event` | 明确标记真实发送事件 |
| `command_seq` / `sequence` | 发送序号，可区分相同三元组的重复发送 |
| `command_timestamp_ns` / `command_time_s` | 实际发送时间；缺失则只能使用行时间 |
| `dof_mask` / `command_mask` | 实际指令选中位；仅有 mask 不证明发生新发送 |
| `is_fallback` | 采集时的控制分支，缺失不能假定已验证 |

`--command-frames auto` 根据事件标记/序号区分事件和采样状态；相同序号的采样副本不会重复触发。`changes` 适用于仅记录缓存指令状态的旧文件；`each-row` 仅适用于每行确实是发送事件的文件。稀疏指令到达时，同指另一关节按当时反馈位置补齐；不能每个反馈采样都重新补齐。新的非空 active mask 在下一个控制 tick 生效；空 A0 mask 没有可靠的硬件停机语义，因此显式空事件会被运行器拒绝。仅表示缺少指令的空采样行不等同于停机。

旧 benchmark CSV 常只保存位置指令、反馈角和 `effort_raw`，没有完整实际三元组和发送事件。后续采集应补齐这些字段以及实际固件版本/分支、传感器零点与刻度、手姿态。不能用生成的仿真 CSV 充当实机数据。

## 绘图与评价

```bash
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/real2sim_curves.py \
  --real /path/to/real.csv --sim /tmp/replay_sim.csv \
  --joints FFJ2 FFJ3 --config configs/real2sim/s1.yaml \
  --out /tmp/ff_real2sim.html
```

支持 `.png`、单文件交互 `.html` 和配套 JSON 指标。默认显示关节角、电机角、电机张力、关节张力；缺失通道自动省略，同时给 real+sim 时追加对应残差。没有实测关节侧张力时该残差为空，绝不复用电机传感器值。可用 `--channels` 选择速度、加速度、内部长度、`tension_drop`、`contact`、两区摩擦/微滑和耗散；完整示例见 [调参说明](REAL2SIM_GEOMETRY.md)。第三指令字段默认不画，避免 20 N 等指令值压缩实际张力显示范围；需要时用 `--show-tension-command` 或 YAML `curves.show_tension_command: true`，并明确标注 Fallback 不执行该上限。CSV 和指标仍保留该字段。近恒定角度/张力曲线采用最小 0.01°/0.01 N 显示跨度，仅影响坐标轴，不表示传感器分辨率、标定容差或数据修改。默认按首个超过 0.5° 的指令边沿对齐，静态记录用 `--align start`。`--sim-shift-s` 显式设置额外偏移；不要通过任意平移隐藏系统延迟。已测传感器偏置可用 `--real-tension-bias-N` 扣除；`--fallback-only` 可筛选带分支记录的数据。

报告包含关节角/电机角/张力误差、偏差、最大误差和尾窗误差；尾窗统计不自动代表机械已静止。只在重叠时段比较，不外推，不跨 `max_interp_gap_s` 缺口插值。查看 rise/settling、超调、张力峰值、反向运动、同指 partner 扰动和有效采样覆盖；缺失实际分支/速度/第三字段会限制结论。先核对指令及来源，再判断动力学误差。

建议顺序：

1. 确认协议、张力刻度/零点、角度正方向、crossing 符号、mimic 和手姿态。
2. 低速/准静态多个角度辨识回位弹簧与预紧；避免把传感器偏置当弹簧预紧。
3. 用双向慢速、不同加载历史辨识走线柔顺性、静摩擦和接触参数；再用不同加速度及停留时间检查有效质量、接触时间常数和电机速度响应。
4. 以未用于拟合的幅度、速度和同指组合验证关节角、电机角与张力三类曲线。
5. 固定未经辨识的强相关参数，分别检查数值耗散与物理耗散；仅在残差支持时增加更复杂的三维接触模型。

## 检查

纯逻辑检查不启动 Isaac Sim：

```bash
PYTHONPATH=$PWD/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  ~/IsaacLab/.venv/bin/python -m pytest -q tests/test_real2sim_*.py
```

单元测试验证接口和力学不变量，不能替代 PhysX 实际运行、步长/armature 收敛检查或真实数据验证。应分别保存实际执行过的检查结果。

## 动态走线验证记录（2026-11-18）

- 全仓库 **202 项 pytest 通过**；相关 Python 文件的 Ruff lint/format 和 `git diff --check` 通过。
- 新测试覆盖无摩擦惯性张力差、静摩擦保持、松弛/反向、最大长度和停止冲量、摩擦/数值耗散分离、接触历史、状态恢复、远端力矩及 CSV 字段对应关系。
- FFJ2 20° 阶跃：1.2 s、2,400 物理步、401 行，物理步 0.5 ms、内部步 0.1 ms；无力矩裁剪，mimic 最大误差 `3.80e-6 rad`。
- 上述未拟合例子中，末点电机侧/关节侧张力约 `6.962/5.570 N`，全程最大差约 `1.488 N`。这些数值仅证明两通道独立求解，不能当成真机拟合目标；末点不自动代表已达稳态。
- 同时给 THJ1/THJ2 和 FFJ2/FFJ3 加载、降低目标、再加载：1.1 s、2,200 物理步、368 行，四条完整三字段指令；无力矩裁剪，mimic 最大误差 `2.82e-6 rad`。拇指电机速度及两侧张力差发生反转；此短实验中 FFJ2/FFJ3 电机没有反转，不把它当作四指充分卸载实验。
- 两次回放的十通道均检查非负张力、有限状态、接触范围、弹簧行程和长度平衡；往返指令回放另检查三类累计耗散不减。第一点关节/电机加速度仍按定义为 NaN。结果和图保存到 `examples/reference_runs/dynamic/`，`*.validation.json` 保存数值检查。
- 独立指定边界的 20 ms 平滑收绳测试比较 200/100/50/25/12.5 μs 内部步长。100 μs 相对 12.5 μs 参考的电机/远端张力 RMSE 约 `0.00360/0.00571 N`，最大绝对差约 `0.00676/0.01308 N`；误差不严格单调。见 `step_size_probe.json` / `.png`，它是敏感性记录，不是对真机精度或 PhysX 耦合收敛的证明。

直接查看 `j2_curves.html`（角度、电机角、两侧张力）、`j2_routing.html`（张力差和接触状态）、`j2_states.html`（摩擦/微滑/耗散）及 `reversal_curves.html`。这些输入都是生成的测试指令，所有摩擦和质量参数仍待实测数据辨识。

## S1 准静态版本验证记录（动态走线更新前）

- 全仓库 **184 项 pytest 通过**；Ruff 检查与格式检查通过。
- 几何解析梯度通过有限差分检查；0–90° 双关节网格检查路径连续性、力臂符号和回位方向；验证 J1=J2 虚功约化。
- 串联弹簧检查松弛、硬停止、停止后张力继续增长、长度守恒和能量导数；验证独立电机初态、部分环境重置与参数快照。
- 当前预设 FFJ2 20° 阶跃：1.2 s、2,400 物理步、401 行；J1=J2 最大误差 `3.86e-6 rad`，无力矩裁剪，无意外非有限值，长度平衡残差为零。第一点加速度按约定未定义。
- FFJ3 阶跃、包含电机列的严格三字段回放、走线刚度 500/1500 N/m 实际扫描完成。每次保留自己的完整参数快照。
- 曲线、CSV、元数据与扫描结果位于 `examples/reference_runs/quasistatic/`；当前预设主示例为 `finger_j2_final.html` / `.png`。

以上是软件和数值验证，输入仍为仿真生成的测试记录，不是实机标定数据。

## 早期模型验证记录（更新前）

已在当前 PhysX 环境完成以下功能检查：

- FFJ2 30° 生成阶跃：1.0 s、10,000 物理步、335 行输出；四指 J1=J2 最大误差约 `5.96e-8 rad`，无力矩裁剪或非有限值。
- THJ1 从 5° 到 25° 的生成阶跃，使用负 crossing 假设：0.6 s、6,000 物理步、201 行；无力矩裁剪或非有限值。
- 用仿真 CSV 进行严格三字段回放，以及两个参数值的实际批量扫描均通过。
- 最终全仓库 **163 项 pytest 通过**；Ruff lint 全仓库通过，real2sim 新增/修改 Python 文件的格式检查通过。依赖弃用提示和旧采集数据缺少指令字段的预期提示保留。

这些是软件接口、时序、PhysX 力矩施加及数值有限性的 smoke 检查。回放输入中的仿真 CSV 只是测试产物，不是实机记录；负 crossing 也不是经过验证的实际绕线方向。尚未使用真实角度/张力曲线辨识参数，不能由这些检查推断实机精度或已完成 real2sim 标定。
