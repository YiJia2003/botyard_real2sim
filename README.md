# botyard_real2sim

Botyard 灵巧手的独立空载 real2sim 调试项目。包含 Fallback 控制、变力臂几何、腱绳走线摩擦、有限行程弹簧与等效惯量、电机角观测，以及实际连接 Isaac Lab / Isaac Sim **PhysX** 的运行入口。

运行只需要本项目和已经配置好的 Isaac Lab 环境。手部 USD、关节/引擎配置、测试、S1 绕线图片及 STEP 均已复制到本项目，不从原仓库导入代码或引用原仓库的资产。原项目文件保持不变。

## 开始调试

当前机器直接复用 `/home/cyj/IsaacLab/.venv/bin/python`，无需重新安装仿真引擎。启动脚本会设置本项目的 Python 路径，输出默认写到本项目当前目录：

```bash
cd /home/cyj/botyard_real2sim

# 先查看命令选项，不启动引擎
./run.sh replay --help

# 带窗口的单关节阶跃；去掉 --gui 则无窗口运行
./run.sh replay --gui --joint FFJ2 --target-deg 20 \
  --max-speed-deg-s 120 --tension-command-n 20 \
  --duration 2 --out logs/demo/sim.csv

# 打开生成的 HTML，查看关节角、电机角及两侧张力
./run.sh curves --sim logs/demo/sim.csv --joints FFJ2 FFJ3 \
  --out logs/demo/curves.html
```

每次仿真输出 `sim.csv`、`sim.params.yaml`（完整有效参数）及 `sim.metadata.json`（初态、引擎版本和运行检查）。`--gui` 中可见手部运动；绳和弹簧由力学层计算后向引擎施力，USD 暂无可见绳/弹簧实体。

其他环境用 `BOTYARD_REAL2SIM_PYTHON=/path/to/isaac/python ./run.sh ...` 指定兼容的解释器。当前验证环境为 Python 3.12、Isaac Lab 42.1.0、Isaac Sim 6.2.0.0、Torch 2.11.0+cu130；Isaac Lab 3.0 源码接口中的 `.torch`、`launch_simulation` 和 `set_effort_index` 均被运行入口使用。

## 用真实记录做 real2sim

真实记录须同步保存实际发送的三字段指令和三类反馈。字段单位和事件语义见 [CSV 契约](docs/REAL2SIM.md#csv-契约反馈采样指令与发送事件)：

- `time_s` 或整数 `timestamp_ns`；建议记录 `command_seq` / `command_event` 及 `is_fallback`。
- `<joint>/cmd_rad`、`cmd_speed_rad_s`、`cmd_tension_N`：实际发送的目标角、最大速度、第三张力字段。
- `<joint>/position_rad`：关节角。
- `<joint>/motor_position_rad`：连续、不取模的电机**绕线轮输出轴角**，并正确换算减速比和方向。
- `<joint>/tension_N`：电机侧应力片张力，位于走线摩擦和中间弹簧之前。

先记录真实实验，然后将同一条指令时间线交给仿真：

```bash
./run.sh replay --commands /path/to/real.csv --strict-commands \
  --config configs/real2sim/s1.yaml --out logs/calibration/sim.csv

./run.sh curves --real /path/to/real.csv --sim logs/calibration/sim.csv \
  --joints FFJ2 FFJ3 --channels angle motor tension \
  --out logs/calibration/compare.html
```

只使用实测反馈初始化并评价误差，后续关节运动由 PhysX 自行求解。当前是离线 CSV 回放入口，没有实时 ROS2 bridge。
没有实测远端张力时，`distal_tension_N` 仅用于查看仿真内部响应；不会用电机张力代替远端实测值。

仓库提供 [生成指令示例](examples/commands/finger_reversal.csv)，用于检查双关节加载/放绳流程，**不是实测数据**：

```bash
./run.sh replay --commands examples/commands/finger_reversal.csv --strict-commands \
  --duration 2 --out logs/example/sim.csv
```

agent 标定入口见[项目 skill 路由](.agents/skills/real2sim-agent-router/SKILL.md)、[标定流程](docs/AGENT_CALIBRATION.md)和仓库级 [real2sim-calibration skill](.agents/skills/real2sim-calibration/SKILL.md)。技能放在 `.agents/skills/`；Codex 可通过仓库指引读取，DeepSeek Harness 可在其运行时加载仓库 `AGENTS.md` 和项目 skills 时使用相同路径。本仓库未验证 DSH 的自动启动加载。先用 `data` 校验含硬件信息和 CSV 哈希的 manifest，再用 `parameters` 检查有效参数；`evaluate` 执行严格评分，`fit` 在明确的参数边界内搜索，并用独立 validation manifest 对比基线：

```bash
./run.sh data --manifest /data/train/manifest.json --strict
./run.sh parameters --config configs/real2sim/s1.yaml
./run.sh evaluate --real /data/train/capture.csv --sim logs/calibration/sim.csv \
  --joints FFJ2 --tolerance angle_deg=2 --tolerance motor_angle_deg=3 --tolerance tension_N=1
./run.sh fit --help
```

拟合阈值与参数边界必须有测量或实验依据。fit 会报告训练/验证基线与逐通道结果；留出集通过显式容差且相对留出基线改善后才接受为曲线拟合候选，这仍不等于已证明物理模型完成真机标定。

## 本地 ↔ 远程数据桥（real2sim experiment）

真机采集在本机、PhysX/拟合在远程时，用 `experiment/` 数据包和 `real2sim/experiment/` 对接。桥保证**先上传到 staging、复核哈希与 manifest、写 READY、再原子改名**，远程只消费 `READY.json` 存在的包；远程结果同样先 staging 再发布。

```bash
cd /home/botyard/real2sim/botyard_real2sim

# 本机只需 Python 3.12 + ssh + rsync；不需要 Isaac 环境
python3 scripts/real2sim_bridge.py doctor

# 原始采集目录 -> 数据包（需要 capture_meta.json，显式容差）
python3 scripts/real2sim_bridge.py build-capture \
  /home/botyard/real2sim/embedding_ws/runs/<run> \
  --meta /path/to/capture_meta.json --joints MFJ2 MFJ3 --split train \
  --tolerance angle_deg=2 --tolerance motor_angle_deg=3 --tolerance tension_N=1

python3 scripts/real2sim_bridge.py verify <bundle_id>
python3 scripts/real2sim_bridge.py publish <bundle_id>
python3 scripts/real2sim_bridge.py notify <bundle_id> --timeout-s 3600
python3 scripts/real2sim_bridge.py results show <bundle_id>
```

远程目录、bundle/result/next-data 契约、staging 生命周期、SSH 密钥与故障处理见 [数据桥文档](docs/REAL2SIM_BRIDGE.md)。本地目录说明见 `/home/botyard/real2sim/experiment/README.md`。

## 修改和扫描参数

默认使用 [configs/real2sim/s1.yaml](configs/real2sim/s1.yaml)。它启用变力臂几何和有状态动态走线模型；[default.yaml](configs/real2sim/default.yaml) 保留旧准静态分支，仅用于复现旧实验。

可编辑 YAML，也可逐次覆盖单关节参数：

```bash
./run.sh replay --joint FFJ2 --target-deg 20 --duration 2 \
  --set tendon.routing_mass_kg.FFJ2=0.003 \
  --set tendon.routing_friction_mu_static.FFJ2=0.30 \
  --set tendon.routing_contact_release_tau_s.FFJ2=0.10 \
  --out logs/tuned/sim.csv

# 先检查扫描计划；实际执行时去掉 --dry-run
./run.sh sweep --parameter tendon.routing_mass_kg.FFJ2 --values 0.001 0.002 0.003 \
  --out-dir logs/mass_sweep --dry-run -- \
  --commands examples/commands/finger_reversal.csv --strict-commands --duration 2

# 检查张力差、接触状态和摩擦
./run.sh curves --sim logs/tuned/sim.csv --joints FFJ2 \
  --channels tension distal_tension tension_drop contact \
  --out logs/tuned/routing.html
```

摩擦、接触、等效质量与弹簧参数目前均为**未拟合初值**。先核对慢速几何和预紧，再用不同速度、加速度、停留和往返曲线区分传动柔度、摩擦与惯性。内部积分有数值耗散，拟合前需检查步长敏感性。
电机和主关节转轮均采用**半径 5 mm**。Fallback 沿用当前固件行为：第三张力字段被保留，不直接执行为张力目标或上限。

## 项目结构

```text
src/botyard_real2sim/
  controller.py, runtime.py, protocol.py, commands.py  控制、调度和记录接口
  model.py, geometry.py, transmission.py              腱绳/关节力学
  routing_dynamics.py                                摩擦、接触记忆、惯量和硬停止
  simulation/hand_cfg.py, physics.py, ../scene.py      本地 PhysX 适配
  assets/data/hand/boya.usda                          本地手部资产
  cli/replay.py, curves.py, sweep.py                  回放、曲线和扫描入口
  cli/data.py, parameters.py, evaluate.py, fit.py     agent 数据/参数/评分/拟合入口
  cli/bridge.py, bridge/, scripts/real2sim_bridge.py  本地↔远程 staging/READY 数据桥
configs/real2sim/                                    调试参数
scripts/                                            兼容脚本入口
tests/                                              real2sim 测试
docs/                                               方程、运行说明、数据桥、绕线图片、STEP、CAD 证据
examples/commands/                                  生成的指令示例
examples/reference_runs/                            提取时的历史仿真曲线和数值检查
```

参考入口：[完整运行和 CSV 说明](docs/REAL2SIM.md)、[几何/摩擦/惯性模型与参数](docs/REAL2SIM_GEOMETRY.md)、[历史动态曲线](examples/reference_runs/dynamic/j2_curves.html)。历史 CAD URI 和运行元数据保留提取时原文，表示来源，不参与新项目路径解析。

## 测试与安装

```bash
./run.sh test
./run.sh check
```

已有 Isaac Lab 环境时，也可安装为 Python 包：

```bash
uv pip install --python /home/cyj/IsaacLab/.venv/bin/python --no-deps -e .
/home/cyj/IsaacLab/.venv/bin/botyard-replay --help
/home/cyj/IsaacLab/.venv/bin/botyard-curves --help
```

基础安装只依赖 PyYAML。离线建模需要 `[model]` 的 Torch，绘图需要 `[curves]`，参数拟合需要 `[calibration]` 的 SciPy，测试需要 `[dev]`；启动 PhysX 另需已配置的兼容 Isaac Lab / Isaac Sim 环境，不会通过本包自动安装整套引擎或 RL 依赖。

CLI 实现位于包内，wheel 包含 S1/旧版参数和手部 USD，安装后可在其他目录使用。源码运行优先读取 `configs/real2sim/s1.yaml`；wheel 运行读取打包预设，实际标定建议通过 `--config` 指定自己的 YAML。
提取范围、原始文件哈希和适配说明保存在 [EXTRACTION_MANIFEST.json](EXTRACTION_MANIFEST.json)。

本次提取验证：**181 项测试通过**；阻止原项目导入的 PhysX 1.2 s 回放完成，401 行输出，无力矩裁剪；关节角、电机角和两侧张力与提取前同条件参考逐点一致。wheel 的三个安装入口、内置配置和资产也在独立临时环境验证。详见 [提取验证记录](docs/EXTRACTION.md)。
