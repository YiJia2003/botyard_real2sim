# real2sim 数据桥：本地采集 ↔ 远程仿真

本文档定义硬件主机与远程仿真主机（`cyj@10.249.133.227`）之间的数据包、任务、结果和补采请求契约。**硬件主机**负责真机控制和采集；**仿真主机**运行本仓库，由远程agent完成仿真、模型/参数优化与结果反馈。文中的“本机/本地”均指硬件主机，不随阅读位置变化。

> 当前状态（2026-11-20）：SSH/Rsync 专用密钥、远程 staging→READY→原子改名、远程 DSH agent 通知、结果发布与回读均已实机联通。端到端自检见 [§12](#12-已验证记录)。

**采集适用边界（2026-11-20）：** 硬件主机使用RS485 ROS控制和三板SWD RAM采样。`build-capture`仍仅接收旧host对齐输入；SWD到标准数据的适配、时间误差预算及源质量验收尚未完成，现有SWD记录只作诊断，不能直接标为`fit_ready`。H7本地微秒、主机纳秒和传感器采样时刻是不同概念。SSH/READY自检成功不能覆盖原始运行的失败或拟合限制。

**远程阅读入口：** 本文及仓库内的[标定流程](AGENT_CALIBRATION.md)、[远程桥操作说明](../bridge/README.md)足以说明交接职责与格式。远程仓库无需同步`embedding_ws`、ROS工作区或本机端口配置。硬件主机的详细操作文档位于其`/home/botyard/real2sim/embedding_ws/docs/real2sim/`，只是采集端参考路径，不是远程依赖。

## 1. 拓扑与职责

```text
本机 botyard（真机 / ROS2 / 烧录器）                远程 cyj-B760M（Isaac Lab / PhysX / RTX 4070）
┌─────────────────────────────────────┐            ┌────────────────────────────────────────┐
│ 单/双关节runner + 三板SWD采集器     │            │ botyard_real2sim 仿真与标定工具链        │
│  三板SWD原始记录；适配/误差验收待完成 │            │  run.sh data/parameters/replay/          │
│  ↓ 旧输入导出CSV；SWD适配尚未完成   │            │           evaluate/curves/fit             │
│ experiment/<bundle_id>/               │  rsync     │ real2sim/experiment/<bundle_id>/         │
│  capture.csv manifest.json TASK.json  │ ─────────▶ │   capture.csv manifest.json TASK.json    │
│  BUNDLE.json SHA256SUMS README.md     │  staging   │   BUNDLE.json SHA256SUMS READY.json      │
│                                       │            │   （remote_verify.py 校验+原子改名）      │
│                                       │  DSH API   │                                          │
│                                       │ ─────────▶ │ 远程 DSH agent（botyard_real2sim 会话）  │
│                                       │ 127.0.0.1  │   → baseline/evaluate/fit                │
│                                       │ :3081      │   → real2sim/results → RESULT_READY.json │
│ experiment/<id>/results/              │ ◀───────── │ real2sim/results/<bundle_id>/            │
│  （只读镜像，可选）                    │  ssh cat   │   RESULT.json report.md next_data_request│
└─────────────────────────────────────┘            └────────────────────────────────────────┘
```

| 环节 | 本机（真机侧） | 远程（仿真侧） |
|---|---|---|
| 激励与采集 | 改固件 / ROS2 控制 / 采 J2、J3、H7 数据 | 不接触硬件 |
| 时间基准 | 主机 `CLOCK_MONOTONIC_RAW` 与三板本地原始计数分别保存；SWD诊断仅拟合 G4→H7 本地时间 | 按原始时轴/传输偏差/源年龄证据复核，不能仅信拟合残差 |
| 数据打包 | 仅合格旧host对齐输入可由 `build-capture` 构建；SWD适配待完成 | 只读 `real2sim/experiment/`，不改输入包 |
| 完整性 | 本地先校验，再上传 | 复核 SHA256、manifest/CSV 哈希、严格数据集校验 |
| 发布 | rsync 到 `.staging/*.partial-*` | `remote_verify.py` 写 `READY.json` 后原子改名 |
| 仿真/拟合 | 不跑 Isaac | 严格按 `TASK.json` 容差和项目流程执行 |
| 结果 | 读取或拉取远程结果，按反馈补采 | `remote_publish_result.py` 原子发布 |
| 补采 | `plan-requests` 把远程请求转为激励命令计划 | 在 `next_data_request.json` 写清缺什么数据 |

**硬边界：**
- 本机不把仿真结果当真机证据；远程不把仿真 CSV 当实机数据。
- 远程输入目录`real2sim/experiment/`只读，任务结果写`real2sim/results/`；模型、代码与参数文件仍可按任务修改。
- 第三指令字段在 Fallback 中只是遥测，不是张力目标或上限。
- 远程拟合的 `accepted=true` 仍只是曲线拟合候选，不是“物理模型已标定”。

## 2. 目录布局

### 2.1 本机

```text
/home/botyard/real2sim/
  experiment/
    README.md
    <bundle_id>/                 # 已构建、待/已发布的数据包
      capture.csv                # canonical 宽表 CSV
      manifest.json              # 数据集 manifest（项目 schema_version=1）
      TASK.json                  # 本次远程任务与显式容差
      BUNDLE.json                # 文件表、manifest/csv 哈希
      SHA256SUMS                 # 含 BUNDLE.json 自身
      README.md
      raw/                       # 原始证据；旧输入形态见§9
    .staging/                    # 本地构建临时目录（成功后清空）
    .replaced/                   # 显式 --replace 时保留的旧包（审计）
    .bridge/state/<id>.json      # 发布/通知/最终消息的本地状态
    _plans/<request_id>/         # plan-requests 生成的采集命令计划
  botyard_real2sim/
    src/botyard_real2sim/bridge.py         # 打包/哈希/DSH/SSH 核心
    src/botyard_real2sim/cli/bridge.py     # 本地 CLI
    bridge/remote_verify.py                # 远程校验器（同步到远程）
    bridge/remote_publish_result.py        # 远程结果发布器（同步到远程）
    scripts/real2sim_bridge.py             # 本地 CLI 兼容入口
    docs/REAL2SIM_BRIDGE.md                # 本文
~/.config/botyard-real2sim/bridge.env      # 600，含远程 token；不进 git
~/.ssh/real2sim_bridge_ed25519             # 专用桥密钥
```

### 2.2 远程

```text
/home/cyj/botyard_real2sim/
  run.sh, src/, configs/, ...              # 项目本体
  real2sim/
    experiment/
      .staging/<bundle_id>.partial-<nonce> # 上传中；未 READY 不可消费
      .replaced/                           # --replace 保留的旧正式包（审计）
      <bundle_id>/                         # READY.json 存在才可消费
    results/
      .staging/<bundle_id>.partial/        # 结果写入中
      .replaced/                           # --replace 保留的旧结果包
      <bundle_id>/                         # RESULT_READY.json 存在才可回读
    bridge/
      remote_verify.py
      remote_publish_result.py
    logs/                                  # 可选运行日志
```

`bridge/`是版本控制中的脚本与模板源目录，`real2sim/bridge/`是远程运行副本。硬件主机执行`publish`时默认先同步该目录（除非指定`--no-sync-scripts`）；这不会同步模型源码或`docs/`。因此更新仓库代码后，仍应确认运行副本来自同一版本；需要立即更新时可单独同步`bridge/`到该运行目录。输入/结果目录是运行数据，不能用源码同步覆盖。

## 3. 数据包契约

### 3.1 文件

| 文件 | 必需 | 作用 |
|---|---|---|
| `capture.csv` | 是 | canonical 宽表；列契约见 §3.3 |
| `manifest.json` | 是 | 项目数据集 manifest；`recording.csv` 指向 `capture.csv` |
| `TASK.json` | 是 | 远程任务：intent、split、joints、显式容差、期望产物 |
| `BUNDLE.json` | 是 | 数据包记录：bundle_id、protocol、文件表、manifest/csv 哈希 |
| `SHA256SUMS` | 是 | 所有 payload 文件 + `BUNDLE.json` 的 SHA-256 |
| `README.md` | 是 | 人读摘要；不是证据 |
| `raw/` | 建议 | 旧输入保留 samples/commands/board_samples_aligned/clock_fit/metadata；未来SWD adapter须保留三板BIN/JSONL、配置与质量报告及哈希，当前尚未实现 |
| `READY.json` | 远程生成 | 远程校验通过后写入；**有它才是 ready 数据接口** |
| `VERIFY_FAILED.json` | 失败时 | 远程校验失败原因，留在 staging 供排查 |

`BUNDLE.json` 的 `files` 不含自身；`SHA256SUMS` 包含 `BUNDLE.json`，因此包记录本身也被钉住。`READY.json` 不参与 payload 哈希，避免“写 marker 改哈希”的循环。

### 3.2 manifest 必填证据

严格校验（`./run.sh data --manifest ... --strict`）要求：

- `schema_version=1`、`dataset_id`（必须等于 `bundle_id`）、`split=train|validation|test`；
- `robot.id`、`robot.hardware_revision`；
- `firmware.version`、`control_mode`、`fallback_confirmed=true`、`fallback_evidence`；
- `recording.csv/sha256/source_recording_id/source_sha256/timestamp_column/clock_id/command_semantics/command_time_source`；
- `recording.sensor_sync_method`、`max_sensor_skew_s`、`max_sensor_skew_source`；
- `joints`（本项目 10 个屈曲 DOF 的子集；当前采集通常 `MFJ2 MFJ3`）；
- `signals.position/motor_position/tension/commands`，且每关节 calibration `status=applied` + `source`；
- `motor_position.semantics=unwrapped_spool_output_angle`，每关节 `gear_ratio>0/encoder_sign=±1/unwrapped=true/gear_ratio_applied=true`；
- `tension.sensor_location=motor_side_before_routing`；
- `experiment.protocol/root_pose/prehistory`；`prehistory.recorded=true` + `initial_state_source`；
- `quality_gates.max_sensor_skew_s` + `max_sensor_skew_source`；实测 skew 不得超过限值；
- 每行 `is_fallback=true`；每条指令边沿有 `command_event` + `command_timestamp_ns`；
- `train` 与 `validation` 必须是**两条独立物理录制**（不同 CSV/哈希/source id）。

不要从预设推容差或参数边界；manifest 里的容差/边界必须有测量或明确实验决策来源。

### 3.3 CSV 契约

时间列使用 `timestamp_ns`（整数纳秒，推荐）或 `time_s`。当前桥使用主机 `CLOCK_MONOTONIC_RAW` 纳秒。

必需通道（对每个 `joints`）：

| 列 | 单位/要求 |
|---|---|
| `<joint>/position_rad` | rad；实测关节角 |
| `<joint>/motor_position_rad` | rad；连续、不取模的**绕线轮输出轴角**，已做方向/减速比换算 |
| `<joint>/tension_N` | N；电机侧、走线摩擦之前的应力片张力 |
| `<joint>/cmd_rad` | rad；实际发送的目标角 |
| `<joint>/cmd_speed_rad_s` | rad/s；实际发送的速度字段 |
| `<joint>/cmd_tension_N` | N；实际发送的第三字段（Fallback 不执行） |

事件/分支列：`command_event`、`command_seq`、`command_timestamp_ns`、`is_fallback`。

可选：`velocity_rad_s`、`motor_velocity_rad_s`、`<joint>/command_seq`、原始 `effort_raw`。

**以下为旧 host 对齐输入的解析契约，不是当前 SWD 采集输出：**

- 关节角/速度来自 `samples.csv`；
- 电机角/张力来自 `board_samples_aligned.csv` 的 J2/J3 板（`module_position_rad` / `tension_N_from_firmware_scale`）；
- 指令来自 `commands.csv` 的实际发布事件；
- 旧输入要求板卡时间已映射为主机纳秒。插值约定为最大50ms且不外推；当前导出器仍有ns/s阈值单位问题待修，不能据此段认定异步插值正确。原始文件保留在 `raw/`；
- 第三字段默认按 `effort_raw × 0.01` 记为 `cmd_tension_N`；这个刻度是当前约定的**待固件复核值**，写在 manifest `provenance.third_field`。

### 3.4 `TASK.json`

```json
{
  "schema_version": 1,
  "protocol": "botyard-real2sim-bridge/v1",
  "bundle_id": "r2s-20261120T090000Z-mfj2-step",
  "intent": "calibration",
  "created_at_utc": "2026-11-20T09:00:00Z",
  "split": "train",
  "joints": ["MFJ2", "MFJ3"],
  "requested_actions": ["validate_dataset", "replay_baseline", "evaluate", "fit_if_tolerances_allow"],
  "tolerances": {"angle_deg": 2.0, "motor_angle_deg": 3.0, "tension_N": 1.0},
  "remote": {
    "read_only_experiment": true,
    "result_dir": "real2sim/results/<bundle_id>",
    "staging_dir": "real2sim/results/.staging/<bundle_id>.partial",
    "publish_with": "real2sim/bridge/remote_publish_result.py",
    "ready_marker": "RESULT_READY.json",
    "next_data_request": "next_data_request.json"
  },
  "result_contract": {
    "required_files": ["RESULT.json", "report.md"],
    "allowed_status": ["completed", "partial", "blocked", "failed", "acknowledged"],
    "claims_scope": "curve_fit_candidate_only"
  },
  "notes": "本次实验目的与已知限制"
}
```

`intent=transport_selftest` 只验证桥，不跑 PhysX、不拟合。

## 4. 生命周期（严格顺序）

```text
① 采集 raw run（当前SWD诊断停在此，待适配/质量验收）
   ↓
② build-capture（canonical CSV + manifest + TASK + BUNDLE + SHA256SUMS）
   ↓ 本地 verify（完整性 + 严格数据集）
③ rsync → 远程 experiment/.staging/<id>.partial-<nonce>/
   ↓
④ 远程 remote_verify.py
   ├─ 拒绝符号链接 / 路径穿越 / 多余文件
   ├─ 复核 SHA256SUMS、BUNDLE.files、manifest.recording.sha256
   ├─ ./run.sh data --manifest ... --strict
   └─ 写入 READY.json
   ↓
⑤ 同一文件系统内 mv → experiment/<id>/   （原子改名）
   ↓
⑥ 本地 notify → 远程 DSH 会话（只发已 ready 的 id）
   ↓
⑦ 远程 agent 只读 real2sim/experiment/<id>/，在 real2sim/results/.staging/<id>.partial/ 写结果
   ↓
⑧ remote_publish_result.py 校验 RESULT.json + 哈希，写 RESULT_READY.json
   ↓
⑨ 同一文件系统内 mv → results/<id>/
   ↓
⑩ 本地 results show / pull、watch；读取 report.md + next_data_request.json
   ↓
⑪ 若缺数据：plan-requests 生成单关节/配对采集计划 → 回到 ①（新 bundle）
```

为什么必须 staging + READY + 改名：

- 上传中的半包不会被远程 agent 消费；
- 远程只对完整包做校验，失败留在 staging，可重试而不会污染正式接口；
- 正式目录名只在所有哈希与严格校验通过后出现，消费者只需判断 `READY.json`；
- 结果侧同样原子：`RESULT_READY.json` 出现前，`results/<id>/` 不存在。

## 5. 硬件主机常用命令

先安装/确认：本机只需要 Python 3.12 + `rsync` + `ssh`；不需要 Isaac。远程需要项目现有 Isaac 环境（`run.sh` 会用它）。

```bash
cd /home/botyard/real2sim/botyard_real2sim

# 0. 配置（已生成则只需 doctor）
python3 scripts/real2sim_bridge.py doctor

# 1. 仅从独立验收合格的旧host对齐输入构建；不适用于当前SWD诊断
python3 scripts/real2sim_bridge.py build-capture \
  /path/to/legacy-host-aligned-run \
  --meta /path/to/capture_meta.json \
  --joints MFJ2 MFJ3 --split train \
  --tolerance angle_deg=2 --tolerance motor_angle_deg=3 --tolerance tension_N=1

# 2. 本地复核
python3 scripts/real2sim_bridge.py verify <bundle_id>

# 3. 上传 + 远程 ready
python3 scripts/real2sim_bridge.py publish <bundle_id>

# 4. 通知远程 agent 并等待其最终消息
python3 scripts/real2sim_bridge.py notify <bundle_id> --timeout-s 3600

# 5. 读结果 / 等结果 / 可选拉取
python3 scripts/real2sim_bridge.py results show <bundle_id>
python3 scripts/real2sim_bridge.py watch <bundle_id>
python3 scripts/real2sim_bridge.py results pull <bundle_id> --replace

# 6. 把远程补采请求转成激励命令计划
python3 scripts/real2sim_bridge.py plan-requests <bundle_id> \
  --out-dir /home/botyard/real2sim/experiment/_plans/<request_id>

# 7. 临时问远程 agent 一句话（不经过数据包）
python3 scripts/real2sim_bridge.py say "请检查当前 results 目录并回复状态" --timeout-s 120
```

`build-selftest` 可随时生成一个合成传输自检包：

```bash
python3 scripts/real2sim_bridge.py build-selftest
```

> 本地没有 Isaac venv 时，`./run.sh bridge` 不可用；请直接使用 `python3 scripts/real2sim_bridge.py`。远程则可用 `./run.sh bridge-verify` / `./run.sh bridge-result`。

## 6. 远程结果契约

远程结果目录：

```text
real2sim/results/<bundle_id>/
  RESULT.json                 # 结构化结论
  report.md                   # 人读报告
  next_data_request.json      # 补采请求（可为空）
  baseline/                   # sim.csv、params.yaml、metadata.json、evaluation.json
  fit/                        # 可选；fit.json、参数快照、逐 trial 产物
  curves/                     # 可选；HTML/PNG/JSON
  RESULT_READY.json           # 发布回执：文件表 + 哈希
```

`RESULT.json` 最小结构：

```json
{
  "schema_version": 1,
  "protocol": "botyard-real2sim-bridge/v1",
  "bundle_id": "r2s-...",
  "status": "completed|partial|blocked|failed|acknowledged",
  "claims_scope": "curve_fit_candidate_only",
  "summary": "一句话结论",
  "dataset": {"dataset_id": "...", "split": "train"},
  "checks": {"ready_verified": true, "dataset_strict": true, "replay_status": "completed"},
  "baseline": {"evaluation": "baseline/evaluation.json"},
  "fit": {"accepted": false, "reason": "..."},
  "diagnosis": {"summary": "...", "residual_patterns": []},
  "next_data_request": "next_data_request.json"
}
```

`RESULT_READY.json` 由 `remote_publish_result.py` 写入，包含 `result_sha256`、`report_sha256`、`files` 表；本地 `results show` 会先检查它。

## 7. 补采请求契约

远程agent在`next_data_request.json`中写清残差、欠缺观测和激励条件；硬件主机agent收到后核对机台配置、生成计划、验证并执行。`plan-requests`和所有采集runner均在硬件主机运行，远程无需安装这些工具。

以下为2026-11-20的硬件能力快照，供远程设计补采；每条数据最终以随包归档的固件、配置、实际输入和测量质量为准：

| 项目 | 请求能力与解释 |
|---|---|
| 机台与范围 | 左手、空载、固定手体姿态，针对中指MFJ2/MFJ3 |
| 配对目标角 | MFJ2为0.5–89.5°，MFJ3为2.5–89.5°；目标范围不保证实际可达 |
| 每轴请求速度上限 | 各自独立15–360°/s；保持原Fallback PID及1°位置死区，没有15°/s实际输出下限 |
| 电机轴目标死区 | 200rad/s，与关节°/s不是同一量；原PID较小输出仍可能归零 |
| 板内模块限速读回 | J2为4rad/s、J3为6.25rad/s；不能据请求360°/s宣称实际达到360°/s |
| 单关节profile | 幅值≤0.10rad；用于基础小幅激励，不能指定搭档定姿 |
| 配对激励 | 同帧两关节、互为定姿、独立速度与四方向、全行程、同步姿态穿越、相位/速比、扫频、加载/反转/驻留 |
| 参考轨迹与覆盖 | 穿越轨迹可预加速，并给低速轴留起步行程；参考加速度/恒速平台不等于实测，缺格由真实源新值与时间证据审计 |
| 电机直接速度控制 | 当前通用runner尚未支持，不能以关节位置伺服速度上限替代 |

远程可据此请求某一固定搭档角度、角度/速度组合、相位、方向、驻留或独立validation录制，并说明要区分的参数/结构假设。无需先读取硬件主机的本地文件。单关节请求示例：

```json
{
  "schema_version": 1,
  "protocol": "botyard-real2sim-bridge/v1",
  "bundle_id": "r2s-...",
  "status": "needs_more_data",
  "reason": "缺少低速双向卸载段，回位弹簧与静摩擦不可分",
  "experiments": [
    {
      "id": "mfj2_slow_bidirectional_r1",
      "joints": ["MFJ2"],
      "profile": "triangle",
      "center_deg": 20,
      "amplitude_deg": 5,
      "speed_cap_deg_s": 30,
      "duration_s": 8,
      "frequency_hz": 0.1,
      "directions": ["flex", "extend"],
      "dwell_s": 0.5,
      "repeats": 3,
      "purpose": "分离回位弹簧预紧与低速走线静摩擦"
    }
  ],
  "required_channels": ["position_rad", "motor_position_rad", "tension_N"],
  "acceptance_tolerances": {"angle_deg": 2, "motor_angle_deg": 3, "tension_N": 1}
}
```

上述为`execution_mode=separate`（默认）的旧单关节请求。`directions` / `dwell_s`仅描述意图，现有profile内部固定方向与阶段。模板幅值5°在单关节profile的0.10rad上限内，但仍须核对中心余量。仅把`joints`改成两个关节会生成两次独立运行，不能表达搭档固定角度或同时运动。

硬件主机的`plan-requests`把单关节请求转成`r2s_run.py`命令（默认100Hz、速度上限0.2618rad/s约15°/s、`--effort-raw 0`），输出`plan.json`与`run_plan.sh`。生成脚本只供本机离线预览，实际执行由硬件主机agent另取新`--out`并显式加`--execute`。

需要搭档定姿、两关节独立速度或联合轨迹时，使用`execution_mode=paired`、`joints=[MFJ2,MFJ3]`和完整的`pair_plan`。例如固定J3=30°，J2从10°阶跃到12°后保持：

```json
{
  "id": "j2_step_at_j3_30",
  "joints": ["MFJ2", "MFJ3"],
  "execution_mode": "paired",
  "repeats": 1,
  "pair_plan": {
    "schema": "real2sim-pair-plan-v1",
    "joint_order": ["MFJ2", "MFJ3"],
    "metadata": {"experiment_id": "j2_step_at_j3_30", "family": "remote_supplement", "split": "train", "repeat": 1, "conditions": {"active_joint": "MFJ2", "held_joint": "MFJ3", "held_angle_deg": 30, "history": "hold_step_hold", "dwell_s": 1, "direction": [1, 0]}},
    "points": [
      {"offset_ns": 0, "phase": "hold", "positions_rad": [0.17453292519943295, 0.5235987755982988], "speed_limits_rad_s": [0.2617993877991494, 0.2617993877991494]},
      {"offset_ns": 1000000000, "phase": "hold", "positions_rad": [0.17453292519943295, 0.5235987755982988], "speed_limits_rad_s": [0.2617993877991494, 0.2617993877991494]},
      {"offset_ns": 1010000000, "phase": "step", "positions_rad": [0.20943951023931956, 0.5235987755982988], "speed_limits_rad_s": [0.2617993877991494, 0.2617993877991494]},
      {"offset_ns": 3010000000, "phase": "hold", "positions_rad": [0.20943951023931956, 0.5235987755982988], "speed_limits_rad_s": [0.2617993877991494, 0.2617993877991494]}
    ]
  }
}
```

把该条放进外层`experiments`，硬件主机的桥会冻结payload，为每次重复生成一条`r2s_pair_run.py --plan ... --out ...`离线检查命令。实际执行时保持同帧双关节输入，不拆成两个单关节运行。未知mode、缺payload或两关节错序会拒绝；未填mode但有`pair_plan`时也按paired处理。桥只作结构检查，目标范围、时间间隔、速度、wire量化和ROS控制器顺序由硬件主机的配对校验器完整检查。

`repeats>1`时保留原始payload，再生成每次独立冻结计划：轨迹points保持一致，`metadata.experiment_id`追加`-r01`等后缀，`repeat`为1起整数，`source_experiment_id`保留原ID，避免重复录制在覆盖审计中混为一个实验。一次请求只录制一条既定campaign ID时使用`repeats=1`，保留原metadata。

配对计划包含2–6001点、时间从0递增、间隔≥10ms、总长≤60s，位置与速度单位分别为rad和rad/s。精确方向、停留和速度变化写入points；metadata保留experiment_id、family、split、repeat及conditions。远程可使用本仓库模板表达完整payload；参考轨迹和候选矩阵由硬件主机维护，无需在远程安装。生成计划、成功执行、实际状态覆盖和参数可辨识性分别判断。

若远程只确定了观测缺口，可先给出`purpose`、`requested_conditions`和`local_preparation_required=true`，由硬件主机agent补齐具体计划。此类语义请求尚不可直接交给`plan-requests`，paired请求必须先形成完整`pair_plan`；不能因为缺少本机工具而伪造已执行或已覆盖的状态。

例如下面一项只描述联合姿态附近的速度缺口，不要求远程生成波形：

```json
{
  "id": "coupled_speed_gap",
  "joints": ["MFJ2", "MFJ3"],
  "execution_mode": "paired",
  "local_preparation_required": true,
  "requested_conditions": {
    "target_pose_deg": [45, 30],
    "signed_velocity_deg_s": [60, -30],
    "history": "start from rest, accelerate before the target pose, then hold",
    "split": "validation"
  },
  "repeats": 3,
  "purpose": "检查该联合姿态下的反向耦合残差，区分走线摩擦与控制/采样延迟"
}
```

`requested_conditions`表达观测意图，字段由两侧agent解释；它不是已验收的实际速度或机器可直接执行的计划。硬件主机应反馈能否实现、最终计划和未满足条件，不得静默改成两个独立run或将目标速度当成实测。

可直接复制的模板：

- [capture_meta.example.json](../bridge/capture_meta.example.json)
- [next_data_request.example.json](../bridge/next_data_request.example.json)
- [next_data_request_pair.example.json](../bridge/next_data_request_pair.example.json)

## 8. 远程 agent 命令协议

本地 `notify` 通过 SSH 隧道调用远程 DSH web API（默认 `127.0.0.1:3081`），只向 `cwd=/home/cyj/botyard_real2sim` 的**主会话**发消息，不向 subagent 发。消息固定含：

- `[R2S-BRIDGE botyard-real2sim-bridge/v1]` 前缀；
- `bundle_id`、`intent`、只读数据目录、结果目录、显式容差；
- 结果契约与 `next_data_request` 字段要求；
- 只读 `experiment/`、必须用 `remote_publish_result.py` 发布。

远程 agent 的职责：

1. 在仿真主机读取`real2sim/experiment/<bundle_id>/`中的`READY.json / TASK.json / manifest.json / README.md`，复核READY与哈希；
2. `intent=transport_selftest`：只写确认性 result，不跑 PhysX/fit；
3. `intent=calibration`：按 `data → parameters → baseline replay → evaluate → curves` 执行，容差只用 TASK 中的显式值；
   - `split=train`：只有对应的**独立 validation bundle 已 ready** 时才允许 `fit`；否则只做 baseline/evaluate/diagnose，并在 `next_data_request.json` 申请独立 validation 录制；
   - `split=validation/test`：只做评估/对比，不得用于训练或拟合；
4. 结果写`real2sim/results/.staging/<id>.partial/`，保持输入包只读；模型与参数修改按项目标定流程进行；
5. 用 `remote_publish_result.py` 发布；
6. 返回结论摘要；缺数据时写 `next_data_request.json`。

## 9. 当前 SWD 输出与历史构建输入

当前`r2s_run.py`与`r2s_pair_run.py`输出samples/commands/metadata/command_plan/joint_state_quality，以及`swd/`下三板`.bin` / `.jsonl`、OpenOCD日志、`capture_manifest.json`、`swd_quality.json`。配对版本额外冻结`input_plan.json`与其SHA256，commands每行记录同次publish的两个关节字段。离线分析可生成`aligned_swd.csv` / `swd_analysis.json`；它保留原始异步样本并映射至H7本地时钟，尚不是canonical capture。后续adapter须保留原始证据、真正的控制分支/命令事件与独立质量失败，不能由重新打包覆盖aborted或fitfalse。

下面仅记录 `build-capture` **历史 UART / host 对齐输入schema**，不要求为新SWD采集生成这些旧文件：

```text
samples.csv               # ROS /joint_states 反馈，主机 CLOCK_MONOTONIC_RAW
commands.csv              # 每次实际 publish 的三字段 + 时间
boards.csv                # J2/J3 原始 UART 行
board_samples_aligned.csv # 历史已映射至host ns的板卡采样
clock_fit.json            # H7/G4/host 时钟拟合与质量门
joint_state_quality.json  # 关节反馈质量门
metadata.json             # 采集参数、控制器、量化说明
command_plan.csv          # 计划轨迹
```

对上述合格历史输入，`build-capture`默认从`commands.csv`推断要导出的关节，因此一次只激励MFJ2的采集会得到单关节bundle；这合法。本机现有配对执行器可在同一次run中同时下发两关节，但其SWD输出仍需新adapter，不能直接喂给历史输入接口。不要把两条独立run拼成一个train/validation包。

`build-capture` 需要一份 `capture_meta.json`，至少包含：

```json
{
  "robot": {"id": "botyard-s1", "hardware_revision": "..."},
  "firmware": {
    "version": "...",
    "control_mode": "Fallback_pidControl",
    "fallback_confirmed": true,
    "fallback_evidence": "固件版本/分支/日志位置"
  },
  "calibration": {
    "position_source": "...",
    "tension_source": "...",
    "motor_position": {
      "source": "输出轴角，已做方向/减速比换算",
      "gear_ratio": {"MFJ2": 1.0, "MFJ3": 1.0},
      "encoder_sign": {"MFJ2": 1, "MFJ3": 1}
    }
  },
  "experiment": {
    "protocol": "H7 A0 / Fallback",
    "root_pose": {"position_m": [0, 0, 0.5], "quaternion_xyzw": [0, 0, 0, 1]},
    "prehistory": {"recorded": true, "duration_s": 0.2, "initial_state_source": "..."},
    "load_condition": "unloaded",
    "temperature_c": 25
  },
  "quality_gates": {
    "max_sensor_skew_s": 0.002,
    "max_sensor_skew_source": "REPLACE: measured transport bias, clock mapping and sensor acquisition-age budget"
  }
}
```

不要伪造 `calibration.status=applied`，或用常量metadata替代实际逐行Fallback证据。当前源运行质量尚未完整传递至导出资格，因此严格校验通过也不能覆盖原始aborted/fitfalse。可直接复制 [capture_meta.example.json](../bridge/capture_meta.example.json) 并逐项替换；标记为 `REPLACE` 的字段必须按实测/实验决策填写。

`calibration.motor_position.angle_basis` 支持：
- `output_shaft`（默认）：`module_position_rad` 已经是绕线轮输出轴角，桥不再乘/除减速比；
- `motor_shaft`：桥按 `encoder_sign × value / gear_ratio` 换算到输出轴，并在 manifest 标记 `gear_ratio_applied=true`。

## 10. 安全与配置

- 专用密钥：`~/.ssh/real2sim_bridge_ed25519`；远程 `~/.ssh/authorized_keys` 只追加 `real2sim-bridge@botyard-local`，权限 700/600。
- `bridge.env` 权限 600，含远程 DSH 启动 token；**不要提交到 git**。
- DSH token 会在远程 `dsh web` 重启后轮换；`doctor` 报 `DSH token FAIL` 时，重新打开远程 URL 并用 `init-config --dsh-token ...` 或环境变量更新。
- 远程输入包`real2sim/experiment/`只读；任务结果写入`real2sim/results/.staging/`并发布到`real2sim/results/`。该约束不禁止按任务修改模型代码和参数。
- 远程校验拒绝符号链接、绝对路径、`..` 路径和 SHA 表中未登记的文件。
- `rsync --checksum` 只同步内容差异；正式目录发布后不要原地改文件，追加数据请新建 bundle。

## 11. 故障处理

| 现象 | 处理 |
|---|---|
| `publish` 卡在 rsync | 检查 `ssh -i ~/.ssh/real2sim_bridge_ed25519 cyj@10.249.133.227 true`；staging 可安全重传 |
| `remote_verify.py` 返回 `VERIFY_FAILED` | 读 staging 下 `VERIFY_FAILED.json`；修本地包后新 nonce 重传，或 `--replace` 重验（旧正式包移到 `experiment/.replaced/` 保留审计） |
| `staging must live under ...` | 不要手工指定任意目录；使用 `publish` 生成的 staging |
| `session/not-found` / `DSH token` | `doctor`；更新 `R2S_REMOTE_DSH_TOKEN`；`R2S_REMOTE_SESSION=auto` 会自动选 `botyard_real2sim` 最新主会话 |
| 远程 agent 忙 | `status` 看 `dsh_running`；`notify` 默认排队，`watch` 等结果 |
| `RESULT_READY.json` 不存在 | 远程可能只写了 staging；让 agent 运行 `remote_publish_result.py`；不要手工改名 |
| 结果里 `next_data_request` 为空 | 表示当前包无需补采；继续下一轮或结束 |
| 本地 `run.sh` 找不到 Python | 本地桥不需要 Isaac：用 `python3 scripts/real2sim_bridge.py` |

## 12. 已验证记录

2026-11-20 端到端自检（合成夹具，不是硬件证据）：

1. `build-selftest` 生成 `r2s-20261120T080141Z-selftest`；
2. 本地完整性 + `./run.sh data --strict`：`evaluation_ready`，1001 行 / 2.0 s / 42 事件 / 三通道 coverage 1.0；
3. `publish`：rsync 到远程 `.staging`，`remote_verify.py` 复核 5 个 SHA256、READY 哈希、严格数据集，原子改名为 `experiment/<id>`；
4. `notify`：远程 `session-0a304202-8881-4b10-b176-d605d13a6b8f`（`botyard_real2sim` 最新主会话）确认 READY/哈希/严格校验，未跑 PhysX；
5. 远程用 `remote_publish_result.py` 原子发布 `results/<id>`，生成 `RESULT.json`、`report.md`、`next_data_request.json`、`RESULT_READY.json`；
6. 本地 `results show` 成功读取结果与报告；`next_data_request` 为空。
7. 第二次自检 `r2s-bridge-check` 重复了完整发布→远程确认→结果回读，`next_data_request.json` 已按新约定写 `status=none`、`experiments=[]`。
8. 当时桥单测4项通过（`tests/test_real2sim_bridge.py`），覆盖selftest、篡改检测、旧输入导出、远程verify/result publish；当时远程完整测试245项通过，Ruff check/format通过。这是该次自检记录，不代表后续修改已经在远程重验。

该记录只证明传输、完整性、READY 握手和数据集结构校验可用；**不**证明物理模型、参数或真机曲线拟合。

## 13. 当前限制与下一步

- 尚未有真实硬件 bundle；`experiment/` 里的 selftest 明确标记为合成夹具。
- `cmd_tension_N = effort_raw × 0.01` 的刻度需与固件/驱动再核对；在 manifest `provenance.third_field` 中保留来源。
- 桥的插值只用于把板卡/指令事件对齐到主机网格；原始文件保留在 `raw/`，不替代原始时间戳。
- 远程 `fit` 每次候选重启 Isaac Sim，100 次候选约 3 小时量级；`TASK.json` 的 `--max-evaluations`/预算应由实验决策给出。
- 下一步：用真实 `train` + `validation` 两条独立录制走完整闭环；远程缺数据时按 `next_data_request.json` 生成新激励，再建新 bundle。
