# real2sim 数据桥：本地采集 ↔ 远程仿真

本文档定义真机侧（本机）与仿真侧（远程 `cyj@10.249.133.227`）之间的数据包、命令、结果和补采请求契约。它把 local 的 `experiment/`、远程的 `real2sim/experiment/` 与 `real2sim/results/` 串成可审计的闭环。

> 当前状态（2026-11-20）：SSH/Rsync 专用密钥、远程 staging→READY→原子改名、远程 DSH agent 通知、结果发布与回读均已实机联通。端到端自检见 [§12](#12-已验证记录)。

## 1. 拓扑与职责

```text
本机 botyard（真机 / ROS2 / 烧录器）                远程 cyj-B760M（Isaac Lab / PhysX / RTX 4070）
┌─────────────────────────────────────┐            ┌────────────────────────────────────────┐
│ embedding_ws/tools/r2s_run.py        │            │ botyard_real2sim 仿真与标定工具链        │
│  采样/指令/板卡日志 + 时钟对齐        │            │  run.sh data/parameters/replay/          │
│  ↓ 导出 canonical CSV + manifest     │            │           evaluate/curves/fit             │
│ experiment/<bundle_id>/               │  rsync     │ real2sim/experiment/<bundle_id>/         │
│  capture.csv manifest.json TASK.json  │ ─────────▶ │   capture.csv manifest.json TASK.json    │
│  BUNDLE.json SHA256SUMS README.md     │  staging   │   BUNDLE.json SHA256SUMS READY.json      │
│                                       │            │   （remote_verify.py 校验+原子改名）      │
│                                       │  DSH API   │                                          │
│                                       │ ─────────▶ │ 远程 DSH agent（botyard_real2sim 会话）  │
│                                       │ 127.0.0.1  │   → baseline/evaluate/fit                │
│                                       │ :3081      │   → results/.staging → RESULT_READY.json │
│ experiment/<id>/results/              │ ◀───────── │ real2sim/results/<bundle_id>/            │
│  （只读镜像，可选）                    │  ssh cat   │   RESULT.json report.md next_data_request│
└─────────────────────────────────────┘            └────────────────────────────────────────┘
```

| 环节 | 本机（真机侧） | 远程（仿真侧） |
|---|---|---|
| 激励与采集 | 改固件 / ROS2 控制 / 采 J2、J3、H7 数据 | 不接触硬件 |
| 时间基准 | `CLOCK_MONOTONIC_RAW` 主机时钟；板卡经 `r2s_align.py` 映射 | 信任 manifest 的时钟与 skew 证据 |
| 数据打包 | `build-capture` 生成 CSV/manifest/BUNDLE/SHA256SUMS | 只读 `experiment/`，不改包 |
| 完整性 | 本地先校验，再上传 | 复核 SHA256、manifest/CSV 哈希、严格数据集校验 |
| 发布 | rsync 到 `.staging/*.partial-*` | `remote_verify.py` 写 `READY.json` 后原子改名 |
| 仿真/拟合 | 不跑 Isaac | 严格按 `TASK.json` 容差和项目流程执行 |
| 结果 | 读取/按需拉取，不回传 | `remote_publish_result.py` 原子发布 |
| 补采 | `plan-requests` 把远程请求转为激励命令计划 | 在 `next_data_request.json` 写清缺什么数据 |

**硬边界：**
- 本机不把仿真结果当真机证据；远程不把仿真 CSV 当实机数据。
- `experiment/` 在远程是**只读**目录；结果只能写 `results/`。
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
      raw/                       # 可选：samples/commands/boards/clock_fit/...
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
| `raw/` | 建议 | 原始 `samples.csv`、`commands.csv`、`board_samples_aligned.csv`、`clock_fit.json`、`metadata.json` 等；发布时一并完整性校验 |
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

采集端从 `r2s_run.py` 原始目录导出时：
- 关节角/速度来自 `samples.csv`；
- 电机角/张力来自 `board_samples_aligned.csv` 的 J2/J3 板（`module_position_rad` / `tension_N_from_firmware_scale`）；
- 指令来自 `commands.csv` 的实际发布事件；
- 板卡与主机时间已由 `r2s_align.py` 映射；桥只在 50 ms 内对指令时间点做线性插值，不外推，并把原始文件放 `raw/`；
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
    "result_dir": "results/<bundle_id>",
    "staging_dir": "results/.staging/<bundle_id>.partial",
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
① 采集 raw run
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
⑦ 远程 agent 只读 experiment/<id>/，在 results/.staging/<id>.partial/ 写结果
   ↓
⑧ remote_publish_result.py 校验 RESULT.json + 哈希，写 RESULT_READY.json
   ↓
⑨ 同一文件系统内 mv → results/<id>/
   ↓
⑩ 本地 results show / pull、watch；读取 report.md + next_data_request.json
   ↓
⑪ 若缺数据：plan-requests 生成 r2s_run.py 采集计划 → 回到 ①（新 bundle）
```

为什么必须 staging + READY + 改名：

- 上传中的半包不会被远程 agent 消费；
- 远程只对完整包做校验，失败留在 staging，可重试而不会污染正式接口；
- 正式目录名只在所有哈希与严格校验通过后出现，消费者只需判断 `READY.json`；
- 结果侧同样原子：`RESULT_READY.json` 出现前，`results/<id>/` 不存在。

## 5. 本地命令

先安装/确认：本机只需要 Python 3.12 + `rsync` + `ssh`；不需要 Isaac。远程需要项目现有 Isaac 环境（`run.sh` 会用它）。

```bash
cd /home/botyard/real2sim/botyard_real2sim

# 0. 配置（已生成则只需 doctor）
python3 scripts/real2sim_bridge.py doctor

# 1. 从一次 r2s_run 采集目录构建 train/validation bundle
python3 scripts/real2sim_bridge.py build-capture \
  /home/botyard/real2sim/embedding_ws/runs/<run> \
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

`next_data_request.json` 是本地下一轮采集的唯一输入。推荐结构：

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
      "amplitude_deg": 10,
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

`plan-requests` 会把它转成 `embedding_ws/tools/r2s_run.py` 命令（默认 100 Hz、`--effort-raw 0`），输出 `plan.json` 与 `run_plan.sh`。计划不会自动执行；必须在确认端口、限位、手姿态与安全范围后自行加 `--execute`。

可直接复制的模板：
- [capture_meta.example.json](../bridge/capture_meta.example.json)
- [next_data_request.example.json](../bridge/next_data_request.example.json)

## 8. 远程 agent 命令协议

本地 `notify` 通过 SSH 隧道调用远程 DSH web API（默认 `127.0.0.1:3081`），只向 `cwd=/home/cyj/botyard_real2sim` 的**主会话**发消息，不向 subagent 发。消息固定含：

- `[R2S-BRIDGE botyard-real2sim-bridge/v1]` 前缀；
- `bundle_id`、`intent`、只读数据目录、结果目录、显式容差；
- 结果契约与 `next_data_request` 字段要求；
- 只读 `experiment/`、必须用 `remote_publish_result.py` 发布。

远程 agent 的职责：

1. 读 `READY.json / TASK.json / manifest.json / README.md`，先复核 READY 与哈希；
2. `intent=transport_selftest`：只写确认性 result，不跑 PhysX/fit；
3. `intent=calibration`：按 `data → parameters → baseline replay → evaluate → curves` 执行，容差只用 TASK 中的显式值；
   - `split=train`：只有对应的**独立 validation bundle 已 ready** 时才允许 `fit`；否则只做 baseline/evaluate/diagnose，并在 `next_data_request.json` 申请独立 validation 录制；
   - `split=validation/test`：只做评估/对比，不得用于训练或拟合；
4. 结果写 `results/.staging/<id>.partial/`，不得写 `experiment/`；
5. 用 `remote_publish_result.py` 发布；
6. 返回结论摘要；缺数据时写 `next_data_request.json`。

## 9. 从 `r2s_run.py` 采集到 bundle

采集侧原始目录（`r2s_run.py --execute --out <run>`）应包含：

```text
samples.csv               # ROS /joint_states 反馈，主机 CLOCK_MONOTONIC_RAW
commands.csv              # 每次实际 publish 的三字段 + 时间
boards.csv                # J2/J3 原始 UART 行
board_samples_aligned.csv # r2s_align.py 映射后的板卡采样
clock_fit.json            # H7/G4/host 时钟拟合与质量门
joint_state_quality.json  # 关节反馈质量门
metadata.json             # 采集参数、控制器、量化说明
command_plan.csv          # 计划轨迹
```

`build-capture` 默认从 `commands.csv` 推断要导出的关节，因此一次只激励 MFJ2 的采集会得到单关节 bundle；这合法。需要一条 bundle 覆盖 MFJ2+MFJ3 时，采集端必须在同一次 run 中同时下发两关节指令（或扩展采集器），不要把两条独立 run 拼成一个 train/validation 包。

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
    "max_sensor_skew_source": "由 r2s_align 的残差和传感器规格决定"
  }
}
```

不要伪造 `calibration.status=applied`。没有证据时严格校验会拒绝发布，这是预期行为。可直接复制 [capture_meta.example.json](../bridge/capture_meta.example.json) 并逐项替换；标记为 `REPLACE` 的字段必须按实测/实验决策填写。

`calibration.motor_position.angle_basis` 支持：
- `output_shaft`（默认）：`module_position_rad` 已经是绕线轮输出轴角，桥不再乘/除减速比；
- `motor_shaft`：桥按 `encoder_sign × value / gear_ratio` 换算到输出轴，并在 manifest 标记 `gear_ratio_applied=true`。

## 10. 安全与配置

- 专用密钥：`~/.ssh/real2sim_bridge_ed25519`；远程 `~/.ssh/authorized_keys` 只追加 `real2sim-bridge@botyard-local`，权限 700/600。
- `bridge.env` 权限 600，含远程 DSH 启动 token；**不要提交到 git**。
- DSH token 会在远程 `dsh web` 重启后轮换；`doctor` 报 `DSH token FAIL` 时，重新打开远程 URL 并用 `init-config --dsh-token ...` 或环境变量更新。
- 远程 `experiment/` 只读；所有写入限制在 `results/.staging` 与 `results/`。
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
8. 桥单测 4 项通过（`tests/test_real2sim_bridge.py`），覆盖 selftest、篡改检测、raw capture 导出、远程 verify/result publish；远程完整测试 **245 passed**，Ruff check/format 通过。

该记录只证明传输、完整性、READY 握手和数据集结构校验可用；**不**证明物理模型、参数或真机曲线拟合。

## 13. 当前限制与下一步

- 尚未有真实硬件 bundle；`experiment/` 里的 selftest 明确标记为合成夹具。
- `cmd_tension_N = effort_raw × 0.01` 的刻度需与固件/驱动再核对；在 manifest `provenance.third_field` 中保留来源。
- 桥的插值只用于把板卡/指令事件对齐到主机网格；原始文件保留在 `raw/`，不替代原始时间戳。
- 远程 `fit` 每次候选重启 Isaac Sim，100 次候选约 3 小时量级；`TASK.json` 的 `--max-evaluations`/预算应由实验决策给出。
- 下一步：用真实 `train` + `validation` 两条独立录制走完整闭环；远程缺数据时按 `next_data_request.json` 生成新激励，再建新 bundle。
