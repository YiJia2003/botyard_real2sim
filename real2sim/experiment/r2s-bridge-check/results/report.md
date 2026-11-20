# 数据桥校验回执 `r2s-bridge-check`

- intent：`transport_selftest`（仅验证传输；不跑仿真、不拟合）
- split：`test`　joints：`MFJ2`、`MFJ3`
- 数据种类：`synthetic_transport_selftest`（合成夹具，**不是实机证据**）
- 处理时间：2026-11-20T08:10:12Z
- 结论：**传输链路通过，包完整且自洽，未运行 PhysX / curves / sweep / fit**

## 1. 控制文件与 READY

`READY.json`、`TASK.json`、`BUNDLE.json`、`manifest.json`、`README.md`、`SHA256SUMS`、`capture.csv` 均存在且可读。

- `READY.ok = true`，`protocol = botyard-real2sim-bridge/v1`；
- `final_path` 与本 bundle 实际目录一致；
- `READY.staging`（`.partial-25873e12`）在重命名后已不存在，`experiment/.staging/` 为空 → 符合原子发布语义；
- `dataset_status = evaluation_ready`，与本机复算结果一致。

## 2. 校验和复核

`sha256sum -c SHA256SUMS` → 5/5 通过（BUNDLE、README、TASK、capture、manifest）。

与 `READY.json` 声明逐项一致：

| 项 | 声明 | 实测 |
|---|---|---|
| `manifest_sha256` | `906497ac…9218b` | 一致 |
| `csv_sha256` | `b96c1aef…958f56` | 一致 |
| `bundle_record_sha256` | `05538523…eff63` | 一致 |

补充一致性检查全部通过：`BUNDLE.files` 的 sha256 与字节数、`BUNDLE.task_sha256`、`manifest.recording.sha256`、
`dataset_id == bundle_id == TASK.bundle_id`。与消息给出的 bundle_id 和三个容差无冲突，未触发停止条件。

## 3. 数据集严格校验

```
./run.sh data --manifest <bundle>/manifest.json --strict   → exit 0
status = evaluation_ready   valid = true   data_ready = true   fit_ready = false
errors = []   blocking_reasons = []   warnings = []
rows = 1001   duration = 2.0 s   command_event_count = 42
fallback_fraction = 1.0   max_sensor_skew_s = 0.001（限值 0.002）
逐通道 time_coverage 最小值 = 1.0（MFJ2/MFJ3 × position/motor_position/tension）
```

`fit_ready=false` 属预期：本包 `split=test`，按契约只能进入评估侧，不能作训练集。

## 4. 只读与安全

`experiment/` 下未创建或修改任何文件；bundle 内符号链接 0 个。
全部写操作仅发生在 `results/.staging/r2s-bridge-check.partial/`，随后由
`real2sim/bridge/remote_publish_result.py` 校验并原子发布。

## 5. 证据边界

本次只证明：数据桥、checksum、READY 握手、manifest 结构与严格校验可用。
**不**证明任何物理模型、参数或曲线拟合结果。

另注：本包 `capture.csv` 与上一份自检包 `r2s-20261120T080141Z-selftest` 字节完全相同
（`sha256=b96c1aef…958f56`），manifest/TASK 已按新 bundle 重建。作为传输自检无影响，
但两包不可当作两条独立录制，也不可分别用作 train/validation。

## 6. 下一步

`next_data_request.json` 为 `status=none`、`experiments=[]`（不索取额外数据）。
收到真实硬件 bundle 后，按 `data → parameters → replay → evaluate → curves → fit`
执行；`fit` 需独立的 `train` 与 `validation` 两条物理录制。
