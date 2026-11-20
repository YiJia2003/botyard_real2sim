# 数据桥校验回执 `r2s-20261120T080141Z-selftest`

- intent：`transport_selftest`（仅验证传输，不做仿真、不拟合）
- split：`test`　joints：`MFJ2`、`MFJ3`
- 数据种类：`synthetic_transport_selftest`（合成夹具，**不是实机证据**）
- 处理时间：2026-11-20T08:03:00Z
- 结论：**传输链路通过，包完整，未运行 PhysX / curves / sweep / fit**

## 1. 控制文件

`READY.json`、`TASK.json`、`BUNDLE.json`、`manifest.json`、`README.md`、`SHA256SUMS`、`capture.csv` 均存在且可读。
`READY.json` 的 `final_path` 与实际 bundle 目录一致；其 `staging` 路径在重命名后已不存在（`experiment/.staging/` 为空），符合原子发布语义。

## 2. 校验和复核

`sha256sum -c SHA256SUMS` → 5/5 成功（BUNDLE.json、README.md、TASK.json、capture.csv、manifest.json）。

与 `READY.json` 声明逐项比对，全部一致：

| 项 | READY.json | 实测 |
|---|---|---|
| `manifest_sha256` | `200931e1…299386` | 一致 |
| `csv_sha256` | `b96c1aef…958f56` | 一致 |
| `bundle_record_sha256` | `ea1a3d6d…51ca09` | 一致 |

`BUNDLE.json` 内部登记的 4 个 payload 文件哈希与字节数也全部一致；`manifest.json` 的 `recording.sha256` 等于 `capture.csv` 实际哈希。
与消息中给出的 bundle_id、容差（`angle_deg=2`、`motor_angle_deg=3`、`tension_N=1`）无冲突，未触发停止条件。

## 3. 数据集严格校验

```
./run.sh data --manifest <bundle>/manifest.json --strict   → exit 0
status = evaluation_ready   valid = true   data_ready = true   fit_ready = false
errors = []   blocking_reasons = []   warnings = []
rows = 1001   duration = 2.0 s   command_event_count = 42
fallback_fraction = 1.0   max_sensor_skew_s = 0.001（限值 0.002）
逐通道 time_coverage = 1.0（MFJ2/MFJ3 × position/motor_position/tension）
```

`fit_ready=false` 是预期的：该包 `split=test`，按设计只能用于评估侧验收，不能作为训练集。

## 4. 只读约束

`experiment/` 下未创建或修改任何文件；bundle 内无符号链接（扫描 0 个）。
所有写操作仅发生在 `results/.staging/r2s-20261120T080141Z-selftest.partial/`，随后由
`real2sim/bridge/remote_publish_result.py` 原子发布。

## 5. 证据边界

本次只证明：数据桥、checksum、READY 握手、manifest 结构与严格校验流程可用。
**不**证明任何物理模型、参数或曲线拟合结果——原始数据本身声明为合成夹具。

## 6. 下一步

`next_data_request.json` 为空请求（不索取额外数据）。待收到真实硬件 bundle 时，再按
`data → parameters → replay → evaluate → curves → fit（需独立 train/validation 两条录制）`
的流程执行。
