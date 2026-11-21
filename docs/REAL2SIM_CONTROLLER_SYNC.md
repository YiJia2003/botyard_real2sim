# 控制器源码同步：2026-11-21

本次将当前 `s1.yaml` 和 `default.yaml` 的控制段同步到
`/home/cyj/botyard_embedded` 内记录的 real2sim 实验配置。物理参数、收到的实验包和
历史结果不修改。本次是控制指令路径的同步，不是一次参数拟合或真机一致性验收。

## 来源与适用范围

| 仓库 | 本地版本（核对时工作区干净） |
|---|---|
| `botyard-hand-d22-core-h7` | `53d820341688b4b875a6009ac2fd54e57adbe01b` |
| `botyard-motor-bl` | `d395c9ff50e62faf42c1d39b2c9175ee124f4b56` |

H7 的 `BOTYARD_REAL2SIM_TRACE=ON` 选择 MFJ3 的 2.5° 下限。G4 实验说明
`docs/real2sim-position-limit.md` 选择 `BOTYARD_VELOCITY_TARGET_DEADBAND_RAD_S=200`；
**仅开启 TRACE 不会自动选择 200**，CMake/头文件的普通默认值是 250。
本地没有对应 ELF、CMakeCache 或设备读回值，因此这里锁定的是当前源码和文档中的
实验配置；实际录制的固件哈希、编译开关和运行参数优先。

| 设置 | 当前仿真预设 | 源码依据（相对于上表仓库） |
|---|---|---|
| MFJ3 目标角下限 | **2.5°** | H7 `projects/main/inc/real2sim_middle_control.h:11`，`src/control.c:301`（src 相对于 projects/main） |
| 其余九个屈曲关节下限 | 0.5° | H7 `projects/main/src/control.c:253–262`、`:277–286` |
| 所有目标角上限 | 89.5° | 同上 |
| 关节角误差死区 | **严格 \|误差\| < 1°** | H7 `projects/main/src/pid.c:430–457` |
| 积分分离、限幅 | \|误差\| < 2° 时积分，I ∈ [−10,+10] °/s | H7 `projects/main/src/pid.c:430–465`、`inc/pid.h:33` |
| P / I / D | P 为拇指 15/10，四指 J2/J3 为 20/10；I=1，D=0 | H7 `projects/main/src/control.c:767–782` |
| 指令速度处理 | 取绝对值，上限 360°/s；≤1°/s 改用 120°/s | H7 `projects/main/src/control.c:1714–1717` |
| H7 输出限制与量化 | ±7000 mrad/s；`trunc(trunc(x)+0.5)` | H7 `projects/main/src/pid.c:472–504`、`control.c:1819–1820` |
| G4 速度目标死区 | **转子侧 200 rad/s，严格小于才归零** | G4 `projects/main/src/condition_monitor.cpp:83`，实验配置见上文 |
| 转子/绕线轮速度比 | **800** | G4 `projects/main/src/module_info.cpp:38–58`、`mem_object.cpp:281–284` |
| G4 绕线轮速度上限 | **6.25 rad/s** | G4 `projects/main/src/mem_object.cpp:117`、`motion_control.cpp:528`、`:797–813` |
| H7 控制周期 | 3 ms；估计器的 2 ms 截断量不作控制周期 | H7 `projects/main/inc/user_config.h:57` |

`motor_gear_ratio` 是转子到绕线轮的速度比，仅用于换算速度死区；不改变腱绳力臂、
绕线轮半径，也不再缩放已经是绕线轮输出轴角的实测数据。该比值和 `moduleMaxVel`
均可被设备运行参数覆盖，回放时应使用录制归档中的实际值。

## 生效顺序与回放列

1. 同指补齐仍在收到指令时捕获伙伴角度；随后 H7 对目标角限幅。因此 MFJ3 即使作为
   保持角度的伙伴，目标仍会被抬到 2.5°。实测关节角不裁剪，机械限位不修改。
2. H7 按原来的死区、积分、速度限制和整数转换计算请求。
3. 运行时把已量化的绕线轮请求乘以 800，比较转子侧死区；低于 200 rad/s 则归零。
   等价绕线轮阈值为 **0.25 rad/s ≈ 14.32394°/s**，等于阈值时保留。
4. 通过死区后限幅到 ±6.25 rad/s，再交给原有传动模型。

驱动器死区不会复位 H7 积分，也不会把小速度强制抬到 15°/s。
H7 负数整数转换的不对称行为保留，例如中间整数 −250 在加 0.5 后变成线上的
−249 mrad/s，随后会落入死区。

CSV 中 `controller_speed_rad_s` 保留原有含义：H7 量化后的请求。
新增 `motor_target_speed_rad_s` 是经过 G4 死区和速度上限后的**绕线轮侧**目标。
`motor_velocity_rad_s` 仍是传动模型的绕线轮速度，不能混作转子速度。
参数快照保存所有新字段，`./run.sh parameters` 把它们列为应与固件锁定的控制参数。

## 历史兼容与尚未建模的路径

没有新字段的旧 YAML/`FallbackControllerConfig()` 保持旧行为：角度下限默认均为
0.5°，驱动器死区和速度上限为 0（禁用）。当前两个 YAML 预设显式选择本次配置。
重新跑历史记录时使用其原始参数快照，或明确按归档固件修改新字段；不要把当前预设
自动套到所有旧记录上。普通固件的 250 rad/s 和早期诊断固件的 0.5 rad/s 都不是
本次实验配置。

本次同步的是 Fallback 及其下游的静态速度指令规则。以下区别仍需区分：

- H7 trace 仅对 MFJ2/MFJ3（mask 0x30）强制 Fallback，其余关节需有 `is_fallback`
  记录才能用于本仿真。现有回放仍拒绝明确标为非 Fallback 的记录。
- G4 内部速度/电流环、5 ms 指令低通、启动逻辑和故障/张力保护没有逐周期复刻。
  `tendon_speed_lag_s` 仍是原有可选速度端口近似，不自动当作已辨识的电机动态。
  因此新增列表示低通前的目标，不保证物理电机瞬间达到或归零。
- G4 的位置方向拦截与 H7 目标角下限是不同的规则。实验文档选择位置方向拦截 OFF，
  但仿真不新增或修改机械限位。设备实际开关、标定限位、方向及故障需查归档读回值。
- Fallback 的外部第三指令字段仍不决定张力；这不代表 G4 没有独立过张力保护。

单元测试覆盖角度地板、严格死区边界、正负整数转换、转速比、双精度/单精度与批量
输入、积分继续累加、传动入口和 CSV。PhysX 冒烟回放只能验证链路可运行，不能
代替真机三通道对比和留出验证。

本次实际验证：`./run.sh test` **317 passed**（两个已有的旧格式缺失字段提示），
`./run.sh check` 通过，独立代码复核通过。PhysX 目标下限与低速死区两次冒烟回放
分别完成 1500 / 300 步，无 effort 限幅；低速回放的 11 个样本均保留非零 H7 请求，
驱动器目标和绕线轮位移均为零。CSV、参数快照、元数据和比较报告保存在
`real2sim/results/.staging/controller-sync-20261121.partial/`。
