# S1 空载 real2sim：几何、走线摩擦与惯性传动

2026-11-18。本次目标是同一条 Fallback 指令下同时比较**关节角、电机绕线轮角、应力片腱绳张力**。
当前已实现可执行模型和观测接口；尚无真实响应数据，以下可运行参数不是辨识结果。
电机和关节主转轮均采用用户最终确认的**半径 5 mm（0.005 m）**，不是直径 5 mm。

## 入口与文件

- `scripts/real2sim_replay.py` 未指定 `--config` 时使用 `configs/real2sim/s1.yaml`。
- `configs/real2sim/default.yaml` 保留旧常力臂模型，显式传入可复现旧实验。
- `geometry_config.py`：无引擎依赖的几何配置检查。
- `geometry.py`：附着于运动连杆的端点、圆导向件切线/圆弧、脱离接触、路径雅可比。
- `transmission.py`：保留旧的准静态串联弹簧，用于旧实验和动态模型的初态逆解。
- `routing_dynamics.py`：有限质量走线弹簧、两段摩擦、接触/微滑状态和隐式求解。
- `model.py`：电机收绳状态、单边张力、回位弹簧、关节摩擦和虚功力矩。
- `runtime.py` / `controller.py`：保留 Fallback 和同指补齐控制，不引入固件逆动力学分支。
- `real2sim_curves.py` / `real2sim_sweep.py`：三类主曲线、运动学/内部状态图、指标和参数扫描。

## CAD 证据和参数等级

通过 **cad-mcp-server** 执行 `inspect → find_entities → measure(section)`。
STEP 长度单位为 **mm**，配置转换为 **m**。

```text
docs/S1手指-V3.2.STEP
sha256:34a46bdc8ed93c6e918df4d2936297cf187681ac224042b214863bd0dd0c86c7
```

原始检查及截面结果保存在 `docs/cad/`，确切实体编号只对这个 SHA-256 版本有效。
CAD 返回的是几何事实；实体的关节/导向/锚点功能分配结合图片推定，不能把这个推定称作 CAD 自动确认。

| 项目 | 数值 / 来源 | 使用方式 |
|---|---|---|
| J3 候选轴心 | STEP `(0, 0, 0.010002779)` mm，`body:157` 的圆截面 | 平面坐标原点 |
| J2 候选轴心 | `(-0.499999577, 0, 45.010531601)` mm，`body:114` | J3→J2 的参考位移 |
| J1 候选轴心 | `(-0.499999577, 0, 70.010531601)` mm，`body:82` | J2→J1 的参考位移 |
| 三处主转轮截面 | 半圆弧长 `15.707963268 mm`，推出半径 `5 mm` | 几何路径绕行曲面 |
| 中间小导向件 | `(4.500000423, 0, 54.003117088)` mm，`body:4` 截面半径 `2 mm` | J2 路径中间导向 |
| 下方小导向件 | `(0.5, -2.05, -9.089997221)` mm，`body:11` 截面半径 `2 mm` | J3 出线侧导向 |
| J3 系绳位置候选 | `body:193` 圆截面中心约 `(0.494246, -3.5, 34.110532)` mm | 附着于 proximal，不是世界固定点 |
| J3 回位支点候选 | `body:192/194` 截面及实体位置 | 推定连杆归属，配置可改 |
| 回位拉簧 | 图片：线径 0.4 mm，有效圈数 38，长度 20 mm，外径 2.5 mm，安装预紧长 24 mm，72A 钢 | 自由长和安装长度取图片值 |
| 回位短绳 | 图片：J2 17 mm，J3 19 mm | 不伸长的回位连接绳 |

轴距应区分轴向分量和欧氏距离：J3→J2 的 Δz 是 `45.000528822 mm`，Δx 是 `−0.499999577 mm`；J2→J1 为 `25 mm`。
已有 USD 的对应 Δz 为约 45.001 mm / 25.106 mm。本次更新的是独立标定力学层，未重建共享网格、刚体惯量或碰撞资产；这约 0.106 mm 的末节差异在后续刚体资产更新时需统一。

以下仍属**推定/待拟合**，在 YAML 中可修改：

- 图中 J2 顶部系绳点取 STEP 平面约 `(6.25, 81)` mm；两个未画出的电机出线点也是可编辑假设。
- J2 回位绳的连杆锚点 `(7.5, 3)`、`(7.5, 43)` mm 是图片近似，不是已测出的孔心。
- `frame`、绕行方向 `winding`、可脱离导向件的对应关系由图示推定；绿色标记本身从未当作实体圆轮测量。
- 把这份 S1 指体作为四指共同起点，并不表示已测过四根手指各自的制造差异。拇指没有对应 CAD，保留原常力臂假设；新串联弹簧和电机观测对拇指同样可用。
- 回位拉簧刚度暂为 **1000 N/m**，初拉力暂为 0 N。没有用未知的 72A 剪切模量冒充刚度实测值；线径/圈数/外径尚不足以确认弹簧端钩柔度及初拉力。
- 走线等效弹簧的 **1000 N/m、自由长 20 mm、最大长 23 mm** 全部是可运行初值，与图片中的实体回位拉簧是两套独立参数。

## 运动几何与虚功

使用 STEP 的 XZ 平面，正角从 +z 向 −x 弯曲。每指独立坐标 `q=(J2,J3)`；三处转轴的累计转角依次是 `J3`、`J3+J2`、`J3+2*J2`。
末项来自四指 `J1=J2`，因此顶部系绳点随 distal 连杆运动。图中“系死无位移”解释为对该连杆固定。

路径包含直线公切线和圆弧，圆心随所属连杆更新：

```text
L(q) = sum(tangent_lengths) + sum(radius * wrap_angle)
h(q) = L(0) - L(q)
R(q) = dh/dq
tau_tendon = R(q)^T T_distal
```

长度变化和力矩共用解析雅可比，满足 `T_distal·h_dot = tau·q_dot`。这里使用经过走线摩擦和惯性传动后的关节入口张力，不是电机传感器张力。J1 约束先约化进几何梯度，再把合力矩提交给独立 J2；不再额外向从动 J1 重复加力矩。
轴心重合的理想圆轮在持续绕行时本来就会产生常力臂；非同轴导向、移动系绳点、离开圆轮后的直线路径以及回位弹簧方向使有效力臂随角度变化。

当前图示解释下，J2 绳在 J3 上的带符号力臂约为 **−5 mm**，不同于旧模型临时使用的 +5 mm。
J2 自身小角度力臂约 10 mm，包含 J1 同步运动的贡献。必须后续用单关节慢速运动和电机位移确认符号及绕行分配。

`detachable: true` 允许导向包角缩到零时平滑转为直线。不会在角度过零时误增 `2πr` 绳长，也不会提前跳到导向件另一侧的较短路径。
当前每条路径支持一个可脱离导向件、短包角、二维布线；这些 S1 预设已覆盖 0–90° 检查。任意复杂三维碰撞/改道不在该解析模型中，超出拓扑范围会报错。

## 两种弹簧和收绳长度守恒

### 回位实体拉簧

```text
L_s(q) = L_return_path(q) + return_path_length_offset_m - return_tether_length_m
F_s = 0                                     L_s <= L_free
F_s = initial_tension + k_s*(L_s-L_free)      L_s > L_free
tau_return = -(dL_return_path/dq)^T F_s
```

绳松弛时实际弹簧长度停留在自由长，张力为零。记录的 `return_spring_length_m` 不会把拉簧当成受压弹簧。
常量路径补偿使参考姿态下的实体弹簧长为图片给出的 24 mm，补偿未建模端钩/固定路径长度，**不改变力臂**。
安装长 24 mm 对应 STEP 参考姿态是当前初始化假设。几何回位弹簧替代对应通道旧的扭转弹簧，不叠加两遍。

### 电机—关节之间的动态走线模型

```bash
motor x  ──[ 近端材料弹簧 kp ]──  u  ══[ 走线弹簧 kr ]══  v  ──[ 远端材料弹簧 kd ]──  y(q)
(绕线轮)                         近端节点            远端节点                    关节侧
                                  ↑ 近端摩擦区                      ↑ 远端摩擦区
```

S1 预设启用 `tendon.routing_model: dynamic`，十条屈曲腱绳各有独立状态。传感器和元件的顺序为：

```text
电机绕线轮 x（半径 5 mm）
  → 电机侧张力观测 Tp
  → 近端材料柔度 kp
  → 节点 u：等效质量 m/2、近端走线摩擦 Fu
  → 有限行程走线弹簧 kr
  → 节点 v：等效质量 m/2、远端走线摩擦 Fv
  → 远端材料柔度 kd
  → 关节入口张力 Td，边界 y=h(q)+slack
```

摩擦从电机张力观测之后、走线弹簧之前开始，按两个集总接触区覆盖到关节入口。
两区的位置是等效离散化，不能解释为已经测得两个真实接触点。
`m=routing_mass_kg` 是走线和弹簧参与运动的**等效质量**，不是已测弹簧质量，也不是电机转子惯量。
旧材料刚度 `k_c` 按柔度比例 `f` 分成 `kp=k_c/f`、`kd=k_c/(1-f)`，其串联材料柔度保持不变。

```text
ep = x-u,  es = u-v,  ed = v-y
x-y = ep + es + ed
Tp = kp * max(ep,0)
Td = kd * max(ed,0)
Ts = kr * max(es,0) + stop_reaction
(m/2) * u_ddot = Tp - Ts - Fu
(m/2) * v_ddot = Ts - Td - Fv
```

三个元件都只能受拉；负 `ep/ed/es` 表示对应间隙，不产生压缩力。
实际走线弹簧长为 `L_free+max(es,0)`，并满足 `es <= L_max-L_free`。
达到最大长度后，约束反力承担新增内力，弹簧不再伸长；拉紧瞬间用等大反向冲量消除继续伸长的相对速度，恢复系数为零。
把自由长设成最大长只会禁止该弹簧继续伸长，**动态分支的质量和摩擦仍存在**。

`Tp-Td = Fu+Fv+(m/2)*(u_ddot+v_ddot)`，因此动态张力差由惯性和摩擦共同决定。
静止时惯性项消失，但静摩擦仍可维持张力差；没有强制两侧最终相等，也没有人为添加固定延迟。
放绳/反转时张力差允许换号，不能把 `Td` 恒定钳制为小于 `Tp`。

### 随走线变化的摩擦与内部状态

每个节点保留一个弹塑性微滑状态 `z`，一个腱绳通道另保留接触状态 `c∈[0,1]`：

```text
c_target = clip(c0 + aT*Tmean/(Tscale+Tmean)
                   + ae*clip(max(es,0)/travel,0,1)
                   + ay*abs(y)/(yscale+abs(y)), 0, 1)
c_dot = (c_target-c)/tau_loading_or_release
wrap = max(wrap_base + wrap_gain*c, 0)
Nu = 0.5*(normal_preload + 0.5*(Tp+Ts)*wrap)
Nv = 0.5*(normal_preload + 0.5*(Td+Ts)*wrap)
B = N*(mu_dynamic + (mu_static-mu_dynamic)*exp(-(velocity/vs)^2))
z_new = clip(z_old + displacement, -B/kb, B/kb)
F = kb*z_new + b*N/(Tscale+N)*velocity
```

接触状态随张力、走线收紧量和几何收绳量改变，加载/释放分别使用时间常数。
`z` 保存预滑、静止保持和反向运动历史；这两层状态允许同一瞬间的角度和张力因先前路径不同而产生不同响应。
模型采用弹塑性投影，并非完整 LuGre 模型，也没有假定 CAD 已提供未知的三维接触分布。
三个接触增益以及 `wrap_contact_gain` 允许为负：实际绷紧可能增加接触，也可能使绳离开某些接触面，需由数据确定。
微滑刚度 `kb` 保持常量，接触变化只调整摩擦边界和黏性项，避免改变刚度直接制造切向储能。

这是电机观测点到关节入口的集总损失模型；指内逐个导轮的摩擦尚未分段求解，旧 `crossing_guide_mu` 仍只允许零，避免重复计入。
动态分支要求 `cable_damping_n_s_per_m=0`，耗散通过新的摩擦参数建模。关节轴上的原摩擦仍是独立项。

### 求解、能量与兼容性

内部使用双精度后向欧拉，每步枚举张紧/松弛和黏着/滑移组合，求解两个惯性节点。
`routing_max_substep_s` 限制内部步长，S1 初值为 0.1 ms；PhysX 步长独立设置。
该方法包含数值耗散，不能把其平滑效果当成已辨识的真实摩擦。拟合高频/加速度响应时须减小内部步长和 PhysX 步长检查整段曲线敏感性。

弹性储能含两段材料、走线弹簧和两个微滑元件，动能来自两个等效质量。
分别记录摩擦耗散、硬停止碰撞耗散、积分数值耗散、边界输入功以及**未修正的**能量残差。
能量台账检查的是内部子步的指定边界：当前与 PhysX 的耦合用 `h(q)+h_dot*dt` 预测下一步关节边界。
实际 PhysX 关节终点可以与预测不同，故内部残差很小**不等于整个 PhysX 耦合系统能量守恒或步长已收敛**。

显式设置 `routing_model: quasistatic` 可继续使用旧无质量串联弹簧，近远端张力相等且没有走线历史状态。
库的 `TendonModelConfig()` 和旧 `default.yaml` 保持该模式；默认回放使用的 `s1.yaml` 已启用动态模式。

### 电机角度及初态

```text
motor_position_rad = motor_zero_angle_rad + motor_encoder_sign * x / drum_radius_m
motor_velocity_rad_s = motor_encoder_sign * x_dot / drum_radius_m
```

电机角独立积分得到，**不会从关节角拷贝**。电机运动定义为绕线轮输出轴、连续不取模的弧度。
若实际采集的是减速器前转子编码器，先按实际减速比和极性转换；这里没有假设未知减速比。
一阶 `tendon_speed_lag_s` 可拟合速度端口响应，其积分为精确常指令解，仍不是电机电气模型。

初始化同一 CSV 行的 q/T 后，按串联弹性逆解得到 x，再用该行电机角求编码器参考零位。
记录中的关节/电机速度也可用于初始化；缺失通道明确记录为假定静止，电机角缺失时使用配置零位。
默认初始化假定两端初始张力相等、两个微滑状态为零，节点初速取电机绳速与几何路径速度的平均值。
这只是初态约定；单行关节角、电机角和电机张力不能唯一反推未知的远端张力、摩擦历史和内部速度。推荐保留完整加载前史；必要时把未知初态也纳入辨识。
旧准静态分支中，非零材料阻尼与非静止初态的组合不受支持。
角加速度记录为物理步上的后向差分，第一点缺少前一采样，CSV 为 NaN；理想速度阶跃的瞬时加速度不应当作有限物理量拟合。

`plant.advance(speed, dt, q_rad=q, qd_rad_s=dq)` 在动态分支要求当前关节边界；`evaluate(q,dq)` 只读状态。
Python 层可用 `plant.routing_dynamics.state_dict()` / `load_state_dict(snapshot)` 保存和恢复完整走线内部状态，参数配置必须相同。
这不是完整仿真 checkpoint：恢复时还需调用方同步电机状态、PhysX 关节状态、控制器和时钟。仅重新加载一行 CSV 不会恢复摩擦历史。

## 曲线与调参

三类主要 CSV 列是 `FFJ2/position_rad`、`FFJ2/motor_position_rad`、`FFJ2/tension_N`，其余关节同命名。
其中 `tension_N` 始终是**弹簧和走线摩擦之前的电机侧观测**；`distal_tension_N` 是关节入口张力。
如果现有 ROS 采集只有关节角和 effort，需要另外采集并同步电机编码器列；本次没有改动外部 ROS/固件仓库，工具也不会用关节角伪造真实电机反馈。
另有 `velocity_rad_s`、`acceleration_rad_s2`、`motor_velocity_rad_s`、`motor_acceleration_rad_s2`。
内部诊断含 `routing_extension_m`、`routing_length_m`、`routing_locked`、`material_extension_m`、`return_spring_length_m`、`return_spring_force_N`、`own_moment_arm_m`、`paired_moment_arm_m`、`length_balance_error_m`。
旧 `cable_extension_m` 指总伸长/间隙，不应误读成材料绳伸长。

新增内部曲线列如下，全部带 `<joint>/` 前缀：

| 后缀 | 含义 |
|---|---|
| `routing_tension_drop_N` | `Tp-Td`，有符号 |
| `routing_contact_state` | 随加载历史变化的接触状态 |
| `routing_bristle_proximal_m` / `routing_bristle_distal_m` | 两区微滑记忆 |
| `routing_friction_proximal_N` / `routing_friction_distal_N` | 有符号摩擦力，在运动方程中取负 |
| `routing_proximal_*` / `routing_distal_*` | 两节点的位置、速度、加速度 |
| `routing_spring_tension_N` / `routing_stop_reaction_N` | 内部弹簧总内力 / 最大长度约束反力 |
| `routing_material_proximal_extension_m` / `routing_material_distal_extension_m` | 两段材料伸长或负间隙 |
| `routing_internal_slack_m` | `min(es,0)`，长度平衡中须保留的负间隙 |
| `routing_kinetic_energy_J` | 内部等效质量动能 |
| `routing_friction_dissipated_J` / `routing_impact_dissipated_J` | 累计摩擦 / 停止碰撞耗散 |
| `routing_numerical_dissipated_J` | 累计数值耗散，不能算作实测摩擦 |
| `routing_port_work_J` / `routing_energy_residual_J` | 内部台账边界功 / 能量残差 |
| `routing_solver_residual_N` / `routing_stop_impulse_Ns` | 力平衡残差 / 当前子步停止冲量 |

在准静态兼容模式下，新增动态专用诊断列写零，占位不是测量值；以元数据中的 `routing_model` 为准。

```bash
# 默认已经选择 S1 预设；输出 CSV + 完整参数 + 元数据
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py \
  --joint FFJ2 --target-deg 30 --duration 2 --out logs/s1/run.csv

# 三类主曲线和三类误差；缺失真实电机列时不会伪造电机指标
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_curves.py \
  --real real.csv --sim logs/s1/run.csv --joints FFJ2 FFJ3 --out logs/s1/compare.html

# 运动学曲线；仅提供角度时可显式加 --derive-kinematics
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_curves.py \
  --sim logs/s1/run.csv --joints FFJ2 --channels velocity motor_velocity acceleration motor_acceleration \
  --out logs/s1/kinematics.html

# 内部长度分解
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_curves.py \
  --sim logs/s1/run.csv --joints FFJ2 --channels routing material return_spring --out logs/s1/lengths.html

# 两侧张力、差值和随加载改变的接触状态
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_curves.py \
  --sim logs/s1/run.csv --joints FFJ2 FFJ3 \
  --channels tension distal_tension tension_drop contact --out logs/s1/transmission.html

# 摩擦记忆与耗散；无需真实关节侧张力传感器即可检查仿真内部状态
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_curves.py \
  --sim logs/s1/run.csv --joints FFJ2 \
  --channels routing_friction_proximal routing_friction_distal routing_bristle_proximal routing_bristle_distal \
  --out logs/s1/friction.html

# 标量、指定关节参数、几何点均可覆盖；快照记录实际使用值
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py \
  --commands real.csv --strict-commands --duration 3 --out logs/s1/tuned.csv \
  --set tendon.routing_spring_stiffness_n_per_m.FFJ2=1500 \
  --set tendon.routing_spring_max_length_m.FFJ2=0.024 \
  --set tendon.tendon_speed_lag_s.FFJ2=0.01 \
  --set 'tendon.geometry.index.tendons.first.5.point_m=[0.006,0.081]'

# 同一真实指令下扫描走线刚度；真实曲线用于输出误差报告
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_sweep.py \
  --parameter tendon.routing_spring_stiffness_n_per_m.FFJ2 --values 500 1000 1500 \
  --out-dir logs/s1/sweep --real real.csv -- \
  --commands real.csv --strict-commands --duration 3
```

时间对齐只使用统一指令事件和显式时间偏移，不会为关节、电机、张力分别寻找“最佳相位”。
可选差分不插值跨过长缺口、不自动滤波；真实高频噪声会放大，建议优先提供实测速度和采样时间定义。
每个可比较通道输出 RMSE、MAE、偏差、最大绝对误差和尾窗误差；没有真实电机数据时电机指标为空。
默认图显示关节角、电机角、电机张力、关节张力，缺失通道自动跳过；真实记录没有远端传感器时，不会用电机张力充当远端实测值。

### 摩擦与惯量调参入口

以下参数均在 `tendon` 中，除 `routing_model` 和标量 `routing_max_substep_s` 外都支持单关节覆盖：

| 参数 | S1 初值 | 作用 |
|---|---|---|
| `routing_mass_kg` | 0.002 kg | 两节点总等效质量，影响两端动态差异 |
| `routing_proximal_compliance_fraction` | 0.5 | 材料柔度在两端的分配，严格在 0–1 之间 |
| `routing_friction_mu_static` / `routing_friction_mu_dynamic` | 0.25 / 0.15 | 静止上限和滑动摩擦，前者不得小于后者 |
| `routing_bristle_stiffness_n_per_m` | 2000 N/m | 预滑刚度、反向微滑记忆 |
| `routing_viscous_friction_n_s_per_m` | 0.05 N·s/m | 接触加权的速度耗散 |
| `routing_stribeck_speed_m_s` | 0.002 m/s | 摩擦随速度变化的尺度 |
| `routing_normal_preload_n` | 0 N | 非张力贡献的接触预载 |
| `routing_wrap_base_rad` / `routing_wrap_contact_gain_rad` | 0.25 / 1.5 rad | 等效接触量，非逐导轮 CAD 包角 |
| `routing_contact_baseline` | 0.1 | 接触基准 |
| `routing_contact_tension_gain` / `routing_contact_extension_gain` / `routing_contact_path_gain` | 0.5 / 0.3 / 0.1 | 张力、弹簧收紧量、路径变化的影响 |
| `routing_contact_tension_scale_n` / `routing_contact_path_scale_m` | 5 N / 0.02 m | 饱和尺度 |
| `routing_contact_load_tau_s` / `routing_contact_release_tau_s` | 0.02 / 0.08 s | 接触形成/释放时间常数 |
| `routing_max_substep_s` | 0.0001 s | 内部积分上限，数值设置而非物理待拟合量 |

所有初值均未拟合；不要同时自由拟合所有强相关参数。比如：

```bash
PYTHONPATH=src ~/IsaacLab/.venv/bin/python scripts/real2sim_replay.py \
  --joint FFJ2 --target-deg 20 --duration 1.2 --out logs/s1/friction_tuned.csv \
  --set tendon.routing_mass_kg.FFJ2=0.003 \
  --set tendon.routing_friction_mu_static.FFJ2=0.30 \
  --set tendon.routing_contact_extension_gain.FFJ2=-0.15 \
  --set tendon.routing_max_substep_s=0.00005
```

## 后续数据和辨识顺序

先收集同一时间基准的目标三元组、命令序号/掩码/Fallback 标志、两关节角、两电机绕线轮角和每条腱绳的电机侧张力。
“两路张力”指同指两条腱绳各自的电机侧传感器，不要求当前硬件已有弹簧两侧传感器；若能临时测远端张力，将有助于区分关节摩擦和走线摩擦。
推荐包含静止预段、低速分关节运动、不同速度/加速度阶跃和往返卸载，注明手根朝向、传感器延迟/滤波与单位。

先用慢速数据核对绕线符号、锚点和几何；再拟合回位拉簧/预紧与低速往返摩擦；用电机—关节长度差对张力拟合走线刚度和行程；最后用不同加速度、停留和反向段区分等效质量、接触时间常数和电机速度响应。
只测电机张力时，这些参数与关节摩擦可能高度相关；必须使用不同速度/幅度/历史的留出曲线检验，不能从一条阶跃宣称唯一辨识。
自由长与最大长在运动学中主要通过二者之差出现：缺少绝对绳长/电机零位基准时，不能同时唯一辨识两个绝对长度和一个自由零位。
本次实现不把参数扫描自动称作“已标定”，也不把数值 armature 当成真实电机惯量。
