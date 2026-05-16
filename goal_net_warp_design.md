# 球网 XPBD 仿真 + 神经代理模型 — 设计文档

> **终极目标**：在移动端实时跑出精确且稳定的球 + 球网运动。
> 路线：Warp GPU teacher 仿真 → 离线/在线训练 MLP 代理 → 蒸馏到 ≤5M 参数的 student → 部署。

本文档与 [`README.md`](README.md) 配套。README 是 CLI 速查，本文档是物理 / 算法 / 训练 / Roadmap 的工程参考。

---

## 1. 总览

| 组件 | 实现 |
|---|---|
| 物理后端 | NVIDIA Warp（GPU CUDA 内核，单进程并行 batch 多样本） |
| 粒子模型 | XPBD 距离/弯曲约束 + 硬锚 + 2 根支撑绳，**514 粒子 / 1872 约束** |
| 碰撞 | 球-粒子离散 / 球-段 swept CCD / 球门柱胶囊 / 草地 5 段反弹 |
| 样本 | 默认 601 帧（10 s @ 60 Hz） |
| 数据集 | HDF5 chunked（推荐）、npz、json 三种格式 |
| 训练 | 离线 `train`（dataset.h5）+ 在线 `train-online`（sim 实时喂 trainer） |
| 性能 | 4090 上 **~43 sample/s**（B=512, raw=h5），离线 epoch ~0.1 s |

**关键不变量**（任何重写/优化都不能动）：
- 所有物理参数与判定阈值，包括 `bounce_floor_velocity_offset=0.1`、`crossbar_z_min_speed=0.1` 这种细节
- 求解循环顺序（§5.1）
- 输出 schema（§6）

---

## 2. 物理模型与世界坐标系

- **右手系**，y 轴竖直向上
- 球门门口位于 z=0 平面，**网袋向 -z 方向延伸**（`-goal.depth ≤ z ≤ 0`）
- 球门宽度沿 x 轴：`-goal.width/2 ≤ x ≤ +goal.width/2`
- 球员从 +z 一侧射门
- 重力 `(0, -9.81, 0)`
- 地面在 `y = ground.y`（默认 0），网粒子和球都不能穿过

球门尺寸默认（m）：`width=7.32, height=2.44, depth=2.0`（标准足球门）。
网格离散：`cell_size_{x,y,z} = 0.305`（约 1 英尺）。

---

## 3. 完整参数表

全部在 `params.py` 的 dataclass 中定义。

### 3.1 `RopeParams`（最重要）

| 字段 | 默认 | 含义 |
|---|---|---|
| `radius` | 0.018 | 渲染用绳半径 |
| `collision_radius` | **0.16** | 球-网碰撞膨胀半径，封 0.305 m 网孔（≤0 时回退到 `radius`） |
| `particle_mass` | 0.035 | 单粒子质量 (kg) |
| `stretch_stiffness` | 0.92 | XPBD 拉伸约束 stiffness（[0,1]） |
| `bend_stiffness` | 0.18 | 二阶邻接（弯曲）约束 |
| `damping` | 0.035 | 速度阻尼（每 substep 末乘 `1 - damping`） |
| `friction` | 0.32 | 默认切向摩擦 |
| `restitution` | 0.55 | 默认法向恢复 |
| `panel_restitution_back/top/side` | 0.10 / 0.30 / 0.20 | 各 panel 法向系数 |
| `panel_friction_back_tangent` | 0.80 | BACK 面切向摩擦 |
| `impulse_clamp` | 6.0 | 单次 swept 命中时注入到绳粒子的最大 Δv (m/s) |
| `impulse_speed_threshold` | 1.0 | 球速 < 此值时 swept 不再注能给网（杜绝低速能量泵入循环） |

### 3.2 `SolverParams`

| 字段 | 默认 | 含义 |
|---|---|---|
| `frame_dt` | 1/60 | 输出帧间隔 |
| `substeps` | **12** | 每帧 substep 数（必须 ≥ 12 防穿网，与 `collision_radius=0.16` 配套） |
| `iterations` | 8 | 每 substep 内 XPBD 约束 GS 迭代次数 |
| `duration` | **2.0** | 单样本时长 s（要捕到落地后滚动） |
| `gravity` | `[0, -9.81, 0]` | |
| `enable_bend_constraints` | True | bend 约束开关 |
| `stuck_speed_threshold` | 0.5 | 球速 ≤ 此值视为低速 |
| `stuck_duration_seconds` | 0.3 | 低速持续 ≥ 此时长后冻结剩余 substep |

### 3.3 `CollisionParams`（quality 阈值）

| 字段 | 默认 | 含义 |
|---|---|---|
| `severe_penetration_threshold` | **0.15** | 球 z < `safety_back_z` 的最大穿透量 |
| `safety_back_z` | -2.25 | 球门后部安全边界（=`-depth - 0.25`） |
| `max_ball_speed` | 80.0 | 球速爆炸阈值 |
| `max_particle_speed` | 60.0 | 网粒子速度爆炸阈值 |
| `max_net_displacement` | 4.0 | 网粒子相对 rest 位移上限 |
| `stuck_speed_threshold` | 0.05 | 接触后球速过低标记 stuck |
| `stuck_duration` | 0.25 | stuck 持续时长阈值 |

### 3.4 `GroundParams`（草地反弹）

| 字段 | 默认 | 含义 |
|---|---|---|
| `enabled` | True | |
| `y` | 0.0 | 地面高度 |
| `bounce_restitution` | 0.8 | 垂直反弹系数 |
| `bounce_speed_loss` | 12.0 | 弹跳模式整体速度衰减系数（× 9.8 × dt / |v|） |
| `bounce_to_roll_vertical_threshold` | 2.0 | 反弹后垂直速度 < 此值切换到滚动 |
| `bounce_to_roll_total_threshold` | 3.0 | 反弹时总速度 < 此值切换到滚动 |
| `roll_speed_loss` | 2.0 | 滚动模式水平衰减（× dt / |v|） |
| `bounce_floor_velocity_offset` | 0.1 | 反弹后 vy 减 0.1 m/s（残余能量损失） |

### 3.5 `GoalpostParams`（球门柱）

| 字段 | 默认 | 含义 |
|---|---|---|
| `enabled` | True | |
| `radius` | 0.06 | =HALF_GOALPOST_WIDTH |
| `speed_change_factor` | 0.6 | 法向 ×0.6（切向不损失） |
| `crossbar_z_min_speed` | 0.1 | 横梁反弹后 \|vz\| 钳到 ≥ 0.1（防原地冻结） |

### 3.6 `ShapeParams`（网形态）

| 字段 | 默认 | 含义 |
|---|---|---|
| `top_sag` | 0.16 | 顶网/后网下垂量 |
| `side_slope` | 0.15 | 侧网外扩斜度 |
| `back_slope` | 0.05 | 后网底部 z 略后倾 |
| `back_pocket_depth` | 0.30 | 后网中心向 -z 兜状鼓包，四边为零 |
| `stay_count` | 2 | 物理支撑绳数（0 / 2 / 4） |
| `stay_anchor_offset_x/y/z` | 0.3 / 0.6 / 0.4 | stay 远端锚点相对后上角的偏移（外/上/后） |

### 3.7 `BallState`（射门输入）

| 字段 | 默认 | 含义 |
|---|---|---|
| `position` | — | 起始位置 |
| `velocity` | — | 初速度 (m/s) |
| `angular_velocity` | (0,0,0) | 角速度（仅地面反弹时按比例衰减，不影响碰撞响应） |
| `radius` | 0.13 | 球半径 |
| `mass` | 1.0 | 球质量 |

> **batch 内 ball radius/mass 必须一致**。异构需把这两个量改成 `wp.array(B,)`。

---

## 4. 拓扑生成（`topology.py`）

### 4.1 数据结构

```python
@dataclass(frozen=True)
class Particle:
    index: int; position: Vec3
    panel: str            # "back" / "left" / "right" / "top"
    u: int; v: int        # panel 网格内 2D 坐标
    anchored: bool        # 边缘是否固定

@dataclass(frozen=True)
class DistanceConstraint:
    index: int; i0: int; i1: int
    rest_length: float; stiffness: float
    kind: str             # "stretch" / "bend"
    panel: str

@dataclass(frozen=True)
class AnchorConstraint:
    index: int; particle: int
    target: Vec3; stiffness: float; hard: bool

@dataclass(frozen=True)
class GoalpostSegment:
    index: int; name: str
    p0: Vec3; p1: Vec3
    radius: float; kind: str  # "post" / "crossbar"
```

### 4.2 锚定边缘规则

| Panel | 锚定粒子（`anchored=True`） |
|---|---|
| back | `iy == ny`（顶/横梁）或 `ix in {0, nx}`（连立柱）或 `iy == 0`（底缘连地面） |
| left/right | `iy == ny`（顶）或 `iz == 0`（前缘连立柱）或 `iy == 0`（底缘连地面） |
| top | `iz == 0`（前缘连横梁）或 `ix in {0, nx}`（连立柱） |

锚定 → `inverse_mass = 0`，所有约束求解中只读不写。

**例外**：`shape.stay_count > 0` 时，后上 2 个角粒子（back panel 的 `(0, ny)` 和 `(nx, ny)`，dedup 共享 side/top 同位置粒子）被显式改为 `anchored=False`，由物理 stay 绳承重（§4.4）。

### 4.3 球门柱 3 段胶囊

设 `W=goal.width, H=goal.height, r=goalpost.radius=0.06`：

| name | p0 | p1 | kind |
|---|---|---|---|
| post_left | `(W/2 + r, 0, 0)` | `(W/2 + r, H+r, 0)` | post |
| post_right | `(-W/2 - r, 0, 0)` | `(-W/2 - r, H+r, 0)` | post |
| crossbar | `(-W/2, H+r, 0)` | `(W/2, H+r, 0)` | crossbar |

### 4.4 物理支撑绳（`_add_support_stays`）

真实球门网后侧由 2~4 根绳拉到斜上方锚点。建模成普通粒子 + 距离约束：

1. **角脱锚**：取 back 面的后上左/右角粒子（dedup 共享 side/top 顶后角同位置粒子），把 `anchored` 改为 `False`
2. **加 stake ghost 粒子**：在 `(±(W/2 + offset_x), corner_y + offset_y, -depth - offset_z)` 处加 `Particle`，`anchored=True, inverse_mass=0`，索引记入 `topo.stake_particle_indices`
3. **加 stretch 距离约束**：corner ↔ stake 之间一条 `DistanceConstraint`，`stiffness = rope.stretch_stiffness`，`rest_length = 几何距离`
4. **viewer**：stake 不在 `panel_particle_indices` 里所以不画成网粒子；stay 约束从 mesh rope 渲染中排除；每帧动态画 `corner ↔ stake` 一条独立 stay 线（颜色 `(220,200,80)` 区分 mesh）

默认 `stay_count=2, offset=(0.3, 0.6, 0.4)`：每根绳 ~0.78 m，角粒子被斜上拉向 `(±3.96, 3.04, -2.4)`。**拿掉 stay 绳后整个后网会塌**——后上角没有任何其他锚定路径。

### 4.5 Warp 需要的扁平 ndarray

```python
# Per-particle (N,)
particle_pos_init: float32[N, 3]
particle_inv_mass: float32[N]               # anchored 为 0
particle_panel_id: int32[N]                 # 0=back, 1=left, 2=right, 3=top

# Per-distance-constraint (M,)
constraint_i0:        int32[M]
constraint_i1:        int32[M]
constraint_rest:      float32[M]
constraint_stiffness: float32[M]
constraint_panel_id:  int32[M]
constraint_kind:      int32[M]              # 0=stretch, 1=bend

# Per-anchor (A,)
anchor_particle:  int32[A]
anchor_target:    float32[A, 3]
anchor_stiffness: float32[A]
anchor_hard:      int32[A]                  # 0/1

# Per-goalpost-segment (3,)
post_p0:     float32[3, 3]
post_p1:     float32[3, 3]
post_radius: float32[3]
post_kind:   int32[3]                       # 0=post, 1=crossbar

# Per-panel collision coefficients (4,) — 预查表
panel_restitution: float32[4]
panel_friction:    float32[4]
```

**Panel id 映射（必须固定）**：`back=0, left=1, right=2, top=3`。Warp kernel 内通过 `panel_restitution[panel_id]` 取系数，避免字符串。

### 4.6 索引稳定性

`tests/test_topology.py::test_topology_is_stable` 要求：相同参数两次生成的 topology 字节级一致（`stable_signature`）。Warp 不影响——拓扑生成在 Python 端，Warp 只消费。

---

## 5. 求解算法（`solver_warp.py`）

### 5.1 主循环骨架（**顺序极其重要**）

```
for frame_index in 0..total_frames:
    record frame
    if frame_index == total_frames: break

    for substep in 0..substeps:
        if frozen[b]: skip                                 # 早停后只填帧

        # ---- 网粒子积分 ----
        previous_positions = positions.copy()
        for i where inv_mass[i] > 0:
            velocities[i] += gravity * sub_dt
            positions[i]  += velocities[i] * sub_dt
        resolve_particle_ground()                           # y < ground.y 拉回

        # ---- 球积分 + 离散外部碰撞 ----
        ball.velocity += gravity * sub_dt
        ball_prev      = ball.position
        ball.position += ball.velocity * sub_dt
        resolve_ball_vs_goalposts(ball, ball_prev)          # 必须先于 swept_net
        resolve_ball_swept_collisions(ball, ball_prev)      # CCD 球-绳
        resolve_ball_ground(ball)                           # 草地反弹/滚动

        # ---- XPBD 约束求解（GS 迭代）----
        for it in 0..iterations:
            solve_anchors()                                 # 硬锚强制
            solve_distance_constraints(stretch)
            if enable_bend: solve_distance_constraints(bend)
            solve_ball_particle_collisions(ball)            # 离散，残余穿透清理
            solve_ball_segment_collisions(ball)             # 离散，残余穿透清理

        # ---- 速度 update + 全局阻尼 ----
        velocities[i] = (positions[i] - previous_positions[i]) / sub_dt
        velocities[i] *= (1 - damping)

        update_stuck_timers()                               # 早停统计
```

**为什么 goalposts 必须在 swept_net 前**：swept_net 用的 `ball_prev` 是子步开始的位置；如果先撞柱，`ball.position` 已被改写到 TOI 处，传给 swept_net 就错了。

### 5.2 XPBD 距离约束

```
for each constraint c with (i0, i1, rest, stiffness):
    delta = pos[i1] - pos[i0]
    L     = |delta|
    if L < 1e-8: continue
    error = L - rest
    w0, w1 = inv_mass[i0], inv_mass[i1]
    if w0 + w1 <= 0: continue
    direction      = delta / L
    correction_mag = error * stiffness / (w0 + w1)
    pos[i0] += direction * correction_mag * w0
    pos[i1] -= direction * correction_mag * w1
```

> 这是简化的 XPBD（没真用 compliance/lambda）。**不能改公式**，否则数值漂移。

**GS GPU 化**：单 kernel + `wp.atomic_add` 全约束并行（Jacobi 风格）。一次 launch 内所有约束读 launch 入口的 `positions` 快照，写入用 atomic 累加到下次 launch 才可见。比 CPU 串行 GS 收敛慢 1.3~1.5×，默认 `iterations=8` 已足够；如要更紧约束可调到 12。

### 5.3 锚点求解

```
for each anchor a:
    if a.hard:
        pos[a.particle] = a.target
        vel[a.particle] = 0
    else:
        pos[a.particle] += (a.target - pos[a.particle]) * a.stiffness
```

默认所有 anchor 都是 hard，等价于 `inv_mass=0` + 强制位置。

### 5.4 球-粒子离散碰撞

```
collision_radius_total = ball.radius + rope.collision_radius
for each particle i:
    delta = pos[i] - ball.position
    d     = |delta|
    if d <= 1e-8 or d >= collision_radius_total: continue
    n   = delta / d
    pen = collision_radius_total - d
    inv_p, inv_b = inv_mass[i], 1/ball.mass
    pos[i]        += n * pen * inv_p / (inv_p + inv_b)
    ball.position -= n * pen * inv_b / (inv_p + inv_b)
    (e, f) = panel_coeffs(particle_panel[i])
    ball.velocity = reflect(ball.velocity, n, e, f)
    contacts.append(type="particle", ...)
```

`reflect(v, n, e, f)`：
```
vn = v · n
if vn >= 0: return v        # 已分离
v_normal  = n * vn
v_tangent = v - v_normal
return v_normal * (-e) + v_tangent * (1 - f)
```

### 5.5 球-段离散碰撞

```
for each distance_constraint c:
    p0, p1   = positions[c.i0], positions[c.i1]
    closest, t = closest_point_on_segment(ball.position, p0, p1)
    delta    = closest - ball.position
    d        = |delta|
    if d >= collision_radius_total: continue
    n   = delta / d
    pen = collision_radius_total - d
    w0  = inv_mass[c.i0] * (1 - t)
    w1  = inv_mass[c.i1] * t
    wb  = 1 / ball.mass
    total = w0 + w1 + wb
    pos[c.i0]     += n * pen * w0 / total
    pos[c.i1]     += n * pen * w1 / total
    ball.position -= n * pen * wb / total
    (e, f) = panel_coeffs(c.panel)
    ball.velocity = reflect(ball.velocity, n, e, f)
    contacts.append(type="segment", ...)
```

### 5.6 球-段 swept CCD（**核心防穿网**）

```
delta = ball.position - ball_prev
if |delta| <= 1e-8: return
collision_radius = ball.radius + rope.collision_radius

# 对每段绳，用子步起点 previous_positions 作静态端点
best_toi = 2.0; best_constraint = None
for each constraint c:
    p0 = previous_positions[c.i0]; p1 = previous_positions[c.i1]
    toi = swept_sphere_vs_segment_toi(ball_prev, ball.position, collision_radius, p0, p1)
    if toi is not None and toi < best_toi:
        best_toi = toi; best_constraint = c
if no hit: return

toi         = clamp(best_toi, 0, 1)
contact_pos = ball_prev + delta * toi
ball.position = contact_pos

closest, t = closest_point_on_segment(contact_pos, best_seg_p0, best_seg_p1)
offset = contact_pos - closest
n = offset / |offset| if |offset| > 1e-8 else -ball.velocity / |ball.velocity|

ball.position = closest + n * (collision_radius + 0.001)   # 1mm skin

velocity_before = ball.velocity
(e, f) = panel_coeffs(best_constraint.panel)
ball.velocity = reflect(ball.velocity, n, e, f)

# 球速 < impulse_speed_threshold 时跳过冲量注入
if |velocity_before| >= impulse_speed_threshold:
    apply_segment_impulse_to_endpoints(...)

contacts.append(type="segment_swept", ...)
```

**swept TOI 数学**（`swept_sphere_vs_segment_toi`）：球心运动 `C(t) = ball_a + t*(ball_b - ball_a)` 与线段 `L(s)` 之间最近距离 = `radius` 求最早 `t ∈ [0,1]`。等价于 ray-vs-capsule。

```
# Step 1: early-out
closest_a, _ = closest_point_on_segment(ball_a, seg_p0, seg_p1)
if |ball_a - closest_a| <= radius:
    return 0.0  # 起点已穿透

# Step 2: 无限圆柱
d = ball_b - ball_a; seg = seg_p1 - seg_p0; m = ball_a - seg_p0
seg_len2 = seg · seg
if seg_len2 < 1e-12:
    return ray_sphere_first_hit(ball_a, d, seg_p0, radius)
a_coeff = seg_len2 * (d·d) - (d·seg)^2
b_coeff = seg_len2 * (m·d) - (d·seg)*(m·seg)
c_coeff = seg_len2 * (m·m - radius^2) - (m·seg)^2
解 a*t^2 + 2b*t + c = 0
对 t ∈ [0,1] 检查 s = ((m·seg) + t*(d·seg)) / seg_len2 ∈ [0, 1]，取最小 t

# Step 3: 两端球帽
cap0 = ray_sphere_first_hit(ball_a, d, seg_p0, radius)
cap1 = ray_sphere_first_hit(ball_a, d, seg_p1, radius)
return min(best_t, cap0, cap1)
```

### 5.7 冲量注入两端粒子

swept 命中时只反弹球、绳子纹丝不动是不真实的。把球损失的法向动量按 `(1-t):t` 权重分给两端粒子：

```
delta_v_ball = velocity_new - velocity_old
impulse_n    = ball.mass * (delta_v_ball · normal)
if impulse_n <= 0: return

clamp_speed = rope.impulse_clamp     # 6.0 m/s
for (idx, w) in [(c.i0, 1-t), (c.i1, t)]:
    if inv_mass[idx] <= 0 or w <= 0: continue
    dv_mag = impulse_n * w * inv_mass[idx]
    dv     = -normal * dv_mag        # rope 受 -normal 方向
    if |dv| > clamp_speed: dv *= clamp_speed / |dv|

    velocities[idx]         += dv
    shift                    = dv * sub_dt
    positions[idx]          += shift
    previous_positions[idx] -= shift     # 关键！
```

**关键陷阱**：XPBD 末尾用 `velocity = (positions - previous_positions) / sub_dt` 反算速度，会把直接写入 `velocities` 的注入抹平。所以必须**同时**修改 `positions`（+shift）和 `previous_positions`（-shift），让反算自然得到 `v_old + dv`。`velocities[idx] += dv` 是为了下个 substep 的积分立刻看到注入。

### 5.8 球-球门柱碰撞

```
total_radius = ball.radius + post.radius
找到最早 toi（同 5.6 swept），命中段 best_segment

contact_pos = ball_prev + delta * toi
closest, _ = closest_point_on_segment(contact_pos, p0, p1)
offset     = contact_pos - closest
n          = offset / |offset|       # 退化时 fallback 速度反向
ball.position = closest + n * (total_radius + 0.001)

# 球门柱反弹公式（不同于网！）：(v - va) - va * 0.6
vn = ball.velocity · n
va = n * vn
new_v = (ball.velocity - va) - va * 0.6      # 切向不损失，法向反向 ×0.6

if best_segment.kind == "crossbar":
    if |new_v.z| < crossbar_z_min_speed:     # 0.1
        new_v.z = sign(new_v.z) * 0.1        # vz=0 时取 +0.1

ball.velocity = new_v
contacts.append(type="goalpost"|"crossbar", ...)
```

### 5.9 草地反弹（5 段，严格对齐）

```
触发：ball.position.y - ball.radius < ground.y
保存 old_velocity, old_speed = |old_velocity|

# 1) 抬到 floor + radius
ball.position.y = ground.y + ball.radius

# 2) 垂直反弹（含 -0.1 残余损失）
new_vy = max(0, -old_velocity.y * bounce_restitution - bounce_floor_velocity_offset)

# 3) 弹跳 vs 滚动
if new_vy > 2.0 AND old_speed >= 3.0:
    v = (old_velocity.x, new_vy, old_velocity.z)
    if |v| > 1e-8:
        scale = max(0, 1 - bounce_speed_loss * 9.8 * sub_dt / |v|)
        v *= scale
    ball.velocity = v
    mode = "bounce"
else:
    v = (old_velocity.x, 0, old_velocity.z)
    if |v| >= 0.01:
        scale = max(0, 1 - roll_speed_loss * sub_dt / |v|)
        ball.velocity = v * scale
    else:
        ball.velocity = 0
    mode = "roll"

# 4) 角速度按线速度比例衰减
if old_speed >= 0.01:
    ball.angular_velocity *= |ball.velocity| / old_speed
else:
    ball.angular_velocity = 0

# 5) 写 contact (type = ground_bounce | ground_roll)
```

### 5.10 网粒子地面 + 帧末

```
# 网粒子 ground
for each particle i where inv_mass[i] > 0:
    if positions[i].y < ground.y:
        positions[i].y = ground.y
        if previous_positions[i].y < ground.y:
            previous_positions[i].y = ground.y
        if velocities[i].y < 0:
            velocities[i].y = 0

# 帧末速度更新与阻尼
velocities[i]  = (positions[i] - previous_positions[i]) / sub_dt
velocities[i] *= max(0, 1 - rope.damping)
```

### 5.11 Warp kernel 列表（按主循环顺序）

| Kernel | 维度 | 作用 |
|---|---|---|
| `k_save_previous_positions` | (B, N) | 保存上一子步位置 |
| `k_integrate_particles` | (B, N) | 重力 + 半隐式 Euler，受 frozen_mask 控制 |
| `k_resolve_particle_ground` | (B, N) | y < ground.y 拉回 |
| `k_integrate_ball` | (B,) | 球积分 |
| `k_swept_ball_vs_posts` | (B, 3) | atomic_min(toi_packed) |
| `k_apply_post_response` | (B,) | 解码、回退球、反弹、写 contact |
| `k_swept_ball_vs_segments` | (B, M) | atomic_min |
| `k_apply_segment_response` | (B,) | 回退球、反弹、注入冲量到两端粒子 |
| `k_resolve_ball_ground` | (B,) | 草地 5 段反弹 |
| `k_solve_anchors` | (B, A) | 硬锚强制 |
| `k_solve_distance_constraints` | (B, M_stretch) | atomic_add |
| `k_solve_distance_constraints_bend` | (B, M_bend) | atomic_add |
| `k_solve_ball_particle_collisions` | (B, N) | 离散球-粒子 |
| `k_solve_ball_segment_collisions` | (B, M) | 离散球-段 |
| `k_update_velocities_and_damp` | (B, N) | velocities = (pos-prev)/dt; *= (1-damping) |
| `k_update_stuck_timers` | (B,) | 早停统计、frozen_mask 设置 |
| `k_record_frame` | (B, N) | 写帧缓冲 |
| `k_check_quality` | (B,) | 累计 max_penetration / target_hit |

### 5.12 swept argmin（packed atomic_min）

每子步对每 batch 找最早 TOI 的段。用 64-bit packed atomic_min：

```python
# 主机端初始化
swept_hit_packed.fill_(wp.uint64(0xFFFFFFFFFFFFFFFF))

@wp.func
def encode_toi_idx(toi: float, idx: int) -> wp.uint64:
    bits = wp.bitcast_uint32(toi)            # 假设 toi >= 0，IEEE 754 单调
    return (wp.uint64(bits) << 32) | wp.uint64(idx)

# kernel 内
toi = swept_sphere_vs_segment_toi(...)
if 0.0 <= toi <= 1.0:
    wp.atomic_min(swept_hit_packed, b, encode_toi_idx(toi, c))
```

随后一个 `(B,)` kernel 解码 `swept_hit_packed[b]`，做反弹 + 冲量注入。

### 5.13 Warp 常见坑

- **kernel 编译慢**：第一次 launch 1~2 s（cache 后秒起）
- **atomic_add 浮点不确定性**：同样输入两次跑结果末位 ulp 差 1e-7（topology 不受影响，topology 在 Python 端生成）
- **NaN 调试**：kernel 内除零静默 NaN。每个 `1.0/x` 之前必须 `if x <= 1e-8: return`
- **bool 类型**：Warp 内用 `int` 0/1 替代更稳妥
- **不要在 kernel 里调用 Python 函数**，包括 `print`。要 debug 就拷回 numpy 再 print
- **CPU device 调试**：`device='cpu'` 时 kernel 内可以 print 到 stdout
- **`@wp.func`** vs `@wp.kernel`：辅助函数用 `@wp.func`（kernel 内调用）；`@wp.kernel` 是 launch 入口

---

## 6. 采样器（`sampler.py`）

### 6.1 默认配置

| 字段 | 默认 |
|---|---|
| `panels` | `[back, left, right, top, corner]` |
| `styles` | `[ground, low, mid, high, lob]` |
| `speed_range` | (22, 32) m/s |
| `spin_range` | (-18, 18) rad/s |
| `azimuth_range_deg` | (-45, 45) |
| `elevation_range_deg` | (-2, 25) |
| `distance_range` | (6, 18) m |
| `target_jitter` | 0.25 m |
| `corner_post_ratio` | 0.15 | ~15% 样本贴近立柱/横梁，提升擦柱多样性 |
| `source_x_clamp / source_y_clamp / source_z_min` | 9 / (0.2,4) / 1.5 | 源点 envelope（防止极端 grazing 样本） |

### 6.2 风格 profile

| style | elevation_bias | speed_range |
|---|---|---|
| ground | -2° | (22, 28) |
| low | +2° | (24, 30) |
| mid | +6° | (26, 32) |
| high | +12° | (24, 30) |
| lob | +18° | (18, 24) |

### 6.3 采样流程

对每个样本 `i`：
1. `panel = panels[i % 5]`，`style = styles[i % 5]`
2. `target = _sample_panel_target(rng, config, panel)`（panel/角落内随机 + jitter）
3. 抽 azimuth ∈ range，elevation ∈ range + style_bias（钳到边界）
4. 抽 distance ∈ range，speed ∈ style.speed_range
5. `direction = _direction_from_angles(...)`（azimuth=0,elevation=0 → `(0,0,+1)`）
6. `source = target - direction * distance`，强制 `source.y >= 0.05, source.z >= 0.5`
7. `velocity = _solve_initial_velocity(source, target, speed)`（一次牛顿迭代解抛物线）
8. `spin = (Uniform(spin_range), …, …)`
9. 输出 `ShotInput(sample_id, target_panel, seed, template, ball=BallState(...))`

### 6.4 panel target 区间

| panel | x | y | z |
|---|---|---|---|
| back | `Uniform(-W/2+m, W/2-m)` | `Uniform(0.2, H-m)` | `-depth + 0.05` |
| left | `-W/2 + 0.05` | `Uniform(0.3, H-m)` | `Uniform(-depth+m, -0.2)` |
| right | `+W/2 - 0.05` | `Uniform(0.3, H-m)` | `Uniform(-depth+m, -0.2)` |
| top | `Uniform(-W/2+m, W/2-m)` | `H - 0.05` | `Uniform(-depth+m, -0.1)` |
| corner | `±(W/2-0.05)` | `H - Uniform(0.05, 0.45)` | `Uniform(-depth+m, -0.2)` |

最后整体加 `Uniform(-jitter, jitter)`（z 轴 jitter 缩 0.4×）；`m = max(0.1, jitter)`。

---

## 7. 质量检查 / 异常分类

每个样本结束时打 `quality.issues` 标签，**任一非空都标记 `clean=False`**：

| issue | 触发条件 |
|---|---|
| `severe_penetration` | `max_penetration > 0.15` |
| `nan_or_inf` | ball pos/vel 任意分量非有限 |
| `velocity_explosion` | `\|ball.velocity\| > 80` |
| `particle_velocity_explosion` | `max(\|particle.velocity\|) > 60` |
| `constraint_divergence` | `max(\|positions[i] - rest_positions[i]\|) > 4.0` |
| `stuck` | 接触发生后球速 < 0.05 累计 > 0.25 s |
| `target_panel_missed` | 整个 simulation 期间无任何 contact |

`max_penetration` 定义：`max(0, safety_back_z - ball.position.z)`，即球穿过 `safety_back_z = -2.25` 平面的深度。

`stats` 还输出：`frame_dt / substeps / iterations / duration / frame_count / contact_count / max_constraint_error / max_net_displacement / ball_came_to_rest`。

clean ratio 实测约 55–65%。剩余失败基本是 `severe_penetration` + `stuck`——sampler 偶尔生成极端入射角/源点的样本被网"漏"出去；或球停下后被网某处轻微卡住。训练时按 `quality_clean=True` 过滤。

---

## 8. 输出 schema

### 8.1 HDF5 单文件（推荐，`--raw-format h5`）

```
<output_dir>/
├── dataset.h5            # 所有样本 raw + features 都在里头
├── topology.json         # 共享拓扑（拷贝一份方便人读）
├── metadata.json         # params snapshot
└── batch_report.json
```

**`dataset.h5` 内部布局**（S=样本数, F=601, N=514, K=7 种 issue）：

```text
attrs:
  schema_version, frame_dt, frame_count, particle_count
  topology_json   # 完整拓扑
  metadata_json   # params snapshot
  issue_names     # K=7 种 quality issue 名字数组
  include_raw

datasets:
  sample_id            (S,)         vlen str       # "sample_00000"
  target_panel         (S,)         vlen str       # back/left/right/top/corner
  template             (S,)         vlen str
  seed                 (S,)         int64
  input_position       (S, 3)       float32
  input_velocity       (S, 3)       float32
  input_angular        (S, 3)       float32
  input_radius         (S,)         float32
  input_mass           (S,)         float32

  ball_position        (S, F, 3)    float32
  ball_velocity        (S, F, 3)    float32
  particle_position    (S, F, N, 3) float32        # chunked，单样本 ~3.5 MB

  # CSR-encoded contact stream
  contact_offset       (S+1,)       int64
  contact_time         (TotalC,)    float32        # 按时间排序
  contact_object_type  (TotalC,)    int32          # 0=particle 1=segment 2=segment_swept
                                                   # 3=goalpost 4=crossbar 5=ground_bounce 6=ground_roll
  contact_object_index (TotalC,)    int32
  contact_position     (TotalC, 3)  float32
  contact_normal       (TotalC, 3)  float32
  contact_strength     (TotalC,)    float32

  quality_clean        (S,)         bool
  quality_target_hit   (S,)         bool
  quality_issue_mask   (S, K)       bool           # 每位对应 issue_names[i]
  quality_max_pen      (S,)         float32
  quality_max_pen_time (S,)         float32
  stats_contact_count  (S,)         int32
  stats_max_disp       (S,)         float32
  stats_came_to_rest   (S,)         bool
```

### 8.2 npz / json（小数据集 / 兼容）

```
<output_dir>/
├── topology.json
├── metadata.json
├── summary.jsonl        # 每行一条样本摘要
├── batch_report.json
├── features/sample_NNNNN.json
└── raw/sample_NNNNN.npz       # 或 .json
```

`features/sample_*.json`：
```jsonc
{
  "sample_id": "sample_00000",
  "schema_version": "goal_net_params.v1",
  "input_features": { "position", "velocity", "angular_velocity",
                      "target_panel", "template", "seed", "radius", "mass" },
  "ball_trajectory":   [ {"time", "position", "velocity"}, ... ],   // F 帧
  "net_control_points":[ {"time", "positions": [16 抽样粒子]} ],
  "quality": { "clean", "issues", "target_hit",
               "max_penetration_depth", "max_penetration_time" }
}
```

`raw/sample_*.npz` 内含：`frame_time` `ball_position` `ball_velocity` `particle_position` `contact_*` `meta_json`（JSON 字符串包含 shot/quality/stats）。

`raw/sample_*.json` 同 npz 但为文本格式，每文件还内嵌一份 topology（约 250 KB），文件大小约比 npz 大 5~10 倍——仅在需要"单文件自包含"时使用。

`contacts.object_type` 取值字符串：`"particle" / "segment" / "segment_swept" / "goalpost" / "crossbar" / "ground_bounce" / "ground_roll"`。

### 8.3 `batch_report.json`

```json
{
  "sample_count": 0,
  "clean_count": 0,
  "abnormal_count": 0,
  "abnormal_types": {"severe_penetration": 0},
  "panel_stats": {"back": {"samples": 0, "contacts": 0, "abnormal": 0}}
}
```

### 8.4 PyTorch 读取

```python
from torch.utils.data import DataLoader
from dataset_loader import GoalNetH5Dataset

ds = GoalNetH5Dataset("dataset.h5", clean_only=True)
loader = DataLoader(ds, batch_size=8, num_workers=4, persistent_workers=True)
for b in loader:
    ball_traj = b["ball_position"]      # (8, 601, 3)
    net_traj  = b["particle_position"]  # (8, 601, 514, 3)
    cond      = b["input_state"]        # (8, 9)  pos+vel+ang
```

---

## 9. 训练管线

模型与训练目标：

- **输入** `(球初态: pos_xy, vel, ang, radius, mass) + 归一化时间 t_norm = f / (F-1)`，`in_dim = 10 + 2·n_time_freq`（默认 18）
- **输出** `(ball_pos, ball_vel, particle_pos[N])`
- **架构**：`GoalNetMLP`（`model.py`），带内置归一化 buffer（`in_mean / in_std` + 各 head 的 `*_mean / *_std`），**checkpoint 自包含归一化**
- **损失**：归一化空间 MSE，三个 head 加权求和（`--w-ball-pos --w-ball-vel --w-net`）

两条训练路径：

| 路径 | CLI | 数据来源 | 优劣 |
|---|---|---|---|
| 离线 | `train` | dataset.h5 | preload 后 epoch ~0.1 s；但容量饱和后必然 epoch 过拟合 |
| 在线 | `train-online` | sim 实时产 | 每 batch 都是新轨迹，等价 epoch=∞，无过拟合；refill 间隔 12 s 是 wall-clock 主要时间 |

### 9.1 归一化策略（**关键，避开过的坑**）

`--norm-scale-mode` 选 target 的 scale：

- `global`（旧默认）：stddev over all frames。**有 leak**：球停下后帧占比大，导致 std 被压低，loss 不能区分"球停在哪儿"——即使预测全恒等于均值也能拿到 ~3% 的 loss
- `init`：用 frame-0 std + input_velocity.std（比 global 强但 net_pos 仍走全局）
- `robust`（**推荐**）：half-range = `(max - min) / 2`，对 endgame leak 鲁棒

`--drop-last-frames K`：训练时排除最后 K 帧（默认 5）。原因：(a) `t_norm = 1` 处 sin/cos 周期 collide；(b) endgame 大量"球已停"的静帧过拟合。

**loss 权重重平衡**（推荐 `--w-ball-pos 50 --w-ball-vel 20 --w-net 1`）：
- robust scale 把 ball head 压到全部 loss 的 ~3%
- net head 1 cm 已足够小，无需主导梯度
- 50 / 20 / 1 让 ball 误差成为优化器主关切

### 9.2 离线 `train`

```bash
python cli.py train \
    --dataset /data3/netsim/dataset_v2/dataset.h5 \
    --output runs/baseline \
    --epochs 500 --batch 4096 --device cuda \
    --hidden 2048 2048 2048 2048 2048 2048 2048 2048 \
    --norm-scale-mode robust --drop-last-frames 5 \
    --lr-min-frac 0.01 \
    --w-ball-pos 50 --w-ball-vel 20 --w-net 1
```

**性能**：
- preload train+val（默认）：4090 上 epoch ~0.1 s，但 50+ GB RAM 占用
- 启动前 `psutil` 1.4× 安全裕度检查
- 不够 RAM 时用 `--no-preload` 走 HDF5 流式读（慢 ~280×）

**当前最佳离线 baseline**：8 层 × 2048 wide MLP（29M 参数），500 epoch，ball_pos RMSE **~0.80 m**。再延长 epoch 到 1000 反而恶化（train/val ratio 3.4× 过拟合）——容量在 16k clean 样本上**已饱和**。

### 9.3 在线 `train-online`

架构：

```
                      +--------------+         +-----------+
   sample_seed -----> | XpbdWarpSolver|--arrays-->| Online    |
                      | (B=512)       |  (RAM)  | Frame Pool |
                      +--------------+         | (K=4 slots)|
                                               +-----------+
                                                     |
                                            pool.sample(B_train)
                                                     v
                                              +-----------+
                                              | Train Loop|---> AdamW
                                              +-----------+
```

主循环：

```python
warmup K refills:
    arrs = solver.simulate_arrays(...)   # ~12 s
    pool.push(arrs, balls)               # filters by quality_clean

stats = compute_norm_stats_once(pool)    # 训练全程不再更新

for step in range(total_steps):
    if step > 0 and step % refill_every == 0:
        arrs = solver.simulate_arrays(...)
        pool.push(arrs, balls)
        if n_refills % val_every_refills == 0:
            val_metrics = evaluate_on_val_pool(...)
            best.pt update
    batch = pool.sample(train_batch)
    train_step(batch)
```

**关键设计决策**：

1. **ring-buffer 容量以 sim batch 为单位**（不是 sample）
   - K=4 槽位，每槽位最多 `B_max = sim_batch` clean sample
   - 新 batch 覆盖最旧槽位
   - 内存：`K * B * F * N * 3 * 4 ≈ 7.5 GB`（K=4, B=512, F=601, N=514）
   - 4090 主机 64 GB RAM 完全装得下

2. **clean 过滤在 push 时一次性完成**：dirty sample 直接丢，pool 里全是 clean。`pool.stats.total_dropped_dirty` 监控 sampler 健康度

3. **val pool 一次生成、永不刷新**：复现性原则。`val_seed=999_001` 与 train seed 0 起步隔离。`best.pt` 选择标准是 val total_norm 单调递降

4. **归一化 stats 一次性算完**：sampler 是固定分布，warmup 后 4 batch（~720k frame）已是 robust scale 收敛量级。再做 EMA 更新只会引入噪声且影响 ckpt 复现

5. **调度：refill 间隙 train 多步**
   - sim ~12 s, train ~0.05 s/step
   - `refill_every=50` ⇒ sim 摊到 0.24 s/step，与 train 比 3× 占比
   - 整体 wall-clock：50000 × (0.05 + 0.24) ≈ 4 小时

6. **device 共享**：sim 和 train 都在 cuda:0
   - Warp `simulate_arrays` 末尾 `wp.synchronize_device(d)` 串行；torch 后续从 host 重新搬上 GPU
   - 无锁需求；如要重叠需要 pinned buffer + cuda stream（复杂度高，暂不做）

**与离线 `train` 的兼容性**：

| 项 | 离线 `train` | 在线 `train-online` |
|---|---|---|
| `config.json` schema | `mode` 字段缺省 | `mode="online"` + sim/pool/sched 参数 |
| `metrics.jsonl` 单位 | per-epoch | per-validation（含 step / refill / lr / sim_time / train_time） |
| `best.pt / last.pt` 内 state_dict | 同 | 同 |
| `predict.py` 消费 | OK | **OK，未改一行**（state_dict + config 都在 ckpt 里） |

也就是说，在线训练产生的 `best.pt` 可以直接喂给 `cli.py predict --dataset dataset_v2/dataset.h5`，在 dataset_v2 的 test split 上做与离线 baseline **同口径** RMSE 评估。

### 9.4 已知限制

1. **sim 是 refill-period 串行瓶颈**：12 s sim 期间 GPU 上的 train 完全停。后续可考虑：
   - refill 启动时异步 preload 到 pinned memory，把拷贝时间隐藏掉
   - cuda stream 让 sim kernel 与 train kernel 真正并发（需 Warp + torch 共用 stream）
2. **clean ratio 抖动**：sim_batch=512 时 ~60% 稳定；极端 batch 可能拉低 pool 总量。建议 `pool_batches >= 4` 缓冲
3. **`per_sample_stats` 字段未入 pool**：当前 OnlineFramePool 只保留 ball/net/inputs，不保留 contact/max_pen 等 stats。若未来要用接触事件做监督需扩展 pool schema
4. **val pool 一次生成可能漏掉某些 panel**：`val_shots=1024` × 5 panels × 5 styles 分布 OK，但样本数小时尾分布敏感。如发现 val 不稳定，调高 `--val-shots`

---

## 10. Roadmap：通向移动端实时

终极目标：**移动端实时跑出精确且稳定的球 + 球网运动**。

约束（推断）：
- **延迟**：每帧 ≤ 16.7 ms（60 Hz）
- **参数量**：≤ 5M（推断的移动 GPU/NPU 上限；当前 29M baseline 是 6× 超出）
- **精度**：ball_pos 视觉无明显偏离（球门 7 m 宽，目标 RMSE ≤ 30 cm）
- **稳定**：无 NaN，无网粒子穿模/震荡

### 10.1 当前距离目标

| 指标 | 当前最佳（离线 mlp baseline，29M）| 目标 | 差距 |
|---|---|---|---|
| ball_pos RMSE | ~0.80 m | ≤ 0.30 m | 2.7× |
| ball_vel RMSE | ~0.51 m/s | — | — |
| net particle RMSE | ~0.60 cm | — | OK |
| 参数量 | 29M | ≤ 5M | 6× |
| 推理延迟（4090） | ~未测 | ≤ 16.7 ms（移动） | 待评估 |

**网 head 已达标**（cm 级），瓶颈全在 ball head。

### 10.2 路径

按优先级排序：

**P0：在线训练 baseline**（已实现，待跑）
- 跑 `train-online`，与离线 `mlp_v6` 在 dataset_v2 test split 上同口径对比
- 预期：消除 epoch 过拟合后 ball_pos 至少回到 mlp_v6 水平，理想下降到 0.5–0.7 m
- 验证 P1/P2 的设计前提（容量瓶颈到底是数据多样性还是表达能力）

**P1：delta-vs-baseline 目标重设计**
- 当前模型直接预测 `ball_pos`，要从无中生有学抛物线
- 改为预测**相对解析抛物线（重力 + 初速度）的偏差** `Δpos`，学习量从 ~5 m 量级降到 ~50 cm 量级
- 对训练几乎零代价：encoder 加抛物线分支，loss 在 Δ 空间算
- 直觉：相同模型容量下精度可能直接降到 < 30 cm（与目标对齐），但需实测

**P2：知识蒸馏到 ≤ 5M student**
- teacher = P0/P1 跑出的 29M 在线 baseline
- student 候选：8 层 × 768（≈ 5M）/ 6 层 × 1024（≈ 6.3M）
- 损失：student MSE 同时对齐 ground-truth + teacher hidden state
- 目标：student RMSE 不超过 teacher 的 1.2×，同时参数量满足部署约束

**P3：双模型分拆**
- 球动力学（ball_pos / ball_vel）和网形变（net_pos）解耦成两个独立小模型
- 球模型可能极简（4 层 × 256 ≈ 0.3M，因为球只受重力 + 几次离散反弹）
- 网模型仍 5M 但只学 cm 级形变，学习量小
- 推理时两模型并行，总参数 ≤ 5M、总延迟 ≈ max(两路)

**P4：架构层面突破**（兜底）
- 利用粒子拓扑结构（GNN / 卷积 over panel grid）
- 时序模型替代 MLP+t_norm（RNN / Transformer / Neural ODE）
- 物理 informed loss（约束残差作正则项）

### 10.3 不在路径上的工作（已废弃 / 推迟）

- **数据集扩充到 60k+**：在线训练已绕过这个问题，离线扩数据无意义
- **部分 raw_format 优化**：`json` 只是兼容旧版，`npz` 中等场景；正式数据集统一 h5
- **CPU↔Warp 字节级 parity**：本仓库没有 CPU reference solver，物理合理性由 smoke 测试 + 视觉检查保障

### 10.4 评估口径

任何模型的最终评估都跑：

```bash
python cli.py predict \
    --ckpt <best.pt> \
    --dataset /data3/netsim/dataset_v2/dataset.h5 \
    --output <run>/eval \
    --device cuda --batch 16 --worst-k 16
```

产出：`per_frame_rmse.json`（F 维曲线，看哪一帧崩）、`per_sample_summary.json`、`worst_k.json`。**不要**只看 mean RMSE，必须看：
- 时间分布：早期 vs 晚期帧 RMSE 形态
- worst-K 的 panel/style 分布：失败案例集中在哪类射门

---

## 11. 工作目录与基础设施（`/data3/netsim/`）

工作机：单卡 RTX 4090，Linux，无桌面。

```
/data3/netsim/
├── net-sim/                # git clone（即本仓库）
│   ├── .venv/              # python3.10 + warp-lang 1.13 + torch 2.5.1+cu121 + h5py 3.16
│   └── …
├── dataset_v2/             # 105 GB，30000 raw / 16198 clean
│   ├── dataset.h5
│   ├── topology.json
│   ├── metadata.json
│   └── batch_report.json
└── runs/                   # 训练产物
    └── <run_name>/
        ├── config.json
        ├── metrics.jsonl
        ├── best.pt  last.pt
        └── eval/           # predict.py 输出
```

激活环境：
```bash
cd /data3/netsim/net-sim && source .venv/bin/activate
```

### 11.1 完整复现命令

数据集生成（已完成；如要重做）：
```bash
python -u cli.py generate \
    --count 30000 --batch 1024 --device cuda \
    --raw --incremental --raw-format h5 \
    --seed 1 --output /data3/netsim/dataset_v2 \
    > /data3/netsim/dataset_v2.log 2>&1 &
```

离线训练 baseline：
```bash
python -u cli.py train \
    --dataset /data3/netsim/dataset_v2/dataset.h5 \
    --output /data3/netsim/runs/offline_baseline \
    --epochs 500 --batch 4096 --device cuda \
    --hidden 2048 2048 2048 2048 2048 2048 2048 2048 \
    --norm-scale-mode robust --drop-last-frames 5 \
    --lr-min-frac 0.01 \
    --w-ball-pos 50 --w-ball-vel 20 --w-net 1 \
    > /data3/netsim/runs/offline_baseline.log 2>&1 &
```

在线训练（约 4 小时）：
```bash
python -u cli.py train-online \
    --output /data3/netsim/runs/online_v1 \
    --hidden 2048 2048 2048 2048 2048 2048 2048 2048 \
    --total-steps 50000 --refill-every 50 \
    --sim-batch 512 --train-batch 4096 --pool-batches 4 \
    --val-shots 1024 --val-every-refills 10 \
    --norm-scale-mode robust --drop-last-frames 5 \
    --lr-min-frac 0.01 \
    --w-ball-pos 50 --w-ball-vel 20 --w-net 1 \
    --device cuda \
    > /data3/netsim/runs/online_v1.log 2>&1 &
```

评估（同口径 A/B 对比）：
```bash
python -u cli.py predict \
    --ckpt /data3/netsim/runs/<run>/best.pt \
    --dataset /data3/netsim/dataset_v2/dataset.h5 \
    --output /data3/netsim/runs/<run>/eval \
    --device cuda --batch 16 --worst-k 16
```
