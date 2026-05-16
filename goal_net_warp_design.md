# 球网 XPBD 数据集生成工具 — Warp GPU 重写设计文档

> **目标读者**：接手"将现有 CPU 标准库 XPBD solver 用 NVIDIA Warp 重写到 GPU"任务的下一位 AI / 开发者。
> **当前来源**：`/mnt/e/trunk_m1/tools/goal_net_xpbd_dataset/`（已闭环的 CPU Python 标准库实现）。
> **本文档作用**：把所有物理参数、数据结构、算法、踩坑、Warp 实现细节、验收标准一次性交付，新实现者无需回查任何运行时源码就能落地。
> **配套规范**：`openspec/changes/neural-goal-net-xpbd-dataset-tool/`（design.md / progress.md / specs / tasks）。

---

## 0. 总览（TL;DR）

- **场景**：足球射向球门，球网用 XPBD（Extended Position-Based Dynamics）粒子+绳段约束模拟，输出每条样本的球轨迹/网粒子状态/接触事件，作为离线 teacher 数据集，用于训练 neural goal-net response model。
- **现状性能瓶颈**：单样本 21 秒（CPU Python 标准库，纯 list[tuple] + Python 循环）。1000 样本 ≈ 5.8 小时，10000 样本 ≈ 58 小时。**不可接受**。
- **新方案**：用 **NVIDIA Warp**（`pip install warp-lang`）把 solver 完全 GPU 化，加 batch 维度并行多个射门样本。预期 4090 上单 batch=64~256 平摊每样本 0.2~1 s，**100x 量级加速**。
- **保留**：`solver.py` 作为 reference / 数值对齐基准，不删除。CLI 加 `--backend {python, warp}` 切换。
- **接口契约**：`SimulationResult` / `ContactEvent` / output schema **保持不变**，下游 `viewer_open3d.py` / `output.py` / 训练 pipeline 零修改。
- **关键不变量**：所有物理参数、所有判定阈值、所有反弹/摩擦公式必须 1:1 复现 CPU 版（小到 `-0.1 m/s` 残余速度修正、`crossbar_z_min_speed=0.1` 这种细节）。

---

## 1. 物理模型与世界坐标系

### 1.1 坐标系约定

- **右手系**，y 轴竖直向上。
- **球门门口**位于 z=0 平面，**网袋向 -z 方向延伸**（`-goal.depth ≤ z ≤ 0`）。
- 球门宽度沿 x 轴：`-goal.width/2 ≤ x ≤ +goal.width/2`。
- 球员从 +z 一侧射门（射门方向 z 分量为负）。
- 重力 `(0, -9.81, 0)`。
- 地面在 `y = ground.y`（默认 0），网粒子和球都不能穿过。

### 1.2 球门尺寸（默认值，单位 m）

| 参数 | 默认 | 含义 |
|---|---|---|
| `goal.width` | 7.32 | 球门左右立柱内侧距离（标准足球门） |
| `goal.height` | 2.44 | 球门高度 |
| `goal.depth` | 2.0 | 网袋深度（z=0 到后网） |

### 1.3 网格离散化

- 网格单元 `grid.cell_size_{x,y,z} = 0.305 m`（约 1 英尺，参考 ClothNetParam.py）。
- **后网（back panel）**：`(nx+1) × (ny+1)` 粒子，`nx = round(width/cell)`，`ny = round(height/cell)`。
- **左/右侧网（left/right panel）**：`(nz+1) × (ny+1)` 粒子，`nz = round(depth/cell)`。
- **顶网（top panel）**：`(nx+1) × (nz+1)` 粒子。
- 4 个 panel **当前实现是各自独立粒子**（角落坐标重合但 index 不同，是已知 bug B，下文 §10）。
- 总粒子数 ≈ `25 × 9 + 2 × 7 × 9 + 25 × 7 = 225+126+175 = 526` 量级。
- 距离约束 ≈ 同量级 × 2~4（每粒子的水平/垂直/二阶邻接边）。

---

## 2. 完整参数表（默认值 + 来源 + 物理含义）

> 所有参数在 `params.py` 的 dataclass 里，默认值见 `defaults/goal_net_params.json`。
> "来源"列指向 Engine 端运行时代码，确保 offline teacher 与 runtime 量级一致。

### 2.1 `GoalSizeParams`

| 字段 | 默认 | 来源 |
|---|---|---|
| `width` | 7.32 | GATE_SCALE in animation/consts |
| `height` | 2.44 | GATE_SCALE |
| `depth` | 2.0 | GATE_SCALE |

### 2.2 `GridParams`

| 字段 | 默认 |
|---|---|
| `cell_size_x` / `cell_size_y` / `cell_size_z` | 0.305 / 0.305 / 0.305 |

### 2.3 `RopeParams`（最重要的一组）

| 字段 | 默认 | 含义 / 来源 |
|---|---|---|
| `radius` | 0.018 | 渲染用绳半径 |
| `collision_radius` | **0.16** | 球-网碰撞专用膨胀半径，封住 0.305 m 网孔。**原值 0.39（沿用 ClothNetParam.particle_radius）会过度膨胀，球进不到网袋；0.16 是几何重新推算后定的（封孔不超调）。≤0 时回退到 `radius`** |
| `particle_mass` | 0.035 | 单粒子质量（kg） |
| `stretch_stiffness` | 0.92 | XPBD 拉伸约束 stiffness（归一化 [0,1]） |
| `bend_stiffness` | 0.18 | 二阶邻接（弯曲）约束 stiffness |
| `damping` | 0.035 | 速度衰减系数（每 substep 末乘 `1 - damping`） |
| `friction` | 0.32 | 默认切向摩擦（[0,1]） |
| `restitution` | 0.55 | 默认法向恢复 |
| `panel_restitution_back` | 0.10 | BACK 面法向系数（运行时 `handle_hit_net_move`） |
| `panel_restitution_top` | 0.30 | TOP 面 |
| `panel_restitution_side` | 0.20 | LEFT/RIGHT 面 |
| `panel_friction_back_tangent` | 0.80 | BACK 面切向摩擦（运行时 `parallel *= 0.2` 即 friction 0.8） |
| `impulse_clamp` | 6.0 | 单次 swept 命中时注入到绳粒子的最大速度增量（m/s），防止近锚段被打飞 |
| `impulse_speed_threshold` | 1.0 | **v2 新增**：球速 < 此值时 swept 不再注能给网，杜绝持续低速接触的能量泵入循环（球落地反弹顶网那种） |

### 2.4 `AnchorParams`

| 字段 | 默认 |
|---|---|
| `stiffness` | 1.0 |
| `soft_stiffness` | 0.65（保留字段，硬锚下不用） |
| `hard` | True（硬锚定，inverse_mass=0） |

### 2.5 `ShapeParams`（网形态微调）

| 字段 | 默认 | 含义 |
|---|---|---|
| `top_sag` | 0.16 | 顶网/后网下垂量（粒子初始 y 偏移） |
| `side_slope` | 0.15 | 侧网外扩斜度（远离 z=0 时 x 略外开） |
| `back_slope` | 0.05 | 后网底部 z 略后倾（沿 u 方向用 `4·u/nx·(1-u/nx)` 调制，边缘归零保持 dedup 焊接） |
| `back_pocket_depth` | 0.30 | **v2 新增**：后网中心向 -z 方向兜状鼓包，四边为零 |
| `stay_count` | 2 | **v2 新增**：物理支撑绳数量（0 / 2 / 4） |
| `stay_anchor_offset_x` | 0.3 | **v2 新增**：stay 远端锚点相对后上角外侧偏移 |
| `stay_anchor_offset_y` | 0.6 | **v2 新增**：stay 远端锚点相对后上角上方偏移 |
| `stay_anchor_offset_z` | 0.4 | **v2 新增**：stay 远端锚点相对后上角后方偏移 |

### 2.6 `SolverParams`

| 字段 | 默认 | 含义 |
|---|---|---|
| `frame_dt` | 1/60 = 0.01666... | 输出帧间隔 |
| `substeps` | **12** | 每帧 substep 数（原 8，因 collision_radius 缩小到 0.16 后必须涨到 12 防隧穿） |
| `iterations` | 8 | 每 substep 内 XPBD 约束 GS 迭代次数 |
| `duration` | **2.0** | 单样本总时长（原 1.25，扩到 2.0 才能捕到落地后滚动） |
| `gravity` | `[0, -9.81, 0]` | |
| `sample_every_frames` | 1 | 输出帧步进 |
| `enable_bend_constraints` | True | 二阶邻接约束开关 |
| `stuck_speed_threshold` | 0.5 | 球速 ≤ 此值视为低速 |
| `stuck_duration_seconds` | 0.3 | 低速持续 ≥ 此时长后冻结剩余 substep（早停） |

### 2.7 `CollisionParams`（质量检查阈值）

| 字段 | 默认 | 含义 |
|---|---|---|
| `severe_penetration_threshold` | **0.15** | 球 z < safety_back_z 的最大穿透量阈值 |
| `safety_back_z` | -2.25 | 球门后部安全边界（=`-depth - 0.25`） |
| `max_ball_speed` | 80.0 | 速度爆炸检测阈值 |
| `max_particle_speed` | 60.0 | 网粒子速度爆炸阈值 |
| `max_net_displacement` | 4.0 | 网粒子相对 rest 位移上限（约束发散检测） |
| `stuck_speed_threshold` | 0.05 | 接触后球速过低标记 stuck |
| `stuck_duration` | 0.25 | stuck 持续时长阈值 |

### 2.8 `GroundParams`（草地反弹，**完全对齐 net_move.cpp**）

| 字段 | 默认 | 含义 |
|---|---|---|
| `enabled` | True | |
| `y` | 0.0 | 地面高度 |
| `bounce_restitution` | 0.8 | 垂直反弹系数 |
| `bounce_speed_loss` | 12.0 | 弹跳模式整体速度衰减系数（× 9.8 × dt / |v|） |
| `bounce_to_roll_vertical_threshold` | 2.0 | 反弹后垂直速度低于此值切换到滚动 |
| `bounce_to_roll_total_threshold` | 3.0 | 反弹时总速度低于此值切换到滚动 |
| `roll_speed_loss` | 2.0 | 滚动模式水平衰减系数（× dt / |v|） |
| `bounce_floor_velocity_offset` | 0.1 | 反弹后 vy 减去 0.1 m/s（运行时 net_move.cpp:198 残余能量损失） |

### 2.9 `GoalpostParams`（球门柱）

| 字段 | 默认 | 含义 |
|---|---|---|
| `enabled` | True | |
| `radius` | 0.06 | =HALF_GOALPOST_WIDTH（consts.h） |
| `speed_change_factor` | 0.6 | =HIT_GOAL_POST_SPEED_CHANGE（ShootModule.json） |
| `crossbar_z_min_speed` | 0.1 | 横梁反弹后 |vz| 钳到 ≥ 此值（防球在横梁下原地冻结） |

### 2.10 `BallState`（射门输入）

| 字段 | 默认 | 含义 |
|---|---|---|
| `position` | — | 起始位置 |
| `velocity` | — | 初速度（m/s） |
| `angular_velocity` | (0,0,0) | 角速度（仅地面反弹时按比例衰减，目前不影响碰撞响应） |
| `radius` | 0.13 | 球半径（=ClothNetParam.radius） |
| `mass` | 1.0 | 球质量（=ClothNetParam.ball_mass） |

---

## 3. 拓扑生成（topology.py 现状 + Warp 需要的预计算）

### 3.1 数据结构

```python
@dataclass(frozen=True)
class Particle:
    index: int
    position: Vec3       # 初始/rest 位置
    panel: str           # "back" / "left" / "right" / "top"
    u: int; v: int       # 在 panel 网格内的 2D 坐标
    anchored: bool       # 边缘是否固定（连接到球门框/地面）

@dataclass(frozen=True)
class DistanceConstraint:
    index: int
    i0: int; i1: int     # 两端粒子索引
    rest_length: float   # 初始距离
    stiffness: float     # XPBD compliance ≈ 1/stiffness（详见 §4）
    kind: str            # "stretch" 或 "bend"
    panel: str           # 该绳所属面（决定撞击系数）

@dataclass(frozen=True)
class AnchorConstraint:
    index: int; particle: int
    target: Vec3
    stiffness: float
    hard: bool           # True 时直接强制位置 = target、速度 = 0

@dataclass(frozen=True)
class GoalpostSegment:
    index: int; name: str
    p0: Vec3; p1: Vec3   # 胶囊两端
    radius: float
    kind: str            # "post" 或 "crossbar"
```

### 3.2 锚定边缘规则（`generate_topology`）

**v2 现行实现**（与原 v1 spec 有差异，详见 §15 v2 changelog）：

| Panel | 锚定的粒子（`anchored=True`） |
|---|---|
| back | `iy == ny`（顶部，连横梁）或 `ix in {0, nx}`（连立柱）或 `iy == 0`（**v2**：底缘连地面） |
| left/right | `iy == ny`（顶）或 `iz == 0`（前缘连立柱）或 `iy == 0`（**v2**：底缘连地面） |
| top | `iz == 0`（前缘连横梁）或 `ix in {0, nx}`（连立柱） |

锚定 → 该粒子 `inverse_mass = 0`，所有约束求解中只读不写。

**v2 例外**：`shape.stay_count > 0` 时，后上 2 个角粒子（`PANEL_BACK` 的 `(0, ny)` 和 `(nx, ny)`，dedup 共享了 side/top 同位置粒子）的 `anchored` 标志在 `_add_support_stays` 里被显式改为 `False`——它们不再被钉在空中，由物理 stay 绳承重（见 §3.6）。

### 3.3 球门柱 3 段胶囊（`_make_goalpost_segments`）

设 `W=goal.width`, `H=goal.height`, `r=goalpost.radius=0.06`：

| name | p0 | p1 | kind |
|---|---|---|---|
| post_left | `(W/2 + r, 0, 0)` | `(W/2 + r, H+r, 0)` | post |
| post_right | `(-W/2 - r, 0, 0)` | `(-W/2 - r, H+r, 0)` | post |
| crossbar | `(-W/2, H+r, 0)` | `(W/2, H+r, 0)` | crossbar |

### 3.4 Warp 重写需要的拓扑预计算（在 Python 端一次性算好后传 GPU）

为了让 GPU kernel 是无分支的纯数值运算，拓扑要预转成扁平 ndarray 形式：

```python
# Per-particle (shape=(N,))
particle_pos_init: float32[N, 3]
particle_inv_mass: float32[N]               # anchored 粒子为 0
particle_panel_id: int32[N]                 # 0=back, 1=left, 2=right, 3=top

# Per-distance-constraint (stretch ∪ bend, shape=(M,))
constraint_i0: int32[M]
constraint_i1: int32[M]
constraint_rest: float32[M]
constraint_stiffness: float32[M]
constraint_panel_id: int32[M]               # 0..3 同上
constraint_kind: int32[M]                   # 0=stretch, 1=bend（用来分组求解）

# Per-anchor (shape=(A,))
anchor_particle: int32[A]
anchor_target: float32[A, 3]
anchor_stiffness: float32[A]
anchor_hard: int32[A]                       # 0/1

# Per-goalpost-segment (shape=(3,))
post_p0: float32[3, 3]
post_p1: float32[3, 3]
post_radius: float32[3]
post_kind: int32[3]                         # 0=post, 1=crossbar

# Per-panel collision coefficients (shape=(4,)) —— 预查表
panel_restitution: float32[4]   # 索引对应 0..3
panel_friction:    float32[4]
```

**Panel id 映射（必须固定）**：`back=0`, `left=1`, `right=2`, `top=3`。在 Warp kernel 里通过 `panel_restitution[panel_id]` 拿系数，避免字符串。

### 3.6 物理支撑绳（v2 新增，`_add_support_stays`）

真实球门网后侧由 2~4 根绳拉到斜上方锚点。v2 把这套机制建模成普通粒子 + 距离约束：

1. **角脱锚**：取 back 面的后上左角 `(0, ny)` 与后上右角 `(nx, ny)`（dedup 共享 side/top 顶后角同位置粒子），把这些粒子的 `anchored` 改为 `False`。
2. **加 stake ghost 粒子**：每根绳在 `(±(W/2 + offset_x), corner_y + offset_y, -depth - offset_z)` 处加一个 `Particle`，`anchored=True`、`inverse_mass=0`、`panel=PANEL_BACK`（panel id 仅供 collision restitution 查表使用；stake 几乎不会被球碰到）。stake 粒子索引记入 `topo.stake_particle_indices`。
3. **加 stretch 距离约束**：corner ↔ stake 之间加一条 `DistanceConstraint`，`kind=0`（stretch）、`stiffness = rope.stretch_stiffness`、`rest_length = 几何距离`。这样绳在 rest 时正好绷紧。
4. **viewer**：stake 不在 `panel_particle_indices` 里所以不会被画成网粒子；stay 约束索引集合 `{s.constraint}` 在 mesh rope 渲染里排除；每帧从当前 `particle_positions[corner]` 到 `particle_positions[stake]` 画一条独立的 stay 线（颜色 `(220,200,80)` 区别于 mesh）。

默认配置 (`stay_count=2`, `offset=(0.3, 0.6, 0.4)`)：每根绳长 ~0.78 m，角粒子被斜上拉向 `(-3.96, 3.04, -2.4)` / `(3.96, 3.04, -2.4)`。

**物理后果**：拿掉 stay 绳后整个后网会塌——后上角没有任何其他锚定路径。这是设计意图。

### 3.5 索引稳定性保证（`stable_signature` 测试）

`tests/test_topology.py::test_topology_is_stable` 要求：相同参数两次生成的 topology 字节级一致。Warp 不影响这点（拓扑生成在 Python 端，Warp 只消费）。

---

## 4. 求解算法（核心物理逻辑）

### 4.1 主循环骨架（`solver.py::simulate`，**顺序极其重要**）

```
for frame_index in 0..total_frames:
    if frame_index % sample_every_frames == 0:
        emit FrameSample(time, ball, particle_positions)
    if frame_index == total_frames: break

    for substep in 0..substeps:
        if frozen: continue                              # 早停后只填帧

        # ---- 网粒子 ----
        previous_positions = positions.copy()
        for i where inv_mass[i] > 0:
            velocities[i] += gravity * sub_dt
            positions[i] += velocities[i] * sub_dt
        resolve_particle_ground()                        # y < ground.y 拉回

        # ---- 球积分 + 离散外部碰撞 ----（必须在 swept 之前）
        ball.velocity += gravity * sub_dt
        ball_prev = ball.position
        ball.position += ball.velocity * sub_dt
        resolve_ball_vs_goalposts(ball, ball_prev)       # 必须先于 swept_net
        resolve_ball_swept_collisions(ball, ball_prev)   # CCD 球-绳
        resolve_ball_ground(ball)                        # 草地反弹/滚动

        # ---- XPBD 约束求解（GS 迭代）----
        for it in 0..iterations:
            solve_anchors()                              # 硬锚强制 / 软锚拉回
            solve_distance_constraints(stretch)
            if enable_bend: solve_distance_constraints(bend)
            solve_ball_particle_collisions(ball)         # 离散，残余穿透清理
            solve_ball_segment_collisions(ball)          # 离散，残余穿透清理

        # ---- 速度 update + 全局阻尼 ----
        for i: velocities[i] = (positions[i] - previous_positions[i]) / sub_dt
        for i: velocities[i] *= (1 - damping)

        # ---- stuck/早停统计 ----
        update_stuck_timers()
```

**为什么 goalposts 必须在 swept_net 前**：swept_net 用的 `ball_prev` 是子步开始那一帧位置；如果先撞柱，ball.position 已被改写到 TOI 处，传给 swept_net 就错了。

### 4.2 XPBD 距离约束求解（`_solve_distance_constraints`）

伪代码：

```
for each constraint c with (i0, i1, rest, stiffness):
    delta = pos[i1] - pos[i0]
    L = |delta|
    if L < 1e-8: continue
    error = L - rest
    w0 = inv_mass[i0]; w1 = inv_mass[i1]
    if w0 + w1 <= 0: continue
    direction = delta / L
    correction_mag = error * stiffness / (w0 + w1)
    pos[i0] += direction * correction_mag * w0
    pos[i1] -= direction * correction_mag * w1
```

注意这是**简化的 XPBD**（没真用 compliance/lambda），但是离线 teacher 可接受。Warp 重写**必须保持这个公式**，否则数值漂移。

**Gauss-Seidel 串行性**：每条约束读到的 `pos[i0]/pos[i1]` 是**已被前面约束修改过的**值。这是 GPU 并行化的难点（详见 §6 Warp 实现）。

### 4.3 锚点求解（`_solve_anchors`）

```
for each anchor a:
    if a.hard:
        pos[a.particle] = a.target
        vel[a.particle] = 0
    else:
        pos[a.particle] += (a.target - pos[a.particle]) * a.stiffness
```

默认所有 anchor 都是 hard，等价于 inv_mass=0 + 强制位置。

### 4.4 球-粒子离散碰撞（`_solve_ball_particle_collisions`）

```
collision_radius_total = ball.radius + rope.collision_radius
for each particle i:
    delta = pos[i] - ball.position
    d = |delta|
    if d <= 1e-8 or d >= collision_radius_total: continue
    n = delta / d
    pen = collision_radius_total - d
    inv_p = inv_mass[i]
    inv_b = 1 / ball.mass
    pos[i] += n * pen * inv_p / (inv_p + inv_b)
    ball.position -= n * pen * inv_b / (inv_p + inv_b)
    (e, f) = panel_coeffs(particle_panel[i])
    ball.velocity = reflect(ball.velocity, n, e, f)
    contacts.append(type="particle", ...)
```

`reflect(v, n, e, f)`：

```
vn = v · n
if vn >= 0: return v       # 已分离
v_normal = n * vn
v_tangent = v - v_normal
return v_normal * (-e) + v_tangent * (1 - f)
```

### 4.5 球-段离散碰撞（`_solve_ball_segment_collisions`）

```
for each distance_constraint c:
    p0, p1 = positions[c.i0], positions[c.i1]
    closest, t = closest_point_on_segment(ball.position, p0, p1)
    delta = closest - ball.position
    d = |delta|
    if d >= collision_radius_total: continue
    n = delta / d
    pen = collision_radius_total - d
    w0 = inv_mass[c.i0] * (1 - t)
    w1 = inv_mass[c.i1] * t
    wb = 1 / ball.mass
    total = w0 + w1 + wb
    pos[c.i0] += n * pen * w0 / total
    pos[c.i1] += n * pen * w1 / total
    ball.position -= n * pen * wb / total
    (e, f) = panel_coeffs(c.panel)
    ball.velocity = reflect(ball.velocity, n, e, f)
    contacts.append(type="segment", ...)
```

### 4.6 球-段 swept CCD（`_resolve_ball_swept_collisions`，**核心防穿网逻辑**）

输入：`ball_prev`（子步起点位置）、`ball.position`（已积分后位置）、子步时长 `sub_dt`。

```
delta = ball.position - ball_prev
if |delta| <= 1e-8: return        # 不动，跳过
collision_radius = ball.radius + rope.collision_radius

# 对每段绳（用子步起点的 previous_positions 作为静态端点）
best_toi = 2.0
best_constraint = None
for each constraint c:
    p0 = previous_positions[c.i0]
    p1 = previous_positions[c.i1]
    toi = swept_sphere_vs_segment_toi(ball_prev, ball.position, collision_radius, p0, p1)
    if toi is None: continue
    if toi < best_toi:
        best_toi = toi
        best_constraint = c

if no hit: return

toi = clamp(best_toi, 0, 1)
contact_pos = ball_prev + delta * toi
ball.position = contact_pos

closest, t = closest_point_on_segment(contact_pos, best_seg_p0, best_seg_p1)
offset = contact_pos - closest
if |offset| <= 1e-8:
    # 退化（球心在线段上）：用速度反向作为兜底
    n = -ball.velocity / |ball.velocity|
else:
    n = offset / |offset|

# 加 1mm skin 防止下一步又压在表面
ball.position = closest + n * (collision_radius + 0.001)

velocity_before = ball.velocity
(e, f) = panel_coeffs(best_constraint.panel)
ball.velocity = reflect(ball.velocity, n, e, f)

# 把球损失的法向动量注入两端粒子
apply_segment_impulse_to_endpoints(
    c.i0, c.i1, t, ball.mass, velocity_before, ball.velocity, n, sub_dt)

contacts.append(type="segment_swept", ...)
```

### 4.7 swept sphere-vs-segment TOI 数学（`_swept_sphere_vs_segment_toi`）

把球心运动 `C(t) = ball_a + t*(ball_b - ball_a)` 与线段 `L(s) = seg_p0 + s*(seg_p1 - seg_p0)` 之间最近距离 = `radius` 求最早 `t ∈ [0,1]`。等价于 ray-vs-capsule。

**Step 1：early-out**
```
closest_a, _ = closest_point_on_segment(ball_a, seg_p0, seg_p1)
if |ball_a - closest_a| <= radius:
    return 0.0     # 起点已穿透（极少见，但要兜底）
```
> ⚠️ **已知 bug A**：early-out 路径下 normal 方向可能搞反，详见 §10。

**Step 2：无限圆柱**
设 `d = ball_b - ball_a`, `seg = seg_p1 - seg_p0`, `m = ball_a - seg_p0`。
```
seg_len2 = seg · seg
if seg_len2 < 1e-12:
    return ray_sphere_first_hit(ball_a, d, seg_p0, radius)   # 退化为端点球

a_coeff = seg_len2 * (d·d) - (d·seg)^2
b_coeff = seg_len2 * (m·d) - (d·seg)*(m·seg)
c_coeff = seg_len2 * (m·m - radius^2) - (m·seg)^2
```
解 `a*t^2 + 2b*t + c = 0`：
```
disc = b^2 - a*c
if disc < 0: 圆柱无命中
sqrt_disc = sqrt(disc)
for t in {(-b - sqrt_disc)/a, (-b + sqrt_disc)/a}:
    if t in [0, 1]:
        s = ((m·seg) + t*(d·seg)) / seg_len2
        if s in [0, 1]:
            best_t = min(best_t, t)
            break
```

**Step 3：两端球帽**
```
cap0 = ray_sphere_first_hit(ball_a, d, seg_p0, radius)
cap1 = ray_sphere_first_hit(ball_a, d, seg_p1, radius)
best_t = min(best_t, cap0, cap1)
```

**`ray_sphere_first_hit`**：
```
m = origin - centre
b = m · direction
c = m·m - radius^2
if c > 0 and b > 0: return None    # 起点外且远离
a = direction · direction
disc = b^2 - a*c
if disc < 0: return None
t = (-b - sqrt(disc)) / a
if t < 0: t = (-b + sqrt(disc)) / a
return t if 0 <= t <= 1 else None
```

### 4.8 冲量注入两端粒子（`_apply_segment_impulse_to_endpoints`）

**为什么需要这个**：swept 命中时只反弹球，绳子纹丝不动是不真实的。把球损失的法向动量按 `(1-t):t` 权重分给两端粒子。

```
delta_v_ball = velocity_new - velocity_old
impulse_n = ball.mass * (delta_v_ball · normal)   # 必须 > 0（球被推开方向）
if impulse_n <= 0: return

clamp_speed = rope.impulse_clamp                  # 默认 6.0 m/s
weights = [(c.i0, 1 - t), (c.i1, t)]
for (idx, w) in weights:
    if inv_mass[idx] <= 0 or w <= 0: continue
    dv_mag = impulse_n * w * inv_mass[idx]
    dv = -normal * dv_mag                          # 注意负号：rope 受 -normal 方向
    if |dv| > clamp_speed: dv *= clamp_speed / |dv|

    velocities[idx] += dv
    shift = dv * sub_dt
    positions[idx] += shift                        # 同时移位
    previous_positions[idx] -= shift               # 反向移位
```

⚠️ **关键陷阱**：XPBD 末尾用 `velocity = (positions - previous_positions) / sub_dt` 反算速度，会把直接写入 `velocities` 的注入抹平。所以必须**同时**修改 `positions`（+shift）和 `previous_positions`（-shift），让反算结果自然得到 `v_old + dv`。`velocities[idx] += dv` 是为了下个 substep 的 `_integrate_particles` 立刻看到注入。

### 4.9 球-球门柱碰撞（`_resolve_ball_vs_goalposts`）

只处理 3 段胶囊，公式不同于网：

```
total_radius = ball.radius + post.radius
找到最早 toi（同 4.6 swept），命中段 best_segment

contact_pos = ball_prev + delta * toi
closest, _ = closest_point_on_segment(contact_pos, p0, p1)
offset = contact_pos - closest
n = offset / |offset|        # 退化时 fallback 速度反向
ball.position = closest + n * (total_radius + 0.001)

# 球门柱反弹公式（不同于网！）：(v - va) - va * 0.6
vn = ball.velocity · n
va = n * vn
new_v = (ball.velocity - va) - va * 0.6        # 切向不损失，法向反向 ×0.6

if best_segment.kind == "crossbar":
    # 横梁 z 速度保护：|new_v.z| 至少为 0.1 m/s
    if |new_v.z| < 0.1:
        new_v.z = sign(new_v.z) * 0.1   # vz=0 时取 +0.1

ball.velocity = new_v
contacts.append(type="goalpost" or "crossbar", ...)
```

### 4.10 草地反弹（`_resolve_ball_ground`，**5 段逻辑严格对齐 net_move.cpp**）

```
触发条件：ball.position.y - ball.radius < ground.y
之前：保存 old_velocity, old_speed = |old_velocity|

# 1) 把球抬到 floor + radius
ball.position.y = ground.y + ball.radius

# 2) 垂直反弹（含 -0.1 m/s 残余损失）
new_vy = max(0, -old_velocity.y * bounce_restitution - bounce_floor_velocity_offset)
                                                       # = max(0, -vy*0.8 - 0.1)

# 3) 弹跳 vs 滚动判定
if new_vy > 2.0 AND old_speed >= 3.0:
    # 弹跳态：保留水平、整体衰减
    v = (old_velocity.x, new_vy, old_velocity.z)
    if |v| > 1e-8:
        scale = max(0, 1 - bounce_speed_loss * 9.8 * sub_dt / |v|)
        v *= scale
    ball.velocity = v
    mode = "bounce"
else:
    # 滚动态：垂直归零、水平衰减
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

# 5) 写 contact event（type = "ground_bounce" 或 "ground_roll"）
```

### 4.11 网粒子地面约束（`_resolve_particle_ground`）

```
for each particle i where inv_mass[i] > 0:
    if positions[i].y < ground.y:
        positions[i].y = ground.y
        if previous_positions[i].y < ground.y:
            previous_positions[i].y = ground.y
        if velocities[i].y < 0:
            velocities[i].y = 0
```

### 4.12 帧末速度更新与阻尼

```
velocities[i] = (positions[i] - previous_positions[i]) / sub_dt    # 仅 inv_mass > 0
velocities[i] *= max(0, 1 - rope.damping)                          # 全局阻尼
```

---

## 5. 质量检查与异常分类（`_quality_summary`）

每个样本结束时根据下列规则给 `quality.issues` 打标签，**任一非空都标记 `clean=False`**：

| issue | 触发条件 |
|---|---|
| `severe_penetration` | `max_penetration > collision.severe_penetration_threshold` (0.15) |
| `nan_or_inf` | `ball.position` 或 `ball.velocity` 任意分量非有限 |
| `velocity_explosion` | `\|ball.velocity\| > collision.max_ball_speed` (80) |
| `particle_velocity_explosion` | `max(\|particle.velocity\|) > collision.max_particle_speed` (60) |
| `constraint_divergence` | `max(\|positions[i] - rest_positions[i]\|) > collision.max_net_displacement` (4.0) |
| `stuck` | 接触发生后球速持续 < `stuck_speed_threshold`(0.05) 累计 > `stuck_duration`(0.25 s) |
| `target_panel_missed` | 整个 simulation 期间没有任何 contact |

`max_penetration` 定义：`max(0, safety_back_z - ball.position.z)`，即球穿过 `safety_back_z = -2.25` 平面的深度。

`stats` 还输出：`frame_dt / substeps / iterations / duration / frame_count / contact_count / max_constraint_error / max_net_displacement / ball_came_to_rest`。

---

## 6. 采样器（`sampler.py`）

### 6.1 `SamplerConfig` 默认值

| 字段 | 默认 |
|---|---|
| `count` | 5 |
| `seed` | 1 |
| `panels` | `["back", "left", "right", "top", "corner"]` |
| `styles` | `["ground", "low", "mid", "high", "lob"]` |
| `speed_range` | (22, 32) m/s |
| `spin_range` | (-18, 18) rad/s 每轴 |
| `azimuth_range_deg` | (-60, 60) |
| `elevation_range_deg` | (-2, 25) |
| `distance_range` | (6, 25) m |
| `target_jitter` | 0.25 m |
| `goal_width / height / depth` | 7.32 / 2.44 / 2.0（与 topology 同步） |

### 6.2 风格 profile（`STYLE_PROFILES`）

| style | elevation_bias (deg) | speed_range (m/s) |
|---|---|---|
| ground | -2 | (22, 28) |
| low | +2 | (24, 30) |
| mid | +6 | (26, 32) |
| high | +12 | (24, 30) |
| lob | +18 | (18, 24) |

### 6.3 采样流程（`sample_shots`）

对每个样本 `i`：
1. `panel = panels[i % len(panels)]`，`style = styles[i % len(styles)]`。
2. `target = _sample_panel_target(rng, config, panel)` —— 在该面/角落内随机 + jitter。
3. 抽 `azimuth ∈ azimuth_range_deg`、`elevation ∈ elevation_range_deg + style_bias`（钳到 range 上下界）。
4. 抽 `distance ∈ distance_range`、`speed ∈ style.speed_range`。
5. `direction = _direction_from_angles(azimuth, elevation)`，约定 `azimuth=0, elevation=0` → `(0, 0, +1)`。
6. `source = target - direction * distance`；强制 `source.y >= 0.05`、`source.z >= 0.5`。
7. `velocity = _solve_initial_velocity(source, target, speed)` —— 一次牛顿迭代用 `speed^2 - vy^2 = horiz^2` 算抛物线初速度。
8. `spin = (Uniform(spin_range), …, …)`。
9. 输出 `ShotInput(sample_id="sample_%05d" % i, target_panel=panel, seed=config.seed+i, template="%s/%s" % (panel, style), ball=BallState(source, velocity, spin))`。

### 6.4 `_sample_panel_target`

| panel | x | y | z |
|---|---|---|---|
| back | `Uniform(-W/2+m, W/2-m)` | `Uniform(0.2, H-m)` | `-depth + 0.05` |
| left | `-W/2 + 0.05` | `Uniform(0.3, H-m)` | `Uniform(-depth+m, -0.2)` |
| right | `+W/2 - 0.05` | `Uniform(0.3, H-m)` | `Uniform(-depth+m, -0.2)` |
| top | `Uniform(-W/2+m, W/2-m)` | `H - 0.05` | `Uniform(-depth+m, -0.1)` |
| corner | `±(W/2-0.05)`（rng 决定符号） | `H - Uniform(0.05, 0.45)` | `Uniform(-depth+m, -0.2)` |

最后整体加 `Uniform(-jitter, jitter)`（z 轴 jitter 缩 0.4×）；`m = max(0.1, jitter)`。

---

## 7. 输出 schema（**Warp 重写必须保持兼容**）

### 7.1 目录结构

```
<output_dir>/
├── summary.jsonl          # 每行一条样本摘要
├── batch_report.json      # 整批统计
├── features/              # 训练特征（默认产出）
│   ├── sample_00000.json
│   └── ...
└── raw/                   # 可回放原始数据（仅 --raw 时产出）
    ├── sample_00000.json
    └── ...
```

默认 `output_dir = <project_root>/Agent/Temp/goal_net_xpbd_dataset`。

### 7.2 `features/sample_*.json`

```json
{
  "sample_id": "sample_00000",
  "schema_version": "goal_net_params.v1",
  "input_features": {
    "sample_id", "target_panel", "template", "seed",
    "position", "velocity", "angular_velocity", "radius", "mass"
  },
  "ball_trajectory": [
    {"time": 0.0, "position": [...], "velocity": [...]}
  ],
  "net_control_points": [
    {"time": 0.0, "positions": [16 个抽样粒子位置]}
  ],
  "quality": {"clean": bool, "issues": [...], "target_hit": bool,
              "max_penetration_depth": float, "max_penetration_time": float}
}
```

`net_control_points` 抽样规则：每帧从 `particle_positions` 取 `stride = max(1, N // 16)`，最多 16 个点。

### 7.3 `raw/sample_*.json`

```json
{
  "metadata": {"schema_version", "params_source_path", "params_snapshot", "seed"},
  "shot": "<input_features 同上>",
  "topology_summary": "<topology.summary(params)>",
  "topology": {
    "particles", "distance_constraints", "anchor_constraints",
    "bend_constraints", "panel_particle_indices", "goalpost_segments"
  },
  "frames": [
    {"time": 0.0, "ball_position": [0,0,0], "ball_velocity": [0,0,0],
     "particle_positions": "[N×3]"}
  ],
  "contacts": [
    {"time", "object_type", "object_index", "position", "normal", "strength"}
  ],
  "quality": {},
  "stats": {}
}
```

`contacts.object_type` 取值：`"particle"` / `"segment"` / `"segment_swept"` / `"goalpost"` / `"crossbar"` / `"ground_bounce"` / `"ground_roll"`。

### 7.4 `summary.jsonl` 每行

```json
{
  "sample_id", "paths": {"feature": "...", "raw": "... 或 null"},
  "metadata", "input_features", "quality", "stats",
  "contact_count", "topology_summary"
}
```

### 7.5 `batch_report.json`

```json
{
  "sample_count": 0,
  "clean_count": 0,
  "abnormal_count": 0,
  "abnormal_types": {"severe_penetration": 0},
  "panel_stats": {"back": {"samples": 0, "contacts": 0, "abnormal": 0}}
}
```

---

## 8. NVIDIA Warp 实现指南（核心章节）

### 8.1 为什么选 Warp（候选方案对比结论）

在 5 个候选方案（PyTorch / JAX / CuPy / Warp / Taichi）对比后，Warp 是当前场景最优。原因：

1. **GS 串行循环可保留**：用 `wp.atomic_add` 处理同一帧多个约束修改同一粒子的冲突，**算法层面与 CPU 版完全一致**，数值漂移最小。PyTorch 必须改 graph coloring 或 Jacobi，存在算法漂移风险。
2. **加速天花板最高**：原生编译到 CUDA，无 op-launch overhead。预期 4090 + batch 256 → 100~300x。
3. **代码可读性**：kernel 几乎是 CPU Python 的直译，不像 PyTorch 全是 `gather/scatter_add_`。
4. **依赖小**：`pip install warp-lang` ≈ 200MB；不需要装 CUDA toolkit，只要 NVIDIA 驱动 ≥525。
5. **下游对接**：当 neural model 训练（PyTorch）时，用 `wp.to_torch()` 通过 dlpack 零拷贝共享 tensor。
6. **CPU fallback**：开发期没 GPU 时用 `device='cpu'`，kernel 经 LLVM 编译，仍比纯 Python 快 50~100x。

**唯一短板**：跟下游 PyTorch neural pipeline 不如 PyTorch 无缝。但本场景是**离线 imitation learning**（teacher 数据生成与 student 训练完全解耦），不需要可微物理，这个短板不影响。

### 8.2 Warp 速成

```python
import warp as wp
wp.init()                                    # 进程启动一次

# 数组分配
positions = wp.zeros(shape=(B, N), dtype=wp.vec3, device='cuda')
positions = wp.from_numpy(positions_np, dtype=wp.vec3, device='cuda')

# Kernel 定义
@wp.kernel
def integrate_kernel(
    pos: wp.array2d(dtype=wp.vec3),          # shape (B, N)
    vel: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),         # shape (N,)，所有 batch 共享
    gravity: wp.vec3,
    dt: float,
):
    b, i = wp.tid()                          # 二维线程 id
    if inv_mass[i] <= 0.0:
        return
    vel[b, i] = vel[b, i] + gravity * dt
    pos[b, i] = pos[b, i] + vel[b, i] * dt

# 启动
wp.launch(integrate_kernel, dim=(B, N),
          inputs=[positions, velocities, inv_mass, gravity_vec, dt],
          device='cuda')

wp.synchronize_device('cuda')                # 性能 bench 时才需要

result = positions.numpy()                   # shape (B, N, 3)
```

**关键限制**：
- Kernel 内不能用 Python list / dict / 字符串。
- Kernel 内只能用 `wp.range(n)` 风格 for。
- Kernel 内可用：`wp.vec3`, `wp.mat33`, `wp.quat`, `float`, `int`, 算术、`wp.length/dot/cross/normalize`, `wp.atomic_add/min/max`, `if/else/while`, `@wp.func` 辅助函数。
- Kernel 内**不能**调用 numpy / Python 库。
- Kernel 内**不能**抛异常；除以零 → NaN，要自己防御。

### 8.3 数据布局

```python
# B = batch_size, N = particles, M = constraints, A = anchors

# 每 batch 独立的网粒子状态
positions:          wp.array2d(dtype=wp.vec3, shape=(B, N))
previous_positions: wp.array2d(dtype=wp.vec3, shape=(B, N))
velocities:         wp.array2d(dtype=wp.vec3, shape=(B, N))

# 所有 batch 共享拓扑常量
inv_mass:           wp.array(dtype=float,    shape=(N,))
particle_panel:     wp.array(dtype=int,      shape=(N,))
rest_positions:     wp.array(dtype=wp.vec3,  shape=(N,))

constraint_i0:        wp.array(dtype=int,    shape=(M,))
constraint_i1:        wp.array(dtype=int,    shape=(M,))
constraint_rest:      wp.array(dtype=float,  shape=(M,))
constraint_stiffness: wp.array(dtype=float,  shape=(M,))
constraint_panel:     wp.array(dtype=int,    shape=(M,))

anchor_idx:        wp.array(dtype=int,     shape=(A,))
anchor_target:     wp.array(dtype=wp.vec3, shape=(A,))

post_p0:           wp.array(dtype=wp.vec3, shape=(3,))
post_p1:           wp.array(dtype=wp.vec3, shape=(3,))
post_kind:         wp.array(dtype=int,     shape=(3,))

# 每 batch 球状态
ball_pos:          wp.array(dtype=wp.vec3, shape=(B,))
ball_vel:          wp.array(dtype=wp.vec3, shape=(B,))
ball_ang_vel:      wp.array(dtype=wp.vec3, shape=(B,))
ball_radius:       wp.array(dtype=float,   shape=(B,))
ball_mass:         wp.array(dtype=float,   shape=(B,))

# 每 batch 标量状态（GPU 上累计，避免回拷）
frozen_mask:       wp.array(dtype=int,     shape=(B,))
slow_time:         wp.array(dtype=float,   shape=(B,))
stuck_time:        wp.array(dtype=float,   shape=(B,))
target_hit:        wp.array(dtype=int,     shape=(B,))
max_penetration:   wp.array(dtype=float,   shape=(B,))
max_penetration_t: wp.array(dtype=float,   shape=(B,))
contact_started:   wp.array(dtype=int,     shape=(B,))
no_contact_after:  wp.array(dtype=float,   shape=(B,))

# Per-substep swept 命中结果（atomic_min argmin trick，详见 8.6）
swept_hit_packed:  wp.array(dtype=wp.uint64, shape=(B,))

# 接触事件预分配
contacts_buf:      wp.array(dtype=ContactStruct, shape=(B, MAX_CONTACTS))
contact_count:     wp.array(dtype=int,           shape=(B,))
```

### 8.4 Kernel 列表

按主循环顺序：

| Kernel | 维度 | 作用 |
|---|---|---|
| `k_save_previous_positions` | (B, N) | `previous_positions[b,i] = positions[b,i]` |
| `k_integrate_particles` | (B, N) | 重力 + 半隐式 Euler，受 frozen_mask 控制 |
| `k_resolve_particle_ground` | (B, N) | y < ground.y 拉回 |
| `k_integrate_ball` | (B,) | ball_vel += g*dt; ball_pos_prev = ball_pos; ball_pos += vel*dt |
| `k_swept_ball_vs_posts` | (B, 3) | atomic_min(toi_packed) |
| `k_apply_post_response` | (B,) | 解码 swept_hit、回退球、反弹、写 contact |
| `k_swept_ball_vs_segments` | (B, M) | atomic_min |
| `k_apply_segment_response` | (B,) | 回退球、反弹、注入冲量到两端粒子 |
| `k_resolve_ball_ground` | (B,) | 草地 5 段反弹 |
| `k_solve_anchors` | (B, A) | 硬锚强制 |
| `k_solve_distance_constraints` | (B, M_stretch) | XPBD 距离约束（atomic_add） |
| `k_solve_distance_constraints_bend` | (B, M_bend) | bend 约束 |
| `k_solve_ball_particle_collisions` | (B, N) | 离散球-粒子 |
| `k_solve_ball_segment_collisions` | (B, M) | 离散球-段 |
| `k_update_velocities_and_damp` | (B, N) | velocities = (pos-prev)/dt; *= (1-damping) |
| `k_update_stuck_timers` | (B,) | 早停统计、frozen_mask 设置 |
| `k_record_frame` | (B, N) | 写帧缓冲 |
| `k_check_quality` | (B,) | 累计 max_penetration、target_hit 等 |

### 8.5 GS 距离约束的 GPU 化（**最关键设计**）

**做法 1（推荐）：单 kernel + atomic_add，全约束并行**

```python
@wp.kernel
def solve_distance_constraints_kernel(
    positions: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    c_i0: wp.array(dtype=int),
    c_i1: wp.array(dtype=int),
    c_rest: wp.array(dtype=float),
    c_stiff: wp.array(dtype=float),
):
    b, c = wp.tid()
    i0 = c_i0[c]; i1 = c_i1[c]
    p0 = positions[b, i0]; p1 = positions[b, i1]
    delta = p1 - p0
    L = wp.length(delta)
    if L <= 1e-8:
        return
    error = L - c_rest[c]
    w0 = inv_mass[i0]; w1 = inv_mass[i1]
    ws = w0 + w1
    if ws <= 0.0:
        return
    direction = delta / L
    correction = error * c_stiff[c] / ws
    if w0 > 0.0:
        wp.atomic_add(positions[b], i0, direction * (correction * w0))
    if w1 > 0.0:
        wp.atomic_add(positions[b], i1, direction * (-correction * w1))
```

> **数值差异**：这是 **Jacobi 风格的并行 GS**——一次 kernel launch 内所有约束读到的 `positions` 是 launch 入口时的快照，写入用 atomic_add 累加到下次 launch 才可见。
> **后果**：比 CPU GS 收敛慢约 1.3~1.5 倍。
> **建议补偿**：CPU 默认 `iterations=8` → Warp 版调到 `iterations=12`，或保持 8 + 接受网形变略小（teacher 数据可接受）。
> Atomic 浮点求和顺序不确定 → bit 级有 ulp 差异，但物理等价。

**做法 2（次选）：Graph coloring + 多次 launch**

预先按 graph coloring 把约束分成 K 组（每组内无共享端点），每组一次 kernel launch，组内无原子冲突，组间串行。这是 Jacobi 与 GS 之间的折中。但增加预处理 + launch 次数（K 次），收益不一定比做法 1 好。

**默认选做法 1**。

### 8.6 球-段 swept argmin（packed atomic_min）

每子步对每 batch 找最早 TOI 的段。用 64-bit packed atomic_min：

```python
# 主机端初始化
swept_hit_packed.fill_(wp.uint64(0xFFFFFFFFFFFFFFFF))

@wp.func
def encode_toi_idx(toi: float, idx: int) -> wp.uint64:
    # IEEE 754 trick：正 float 的 bit 位单调
    bits = wp.bitcast_uint32(toi)            # 假设 toi >= 0
    return (wp.uint64(bits) << 32) | wp.uint64(idx)

@wp.kernel
def swept_ball_vs_segments_kernel(
    ball_prev:     wp.array(dtype=wp.vec3),
    ball_pos:      wp.array(dtype=wp.vec3),
    ball_radius:   wp.array(dtype=float),
    rope_collision_radius: float,
    prev_positions: wp.array2d(dtype=wp.vec3),
    c_i0: wp.array(dtype=int),
    c_i1: wp.array(dtype=int),
    swept_hit_packed: wp.array(dtype=wp.uint64),
):
    b, c = wp.tid()
    a = ball_prev[b]; bp = ball_pos[b]
    radius = ball_radius[b] + rope_collision_radius
    p0 = prev_positions[b, c_i0[c]]
    p1 = prev_positions[b, c_i1[c]]
    toi = swept_sphere_vs_segment_toi(a, bp, radius, p0, p1)
    if toi >= 0.0 and toi <= 1.0:
        wp.atomic_min(swept_hit_packed, b, encode_toi_idx(toi, c))
```

之后 `(B,)` kernel 解码 `swept_hit_packed[b]`，做反弹+冲量注入。

### 8.7 接触事件收集

`MAX_CONTACTS = total_substeps * (iterations + 3)` ≈ 16000（默认参数）。
VRAM：`16000 × 32 byte × 256 batch ≈ 130 MB`，可接受。

```python
# 添加 contact：
idx = wp.atomic_add(contact_count, b, 1)
if idx < MAX_CONTACTS:
    contacts_buf[b, idx] = ContactStruct(time, type_id, ...)
```

模拟结束后整体 `.numpy()` 拷回 host，每 batch 切片 `[:contact_count[b]]` 转 `ContactEvent` list。

### 8.8 Frozen mask（早停）

```python
@wp.kernel
def integrate_particles_kernel(..., frozen: wp.array(dtype=int)):
    b, i = wp.tid()
    if frozen[b] != 0:
        return
    ...
```

对 frozen batch 仍产生帧（值与上一帧相同），保证 `len(frames)` 与 `duration` 一致。

### 8.9 帧缓冲

```python
# 全粒子帧缓冲（仅 --raw 时分配）
frames_ball_pos:   wp.array2d(dtype=wp.vec3, shape=(B, F))
frames_ball_vel:   wp.array2d(dtype=wp.vec3, shape=(B, F))
frames_particles:  wp.array3d(dtype=wp.vec3, shape=(B, F, N))

# 控制点帧缓冲（默认）—— 对 features 输出足够
frames_control_pts: wp.array3d(dtype=wp.vec3, shape=(B, F, 16))
```

VRAM 估算（B=256, F=121, N=600）：
- 全粒子：`256 * 121 * 600 * 12 byte = 222 MB` → 4090 24GB 完全够
- 仅控制点：6 MB

### 8.10 模块组织

```
tools/goal_net_xpbd_dataset/
├── solver.py              # CPU 标准库版（保留作 reference，不删）
├── solver_warp.py         # 新增：Warp 实现
│   ├── XpbdWarpSolver(params, topology, batch_size, device)
│   ├── simulate_batch(shots: List[ShotInput]) -> List[SimulationResult]
│   └── kernels: 见 §8.4
├── topology.py            # 新增 to_warp_arrays(topology) -> dict
├── cli.py                 # 加 --backend {python, warp} --device {cpu, cuda} --batch N
├── viewer_open3d.py       # 不变（本地有 GUI 时用）
├── viewer_rerun.py        # 新增：Web/远程可视化（详见 §9）
└── tests/
    ├── test_topology.py            # 不变
    └── test_solver_parity.py       # 新增：Warp vs CPU 数值对齐
```

### 8.11 实现顺序（推荐）

**Step A（4~6h）：单 batch + `device='cpu'` Warp 跑通，对齐 reference**
1. 拓扑预转 ndarray（`to_warp_arrays`）
2. 写 integrate / damping / ground / update_velocities kernel（最简单）
3. 写距离约束 + anchor + bend kernel（atomic_add）
4. 写球-粒子离散、球-段离散、swept CCD kernel
5. 写球门柱 swept、草地反弹 kernel
6. 写冲量注入、stuck 早停、帧记录、quality 检查
7. 跑 5 样本对比 reference，contacts 数 / clean / 关键帧球位置容差 1e-3 m

**Step B（1~2h）：加 batch 维 + `device='cuda'`**
1. 所有数组加 `(B, ...)` 维
2. `wp.tid()` → 二维
3. CLI 加 `--batch`，外层 sampler 攒一批 shot 一次性丢给 `simulate_batch`
4. 4090 上压测：`generate --count 1024 --batch 128 --backend warp --device cuda`

**Step C（1~2h）：性能调优**
1. AABB 早 reject（球扫掠 AABB vs 段 AABB）做球-段 swept 的预过滤
2. 检查 atomic 冲突热点（用 `wp.profile`）
3. 必要时切换到 graph coloring（做法 2）

### 8.12 Warp 常见坑

- **kernel 编译慢**：第一次 launch ≈ 1~2s（cache 后秒起）。批跑 1k 样本无感，但单测会觉得慢。
- **atomic_add 浮点不确定性**：同样输入两次跑结果末位 ulp 差 1e-7。`stable_signature` 字节精确测试要绕开（topology 不受影响，因为 topology 是 Python 端生成）。
- **NaN 调试**：kernel 内除零 → NaN 静默扩散。每个 `1.0/x` 之前必须 `if x <= 1e-8: return`。
- **`wp.array2d` vs `wp.array`**：维度不能混。
- **bool 类型**：Warp 内 bool 用 `int` 0/1 替代更稳妥。
- **`wp.tid()` 维度**：与 launch dim 数一致。
- **不要在 kernel 里调用 Python 函数**，包括 `print`。要 debug 就拷回 numpy 再 print。
- **CPU device 调试**：`device='cpu'` 时 kernel 内可以 print 到 stdout，便于定位逻辑 bug。
- **`@wp.func`** vs `@wp.kernel`：辅助函数用 `@wp.func`（在 kernel 内调用）；`@wp.kernel` 是 launch 入口。

---

## 9. Web 可视化方案（适配 Linux 远程 GPU 机）

### 9.1 痛点

- 4090 机器是 Linux，无桌面 GUI，现有 `viewer_open3d.py` 依赖 X server 跑不起来。
- 用户希望浏览器访问 GPU 机器查看效果。
- 数据集生成是离线批处理，但需要随时挑几条样本看物理是否合理。

### 9.2 方案选型

经过对比 Three.js+FastAPI / Plotly Dash / meshcat / **rerun.io** / Open3D WebRTC，**首选 rerun.io**：

- `pip install rerun-sdk`
- 自带 web viewer（基于 wgpu，浏览器原生 3D，性能极好）
- 原生支持时间轴 scrub、batch 多样本切换、record 到 .rrd 文件可离线回放
- ML/机器人圈过去两年崛起的标准答案，NVIDIA Isaac Lab、Hugging Face LeRobot 都在用
- Python API 极简：`rr.log("path/to/thing", rr.Points3D(...))`

**备选 meshcat**：更老牌、更轻，但要自己写动画时间轴（rerun 自带）。

**保留 `viewer_open3d.py`**（本地有 GUI 时仍可用），新增 `viewer_rerun.py` 用于远程 web 访问。两者从同一份 `raw/*.json` 读数据，互不影响。

### 9.3 rerun 工作模式

| 模式 | 用法 | 适合 |
|---|---|---|
| A. Spawn | `rr.spawn()` 起本地 GUI | 本地 Mac/Windows |
| B. Connect | `rr.connect()` 连到运行中的 viewer | 同机协作 |
| **C. Serve** | `rr.serve()` 起 web server | **本场景：Linux GPU 机** |
| **D. Save** | `rr.save("file.rrd")` 落盘 | 离线分发 / CI 录制 |

我们用 **C + D 双轨**：
- 实时查看：`view-rerun --serve` 起 web，本地浏览器访问 `http://gpu-host:9090`
- 离线回放：每个 batch 同步落 `.rrd` 文件，任何机器 `rerun preview.rrd` 即开浏览器回放

### 9.4 `viewer_rerun.py` 设计

CLI 子命令：

```bash
# 单样本 web viewer
python tools/goal_net_xpbd_dataset/cli.py view-rerun \
    Agent/Temp/goal_net_xpbd_vis/raw/sample_00002.json \
    --serve --bind 0.0.0.0:9090

# 整个 batch 多样本切换
python tools/goal_net_xpbd_dataset/cli.py view-rerun \
    Agent/Temp/goal_net_xpbd_vis/raw \
    --serve --bind 0.0.0.0:9090

# 录到 .rrd 离线分发
python tools/goal_net_xpbd_dataset/cli.py view-rerun \
    Agent/Temp/goal_net_xpbd_vis/raw \
    --save Agent/Temp/dataset_preview.rrd
```

实现骨架：

```python
import rerun as rr
import json, signal

def view_rerun(raw_paths, serve=False, bind="0.0.0.0:9090", save=None):
    rr.init("goal_net_xpbd_dataset", spawn=False)
    if serve:
        host, port = bind.split(":")
        rr.serve(open_browser=False, web_port=int(port), ws_port=int(port)+1)
    if save:
        rr.save(save)

    for path in raw_paths:
        sample = json.load(open(path))
        sample_id = sample["shot"]["sample_id"]

        with rr.new_recording(sample_id):
            _log_static(sample)              # 球门柱、地面、锚点
            for frame in sample["frames"]:
                rr.set_time_seconds("sim_time", frame["time"])
                rr.log("ball", rr.Points3D(
                    [frame["ball_position"]],
                    colors=[(255, 100, 0)],
                    radii=[sample["shot"]["radius"]]))
                rr.log("net/particles", rr.Points3D(
                    frame["particle_positions"],
                    colors=[(80, 120, 255)] * len(frame["particle_positions"])))
                rr.log("net/ropes", rr.LineStrips3D(
                    _build_segments(frame, sample["topology"])))
            for c in sample["contacts"]:
                rr.set_time_seconds("sim_time", c["time"])
                rr.log("contacts/" + c["object_type"], rr.Points3D(
                    [c["position"]], colors=[_contact_color(c)]))

    if serve:
        print(f"web viewer: http://<host>:{port}")
        signal.pause()                       # 阻塞直到 Ctrl+C
```

`_build_segments(frame, topology)`：从 `topology.distance_constraints` 取 i0/i1，到 `frame.particle_positions` 取两端坐标，组成 LineStrips3D 输入。

### 9.5 推荐 entity 树结构

```
goal/                          # 静态
├── posts/left
├── posts/right
└── posts/crossbar

ground/                        # 静态 plane（rr.Boxes3D 一个扁盒子）

net/
├── particles                  # 时变 Points3D
├── ropes                      # 时变 LineStrips3D
└── anchors                    # 静态 Points3D，红色

ball/                          # 时变
├── trajectory                 # 累积 LineStrips3D
└── velocity                   # 时变 Arrows3D（可选）

contacts/
├── particle / segment / segment_swept
├── goalpost / crossbar
└── ground_bounce / ground_roll
```

### 9.6 部署细节（Linux GPU 机）

```bash
# 1. 装 rerun
pip install rerun-sdk

# 2. 防火墙开端口
sudo ufw allow 9090
sudo ufw allow 9091           # ws_port = port + 1

# 3. 后台跑
nohup python tools/goal_net_xpbd_dataset/cli.py view-rerun \
    Agent/Temp/goal_net_xpbd_vis/raw \
    --serve --bind 0.0.0.0:9090 > /tmp/rerun.log 2>&1 &

# 4. 本地浏览器访问 http://<gpu-host>:9090
```

如果端口受限，用 SSH tunnel：
```bash
ssh -L 9090:localhost:9090 -L 9091:localhost:9091 user@gpu-host
# 本地浏览器访问 http://localhost:9090
```

### 9.7 集成到数据生成流程

可选 `--save-rrd` 让 generate 直接落 .rrd（方便分享）：

```bash
python cli.py generate --count 1024 --batch 128 --backend warp --device cuda \
    --raw --save-rrd Agent/Temp/dataset_preview.rrd
```

任何人 `rerun preview.rrd` 即可在浏览器查看全部样本。

---

## 10. 已知 bug / 待修问题（**Warp 重写时同步修复**）

CPU 版当前还遗留 2 个视觉验证发现的 bug，Warp 重写时建议**直接修对**，不要照抄旧 bug。

### 10.1 Bug A — Swept early-out 路径反弹方向错

**触发**：球速度极快或上一子步起点已经压在 collision_radius 内，`_swept_sphere_vs_segment_toi` 走 early-out 返回 `0.0`，`contact_pos = ball_prev`，但 ball_prev 可能已穿过对面，导致 `offset = contact_pos - closest_on_segment` 朝向错误。

**症状**：sample_00001 / 00003 / 00004 视觉上看到反弹方向"穿过去了"，z 速度异常爆增。

**修复方案**（推荐选项 3）：early-out 路径单独走 fallback：
```
if early-out triggered:
    closest, _ = closest_point_on_segment(ball_prev, p0, p1)
    offset = ball_prev - closest
    if |offset| > 1e-6:
        n_geom = offset / |offset|
    else:
        n_geom = -velocity / |velocity|
    # 同时检查速度方向：n 与速度反向夹角应小于 90°
    if dot(n_geom, velocity) < 0:
        normal = n_geom
    else:
        # 几何法线方向反了，用速度反向兜底
        normal = -velocity / max(|velocity|, 1e-8)
    ball.position = ball_prev + normal * (collision_radius + 0.001)
    # 反弹同正常路径
```

### 10.2 Bug B — 网角"裂开"

**触发**：4 个 panel（back/left/right/top）在角落几何重合的粒子（< 1 mm）目前是独立 index，撞击时各自变形导致缝合处可见裂缝。

**症状**：sample_00002 视觉看到 side 网和 back 网在角落分开。

**修复方案**（推荐"粒子合并法"）：
- `topology.py::_add_particle` 加 dedupe 字典：以 `round(position*1000)` 作 key，已存在则返回已存在 index
- 修改 `panel_particle_indices` 让一个粒子可同时归属多个 panel
- 修改 `test_topology` 索引稳定性断言（粒子数会减少，但 deterministic）

**Warp 影响**：粒子总数 N 略减，约束总数 M 不变（约束都是邻接的，不会跨 panel）；`particle_panel` 改成多值表示（用 bitmask 或者第一个声明的 panel）。最简单做法：`particle_panel[i]` 取**第一个把它加进来**的 panel 即可，不影响碰撞反弹（角落处反弹哪个 panel 系数都合理）。

### 10.3 Bug C —（待补）擦柱采样不足

`sampler.py` 当前没有专门的"擦柱"瞄准模式，goalpost/crossbar 碰撞类型在数据集中占比极低。

**修复方向**：sampler 增加 `corner_post_ratio` 配置，让 ~15% 样本目标点贴近立柱/横梁（距 ≤ 0.3 m）。

### 10.4 顺序建议

**Warp 重写主线先做**，把 §10.1 / §10.2 这两个 bug 顺便修了（修在 CPU reference 也修一次，保持两版同步）。§10.3 是数据多样性问题，可在所有重写完成后再迭代。

### 10.5 v2 已实施（详见 §15 changelog）

- **D 网面焊接**：`_back_position` 的 z 偏移加 `4·u/nx·(1-u/nx)` 调制，左右边为零 → back 左右边 z 恒 `-depth`，与侧网后边完全重合，dedup 抓住（粒子 540 → 524）
- **E 底缘锚地**：`_is_anchored` 给 back/side 加 `iy == 0` 规则，模拟网底拴到地面
- **F 低速接触跳过 impulse**：`k_apply_segment_response` 检查 `|v_before| >= impulse_speed_threshold`(默认 1.0) 才注入冲量到绳
- **G 物理支撑绳**：见 §3.6
- **H 后网兜状**：`back_pocket_depth` 在 `_back_position` 里加 2D 双向 bulge，四边为零

---

## 11. 验收与数值对齐标准

### 11.1 单元/集成测试

| 测试 | 命令 | 期望 |
|---|---|---|
| 拓扑稳定性 | `PYTHONPATH=tools python tests/test_topology.py` | 同参数两次生成 byte 一致 |
| Warp vs CPU 数值对齐 | `tests/test_solver_parity.py`（新增） | 见 §11.2 容差 |
| 单条 smoke | `cli.py generate --count 1 --seed 7 --raw --backend warp --device cpu` | clean=True，contacts 非空 |
| Batch GPU 跑通 | `cli.py generate --count 64 --batch 64 --backend warp --device cuda --raw` | 正常退出，clean ≥ 60 |

### 11.2 数值对齐容差（5 样本，seed=7，default params）

CPU reference 与 Warp 对比，每个样本：

| 指标 | 容差 |
|---|---|
| `quality.clean` | 完全一致（False/True 必须 match） |
| `quality.issues` | issue 集合相同（顺序可不同） |
| `contact_count` | 相对误差 ≤ 5%（atomic 顺序导致少量差异可接受） |
| `contact.object_type` 比例 | 各类型计数相对误差 ≤ 10% |
| `frames[F].ball_position` | L2 误差 ≤ 1e-3 m（关键帧 = 第一次 segment_swept 命中那帧） |
| `stats.max_net_displacement` | 相对误差 ≤ 5% |

### 11.3 性能验收（4090）

| 配置 | 单样本平摊 | 1k 样本总时长 |
|---|---|---|
| CPU reference | ~21 s | ~5.8 h |
| Warp CPU device, batch=1 | ≤ 2 s | ≤ 35 min |
| Warp CUDA, batch=64 | ≤ 1 s | ≤ 17 min |
| **Warp CUDA, batch=256** | **≤ 0.3 s** | **≤ 5 min** |

如果达不到，先用 `wp.profile` 看 atomic 冲突热点，再决定要不要切 graph coloring。

### 11.4 视觉验收

跑 `view-rerun --serve` + 浏览器看 5 个样本（panel 覆盖 back/left/right/top/corner），人工检查：

1. 球能进入网袋而不是远端被弹飞
2. 网粒子在撞击瞬间有可见变形（5cm 量级帧位移）
3. 落地后球反弹/滚动符合直觉
4. 横/侧弧线射门能命中横梁/立柱（验证 §10.3 修复后）
5. 网角不裂（验证 §10.2 修复后）
6. 反弹方向正常（验证 §10.1 修复后）

---

## 12. 命令速查

### 开发期（CPU device）

```bash
# 拓扑摘要
python tools/goal_net_xpbd_dataset/cli.py topology

# 单条 smoke（Warp CPU，对齐 reference 用）
python tools/goal_net_xpbd_dataset/cli.py generate \
    --count 1 --seed 7 --raw \
    --backend warp --device cpu \
    --output Agent/Temp/warp_cpu_smoke

# 数值对齐测试
PYTHONPATH=tools python tools/goal_net_xpbd_dataset/tests/test_solver_parity.py

# 拓扑稳定性
PYTHONPATH=tools python tools/goal_net_xpbd_dataset/tests/test_topology.py
```

### 生产期（GPU + 4090）

```bash
# 5 样本视觉抽查
python tools/goal_net_xpbd_dataset/cli.py generate \
    --count 5 --seed 7 --raw \
    --backend warp --device cuda --batch 5 \
    --output Agent/Temp/warp_cuda_vis

# 大批量
python tools/goal_net_xpbd_dataset/cli.py generate \
    --count 10000 --seed 1 \
    --backend warp --device cuda --batch 256 \
    --output Agent/Temp/dataset_v1

# 同步落 rrd 方便分发
python tools/goal_net_xpbd_dataset/cli.py generate \
    --count 1024 --seed 1 --raw \
    --backend warp --device cuda --batch 128 \
    --save-rrd Agent/Temp/dataset_v1_preview.rrd \
    --output Agent/Temp/dataset_v1
```

### 远程查看（Linux GPU 机起 web，本地浏览器看）

```bash
# GPU 机
nohup python tools/goal_net_xpbd_dataset/cli.py view-rerun \
    Agent/Temp/warp_cuda_vis/raw \
    --serve --bind 0.0.0.0:9090 > /tmp/rerun.log 2>&1 &

# 本地浏览器：http://<gpu-host>:9090
# 或 SSH tunnel: ssh -L 9090:localhost:9090 -L 9091:localhost:9091 user@gpu-host
```

### 离线回放（任意机器）

```bash
# 直接打开 .rrd 文件（任何装了 rerun-sdk 的机器，浏览器自动开）
rerun Agent/Temp/dataset_v1_preview.rrd
```

---

## 13. 参考代码位置

实现时不再需要回查这些文件，但留作溯源依据。

| 内容 | 文件 |
|---|---|
| CPU reference solver 主循环 | `tools/goal_net_xpbd_dataset/solver.py::XpbdSolver.simulate` |
| 球积分顺序、半隐式 Euler 参考 | `Engine/.../net_move.cpp::RigidBodyBall::advance_timestep` |
| 草地反弹 5 段逻辑参考 | `Engine/.../net_move.cpp` line 189-220 |
| 球门柱反弹公式 | `Engine/.../ball_predict_component.cpp::check_hit_post` line 417-419 |
| 球网 panel 分类吸能 | `Engine/.../ball_predict_component.cpp::handle_hit_net_move` line 469-478 |
| 球门柱半径常量 | `Engine/.../animation/consts.h::HALF_GOALPOST_WIDTH` |
| 球网粒子半径运行时参考 | `Package/Script/Python/dm33/Data/ClothNetParam.py` |
| ShootModule 反弹常量 | `Package/JsonData/CommonData/ShootModule.json::HIT_GOAL_POST_SPEED_CHANGE` |

---

## 14. 与 OpenSpec change 的关系

- 当前 change：`openspec/changes/neural-goal-net-xpbd-dataset-tool/`
- 已完成 task：1.x ~ 9.x（CPU 版基础设施 + 穿网修复 + 草地/球门柱/真实响应）
- **本文档对应的新 task**：建议在该 change 下新增 Section 11："Warp GPU 重写 + Web Viewer"
  - 11.1 拓扑预转 ndarray
  - 11.2 Warp solver kernels（按 §8.4 列表）
  - 11.3 Batch + CUDA
  - 11.4 数值对齐验证（§11.2）
  - 11.5 性能验收（§11.3）
  - 11.6 viewer_rerun.py + CLI 集成
  - 11.7 修复 §10.1 / §10.2 两个遗留 bug
  - 11.8 sampler 增加擦柱模式（§10.3）
  - 11.9 同步更新 proposal.md / design.md / spec.md / progress.md

---

## 15. v2 Changelog（2026-05-16）

v2 是在 Warp 实现首次跑通后基于视觉验证发现的问题做的真实度增强。**所有改动在 `net-sim` 仓库已合入 main**。

### 15.1 拓扑

| 改动 | 文件 | 影响 |
|---|---|---|
| 后网 z 偏移按 u 调制为零边缘 | `topology._back_position` | back 左右边与侧网后边 dedup 焊接 |
| `back_pocket_depth=0.30` 双向 bulge | `topology._back_position` | 后网中心兜状鼓 30cm |
| `iy==0` 加入 back/side anchor 规则 | `topology._is_anchored` | 网底拴地，不再像旗子飘 |
| `_add_support_stays` | `topology` | 后上 2 角脱锚，加 2 个 stake ghost 粒子 + 2 根 stretch 距离约束 |
| `SupportStay` 结构含 `corner_particle / stake_particle / constraint` 索引 | `topology` | viewer 可识别 stay 并独立渲染 |

**新拓扑统计**（默认参数）：粒子 526（= 524 mesh + 2 stake）、距离约束 1898（= 1896 mesh + 2 stay）、anchored 粒子 141（多 35 个底缘 + 2 个 stake，少 2 个后上角）。

### 15.2 物理 / Solver

| 改动 | 文件 | 影响 |
|---|---|---|
| `rope.impulse_speed_threshold=1.0` | `params.RopeParams` + `solver_warp.k_apply_segment_response` | 球速 < 阈值时 swept 不再注能给绳，防止持续低速接触把网"鞭"得震荡 |

### 15.3 Viewer

| 改动 | 文件 | 影响 |
|---|---|---|
| `--public-host` flag | `cli` + `viewer_rerun` | 重写 rerun 0.32 `serve_grpc` 默认硬编码的 `127.0.0.1` URI |
| 启动时打印完整 `?url=` URL | `viewer_rerun` | rerun 0.32 web viewer 不从 HTML 注入连接 URL，必须查询参数传 |
| 每样本独立 `RecordingStream`(recording_id=sample_id) | `viewer_rerun` | 多样本时左上角"Recordings"列表可切换 |
| 根 entity 声明 `RIGHT_HAND_Y_UP` | `viewer_rerun._log_sample` | 防止 rerun 默认 Z-up 把场景画歪 |
| Stake 粒子从网渲染中排除（`stake_particle_indices`） | `viewer_rerun` | ghost 粒子不被画成网粒子 |
| Stay 约束从 mesh rope 渲染中排除，独立动态画 | `viewer_rerun` | 角粒子摆动时 stay 线跟着动 |

### 15.4 SSH tunnel / 网络

rerun 0.32 把 web viewer 拆成两个端口：HTTP HTML 在 9090，gRPC 数据在 9091。SSH tunnel 必须同时转发两个：

```bash
ssh -L 9090:localhost:9090 -L 9091:localhost:9091 user@gpu-host
```

只转发 9090 会看到"Loading Application Bundle"卡住或"Failed to load entries"。

### 15.5 仓库

代码托管于 https://github.com/bayzhen/net-sim 的 `main` 分支。提交历史：
- `d8081ed` Initial Warp 实现（v1）
- `4030d7f` viewer: `--public-host` 修 0.32 硬编码
- 后续 v2 修改尚未单独 commit（本次任务一起 push）

---

## 16 · v3 高吞吐数据管线（HDF5 + simulate_arrays）

### 16.1 动机

v2 跑出来的 `cli.py generate --raw` 在 GTX 1650 SUPER 4 GB 上吞吐只有约 13 samples/s，看起来 GPU 利用率忽高忽低。逐个 batch 排查发现：

| 瓶颈 | 实测 | 原因 |
|---|---|---|
| `_assemble_results` Python 循环 | ~20 s / batch=512 | 每样本每帧都构造 `FrameSample` dataclass，30 万对象 + numpy slice |
| writer 线程持 GIL | sim 12s → 35s | JSON 编码、numpy serialize 都是 GIL-持有，阻塞主线程 launch GPU kernel |
| 16800 个小 npz 文件 | 写盘 1.13 s/sample | 文件 syscall + 每文件独立打开关闭；HDD 写头疯狂寻道 |
| 每 raw 内嵌 topology | 每 sample 多 ~250 KB 文本 | 250 KB × 80000 = 20 GB 重复内容 |

### 16.2 三个针对性改动

**(A) `simulate_arrays()` —— 跳过 B*F 对象循环**

`solver_warp.py` 把 `simulate()` 拆成两层：
- `_simulate_to_arrays()`：跑 GPU + 把所有 17 个 wp.array `.numpy()` 拉回 host，**返回 dict**（无任何 Python loop）
- `_compute_quality_vectorized()`：把原来 per-sample 的 8 项 quality 检查全 vectorize（np.linalg.norm, .any(axis=1) 等）；只在最后构造 B 个 `QualityReport` dataclass（轻量）
- `simulate_arrays()` = 上面两步串起来；返回值附带 `per_sample_quality` / `per_sample_stats`
- 老 `simulate()` 入口保留，包成 `arrs → _assemble_results(arrs)` 的兼容 shim

**(B) HDF5 chunked dataset —— 写盘几乎零 GIL**

`output.py::make_h5_writer()`：单文件 `dataset.h5`，所有大数组 chunked + extendable：
```python
ds = f.create_dataset("particle_position", shape=(0, F, N, 3), maxshape=(None, F, N, 3),
                      dtype=np.float32, chunks=(1, F, N, 3))
# per batch:
ds.resize((i1, F, N, 3))
ds[i0:i1] = arrs["frame_particles"][:B]   # 一次拷贝完成；底层 h5py 释放 GIL
```

contacts 用 CSR 格式扁平存（不浪费 max_contacts × B × 6 字段的零）。topology + metadata + issue_names 进 root attrs，文件自包含。

**(C) ThreadPool 异步写 + back-pressure**

`cli.py`：单 worker 线程；submit 下一 batch 写盘前先 `wait()` 上一 batch 的 future，保证内存占用恒定（不会无限堆积）。配合 (A)+(B) 后，写盘耗时 ≪ sim 耗时，主线程几乎不等。

### 16.3 实测对比（GTX 1650 SUPER, batch=512, --raw）

| 路径 | sim batch 0 | sim batch 1+ | 写盘 | 总速率 |
|---|---|---|---|---|
| v2 json 写盘 | 13 s | 32 s | 113 s/100sample | 13.5 /s |
| v2 npz 写盘 | 13 s | 31 s | 25 s/batch | 16.4 /s |
| v3 simulate_arrays + npz | 13 s | 31 s | 22 s/batch | 23.3 /s |
| **v3 simulate_arrays + h5** | **12 s** | **12 s** | **0.98 s 总 flush** | **42.8 /s** |
| 上限：simulate without raw | 12 s | 12 s | – | 42.1 /s |

h5 路径达到了 GPU 真正的算力上限（vs 不写盘只差 1.5%）。

### 16.4 CLI

```bash
# 大数据集（推荐）
python3 cli.py generate \
    --count 30000 --batch 512 --device cuda \
    --raw --incremental --raw-format h5 \
    --seed 1 --output E:/dataset_v1
```

`--raw-format` 三档可选：
- `h5` — 单文件 chunked HDF5，**推荐**
- `npz` — 每样本一个 numpy 二进制；适合 < 5000 样本、要 viewer 直接看
- `json` — 每样本一个自包含 JSON（含 topology 拷贝），仅作兼容用，不要在大数据集上开

### 16.5 PyTorch dataset 加载

新增 `dataset_loader.py`，封装 `dataset.h5` 为 `torch.utils.data.Dataset` 协议：

```python
from dataset_loader import GoalNetH5Dataset, GoalNetOffsetSampler

base = GoalNetH5Dataset("E:/dataset_v1/dataset.h5", clean_only=True)
ds   = GoalNetOffsetSampler(base, rng_seed=42)  # 任意偏移帧采样
```

`GoalNetOffsetSampler[i]` 返回 `(input_state[12], target_ball[3], target_ball_v[3], target_net[N,3], offset_frame)` —— 正好对应"输入球初态 + 偏移帧 t，输出 t 时刻球+网状态"的训练任务。

### 16.6 已知限制 / 未做事项

- HDF5 dataset 还没接 viewer——要在 rerun 里看 h5 里的某个样本，目前需要先手写 5 行 numpy → 落到 npz → `cli.py view-rerun ...`。下一个版本加 `cli.py view-rerun --h5 path/to/dataset.h5 --index N` 直接看
- `summary.jsonl` + `features/*.json` 在 h5 模式下不再写出（h5 内同等内容覆盖了）。下游若依赖 summary.jsonl，把 h5 的 metadata 数据 dump 一份即可
- `validate_dataset.py` 目前只验证 npz；h5 schema 自身较强、未单独写验证脚本

---

**文档版本**：v3
**最后更新**：2026-05-16
**作者**：CodeMaker AI（v1，2026-05-15）+ Claude（Warp 实现 + v2 增强 + v3 数据管线，2026-05-15..16）

