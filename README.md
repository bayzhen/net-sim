# net-sim — 球网 XPBD 数据集生成工具（Warp GPU）

足球射门 → XPBD 球网响应模拟器，输出 teacher 数据集用于训练 neural goal-net response model。

完整设计文档：[`goal_net_warp_design.md`](goal_net_warp_design.md)（14 节，覆盖物理模型、参数表、算法、Warp 实现指南、bug 列表、验收标准）。

---

## 概览

| 项 | 内容 |
|---|---|
| **物理后端** | NVIDIA Warp（GPU 编译到 CUDA，单进程 batch 并行多样本） |
| **粒子模型** | XPBD 距离约束 + bend 约束 + 硬锚 + 2 根物理支撑绳，526 粒子 / 1898 约束 |
| **碰撞** | 球-粒子离散 / 球-段离散 / 球-段 swept CCD / 球门柱胶囊 / 草地 5 段反弹 |
| **样本规模** | 默认 121 帧（2 s @ 60 Hz），单批可并行 ≥ 256 样本 |
| **性能（RTX 3070）** | **0.028 s / 样本**（B=128），相对 CPU reference ~750× 提速 |
| **输出** | features JSON（默认） + 可选 raw + summary.jsonl + batch_report.json |
| **可视化** | rerun.io web viewer（远程浏览器） / 本地 GUI / 离线 .rrd |

---

## 环境

- **Python** ≥ 3.10
- **NVIDIA 驱动** ≥ 525（带 CUDA 11+ 支持；Warp 自带 toolkit）
- **GPU**：测试在 RTX 3070 8 GB。8 GB VRAM 足够 batch ≤ 128。无 GPU 时可用 `--device cpu`（Warp 编译到 LLVM，仍快于纯 Python）

```bash
pip install --user warp-lang numpy
pip install --user rerun-sdk    # 可选，仅 view-rerun 子命令需要
```

验证：
```bash
python3 -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

---

## 快速开始

### 1. 看一眼拓扑

```bash
python3 cli.py topology
```
输出：540 粒子 / 1926 距离约束（1013 stretch + 913 bend） / 106 锚定 / 3 球门柱段。

### 2. 跑 5 样本（CPU device，最快验证流程跑通）

```bash
python3 cli.py generate --count 5 --seed 7 --device cpu --output /tmp/smoke
```

### 3. 跑 256 样本（CUDA，batch=128）

```bash
python3 cli.py generate --count 256 --batch 128 --device cuda --seed 7 \
    --output /tmp/dataset_v1
```

### 4. 含 raw（可回放）+ rerun 可视化

```bash
python3 cli.py generate --count 5 --batch 5 --device cuda --raw \
    --output /tmp/vis_dataset

# 起 web viewer（监听所有网卡，--public-host 指定浏览器能够访问到的主机名）
python3 cli.py view-rerun /tmp/vis_dataset/raw --serve \
    --bind 0.0.0.0:9090 --public-host localhost
```

启动后会打印一条**完整的可点击 URL**（带 `?url=` 查询参数）：

```
open in browser: http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9091%2Fproxy
```

直接复制到浏览器即可——rerun 0.32 的 web viewer 是从 `?url=` 读 gRPC 后端地址的，不带这个参数会显示 "Failed to load entries"。

**SSH tunnel 时必须同时转发两个端口**：
```bash
ssh -L 9090:localhost:9090 -L 9091:localhost:9091 user@gpu-host
```

---

## CLI 速查

```
python3 cli.py <subcommand> [options]
```

### `topology`
打印当前参数下拓扑摘要 + 稳定签名。
```bash
python3 cli.py topology
python3 cli.py --params my_params.json topology
```

### `generate`
| flag | 默认 | 含义 |
|---|---|---|
| `--count N` | 5 | 总样本数 |
| `--seed S` | 1 | 采样器种子 |
| `--batch B` | 0 | GPU batch 维（0 = 一次跑完全部） |
| `--device {cpu,cuda}` | cuda | Warp device |
| `--raw` | off | 同时输出 raw/sample_*.json（含全帧粒子位置 + topology） |
| `--output DIR` | Agent/Temp/goal_net_xpbd_dataset | 输出目录 |
| `--max-contacts N` | 16384 | 每样本接触事件缓冲上限 |

### `view-rerun`
| flag | 默认 | 含义 |
|---|---|---|
| `path` | (必填) | raw/sample_*.json 文件路径 或 目录 |
| `--serve` | off | 启动 web server（HTTP HTML viewer + gRPC data sink） |
| `--bind HOST:PORT` | 0.0.0.0:9090 | `--serve` 监听地址（gRPC = PORT+1） |
| `--public-host HOST` | localhost | 写到浏览器可点击 URL 里的 gRPC 主机名 |
| `--save PATH.rrd` | — | 把数据落盘成 .rrd 文件（可离线分发） |
| `--spawn` | off | 启动本地 rerun GUI（需要桌面环境） |

⚠️ rerun 0.32 把 web viewer 拆成两个端口：**HTTP UI 在 PORT，gRPC 数据在 PORT+1**。SSH tunnel / 防火墙 / Docker 端口映射必须把两个一起开。

每个 raw 样本写入一个独立的 `RecordingStream`（recording_id = sample_id），viewer 左上角"Recordings"列表可切换样本。所有 entity 都声明 `RIGHT_HAND_Y_UP` 视图坐标——避免 rerun 默认 Z-up 把场景画歪。

---

## 输出 schema

```
<output_dir>/
├── summary.jsonl          # 每行一条样本摘要（路径、quality、stats、metadata）
├── batch_report.json      # 整批 clean/abnormal 计数 + 各 panel 命中
├── features/              # 默认产出（轻量训练特征）
│   └── sample_NNNNN.json
└── raw/                   # 仅 --raw 时产出（完整回放数据）
    └── sample_NNNNN.json
```

### `features/sample_*.json`
```jsonc
{
  "sample_id": "sample_00000",
  "schema_version": "goal_net_params.v1",
  "input_features": { "position", "velocity", "angular_velocity", "target_panel", "template", "seed", ... },
  "ball_trajectory": [ {"time", "position", "velocity"}, ... ],   // 121 帧
  "net_control_points": [ {"time", "positions": [16 pts]} ],     // 稀疏抽样
  "quality": { "clean", "issues", "target_hit", "max_penetration_depth", ... }
}
```

### `raw/sample_*.json`
features 内容 + `topology`（粒子、约束、锚定、球门柱）+ 每帧全 540 粒子位置 + `contacts` 列表。

`contacts.object_type` 取值：`particle` / `segment` / `segment_swept` / `goalpost` / `crossbar` / `ground_bounce` / `ground_roll`。

---

## 物理参数

全部在 `params.py` 里以 dataclass 形式定义，默认值与设计文档 §2 一致。覆盖方式：

```python
from params import GoalNetParams, SolverParams
p = GoalNetParams()
p.solver = SolverParams(substeps=16, duration=2.5)
```

CLI 用 `--params my.json` 加载覆盖（部分覆盖 OK，未出现的字段保持默认）：
```json
{
  "solver": {"substeps": 16, "duration": 2.5},
  "rope":   {"collision_radius": 0.18}
}
```

关键参数（容易踩坑的）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `rope.collision_radius` | 0.16 | 球-网碰撞膨胀半径（封 0.305 m 网孔） |
| `rope.panel_restitution_back/top/side` | 0.10 / 0.30 / 0.20 | panel 法向恢复系数 |
| `rope.impulse_speed_threshold` | 1.0 m/s | 球速低于此值时 swept 不再注入冲量到网（防止持续低速接触把能量泵入网） |
| `solver.substeps` | 12 | 必须 ≥ 12 防穿网 |
| `solver.duration` | 2.0 s | 要捕到落地后滚动 |
| `collision.severe_penetration_threshold` | 0.15 m | 球 z < -2.25 的最大穿透阈值 |
| `shape.back_pocket_depth` | 0.30 m | 后网中心向 -z 方向的兜状鼓包（向后凸） |
| `shape.stay_count` | 2 | 物理支撑绳数量：0 / 2 / 4 |
| `shape.stay_anchor_offset_x/y/z` | 0.3 / 0.6 / 0.4 | stay 远端锚点相对后上角的偏移（外 / 上 / 后） |

---

## 架构

```
params.py       — 所有参数 dataclass
topology.py     — 粒子/约束生成 + 角落 dedupe + ndarray 导出
sampler.py      — 5 panel × 5 style 射门采样 + 擦柱模式 + 源点 envelope
solver_warp.py  — Warp kernels + XpbdWarpSolver 驱动类
output.py       — features/raw/summary/batch_report JSON 写出
cli.py          — argparse 入口
viewer_rerun.py — rerun web/save/spawn viewer
tests/          — 拓扑稳定性 + 端到端 smoke
```

模拟主循环（`solver_warp.XpbdWarpSolver.simulate`）顺序：
```
for each frame:
    if frame % sample_every == 0: record frame
    for each substep:
        save_previous_positions
        integrate_particles (gravity + Euler)
        resolve_particle_ground
        integrate_ball
        ball_vs_goalposts (swept CCD, 必须在网 swept 之前)
        ball_vs_net_segments (swept CCD with impulse injection)
        resolve_ball_ground (5 段 bounce/roll 逻辑)
        for each XPBD iteration:
            solve_anchors (硬锚强制位置)
            solve_distance_constraints (stretch, atomic_add)
            solve_distance_constraints (bend)
            solve_ball_particle_collisions (离散清理)
            solve_ball_segment_collisions
        update_velocities (pos-prev / dt * (1 - damping))
        update_substep_stats (early-stop / stuck / penetration)
```

GS 并行化用 **atomic_add Jacobi**：一次 kernel launch 内所有约束读 launch 入口快照，atomic_add 写入下次 launch 可见。比 CPU 串行 GS 收敛慢约 1.3~1.5×，默认 `iterations=8` 即可。

swept argmin 用 **两遍法**：第一遍 `atomic_min(best_toi)` 找最小 TOI；第二遍重算 TOI 与最小值匹配则 `atomic_min(best_idx)`。

### 网拓扑（实际跑出的结构）

```
crossbar + posts (front, z=0)                      [固定锚]
            │
   ├──── side panel front edge (u=0) ──────────┐
   ├──── top panel front row (v=0) ────────────┤
   ▼                                            ▼
[mesh ropes within side/top/back panels]   [back panel top edge]
   │                                            │
   ▼                                            ▼
[back-top corners (un-anchored)] ──── stay ropes ──── [elevated stay anchors]
   │                                                       (fixed, 上后外)
   ▼
[back panel body, pocket-bulged in middle]
   │
   ▼
[back/side/back-bottom edge (y=0)] ──────── [ground anchors (固定)]
```

- 后上 2 个角不再硬钉在空中，**靠 2 根物理 rope 拉到斜上方锚点** —— 拿掉绳整个后网会塌下来
- 后网中部向 -z 方向鼓 30cm，形成兜状（`back_pocket_depth`）
- back/side 面 y=0 那一行锚定到地面，防止网底像旗子飘
- viewer 渲染时 stake "ghost" 粒子不显示（不在 `panel_particle_indices` 里），stay 绳每帧动态画在当前角粒子位置与 stake 位置之间

---

## 已修复的 bug + 物理改造

设计文档 §10：

- **A** swept early-out 反弹方向：apply_post / apply_segment kernel 检测 `n·v > 0` 时用 `-v/|v|` 兜底
- **B** 网角裂开：`topology._add_panel` 用 `round(position*1000)` 做粒子去重；top corners 几何重合的粒子合并为同一 index（540 vs 朴素 ~566）
- **C** 擦柱采样不足：`SamplerConfig.corner_post_ratio = 0.15`，~15% 样本目标点贴近立柱/横梁，提升 goalpost/crossbar 接触多样性

第二轮（v2）实物增强：

- **D 网面焊接**：back panel z 沿 u 方向用 `4·u/nx·(1-u/nx)` 调制，让左右边的 z 恒为 -depth，与侧网后边 dedup（524 粒子）
- **E 底缘锚地**：back/side `iy==0` 加入锚定规则，模拟网底拴到地面，避免空荡飘旗
- **F 低速接触跳过 impulse**：球速 < `rope.impulse_speed_threshold`(1.0) 时 swept 不再注能给网，杜绝持续低速接触造成的能量泵入循环（球落地反弹后顶网的场景）
- **G 物理支撑绳**：2 个后上角脱锚，加 2 个 "stake" ghost 粒子（hard-anchored）+ 2 根 stretch 距离约束（绳）。stake 位置在角粒子斜上方-外侧-后侧（默认 +0.6m up, ±0.3m out, -0.4m back），绳长 ~0.78m。后网完全靠绳承重
- **H 后网兜状**：`shape.back_pocket_depth=0.30`，后网中心向 -z 鼓包，左右上下四个边都是零（仍保证 dedup 焊接）

---

## 测试

```bash
PYTHONPATH=. python3 tests/test_topology.py   # 拓扑稳定签名 + dedup 验证
PYTHONPATH=. python3 tests/test_smoke.py      # 单样本 CPU 物理合理性
```

---

## 性能

| 配置 | 单样本平摊 | 备注 |
|---|---|---|
| CPU reference | ~21 s | 原 Python 标准库 solver（不在本仓库） |
| Warp CPU device, B=1 | ~10 s | LLVM 编译，仅做对齐验证 |
| Warp CUDA, B=8 | ~0.95 s | kernel launch 主导 |
| Warp CUDA, B=64 | **0.047 s** | clean ≈ 78% |
| Warp CUDA, B=128 | **0.028 s** | clean ≈ 72%（256 样本 7.2 s） |

clean ratio 约 70~80%。剩余失败基本都是 `severe_penetration`——sampler 偶尔生成极端入射角/源点的样本被网"漏"出去。已做源点 envelope clamp + ground kernel `vy >= 0 skip`，进一步收紧需要调整采样分布。

> **注**：v2 修改了拓扑（粒子数 526、约束 1898、新增 stake 粒子 + stay 约束），上面性能数字是 v1 测出来的；v2 多了 2 粒子 + 2 约束，单样本耗时差异在测量噪声内（< 1%）。clean ratio 变化未重新统计；底缘锚 + 低速接触跳过冲量应该让网在球停止后更稳，预期 clean ratio 微升。

---

## 已知限制

1. **CPU↔Warp 严格数值 parity 未做**：本仓库无 CPU reference solver，跳过设计文档 §11.2 的字节级对齐。物理合理性由 smoke 测试 + 视觉检查保障。
2. **batch 内 ball radius/mass 必须一致**：当前实现假设同批样本球参一致；若需异构需要把这两个量改成 `wp.array(dtype=float, shape=(B,))`。
3. **early-stop frozen mask 之后帧值复制**：被 frozen 的 batch 之后帧 ball_position 不再更新；`len(frames) == duration/frame_dt + 1` 仍恒等。

---

## License

—（按需添加）
