# net-sim — 球网 XPBD 数据集生成工具（Warp GPU）

足球射门 → XPBD 球网响应模拟器，输出 teacher 数据集用于训练 neural goal-net response model。

完整设计文档：[`goal_net_warp_design.md`](goal_net_warp_design.md)（17 节，覆盖物理模型、参数表、算法、Warp 实现指南、数据管线、训练管线 v1/v2、bug 列表、验收标准）。

---

## 概览

| 项 | 内容 |
|---|---|
| **物理后端** | NVIDIA Warp（GPU 编译到 CUDA，单进程 batch 并行多样本） |
| **粒子模型** | XPBD 距离约束 + bend 约束 + 硬锚 + 2 根物理支撑绳，514 粒子 / 1872 约束 |
| **碰撞** | 球-粒子离散 / 球-段离散 / 球-段 swept CCD / 球门柱胶囊 / 草地 5 段反弹 |
| **样本规模** | 默认 601 帧（10 s @ 60 Hz），单批可并行 ≥ 512 样本 |
| **性能（GTX 1650 SUPER）** | **0.023 s / 样本**（B=512），**~43 samples/s** GPU 拉满 |
| **输出** | features JSON + 可选 raw + summary.jsonl + batch_report.json，**raw 支持 HDF5（推荐） / npz / json 三种格式** |
| **可视化** | rerun.io web viewer（远程浏览器） / 本地 GUI / 离线 .rrd |

---

## 环境

- **Python** ≥ 3.10
- **NVIDIA 驱动** ≥ 525（带 CUDA 11+ 支持；Warp 自带 toolkit）
- **GPU**：测试在 GTX 1650 SUPER 4 GB（batch=512 + raw 仍能跑）。8 GB 卡上 batch 可拉到 ≥ 4096

```bash
# 数据生成 + 可视化
pip install --user warp-lang numpy h5py psutil
pip install --user rerun-sdk    # 可选，仅 view-rerun 子命令需要

# 训练管线（PyTorch CUDA build；驱动 12.x 用 cu121）
pip install --user torch --index-url https://download.pytorch.org/whl/cu121
```

> `h5py` 是 v3 之后必需，用于 `--raw-format h5` 的高吞吐数据集格式。
> `psutil` 是 v4 之后训练管线的安全检查依赖（preload 前判断 RAM 够不够）。
> `requirements.txt` 已就位，`pip install -r requirements.txt` 一行即可（注意 torch 仍需手动指定 CUDA index）。

验证：
```bash
python3 -c "import warp as wp; wp.init(); print(wp.get_devices())"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

---

## 快速开始

### 1. 看一眼拓扑

```bash
python3 cli.py topology
```
输出：514 粒子 / 1872 距离约束（985 stretch + 887 bend） / 129 锚定 / 3 球门柱段。

### 2. 跑 5 样本（CPU device，最快验证流程跑通）

```bash
python3 cli.py generate --count 5 --seed 7 --device cpu --output /tmp/smoke
```

### 3. 跑大数据集（CUDA，HDF5 单文件，推荐）

```bash
python3 cli.py generate \
    --count 30000 --batch 512 --device cuda \
    --raw --incremental --raw-format h5 \
    --seed 1 --output E:/dataset_v1
```

`--incremental` + `--raw-format h5` = 每 batch 跑完直接 chunked-write 到一个 `dataset.h5`，**GPU 满载、写盘几乎零开销**。30000 样本约 12 分钟、~110 GB（601 帧 × 514 粒子 float32 占大头）。详细 schema 见下文 *输出 schema → HDF5* 一节。

### 4. 含 raw（可回放）+ rerun 可视化（小样本）

```bash
# 生成 5 个 raw 样本（per-sample npz，方便单个看）
python3 cli.py generate --count 5 --batch 5 --device cuda \
    --raw --incremental --raw-format npz \
    --output /tmp/vis_dataset

# 起 web viewer
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
| `--batch B` | 0 | GPU batch 维（0 = 一次跑完全部）。GTX 1650 SUPER 4G 上 raw=True 实测可到 512；8G 卡可上 1024+ |
| `--device {cpu,cuda}` | cuda | Warp device |
| `--raw` | off | 同时输出每帧全粒子位置 + topology + 接触序列；不开 `--raw` 只产 features |
| `--incremental` | off | 每个 batch 跑完立刻 flush 到磁盘，**大数据集必须开**（否则把所有 result 攒在内存等 OOM） |
| `--raw-format {h5,npz,json}` | npz | 仅 incremental 模式下生效。**`h5` = 单文件 chunked HDF5（推荐，GPU 真正拉满）；`npz` = 每样本一个二进制；`json` = 每样本一个 JSON（兼容旧版，慢、大）** |
| `--output DIR` | `Agent/Temp/goal_net_xpbd_dataset` | 输出目录 |
| `--max-contacts N` | 16384 | 每样本接触事件缓冲上限 |

> **大数据集典型用法**：`--raw --incremental --raw-format h5`。h5 写盘几乎零开销、可被 PyTorch 原生 mmap、不会有 16800 个小文件这种文件系统噩梦。

### `view-rerun`
| flag | 默认 | 含义 |
|---|---|---|
| `path` | (必填) | `raw/sample_*.json` / `raw/sample_*.npz` 文件路径 或 目录（自动探测两种） |
| `--serve` | off | 启动 web server（HTTP HTML viewer + gRPC data sink） |
| `--bind HOST:PORT` | 0.0.0.0:9090 | `--serve` 监听地址（gRPC = PORT+1） |
| `--public-host HOST` | localhost | 写到浏览器可点击 URL 里的 gRPC 主机名 |
| `--save PATH.rrd` | — | 把数据落盘成 .rrd 文件（可离线分发） |
| `--spawn` | off | 启动本地 rerun GUI（需要桌面环境） |

> 注：HDF5 数据集目前不能直接喂给 `view-rerun`（每个样本要从大 .h5 切片）。如果要可视化 HDF5 里某个样本，先 export 一份 npz：见下方"H5 → npz 单样本导出"的小脚本片段（暂时手写）。

⚠️ rerun 0.32 把 web viewer 拆成两个端口：**HTTP UI 在 PORT，gRPC 数据在 PORT+1**。SSH tunnel / 防火墙 / Docker 端口映射必须把两个一起开。

每个 raw 样本写入一个独立的 `RecordingStream`（recording_id = sample_id），viewer 左上角"Recordings"列表可切换样本。所有 entity 都声明 `RIGHT_HAND_Y_UP` 视图坐标——避免 rerun 默认 Z-up 把场景画歪。

### `train` —— 训练 MLP 代理模型

在 `--raw-format h5` 产出的 `dataset.h5` 上训练一个神经代理：输入 `(球初态 + 归一化时间 t_norm)`，输出 `(球位置, 球速度, 网粒子位置)`。

```bash
python3 cli.py train \
    --dataset /data3/netsim/dataset_v2/dataset.h5 \
    --output runs/mlp_v2 \
    --epochs 100 --batch 4096 --device cuda \
    --hidden 1024 1024 1024 1024 1024 1024
```

| flag | 默认 | 含义 |
|---|---|---|
| `--dataset` | (必填) | `dataset.h5` 路径 |
| `--output` | (必填) | 训练产物目录 |
| `--epochs` | 100 | 训练轮数 |
| `--batch` | 512 | mini-batch 大小（对 RTX 4090 推荐 4096） |
| `--lr` | 3e-4 | AdamW 学习率（cosine 退火） |
| `--hidden 1024 1024 ...` | `[1024]*6` | MLP 各隐藏层宽度 |
| `--n-time-freq` | 4 | sin/cos 时间编码频率数；in_dim = 10 + 2·N |
| `--no-preload` | preload | 关闭 RAM 预加载，回退到 HDF5 流式读（慢 ~280×，用于 RAM 紧张机器） |
| `--preload-test` | off | 也预加载 test 集（默认 test 流式读，因为只在最后跑一次） |
| `--w-ball-pos / --w-ball-vel / --w-net` | 1.0 | 各 head loss 权重（归一化空间） |
| `--num-workers` | 0 | DataLoader worker（preload 模式下强制 0） |

**性能要点**：默认 `--preload` 会一次性把 train+val 的 `(B, F, N, 3)` 全部读进 RAM（~50 GB for 14k clean × 601 帧 × 514 粒子）。读取完后 epoch 仅 ~0.1 s（GPU 拉满）。
- 启动前 `psutil` 会做 1.4× 安全裕度检查，装不下时给清晰错误信息。
- 大数据集装不下时用 `--no-preload`（慢，但任何机器都能跑）。

**产物**（`--output` 目录）：
```
config.json         # 完整超参 + 数据集统计
metrics.jsonl       # 每 epoch 的 train/val loss（含归一化和物理单位两套）
best.pt             # val loss 最低的 checkpoint（含归一化 buffer）
last.pt             # 最后 epoch checkpoint
test_metrics.json   # best.pt 在 test split 上的 RMSE
```

> 详细技术决策（归一化策略、模型结构、已知 baseline 缺陷与修复路径）见设计文档 §17。

### `predict` —— 评估 / 诊断 checkpoint

跑完训练后用 `predict` 在 test split 上做**全帧**评估（与训练时"每样本随机抽 1 帧"不同），输出 per-frame RMSE 曲线和 worst-K 最差样本，方便定位模型在哪一帧 / 哪类样本上崩。

```bash
python3 cli.py predict \
    --ckpt runs/mlp_v2/best.pt \
    --dataset /data3/netsim/dataset_v2/dataset.h5 \
    --output runs/mlp_v2/eval \
    --device cuda --batch 16 --worst-k 16
```

| flag | 默认 | 含义 |
|---|---|---|
| `--ckpt` | (必填) | `best.pt` / `last.pt` |
| `--dataset` | (必填) | 同训练用的 h5（splits 由 ckpt 内的 seed 复现） |
| `--output` | (必填) | 评估产物目录 |
| `--batch` | 8 | 每次前向多少个完整样本（每样本贡献 F 帧） |
| `--worst-k` | 16 | 输出多少个最差样本 |

**产物**：
- `per_frame_rmse.json` — `(F,)` RMSE 曲线（球位置/速度/网）
- `per_sample_summary.json` — 每个 test 样本的时均 RMSE
- `worst_k.json` — 按球位置 RMSE 排序的最差 K 个样本
- `summary.json` — 整体 mean/median/p95

---

## 输出 schema

根据 `--raw-format` 不同有 3 种产物布局。所有模式都额外产出 `summary.jsonl`（每行一条样本摘要） + `batch_report.json`（整批 clean/abnormal 计数）。

### `--raw-format h5`（推荐，大数据集）

```
<output_dir>/
├── dataset.h5            # 单文件，所有样本的 raw + features 都在里头
├── topology.json         # 共享拓扑（拷贝一份方便人读）
├── metadata.json         # params snapshot + schema 版本
├── batch_report.json
├── summary.jsonl         # （目前不写，已被 dataset.h5 内的 dataset 取代）
└── features/             # （目前不写，已被 dataset.h5 内的 dataset 取代）
```

`dataset.h5` 内部布局（S = 样本数, F = 帧数 = 601, N = 粒子数 = 514, K = 7 种 issue）：

```text
attrs:
  schema_version, frame_dt, frame_count, particle_count
  topology_json   <— 完整 topology JSON（粒子/约束/锚定/球门柱），所有样本共享
  metadata_json   <— params snapshot
  issue_names     <— K=7 种 quality issue 的名字数组
  include_raw

datasets:
  sample_id            (S,)        vlen str  e.g. b"sample_00000"
  target_panel         (S,)        vlen str  back / left / right / top / corner
  template             (S,)        vlen str  采样模板名
  seed                 (S,)        int64
  input_position       (S, 3)      float32   球初始位置
  input_velocity       (S, 3)      float32
  input_angular        (S, 3)      float32
  input_radius         (S,)        float32
  input_mass           (S,)        float32

  ball_position        (S, F, 3)        float32   每帧球位置
  ball_velocity        (S, F, 3)        float32
  particle_position    (S, F, N, 3)     float32   每帧全 N 粒子位置（chunked，单样本 ~3.5 MB）

  contact_offset       (S+1,)      int64    CSR 索引到下面扁平数组
  contact_time         (TotalC,)   float32  按时间排序
  contact_object_type  (TotalC,)   int32    0=particle 1=segment 2=segment_swept 3=goalpost 4=crossbar 5=ground_bounce 6=ground_roll
  contact_object_index (TotalC,)   int32
  contact_position     (TotalC, 3) float32
  contact_normal       (TotalC, 3) float32
  contact_strength     (TotalC,)   float32

  quality_clean        (S,)        bool
  quality_target_hit   (S,)        bool
  quality_issue_mask   (S, K)      bool   每位对应 issue_names[i]
  quality_max_pen      (S,)        float32
  quality_max_pen_time (S,)        float32
  stats_contact_count  (S,)        int32
  stats_max_disp       (S,)        float32  全过程粒子最大位移（vs rest pose）
  stats_came_to_rest   (S,)        bool
```

**PyTorch 读取**：见仓库根的 [`dataset_loader.py`](dataset_loader.py)，里头有 `GoalNetH5Dataset` + `GoalNetOffsetSampler` 两个开箱即用的类。最简：

```python
from torch.utils.data import DataLoader
from dataset_loader import GoalNetH5Dataset

ds = GoalNetH5Dataset("E:/dataset_v1/dataset.h5", clean_only=True)
loader = DataLoader(ds, batch_size=8, num_workers=4, persistent_workers=True)
for b in loader:
    ball_traj = b["ball_position"]        # (8, 601, 3)
    net_traj  = b["particle_position"]    # (8, 601, 514, 3)
    cond      = b["input_state"]          # (8, 9)  pos+vel+ang
    ...
```

### `--raw-format npz`（中等数据集，单样本独立）

```
<output_dir>/
├── topology.json         # 共享拓扑
├── metadata.json
├── summary.jsonl         # 每行一条样本摘要
├── batch_report.json
├── features/sample_NNNNN.json
└── raw/sample_NNNNN.npz  # 每样本一个二进制
```

`raw/sample_*.npz` 内含：`frame_time` `ball_position` `ball_velocity` `particle_position` `contact_*` `meta_json`（JSON 字符串包含 shot/quality/stats）。

### `--raw-format json`（兼容旧版，慢、大）

```
<output_dir>/raw/sample_NNNNN.json   # 每样本独立、含完整 topology 拷贝
```

仅在你需要"单文件自包含"时使用——文件大小约比 npz 大 5~10×，写盘明显成 GPU 拉满的瓶颈。

### `features/sample_*.json`（所有 raw 模式共用）
```jsonc
{
  "sample_id": "sample_00000",
  "schema_version": "goal_net_params.v1",
  "input_features": { "position", "velocity", "angular_velocity", "target_panel", "template", "seed", ... },
  "ball_trajectory": [ {"time", "position", "velocity"}, ... ],   // F 帧
  "net_control_points": [ {"time", "positions": [16 pts]} ],     // 稀疏抽样
  "quality": { "clean", "issues", "target_hit", "max_penetration_depth", ... }
}
```

`contacts.object_type` 取值（npz/json 路径下用字符串名；h5 下用 int 见上表）：
`particle` / `segment` / `segment_swept` / `goalpost` / `crossbar` / `ground_bounce` / `ground_roll`。

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
params.py            — 所有参数 dataclass
topology.py          — 粒子/约束生成 + 角落 dedupe + ndarray 导出
sampler.py           — 5 panel × 5 style 射门采样 + 擦柱模式 + 源点 envelope
solver_warp.py       — Warp kernels + XpbdWarpSolver 驱动类
                       提供 simulate()（构造 SimulationResult 列表）
                       和 simulate_arrays()（直接返回 numpy 大数组，跳过 B*F 对象循环；GPU 拉满路径）
output.py            — 三种 raw writer:
                         · write_outputs (legacy, attic-mode)
                         · make_incremental_writer (json/npz, per-sample)
                         · make_h5_writer (chunked HDF5, recommended)
cli.py               — argparse 入口；--incremental + --raw-format h5 走 simulate_arrays + make_h5_writer 快路径
viewer_rerun.py      — rerun web/save/spawn viewer，支持 raw json/npz 加载
dataset_loader.py    — PyTorch Dataset 包装 dataset.h5（v3 新增）
probe_batch.py       — 探测当前 GPU 上最优 batch（v3 新增）
validate_dataset.py  — 检查 npz 数据集完整性（v3 新增）
tests/               — 拓扑稳定性 + 端到端 smoke
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

第三轮（v3）数据管线改造（核心是把"GPU 算完→落盘"做到 GPU 拉满）：

- **I `simulate_arrays()` 快路径**：原来的 `simulate()` 跑完会构造 B×F 个 `FrameSample` dataclass（B=512、F=601 时是 30 万对象，纯 Python 循环 ~20 s）。新增 `simulate_arrays()` 直接返回 numpy 大数组 + vectorized 算 quality/stats，**单 batch 耗时从 ~30 s 回到 ~12 s**
- **J HDF5 输出格式**：替代每样本一个文件的方案。`make_h5_writer` 用 chunked + extendable dataset，每 batch 一次 `dataset[i:i+B] = arr` 完成写入。**写盘几乎零 GIL**（h5py C 库释放 GIL），不再拖累主线程的 GPU kernel launch
- **K 增量写盘**：`--incremental` 让每 batch 跑完立刻 flush，上一 batch 的写盘和下一 batch 的 sim 异步重叠（`ThreadPoolExecutor(max_workers=1)` + 单条 pending future 做 back-pressure）。大数据集不再需要把全部 result 攒在 host RAM
- **L 写盘格式 picker**：`--raw-format {h5,npz,json}` 三档可选，默认 npz；大数据集统一用 h5
- **M topology 单文件共享**：以前每个 raw 样本里都内嵌一份 topology（约 250 KB 重复内容 × N 样本 = 几 GB 浪费）。现在 topology 提到 dataset 根的 `topology.json`，h5 模式下塞进 `dataset.h5` 的 root attrs

---

## 测试

```bash
PYTHONPATH=. python3 tests/test_topology.py   # 拓扑稳定签名 + dedup 验证
PYTHONPATH=. python3 tests/test_smoke.py      # 单样本 CPU 物理合理性
```

---

## 工具脚本（v3 新增）

### `probe_batch.py` —— 探测当前 GPU 上的最优 batch
```bash
python3 probe_batch.py --candidates 1024,512,256 --raw
```
对每个 candidate 跑 1 次 warm-up + 1 次 timed simulate（不写盘），打印 `samples/s` 和 `samples/hour`。`--raw` 让 probe 模拟 `record_particles=True` 的显存占用，避免后面正式跑时 OOM 才发现 batch 不够大。

### `validate_dataset.py` —— 检查 npz 数据集
```bash
python3 validate_dataset.py /path/to/dataset --sample 500
```
对随机抽样的 N 个 npz 文件检查：key 完整性、shape、NaN/Inf、球轨迹连续性、粒子位置 bbox、接触法线归一化、`meta_json` 可解析。也输出 clean ratio + issue 分布 + panel 分布。

> HDF5 数据集的等价检查暂未抽出独立脚本（文件本身的结构通过 h5py schema 强制保证）；可以直接 `dataset_loader.py /path/to/dataset.h5 --clean-only -n 5` 做粗略 spot-check。

---

## 性能

测试机：GTX 1650 SUPER（4 GB VRAM, sm_75）+ Ryzen 12 核 + SSD。

| 配置 | 单样本平摊 | 速率 | 备注 |
|---|---|---|---|
| Warp CUDA, B=512, no-raw | 0.024 s | **~42 /s** | 纯 GPU 计算上限 |
| Warp CUDA, B=512, raw, **h5** | 0.023 s | **~43 /s** | h5 chunked-write 完全异步、不抢 GIL |
| Warp CUDA, B=512, raw, **npz** | 0.062 s | ~16 /s | writer 线程持 GIL 拖住主线程 kernel launch |
| Warp CUDA, B=512, raw, **json** | 1.13 s | ~0.9 /s | JSON 文本编码 + 重复写 topology = 写盘 100× 慢 |
| Warp CUDA, B=128, raw, h5 | 0.07 s | ~14 /s | batch 太小：固定开销摊不薄 |

**结论**：要批量产数据集，固定使用 `--batch 512 --raw --incremental --raw-format h5`，单卡 30000 样本 ≈ 12 分钟、80000 样本 ≈ 32 分钟。

clean ratio 约 55~65%。剩余失败基本是 `severe_penetration` + `stuck`——sampler 偶尔生成极端入射角/源点的样本被网"漏"出去；或球停下后被网某处轻微卡住。训练时按 `quality_clean=True` 过滤即可（PyTorch loader 内置 `clean_only=True` 选项）。

> **物理瓶颈**：sim 本身大约 12 s/batch 是固定的（B 个样本同时跑 601 帧 × 12 substep × 8 iter，所有 panels 都在）。GPU 不写盘时单 batch 也是 12 s，所以 h5 路径本质就是把"写盘做到不让主线程等"，性能上限就是 GPU 算力。8 GB 卡上 batch 拉到 1024+ 还能继续翻倍。

---

## 已知限制

1. **CPU↔Warp 严格数值 parity 未做**：本仓库无 CPU reference solver，跳过设计文档 §11.2 的字节级对齐。物理合理性由 smoke 测试 + 视觉检查保障。
2. **batch 内 ball radius/mass 必须一致**：当前实现假设同批样本球参一致；若需异构需要把这两个量改成 `wp.array(dtype=float, shape=(B,))`。
3. **early-stop frozen mask 之后帧值复制**：被 frozen 的 batch 之后帧 ball_position 不再更新；`len(frames) == duration/frame_dt + 1` 仍恒等。

---

## License

—（按需添加）
