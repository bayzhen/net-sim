# net-sim — 球网 XPBD 仿真 + 神经代理模型

足球射门 → XPBD 球网响应模拟器（Warp GPU）→ 神经代理模型。

**终极目标**：在移动端实时跑出精确且稳定的球 + 球网运动。

完整设计文档：[`goal_net_warp_design.md`](goal_net_warp_design.md)。

---

## 概览

| 项 | 内容 |
|---|---|
| 物理 | XPBD 距离/弯曲约束 + 硬锚 + 2 根支撑绳，514 粒子 / 1872 约束 |
| 碰撞 | 球-粒子离散 / 球-段 swept CCD / 球门柱胶囊 / 草地 5 段反弹 |
| Solver | NVIDIA Warp GPU kernel，单 batch 并行多样本，4090 上 ~43 sample/s |
| 样本 | 默认 601 帧（10 s @ 60 Hz） |
| 训练 | `train`（离线 dataset.h5）+ `train-online`（sim 实时喂 trainer，无落盘） |
| 输出 | HDF5（推荐） / npz / json，三选一 |
| 可视化 | rerun.io web viewer / 本地 GUI / .rrd 离线 |

---

## 环境

- Python ≥ 3.10
- NVIDIA 驱动 ≥ 525（CUDA 11+；Warp 自带 toolkit）
- GPU：4090 跑 batch=512 + raw 完全无压力；4 GB 卡也能跑

```bash
pip install -r requirements.txt
# torch 单独装 CUDA wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

验证：
```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## CLI 速查

```bash
python cli.py <subcommand> [options]
```

子命令：`topology` / `generate` / `train` / `train-online` / `predict` / `view-rerun`。

### `topology`

打印当前参数下拓扑摘要。
```bash
python cli.py topology
```

### `generate` — 生成数据集

```bash
python cli.py generate \
    --count 30000 --batch 512 --device cuda \
    --raw --incremental --raw-format h5 \
    --seed 1 --output /data3/netsim/dataset_v2
```

| flag | 默认 | 含义 |
|---|---|---|
| `--count` | 5 | 总样本数 |
| `--seed` | 1 | sampler 种子 |
| `--batch` | 0 | GPU batch 维（0 = 全部一次跑完） |
| `--device` | cuda | `cpu` / `cuda` |
| `--raw` | off | 同时输出每帧全粒子位置 + 接触序列 |
| `--incremental` | off | 每 batch 跑完立即 flush 到磁盘（大数据集必须开） |
| `--raw-format` | npz | `h5`（推荐） / `npz` / `json` |
| `--output` | `Agent/Temp/...` | 输出目录 |
| `--max-contacts` | 16384 | 每样本接触缓冲上限 |

> 大数据集固定用 `--raw --incremental --raw-format h5`：单文件 chunked HDF5，写盘释放 GIL，GPU 拉满。30000 样本约 12 分钟、~110 GB。

### `train` — 离线训练（dataset.h5 → MLP）

输入 `(球初态 + 归一化时间 t_norm)`，输出 `(球位置, 球速度, 网粒子位置)`。

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

| flag | 默认 | 含义 |
|---|---|---|
| `--dataset` | (必填) | `dataset.h5` 路径 |
| `--output` | (必填) | 训练产物目录 |
| `--epochs` | 100 | 训练轮数 |
| `--batch` | 512 | mini-batch（4090 推荐 4096） |
| `--lr` | 3e-4 | AdamW（cosine 退火） |
| `--lr-min-frac` | 0.01 | cosine 末段 lr / lr |
| `--hidden` | `[1024]*6` | MLP 各层宽度 |
| `--n-time-freq` | 4 | sin/cos 时间编码频率 |
| `--norm-scale-mode` | `robust` | `global` / `init` / `robust`（推荐） |
| `--drop-last-frames` | 5 | 训练时排除最后 K 帧 |
| `--no-preload` | preload | 禁用 RAM 预加载（慢 ~280×） |
| `--w-ball-pos / --w-ball-vel / --w-net` | 1.0 | head loss 权重；推荐 50 / 20 / 1 |

**离线路径瓶颈**：dataset.h5 100+ GB、preload 50 GB 入 RAM；epoch 受限于固定数据量，容量饱和后开始过拟合。突破路径见 `train-online`。

### `train-online` — 在线训练（sim 直接喂 trainer，推荐）

同一张 GPU 上跑 sim → frame pool → train 闭环。每个 train batch 都是新轨迹，从根本上消除 epoch 过拟合，也不需要 dataset.h5。

```bash
python cli.py train-online \
    --output runs/online_v1 \
    --hidden 2048 2048 2048 2048 2048 2048 2048 2048 \
    --total-steps 50000 --refill-every 50 \
    --sim-batch 512 --train-batch 4096 --pool-batches 4 \
    --val-shots 1024 --val-every-refills 10 \
    --norm-scale-mode robust --drop-last-frames 5 \
    --lr-min-frac 0.01 \
    --w-ball-pos 50 --w-ball-vel 20 --w-net 1 \
    --device cuda
```

| flag | 默认 | 含义 |
|---|---|---|
| `--output` | (必填) | 输出目录 |
| `--total-steps` | 50000 | 总 optimizer step |
| `--refill-every` | 50 | 每 N train step 触发一次 sim refill |
| `--warmup-refills` | 4 | 训练前先灌满几个 batch |
| `--val-every-refills` | 10 | 每 M 次 refill 跑一次 val + best.pt 候选 |
| `--sim-batch` | 512 | 每次 simulate_arrays 样本数 |
| `--train-batch` | 4096 | trainer 从 pool 抽的 mini-batch |
| `--pool-batches` | 4 | ring-buffer 容量（K=4, B=512, F=601, N=514 ≈ 7.5 GB RAM） |
| `--val-shots` | 1024 | 验证池目标样本数（一次性生成、不刷新） |
| `--val-seed` | 999001 | 验证 sampler seed（与 train 隔离） |
| `--max-contacts` | 16384 | 同 generate |
| `--smoke` | off | 极小规模联调（~1 分钟跑通） |

模型/优化器相关 flag（`--lr / --hidden / --n-time-freq / --norm-scale-mode / --drop-last-frames / --w-* / --lr-min-frac` 等）含义同 `train`。

**产物**（与 `train` 兼容，可被 `predict` 直接消费）：
```
config.json              # 含 mode="online" 的完整超参
metrics.jsonl            # 每次 val 一条：step / refill / train_running / val / lr / sim_time / train_time
best.pt                  # val total_norm 最低
last.pt
final_val_metrics.json
```

### `predict` — 评估 checkpoint

在 dataset.h5 的 test split 上做全帧评估（输出 per-frame RMSE + worst-K）。

```bash
python cli.py predict \
    --ckpt runs/baseline/best.pt \
    --dataset /data3/netsim/dataset_v2/dataset.h5 \
    --output runs/baseline/eval \
    --device cuda --batch 16 --worst-k 16
```

产物：`per_frame_rmse.json` / `per_sample_summary.json` / `worst_k.json` / `summary.json`。

### `view-rerun` — 可视化

```bash
# 起 web viewer
python cli.py view-rerun /tmp/dataset/raw --serve --bind 0.0.0.0:9090 --public-host localhost
# 浏览器：http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9091%2Fproxy
```

| flag | 含义 |
|---|---|
| `path` | `raw/sample_*.json` 或 `raw/sample_*.npz` 的文件或目录 |
| `--serve` | 起 web server（HTTP + gRPC，PORT 与 PORT+1） |
| `--bind` | 监听地址 |
| `--public-host` | 浏览器可达的 gRPC 主机名 |
| `--save *.rrd` | 离线落盘 |
| `--spawn` | 启动本地 GUI（需桌面） |

> rerun 0.32 的 web viewer 用两个端口（HTTP + gRPC），SSH tunnel 必须同时转发：`ssh -L 9090:localhost:9090 -L 9091:localhost:9091 user@host`。

---

## 数据集 schema（HDF5）

```
attrs:
  schema_version, frame_dt, frame_count, particle_count
  topology_json     # 完整拓扑（粒子/约束/锚定/球门柱）
  metadata_json     # params snapshot
  issue_names       # 7 种 quality issue 名字
  include_raw

datasets (S=样本数, F=601, N=514, K=7):
  sample_id            (S,)         vlen str
  target_panel         (S,)         vlen str    back / left / right / top / corner
  template             (S,)         vlen str
  seed                 (S,)         int64
  input_position       (S, 3)       float32
  input_velocity       (S, 3)       float32
  input_angular        (S, 3)       float32
  input_radius         (S,)         float32
  input_mass           (S,)         float32
  ball_position        (S, F, 3)    float32
  ball_velocity        (S, F, 3)    float32
  particle_position    (S, F, N, 3) float32     # chunked，单样本 ~3.5 MB

  contact_offset       (S+1,)       int64       # CSR
  contact_time         (TotalC,)    float32
  contact_object_type  (TotalC,)    int32       # 0=particle 1=segment 2=segment_swept 3=goalpost 4=crossbar 5=ground_bounce 6=ground_roll
  contact_object_index (TotalC,)    int32
  contact_position     (TotalC, 3)  float32
  contact_normal       (TotalC, 3)  float32
  contact_strength     (TotalC,)    float32

  quality_clean        (S,)         bool
  quality_target_hit   (S,)         bool
  quality_issue_mask   (S, K)       bool
  quality_max_pen      (S,)         float32
  quality_max_pen_time (S,)         float32
  stats_contact_count  (S,)         int32
  stats_max_disp       (S,)         float32
  stats_came_to_rest   (S,)         bool
```

PyTorch 读取（仅离线 `train` 路径需要；`train-online` 不依赖此文件）：
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

`--raw-format npz` / `json` 的产物布局见设计文档 §5。

---

## 物理参数

全部在 `params.py`（dataclass）。覆盖方式：
```bash
python cli.py --params my.json generate ...
```
```json
{
  "solver": {"substeps": 16, "duration": 2.5},
  "rope":   {"collision_radius": 0.18}
}
```

容易踩坑的几个：

| 字段 | 默认 | 说明 |
|---|---|---|
| `rope.collision_radius` | 0.16 | 球-网碰撞膨胀半径（封 0.305 m 网孔） |
| `rope.panel_restitution_back/top/side` | 0.10 / 0.30 / 0.20 | 各 panel 法向恢复 |
| `rope.impulse_speed_threshold` | 1.0 | 球速 < 此值时 swept 不再注能给网（杜绝低速能量泵入） |
| `solver.substeps` | 12 | 必须 ≥ 12 防穿网（与 collision_radius 0.16 配套） |
| `solver.duration` | 2.0 | 捕到落地后滚动 |
| `collision.severe_penetration_threshold` | 0.15 | 球 z < -2.25 的最大穿透阈值 |
| `shape.back_pocket_depth` | 0.30 | 后网中心向 -z 鼓包深度 |
| `shape.stay_count` | 2 | 物理支撑绳数（0 / 2 / 4），后上角脱锚靠它承重 |

完整参数表见设计文档 §3。

---

## 架构

```
params.py            参数 dataclass
topology.py          粒子/约束生成 + 角落 dedupe + ndarray 导出
sampler.py           5 panel × 5 style 射门采样
solver_warp.py       Warp kernels + XpbdWarpSolver
                       simulate()             — 构造 SimulationResult 列表
                       simulate_arrays()      — 直接返回 numpy 大数组（GPU 拉满路径）
output.py            HDF5 / npz / json 三种 writer
online_pool.py       train-online 用的滑窗 frame pool
model.py             GoalNetMLP（带内置归一化 buffer）
train.py             离线 + 在线训练
predict.py           checkpoint 全帧评估
viewer_rerun.py      rerun web/save/spawn viewer
dataset_loader.py    PyTorch Dataset 包装 dataset.h5
cli.py               argparse 入口
tests/               拓扑稳定性 + 端到端 smoke
```

**模拟主循环顺序**（`solver_warp.simulate`，顺序极其重要）：
```
for each frame:
    record frame
    for each substep:
        save_previous
        integrate_particles (gravity + Euler)
        resolve_particle_ground
        integrate_ball
        ball_vs_goalposts (swept CCD)            ← 必须在网 swept 之前
        ball_vs_net_segments (swept CCD + impulse)
        resolve_ball_ground (5 段 bounce/roll)
        for each XPBD iteration:
            solve_anchors (硬锚)
            solve_distance_constraints (stretch, atomic_add)
            solve_distance_constraints (bend)
            solve_ball_particle_collisions
            solve_ball_segment_collisions
        update_velocities ((pos-prev)/dt × (1-damping))
        update_substep_stats (early-stop / stuck / penetration)
```

GS 并行用 **atomic_add Jacobi**（约束并发写）；swept argmin 用**两遍法**（atomic_min(toi) → atomic_min(idx)）。

### 网拓扑

```
crossbar + posts (front, z=0)              [固定锚]
   ├── side panel front edge (u=0) ───┐
   ├── top panel front row (v=0) ─────┤
   ▼                                   ▼
[mesh]                          [back panel top edge]
   │                                   │
   ▼                                   ▼
[back-top corners (un-anchored)] ── stay ropes ── [elevated stay anchors (固定)]
   ▼
[back panel body, pocket-bulged 中部]
   ▼
[back/side y=0 row] ──── [ground anchors (固定)]
```

- 后上 2 个角脱锚 → 靠 2 根物理 stay 绳拉到斜上方锚点（拿掉绳整个后网会塌）
- 后网中部向 -z 鼓 30 cm 形成兜状（`back_pocket_depth`）
- back/side 面 y=0 那一行锚定到地面（防止网底飘）

---

## 测试

```bash
PYTHONPATH=. python tests/test_topology.py   # 拓扑稳定签名 + dedup
PYTHONPATH=. python tests/test_smoke.py      # 单样本 CPU 物理合理性
```

---

## 性能（4090）

| 配置 | 单样本 | 速率 |
|---|---|---|
| `generate`，B=512，`--raw --incremental --raw-format h5` | 0.023 s | ~43 /s |
| `train` 1 epoch（preload 后）| — | ~0.1 s/epoch（GPU 拉满） |
| `train-online` 1 step（refill 期间不算）| — | ~0.05 s/step |
| `train-online` sim refill | 12 s/refill（B=512） | — |

clean ratio 约 55–65%（剩余多为 `severe_penetration` + `stuck`）。训练时按 `quality_clean=True` 过滤。

---

## 已知限制

1. batch 内 ball radius/mass 必须一致（异构需把这两个量改成 `wp.array(B,)`）
2. early-stop frozen 后帧值复制（被 frozen 的 batch 之后帧 ball_position 不再更新；`len(frames)` 仍恒等）
3. 当前最佳 ball_pos RMSE ~0.8 m，**不能直接产品化**（球门 7 m 宽，0.8 m 视觉明显）。突破路径：`train-online` + 数据扩增 / 知识蒸馏到 ≤5M / delta-vs-baseline 目标重设计。详见设计文档 §7。

---

## License

—（按需添加）
