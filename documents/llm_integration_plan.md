# LiDAR UniAD 端到端 + LLM 集成方案

> 讨论记录 / 2026-06-19。基线 config: `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ.py`。
> 目标场景：自有数据集（港口 / 集装箱场），13 类（IGV、Crane、Forklift、Truck 等）。

---

# 第 0 部分：复现操作手册（给后续 agent / 复现者）

> 本节是可照着复现的操作记录。下面的"第 1~4 部分"是设计讨论与决策依据，先看本节能最快上手。

## 0.1 一句话目标

跨模态蒸馏：用离线大 VLM（Qwen2.5-VL-7B）看**环视相机图（6 路，按目标方位自动路由 2-4 路）**，
结合 LiDAR 几何事实，生成中文场景 summary 作为监督；训练一个挂在 UniAD **LiDAR query** 上的小
LLM（Qwen2.5-0.5B），让它**推理时仅凭 LiDAR query** 就能说出图像老师才看得到的语义。推理零图像依赖，
不动现有 LiDAR 链路。（注：早期落地先只用 front，后因后方/侧方作业目标看不见导致 activity 召回偏低，
已改为 6 路环视，见 4.3-B。）

数据流两步走：**几何事实生成**（`gen_geo_facts.py`，从 GT 算出 `info['geo_facts']`，产物 pkl 后缀 `_geo`）
→ **VLM caption 生成**（`gen_vlm_caption.py`，VLM 看图 + 读 geo_facts 出中文 summary，
sidecar `*_vlmcap_summaries.json`、合并后 pkl 后缀 `_vlmcap`）。

## 0.2 环境矩阵（关键：两套 env，因 numpy/torch 不兼容必须分开）

| env | torch | numpy | transformers | 用途 |
|-----|-------|-------|--------------|------|
| `uniad_train` | 1.12.1+cu116 | 1.22.4 | 4.46.3 (+peft 0.13.2) | 训练主环境；几何事实生成、子集抽样、读写 KL pkl、**LLMBridgeHead 训练**（Qwen0.5B）|
| `qwen_vl`（本方案新建） | 2.6.0+cu124 | 2.2.6 | 4.57.6 | 仅跑 VLM caption 的 Qwen2.5-VL 离线推理 |

> uniad_train 的 transformers/peft 是 LLMBridgeHead 训练所需，**后装**（原始环境无）。
> ⚠️ 装时务必 `--no-deps` 锁 torch，否则 accelerate 会强升 torch→2.4 打断 mmcv.ops（见 4.2）。

⚠️ **numpy 跨版本陷阱**：qwen_vl(numpy2.x) 直接 pickle 重写 KL pkl 后，uniad_train(numpy1.x)
会因 `No module named numpy._core` **读不了**。因此 VLM caption 的产物通过 **token→summary 的 JSON sidecar**
交接，绝不让 qwen_vl 重写训练要读的 pkl。

qwen_vl 重建方法：
```bash
conda create -n qwen_vl python=3.10 -y
conda activate qwen_vl
pip install "torch==2.6.0" torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.57.6" accelerate qwen-vl-utils modelscope pillow
```

## 0.3 数据集事实（KL 港口数据集，已核实）

- pkl：`data/kl_8/kl_infos_{train,val}.pkl`（train 43981 帧 / val 5191 帧）。
  容器结构 `{'metainfo':..., 'data_list':[info,...]}`（旧版可能是 `{'infos':[...]}`，脚本两者都兼容）。
- 每个 info 关键字段：`token`(唯一)、`scene_token`(='<scene>/<record>')、`timestamp`、
  `lidar_points.lidar_path`、`instances`(每个含 `bbox_3d=[x,y,z,l,w,h,yaw,vx,vy]`、`velocity`、
  `track_id`、`gt_fut_traj_locs[12,2]`、`gt_track_traj_locs[8,2]`)、`gt_sdc_bbox`、
  `gt_sdc_fut_traj[1,12,2]`、`command`、`sync_info`。
- **相机**：磁盘有 6 路环视去畸变图（front/left_front/left_rear/rear/right_front/right_rear），
  路径 `data/kl_8/v1.0-trainval/sample/<scene_token>/camera_undist/<view>_image/<ts>.jpg`，
  ~87 万张。**但原始 pkl 未关联相机**（建库 cfg `projects/KL8/configs/kl8_lidar_bevformer.py`
  里 `camera_processing_cfg.enable=False`）。本方案先只用 **front**。
- 13 类（post label_mapping）：0 行人 / 1 小车 / 2 满载IGV / 3 卡车 / 4 空挂车 / 5 满载挂车 /
  6 空载IGV / 7 吊机 / 8 其他车辆 / 9 锥桶 / 10 集装箱叉车 / 11 叉车 / 12 轮胎吊。
- `gt_fut_traj_locs` / `gt_track_traj_locs` 约定（已验证）：**相对自身当前位置的累积位移**，
  LiDAR 帧，步长 0.5s。agent 未来绝对 xy = `bbox_xy + fut_traj[t]`；ego 在原点，未来绝对 xy = `sdc_fut_traj[t]`。

## 0.4 完整数据流水线（脚本 + 命令 + 产物）

所有脚本在 `tools/data_converter/`，都是**增量后处理**（不动原 pkl 其它字段），模仿既有的 `add_sdc.py`。
工作目录均为 `UniAD_train/UniAD/`。

**Step 1 — 图像↔LiDAR 同步**（env: uniad_train）
`add_cam_sync.py`：按 `scene_token` 定位相机目录，对每个 LiDAR 帧做时间戳最近邻匹配，
写 `info['sync_info']['cameras'][<CAM_*>]`（含 path/dt/valid）。容差 50ms（对齐建库 cfg）。
⚠️ `--views` **默认只有 front**；6 路环视必须显式传全部 6 个磁盘视角名。
```bash
python tools/data_converter/add_cam_sync.py \
  --pkl-path data/kl_8/kl_infos_train.pkl data/kl_8/kl_infos_val.pkl \
  --views front left_front left_rear rear right_front right_rear
# 磁盘视角名 -> CAM_*: front=CAM_FRONT, left_front=CAM_FRONT_LEFT, left_rear=CAM_BACK_LEFT,
#   rear=CAM_BACK, right_front=CAM_FRONT_RIGHT, right_rear=CAM_BACK_RIGHT
# 产物（保留副本，不覆盖原 pkl）: data/kl_8/kl_infos_{train,val}_with_cam.pkl
# 命中率(front): train 98.58% / val 98.19%（<50ms）; 缺帧标 valid=False
```

**Step 2 — 几何事实生成**（env: uniad_train）
`gen_geo_facts.py`：从 GT 算每帧 `info['geo_facts']`（per-agent: range/bearing/motion/heading/
load/conflict+ttc/activity_gate；scene: agents_of_interest/ego_advice/congestion/crane_status），
并 dump 静止时长 / 吊机距离分布（用于定 activity 阈值）。
```bash
python tools/data_converter/gen_geo_facts.py \
  --pkl-path data/kl_8/kl_infos_val_with_cam.pkl   # 产物: *_with_cam_geo.pkl
# 关键参数（已数据校准）: --conflict-dist 4.0 --gate-static-s 2.0 --gate-crane-m 30.0
```

**Step 3 —（可选）分层子集**（env: uniad_train）
`make_subset.py`：稀有场景（queue/conflict/activity_gate）全保留 + 普通帧补足到 --size，
固定 seed 可复现。用于快速迭代 VLM caption / LLMBridgeHead。
```bash
python tools/data_converter/make_subset.py \
  --pkl-path data/kl_8/kl_infos_val_with_cam_geo.pkl \
  --out-path data/kl_8/kl_infos_val_sub1k_geo.pkl --size 1500
```

**Step 4 — VLM caption 生成**（env: qwen_vl）
`gen_vlm_caption.py`：读 geo_facts + 渲染中文事实约束文本 + **按目标方位自动路由的 2-4 路环视图**
（`_resolve_views` 按帧内 AOI/conflict/gate 目标方位选相机，每图标【方位相机】）→ VLM → 一句中文 summary。
**关键产物是 JSON sidecar**（`<pkl去后缀>_summaries.json`，token→summary），pkl 输出可丢 /tmp。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/data_converter/gen_vlm_caption.py \
  --pkl-path data/kl_8/kl_infos_val_sub6cam_geo.pkl \
  --model-path /mnt/disk1/models/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --out-path /tmp/val_sub6cam_vlmcap.pkl
# 真正要用的产物: /tmp/val_sub6cam_vlmcap_summaries.json
# 速度 ~1.4-1.8 帧/s（单卡 4090，多相机约 3.4 路/帧）；--limit N 抽样；--dry-run 只渲染 prompt 不加载模型
# 输入 pkl 须已含 6 路 sync_info（add_cam_sync 用 6 路 --views 生成）；无则只路由到 front。
# train 全量请用 7 卡分片：bash tools/run_vlm_caption_train_7gpu.sh（见第 8 步），
#   --num-shards N --shard-id i 把数据切 N 份并行，各写 *_summaries.shard{i}ofN.json
```

**Step 5 — summary 合并回训练 pkl**（env: uniad_train）
`merge_summaries.py`：读 JSON sidecar，按 token 把 summary 写进 pkl 的
`info['geo_facts']['summary']`，输出训练用 pkl（numpy1.x 写，训练可读）。
```bash
python tools/data_converter/merge_summaries.py \
  --pkl-path data/kl_8/kl_infos_val_sub1k_geo.pkl \
  --json-path /tmp/sub1k_vlmcap_summaries.json \
  --out-path data/kl_8/kl_infos_val_sub1k_vlmcap.pkl
# 产物: kl_infos_val_sub1k_vlmcap.pkl（1500 帧, 1466 带 summary, uniad_train 可读）
```

## 0.5 模型与磁盘

- Qwen2.5-VL-7B-Instruct：modelscope 下载到 **`/mnt/disk1/models/Qwen2.5-VL-7B-Instruct`**（16G，5 分片）。
  ⚠️ 系统盘 `/` 仅剩 ~46G，模型与缓存**必须放 disk1**（`export MODELSCOPE_CACHE=/mnt/disk1/...`）。
  下载命令：`modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir /mnt/disk1/models/Qwen2.5-VL-7B-Instruct`
- GPU：本机 8×RTX4090(24G)；7B 单卡可跑。用空闲卡（如 GPU0），勿占训练卡。

## 0.6 已知问题 / 待办

- **流水线 Step 1~5 + LLMBridgeHead 已全跑通**（截至 2026-06-22）：
  - 数据：6 路环视 val 子集 **v3** `data/kl_8/kl_infos_val_sub6cam_v3_vlmcap.pkl`（805 帧, 797 带 summary；
    全为稀有帧，gate 已排除非作业类）。v1/v2 子集与旧单 front `kl_infos_val_sub1k_vlmcap.pkl` 仅作历史对比。
    ⚠️ **子集帧数 ≠ 实际 eval queue 数**：make_subset 按稀有性挑帧，prev/next 可能落在子集外，
    时序 track eval 只能用能组成默认 5 帧 queue 的帧（v3 的 805 帧实际约 438 帧可 eval；
    旧 sub1k 1500→约 542）。teacher caption 评测是逐帧的、用全部 805；但跑 dist_eval 的 track/motion
    指标会少于子集帧数。caption 蒸馏训练本身不受影响（caption 是逐帧监督）。
  - 几何事实：**train/val 全量均已是 6 路 + gate 清理** `kl_infos_{train,val}_with_cam_geo.pkl`
    （train 43981 帧，6 路各 valid 98.45%~98.61%；2026-06-22 重做并排除非作业类 gate）。
  - **train 全量 caption 已完成**：7 卡分片跑 ~2.2h → merge → `kl_infos_train_vlmcap.pkl`
    （43747/43981=99.5% 带 summary）。
  - 模型：`LLMBridgeHead` + config `base_e2e_lidar_occ_llm.py` smoke test 通过
    （caption 注入、llm.loss_llm 有限、与 track/motion/occ 共存、backward OK）。详见第 7 步。
  - 评估：`eval_llm_caption.py` template/teacher 可跑；activity 拆为 addressed/busy（见 4.3-B）。
    6 路 + gate 入 prompt + 排除非作业类后 teacher activity_addressed **74.5%**（单 front 17.0%）。
- **正式训练待办**（第 8 步）：
  - (0) ✅ train pkl 已重做 6 相机 + gate 清理（2026-06-22）。
  - (1) ✅ 7 卡分片 caption 已完成（~2.2h，43747 帧带 summary）。
  - (2) ✅ merge → `kl_infos_train_vlmcap.pkl`；正式 config `base_e2e_lidar_occ_llm_train.py`
    （train→该 pkl，val/test→v3 子集）已建并验证。
  - (3) ⏳ **待跑正式训练**：`uniad_dist_train.sh base_e2e_lidar_occ_llm_train.py <GPUS>`。详见 4.3-A。
- 文件清单（本方案新增）：
  `tools/data_converter/{add_cam_sync,gen_geo_facts,make_subset,gen_vlm_caption,merge_summaries}.py`、
  `tools/run_vlm_caption_train_7gpu.sh`（7 卡并行 VLM caption）、
  `tools/analysis_tools/eval_llm_caption.py`（caption 质量评测）、
  `projects/mmdet3d_plugin/uniad/dense_heads/llm_bridge_head.py`、
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm.py`（smoke）、
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train.py`（正式训练）、本文档。

---

## 1. 当前基线方案（base_e2e_lidar_occ）

纯 LiDAR 的 UniAD 端到端，**没有相机分支**。继承链：

```
base_track_lidar  →  base_track_drivable_lidar  →  base_e2e_lidar  →  base_e2e_lidar_occ
```

数据流（detector `UniADMotionLidar`，继承 `UniADTrackLidar`）：

| 阶段 | 模块 | 产出 | 备注 |
|------|------|------|------|
| 1. BEV | LiDAR backbone + BEV encoder | BEV 特征 120×160 | stage2 中 `freeze_lidar_backbone=True` / `freeze_bev_encoder=True`，BEV 冻结 |
| 2. Track | `forward_track_train` | `track_query` / 检测框 | 13 类港口类别 |
| 3. Seg ("map") | `LidarDrivableHead` | drivable stuff Dice mask | GT 来自 LiDAR 几何 + raycast 地面（`GenerateKLDrivableMapLabels`, `use_map=False`），**不是真 HD-map** |
| 4. Motion | `MotionHeadLidar` (MotionFormer) | 每 agent 6 模态 × 12 步轨迹 | 可吃 `outs_map` 的 HD-map lane prior（`MapLaneEncoder`），occ 配置当前未启用 |
| 5. Occ | `OccHead` | 4 帧未来占据 | FIERY 风格 BinarySeg + Dice，仅 vehicle 类 |
| 6. Plan | `PlanningHead` | — | 模块存在，本配置未启用 |

坐标范围 `point_cloud_range = [-64, -48, -2, 64, 48, 6]`，BEV `120×160`，embed_dims `256`。

关键 query 形状（集成 LLM 的接入点）：
- `outs_track['track_query_embeddings']`：`[num_tracks, 256]`
- `outs_motion['track_query']`：`[1, num_agents, 256]`
- `outs_motion['traj_query']`：`[num_layers, 1, num_agents, num_anchor=6, 256]`
- SDC(ego) query 作为 tail slot 拼在 track_query 末尾，loss 后再 split 出来

## 2. LLM 入手点全景（讨论备忘）

按改动成本 / 收益排序：

1. **query → LLM 高层推理 / VQA**（选定方向）。把 object-centric query 投到 LLM token 空间，输出场景级语义。不动感知主干，加一个 head。类比 DriveGPT4 / DriveVLM，但用 LiDAR query 而非图像 token。
2. **语言驱动的 map prior**。把调度指令 / 作业规则 / 区域语义编码成 query 注入 `outs_map`，喂 MotionFormer 的 MapInteraction，替代几何 raycast 的 lane prior。接口现成（`_build_outs_map` 返回 `lane_query`）。
3. **语言条件 Planning**。PlanningHead 现吃离散 `command`；换成 LLM 把自然语言调度指令编码成 planning 条件向量。港口调度本就是文本，业务价值最高。
4. **离线 LLM 数据/诊断**。VLM 自动场景描述、长尾挖掘、失败帧可读诊断。不进推理图，工程风险最低。

## 3. 选定方向 1：query → LLM，跨模态蒸馏（VLM 老师 + 纯 LiDAR 推理）

> ⚠️ 本节是 2026-06-19 的设计讨论记录，文中多处"先只用 front"是**初始决策**。
> **当前实现已改为 6 路环视**（按目标方位自动路由 2-4 路），活动召回翻倍，见 4.3-B。
> 读本节请把 front-only 字样理解为历史阶段。

### 3.0 定调（2026-06-19 讨论确定）

**一句话**：用离线大 VLM 看相机图（6 路环视，按目标方位自动路由；**2026-06-19 初始落地先只用
front，后已改 6 路环视，见 4.3-B**），
生成"图像才看得到的语义"作为监督；
让挂在 UniAD LiDAR query 上的小 LLM 学会 **只凭 LiDAR 几何推出图像老师才能看到的东西**。

- **推理时纯 LiDAR**，零图像依赖 —— 不破坏现有 LiDAR 链路与 Orin/TensorRT 部署路线。
- 图像只在 **离线生成监督** 时作为"老师"使用，不进模型、不进推理图。
- 学到的不是"把 query 念出来"（感知头已有），而是 **跨模态语义补全**：
  LiDAR 看到一个 ~14m 箱体在动，LLM 要推出"满载 IGV、正驶向泊位"，
  而"满载/作业意图"这种标签来自图像老师。

### 3.0.1 数据集事实（已核实，修正早前判断）

> 早前"纯 LiDAR、无图像"的判断**错误**。实际：

- 磁盘上 **有海量相机图**：`data/kl_8/v1.0-trainval/sample/<scene>/<record>/camera_undist/<view>/*.jpg`，
  6 路环视（front / left_front / left_rear / rear / right_front / right_rear），已去畸变，
  带 `intrinsics.json` / `camera_extrinsics.json`，总计 ~87 万张。
- **但** annotation pkl（`kl_infos_{train,val}.pkl`，5191 帧）**未关联图像**：
  `sync_info` 只有 `label / lidars / localization`，**无 cameras**；`cam_instances` 字段全空；
  dataset `get_data_info` 只读 `lidar_path`。
- ⇒ **前置工程**：需自己做 图像↔LiDAR 帧 时间同步（LiDAR 时间戳 → 每路相机最近邻帧），
  把相机路径补进 pkl 或 sidecar。此工作顺带为将来"LiDAR+相机多模态 UniAD"铺路。

### 3.0.2 同步可行性（已抽样核实，2026-06-19）

> 实测后结论：同步几乎是免费的，关键帧本就对齐。

- 相机 front 与 LiDAR **同为 2Hz（500ms）**，每个 record 帧数一一对应。
- 目录映射通用：`sample/<scene_token>/camera_undist/front_image/<ts>.jpg`，
  **584/584 record 全部存在 front 目录，0 缺失**。
- 最近邻时间 gap：全 train 43981 帧，**median 35.8ms、p95 45.5ms**，
  这 ~33ms 是固定的传感器触发相位差，不是漂移。
- 容差定 **50ms**：98.58% 帧命中；放宽到 500ms 也只到 99.71%
  （剩 ~1.4%≈625 帧是相机整段丢帧，gap 几百 ms~4.5s，放宽也救不回 ⇒ 直接标"无图"，VLM 跳过）。
- ⇒ 同步脚本很轻：按 `scene_token` 定位目录 + 时间戳最近邻 + 50ms 阈值。
- **决策（已更新）**：~~先只用 front 单视角~~ — 初期为省算力先用 front，验证后发现后方/侧方作业
  目标看不见、activity 召回偏低，**已改 6 路环视、按目标方位路由 2-4 路**（见 4.3-B）。
- **VLM 老师**：Qwen2.5-VL（中文/工业场景友好）；本地 8×RTX4090(24G)，7B 单卡可跑，本地批量离线。
- **同步信息存储**：写进 pkl（增量加字段，dataset 直接可读；用户已确认可改 pkl）。

### 3.1 模型侧设计

新增一个 **LLM bridge head**，与 motion/occ 并列，挂在 detector 上：

```
track_query [num_tracks,256]  ┐
motion traj_query             ├─► Projector (MLP, 256 → d_llm) ─► [N object tokens]
(+ box 中心做空间编码)         ┘                                      │
                                                                     ▼
                          [prompt tokens] ⊕ [object tokens] ─► LLM ─► 文本(老师标签)
```

- **底座**：Qwen2.5-0.5B（d_llm=896），中文友好、单卡可 LoRA、最有上车可能（虽暂不上车）。
- **Projector**：小 MLP，256 → 896，每个 agent 一个 token。
- **空间编码**：box 中心 (x,y,z)（来自 `track_bbox_results[0][0].gravity_center`）→ MLP → 加到 token，给 LLM 空间先验。
- **训练**：冻结 LLM 主干（BEV 也已冻结），只训 Projector + LoRA。LM loss(teacher forcing)。
- **token 数**：agent 数不定，截断/padding 到 `max_agents`，截断要 log。


### 3.2 监督数据：几何模板 + VLM 图像（互补）

| 路线 | 来源 | 内容 | 前置成本 |
|------|------|------|----------|
| **几何模板** | 现有 GT 框/轨迹合成 | 确定性事实：方位/距离、运动状态、ego×agent 轨迹冲突/让行、是否在可行驶区 | 低，直接可做 |
| **VLM 图像**（主线） | Qwen2.5-VL 看 front 图 | 作业状态、满载/空载（粗）、吊机工作/空闲 | 中，需先做图像↔LiDAR 同步（已验证轻量） |

- 最终监督 = 几何精确空间/冲突事实 ⊕ VLM 场景语义，合并成每帧的 caption。
- VLM caption 生成时让模型同时吃几何步骤的结构化事实文本作为 prompt 约束，减少幻觉、对齐到 ego 视角。
- ⚠️ 警惕"念 query"：监督价值在 **跨 agent / 跨时间 / 跨模态** 的推理（冲突、意图、作业状态），
  而非"有 3 台 IGV"这种感知头已给的清单。

#### 3.2.1 可学语义边界（2026-06-19 锁定）

标尺：推理时 LLM 只有 ① box 形状/尺寸/类别 ② track 速度/轨迹/朝向 ③ agent 间 + agent↔ego + agent↔可行驶区 的空间关系。能从这三样推出来的才教。

**✅ 教**（LiDAR query 有望推断）

| 语义 | 推理依据 | 来源 |
|------|----------|------|
| 物体类别清单 | 类别已在 track | 几何（仅 caption 底料，非增值） |
| 空间布局（方位/距离/区域） | box 中心 + ego | 几何 |
| 运动状态（静止/行驶/速度档） | track 速度/轨迹 | 几何 |
| 朝向 / 转向意图 | 朝向 + fut_traj 曲率 | 几何 |
| 满载 / 空载（**粗粒度**：有箱/空载/NA） | IGV/Trailer 已分 Full/Empty + 箱体高度 | 几何+VLM |
| **轨迹冲突 / 让行关系** | ego fut × agent fut 相交 | 几何（核心增值） |
| **作业 vs 通行 状态** | 几何门控（久静止+贴吊机/在作业区）→ VLM 才允许标装卸 | 几何门控+VLM |
| 吊机工作 / 空闲 | Crane 静止 + 下方 IGV/Trailer 停靠 | 几何+VLM |
| 拥堵 / 排队 | 多 agent 低速聚集 | 几何 |
| 是否阻塞 ego 路径 | agent 落在 ego fut 走廊内 | 几何 |

**❌ 不教**（图像独有 / LiDAR 无望，学了=幻觉）：颜色、箱号文字、吊具锁扣/灯光等精细部件、
人的姿态/反光衣、地面标线内容、天气/光照、品牌型号、堆叠层数/具体 TEU（细粒度载货）。

**两条灰线裁决**：① 作业状态——教，但 几何门控在先，无几何依据不标。
② 载货——只教粗粒度（有箱/空载/NA），不教层数。

#### 3.2.2 Caption schema 与生成策略（2026-06-19 锁定）

- **训练目标**：仅 **scene-level `summary`** 自由文本（一句/几句中文）。
  per-agent 结构化字段暂不作为输出目标（将来要 grounding 再加）。
- **生成方式 (a)**：几何模板拼几何事实 → Qwen2.5-VL 看 front 图 **改写润色 + 补作业/载货语义** →
  自然中文 summary。VLM 只负责"看图补语义 + 说人话"，几何事实由几何步骤保证准确，防幻觉。
- **scene-level 字段**（几何步骤先算，作为 VLM 的 prompt 约束 + summary 的事实底料）：
  `agents_of_interest`（对 ego 最相关的 id）、`ego_advice`（keep/yield/slow/stop，由冲突导出）、
  `congestion`、`crane_status`、以及每个关键 agent 的 pos/motion/heading/load/conflict。
- **作业状态门控阈值**：**不预设，数据驱动**。
  - 门控只做"宽松预筛"——挡掉几何上绝无可能作业的 agent（如高速行驶中），
    最终"是否装卸"由 VLM 看图裁决。故门控应**故意放松**。
  - 几何生成器先 **dump 分布**（每个 agent 的静止时长、与最近 Crane 的距离），
    看真实直方图再定阈值；阈值写成 config 可调参数，不写死。
  - 宽松默认（待数据校准）：静止 >2s 且距最近 Crane <30m ⇒ "允许"标作业。


### 3.3 落点（代码）

- 新 detector head：`llm_bridge_head.py`（`LLMBridgeHead`，`@HEADS.register_module()`）。
- detector 接线：`uniad_motion_lidar.py` 的 `forward_train` / `simple_test`，motion 之后取
  `outs_motion['track_query']` + `outs_track`（注意 `simple_test` 里要在剥离
  `track_bbox_results`/`track_query_embeddings` 的 cleanup **之前** 调用）。
- config：新建 `base_e2e_lidar_occ_llm.py` 继承 `base_e2e_lidar_occ`，加 `llm_head` 与
  `task_loss_weight['llm']`。caption 不走 pipeline 的 Collect key，而是数据集
  （`kl_dataset.py` `_union2one`）把 `gt_caption` 注入当前帧的 **img_metas**，detector
  （`uniad_motion_lidar.py` `_current_caption`）从 img_metas 取出喂 llm_head。
  - **为什么继承 `base_e2e_lidar_occ`**：这是继承链
    `base_track_lidar`(track) → `base_track_drivable_lidar`(+map seg) → `base_e2e_lidar`(+motion，
    冻结 BEV) → `base_e2e_lidar_occ`(+occ) 的**顶端、感知最全的基线**。LLMBridgeHead 吃
    `outs_motion['track_query']` + box center，监督信号 `geo_facts` 来自多 agent 轨迹+占据语义，
    所以要站在 track+map+motion+occ 都齐全的模型上 query 才有料；继承更低层会缺 motion/occ、
    head 直接拿不到输入。继承它还自动复用 stage2 的冻结策略（`freeze_lidar_backbone/bev_encoder`）。
    注：occ config 未启用 planning，故 LLM 站在 track+map+motion+occ 之上、不含 plan。
  - **感知底座 ckpt**：正式 config `_train` 的 `load_from` 指向
    `base_e2e_lidar_occ/latest.pth`(=epoch_4)——已训练收敛的端到端权重（occ.loss_dice 0.99→0.14、
    motion.loss_traj 0.55→0.44，5 epoch）。该 ckpt 用旧 `kl_infos_train.pkl`(单front)训，但只提供
    点云感知主干（与相机/caption/gate 无关），LLM 训练时全部冻结，故配新 caption pkl 自洽。
    （smoke config 用 stage-1 drivable ckpt 仅为快速验证 llm forward/loss。）
- 数据：图像同步脚本（补相机路径）+ 几何模板生成 + VLM caption 生成，产出每帧 caption，存入 pkl/sidecar。

### 3.4 训练策略

- 阶段化：感知全冻结（BEV 已冻结），只训 Projector + LoRA，保护感知不被语言 loss 破坏。
- loss：LM loss 经 `loss_weighted_and_prefixed(prefix='llm')` 与现有 task loss 合并。
- 验证：先小规模（少量帧 + 0.5B）跑通前向 + loss 下降，再扩。

### 3.5 待定 / 风险

- **图像↔LiDAR 同步质量**：时间最近邻 + 标定，需抽检对齐误差；港口低速，容差相对宽松。
- **VLM 老师质量/成本**：大 VLM 选型、是否本地可跑、生成 ~5191 帧×6 视角的算力与时间。
- **幻觉**：VLM 对港口专有物体（IGV/吊具）可能识别不准；用几何事实约束 + 抽检。
- **部署**：推理纯 LiDAR，LLM 分支默认不上车；若要上车再议量化。
- **token 数**：agent 数不定，截断/padding。

## 4. 进度与下一步

### 4.1 已完成（细节见第 0 部分复现手册 / 第 3 节设计）

1. ✅ LLM 底座选定：Qwen2.5-0.5B（d_llm=896）。
2. ✅ 方案定调：VLM 图像老师 + 纯 LiDAR 推理（跨模态蒸馏）；几何模板互补。
3. ✅ token 边界 + caption schema（scene-level summary）锁定。见 3.2.1 / 3.2.2。
4. ✅ 图像↔LiDAR 同步：`add_cam_sync.py` → `kl_infos_{train,val}_with_cam.pkl`
   （front 命中 train 98.58% / val 98.19%）。**train/val 均已重做成 6 路**（2026-06-22）。见 0.4 Step 1 / 4.3-A。
5. ✅ 几何事实生成：`gen_geo_facts.py` → `..._with_cam_geo.pkl`（train/val 全量）。
   阈值数据化校准（见 0.4 Step 2 与 4.2 经验）。前视 gate 收紧的弯路已回退（见 4.3-B）。
6. ✅ VLM caption 多相机化：`gen_vlm_caption.py`，prompt 设计 + 按方位路由 2-4 路环视 +
   6 路 val 子集批量生成（805 帧，797 带 summary）。teacher activity_addressed 17%→74.5%（见 4.3-B）。
   见 0.4 Step 4 与 4.3-B。
7. ✅ `LLMBridgeHead` + config + smoke test（2026-06-22）：caption 注入正确、
   llm.loss_llm 有限（≈5.2~5.7）、与 track/motion/occ 共存、backward OK。
   实现细节见 3.1 / 3.3；代码 `llm_bridge_head.py`、`uniad_motion_lidar.py`、
   `kl_dataset.py`、config `base_e2e_lidar_occ_llm.py`。
8. ✅ train 全量 caption（2026-06-22）：7 卡分片 ~2.2h（错开启动避 OOM，见 4.2）→ merge →
   `kl_infos_train_vlmcap.pkl`（43747/43981=99.5% 带 summary）。正式 config
   `base_e2e_lidar_occ_llm_train.py` 已建并验证（train→该 pkl，val/test→v3 子集）。
   **单卡 smoke 验证通过**（2026-06-22）：occ ckpt 正确加载（occ.loss_dice 一上来即 0.14 收敛值）、
   `llm.loss_llm` 有限且下降（2.73→1.70）、track/map/motion/occ/llm 全 task 共存、forward+backward
   无报错。**下一步：手动起多卡正式训练**（4.3-A (4)）。

### 4.2 经验教训（踩过的坑，复现必读）

- **bf16 强制**（训练）：Qwen0.5B 在 fp16 下 LM loss=nan，必须 bf16（4090 支持）或 fp32。
  projector/spatial_pe 在 fp32 算、输出再 cast 到 bf16 喂 LLM（否则 Linear dtype 报错）。
- **装包别动 torch**：pip 装 transformers 时 accelerate 会强升 torch→2.4，打断 mmcv.ops
  CUDA 扩展。装 LLM 依赖用 `transformers==4.46.3 peft==0.13.2`，并 `--no-deps` 锁
  `torch==1.12.1+cu116 torchvision==0.13.1+cu116`。
- **numpy 跨版本交接**：qwen_vl(numpy2.x) 不能重写训练 pkl（uniad_train numpy1.x 读不了），
  故 VLM caption 只导出 `*_summaries.json`、训练侧 merge。见 0.2。
- **conflict 锥桶修正**（港口特性）：初版 conflict=轨迹最近距<4m，冲突目标多为静止锥桶，
  VLM 输出"向锥桶让行"。锥桶实为路边禁入区标记、非避让物 ⇒ 把 Cone(cls=9) 加入
  `_NO_CONFLICT_IDS` 跳过。效果：conflict 帧 5.8%→1.6%，目标变为行人/空载IGV 等真动态。
- **congestion 修正**：原"30m内≥3 非移动"误判停车场为排队(66%)；改"只数 moving_slow 且≥4"
  → 1.5%（真缓行队列）。
- **CONFLICT caption 退化**：初版冲突场景退化成套话"前方有障碍物请减速"；prompt 强令点明
  目标(方位+类别)+TTC+建议、禁用"障碍物"泛称、加 few-shot 后具体化。
- **activity 门控数据化**：dump 分布后定 `static≥2s & crane≤30m`（通过率 3.0%，宽松预筛，
  VLM 看图终判），不写死阈值。
- **7 卡 caption 同时启动 OOM**：7 个进程零间隔 fork、各自瞬间加载 16G 模型，分配器峰值碰撞
  导致全部 OOM（`CUDA_VISIBLE_DEVICES` 隔离本身没问题——单跑/错开跑都正常）。修法：分片间
  `sleep` 错开启动 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；脚本加每分片启动自检
  （崩了立即报错中止）+ 前台 `tail -f` 实时观察。见 `run_vlm_caption_train_7gpu.sh`。

### 4.3 下一步（待办）

**A. 正式训练**。(0)(1)(2) 均已于 2026-06-22 完成，命令留作复现记录；当前只剩 (4) 跑训练。
   ```bash
   # (0) ✅ 已完成：train pkl 重做成 6 相机 + gate 排除非作业类。下面两条是当时所跑（留作记录）。
   #     ⚠️ add_cam_sync 的 --views 默认只有 front，6 路必须显式传全部 6 个磁盘视角名。
   #     当前 kl_infos_train_with_cam_geo.pkl 已是 6 路+gate清理，可跳过直接到 (4)。
   python tools/data_converter/add_cam_sync.py \
     --pkl-path data/kl_8/kl_infos_train.pkl \
     --out-path data/kl_8/kl_infos_train_with_cam.pkl \
     --views front left_front left_rear rear right_front right_rear
   python tools/data_converter/gen_geo_facts.py \
     --pkl-path data/kl_8/kl_infos_train_with_cam.pkl \
     --conflict-dist 4.0 --gate-static-s 2.0 --gate-crane-m 30.0
   #     -> data/kl_8/kl_infos_train_with_cam_geo.pkl（含 6 路 + geo_facts）
   # (1) ✅ 已完成：7 卡并行 train 全量 caption（~2.2h；占 GPU1-7，留 GPU0）。
   #     脚本错开启动避 OOM（见 4.2），无需 --views（相机自动路由）。
   bash tools/run_vlm_caption_train_7gpu.sh
   # (2) ✅ 已完成：合并 7 片 sidecar 回 train pkl（43747/43981=99.5% 带 summary）。
   python tools/data_converter/merge_summaries.py \
     --pkl-path data/kl_8/kl_infos_train_with_cam_geo.pkl \
     --json-path /tmp/train_vlmcap_summaries.shard*of7.json \
     --out-path data/kl_8/kl_infos_train_vlmcap.pkl
   # (3) ✅ 正式 config base_e2e_lidar_occ_llm_train.py 已建并验证
   #     （train→kl_infos_train_vlmcap.pkl, val/test→kl_infos_val_sub6cam_v3_vlmcap.pkl）。
   #     smoke config base_e2e_lidar_occ_llm.py 不动。
   # (4) ⏳ 待跑：正式训练
   ./tools/uniad_dist_train.sh \
     projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train.py <GPUS>
   ```
   注意：(0)~(3) 已完成，只剩 (4)。正式训练用 `_train` config，smoke config 保持只指向 val 子集不动。

**B. 评估指标**（已落地 `tools/analysis_tools/eval_llm_caption.py`）：
   规则解析 caption → 关键语义命中率（conflict 类别/方位、TTC 桶、ego_advice、activity、幻觉），
   主对 geo_facts。baseline：`template`（几何直出，floor）/`teacher`（VLM summary，蒸馏上限）
   已可跑；`model`/`shuffle`/`noquery` 待训练出 checkpoint 后接（forward_test 逐帧）。
   - ⚠️ **走过的弯路（已纠正，复现必读）**：曾以为 activity 召回低的根因是"门控把后方目标也
     标了、前视相机看不到"，于是把 activity_gate **收紧到只对前视扇区**（_FRONT_SECTORS +
     gate_visible_m）。方向错了——正解不是"剔掉后方目标"，而是**上 6 路环视让后方目标可见**。
     该收紧逻辑已从 `gen_geo_facts.py` **回退**，activity_gate 恢复全方位（static≥2s & crane≤30m）。
   - ⚠️ **评测指标 bug（已修，2026-06-22）**：旧 `activity` 指标用 `pred==gt['has_activity']`，
     而 `pred` 只匹配"作业/装卸"两词，**把 VLM 正确判出的"空闲/等待"当成漏报**。门控本意是
     三选一（装卸/等待/空闲），"空闲"是有效判断。旧 14.3%~15.6% 严重低估了真实效果。
     已拆成两个指标：`activity_addressed`（gated 帧是否给出三态之一，衡量"VLM 有没有看图回应
     门控"）/ `activity_busy`（其中判"装卸作业"的比例，是 rate 不是命中分）。
   - ✅ **多相机实测（val 子集 teacher）**：单 front → 6 路环视，`activity_addressed`
     **17.0% → 37.3% 翻倍**（gated 分母均=777）。根因证实：单 front 时后方/侧方 gate 目标
     （35-46m 外的吊机居多）看不到只能照搬几何；6 路按目标方位路由 CAM_BACK 后 VLM 才能看图判断。
     连带 conflict_cls 75%→88%、conflict_bearing 75%→84%、ttc 59%→72% 均涨（侧后方冲突有相机佐证）。
     `activity_busy` 92%(n=132)→38%(n=290) 下降是**更真实**：单 front 偏向只说近前方明显在作业的，
     6 路纳入大量后方空闲吊机，VLM 如实判"空闲" ⇒ busy 占比降但 addressed 大涨，正是要的效果。
     幻觉(无幻觉率) 98%→96% 基本持平。
   - ✅ **同帧严格对比（已核实）**：6cam 子集(911)与旧单front 子集(1500)的 gated 帧
     **交集=777、各自独有=0**（两次 make_subset 同 seed，稀有帧确定性全保留），
     即 17.0%→37.3% 是同一批 777 帧上的对比，非近似。6cam(911) 整体亦是 front(1500) 的子集。
   - ⚠️ **路由/渲染/评测口径不一致（已修，2026-06-22）**：`_resolve_views` 按**全部 gate 目标**
     方位选相机，但 `_render_facts` 原本**只渲染 AOI 目标**，导致非 AOI 的 gate 目标"取了图却没在
     prompt 点名"，VLM 没被要求判它，却被 eval 计入 activity 分母。量化：777 gated 帧里仅 395 帧
     gate 目标进 prompt，addressed 全分母 37.3% vs 仅 prompted-gate 57.7%——37.3% 被"没让它判"的帧
     稀释。**修法**：`_render_facts` 渲染 AOI∪conflict∪gate（与 `_resolve_views` 同集），gate 目标
     100% 进 prompt（395→777）。**重跑子集 caption 后 activity_addressed 37.3%→69.4%**（n=777），
     即近 7 成 gated 帧 VLM 做了三态判断；`activity_busy` 38%(n=290)→20%(n=539)，分母涨是因新纳入
     大量"等待/空闲"判断（后方远处吊机），是更全面的分布。conflict/ttc/halluc 不变（只改 activity 渲染）。
     注：6cam vs front 对比同口径仍成立，此修只抬高 teacher 绝对值（单front 17%→6cam+gate入prompt 69%）。
   - ⚠️ **gate 误标非作业类（已修，2026-06-22）**：activity_gate 对静止近吊机的**任意**目标触发，
     含锥桶/行人/小车——它们不可能做装卸，却被标"请判断作业状态"（val 子集 941 个误标：锥桶 491、
     小车 244、行人 206）。加 `_NO_ACTIVITY_IDS={0,1,9}` 排除（吊机仍 gate，其忙/闲正是要判的）。
     gate 目标 2481→1540、gate 帧 777→664。重抽子集(911→805)重跑 caption（v3）：
     **activity_addressed 69.4%→74.5%**（n=664，分母更干净故回应率更高）、**advice 80%→84.8%**
     （清掉"向锥桶让行"的误导帧）、conflict/ttc/halluc 不变。
   - 📊 **三轮演进汇总**（val teacher，activity_addressed）：单front 17.0% → 6cam-only-AOI 37.3%
     → 6cam+gate入prompt 69.4% → +排除非作业类 **74.5%**。幻觉稳定 ~96%（无幻觉率）。
   - `shuffle` 对照（query 配错帧）是证伪试金石：若打乱后命中不掉 = LLM 没真用 LiDAR query。

**C. 优先改进**：
   - **(高) 提升 VLM activity 召回** — ✅ 已根治：6 路环视 + 评测指标修正 + gate 目标入 prompt
     + 排除非作业类，activity_addressed 17%→74.5%（见 B）。train pkl 已是 6 路+gate已清理，可直接跑全量。
   - ✅ **gate 误标非作业类（已做，2026-06-22）**：`_NO_ACTIVITY_IDS={0,1,9}` 排除锥桶/行人/小车
     （参照 `_NO_CONFLICT_IDS`），gate 目标 2481→1540。见 4.3-B。
   - (中) 显式特征入 head：当前只喂 track_query+box center；可拼 class logits / bbox 尺寸/yaw /
     velocity / conflict-advice 结构特征，提升可控性（评注第 2 条，训练后迭代）。
   - (低) 多卡 DDP 全量完整验证（smoke 只跑 3 iter）。
