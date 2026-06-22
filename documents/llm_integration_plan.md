# LiDAR UniAD 端到端 + LLM 集成方案

> 讨论记录 / 2026-06-19。基线 config: `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ.py`。
> 目标场景：自有数据集（港口 / 集装箱场），13 类（IGV、Crane、Forklift、Truck 等）。

---

# 第 0 部分：复现操作手册（给后续 agent / 复现者）

> 本节是可照着复现的操作记录。下面的"第 1~4 部分"是设计讨论与决策依据，先看本节能最快上手。

## 0.1 一句话目标

跨模态蒸馏：用离线大 VLM（Qwen2.5-VL-7B）看**前视相机图**，结合 LiDAR 几何事实，
生成中文场景 summary 作为监督；训练一个挂在 UniAD **LiDAR query** 上的小 LLM（Qwen2.5-0.5B），
让它**推理时仅凭 LiDAR query** 就能说出图像老师才看得到的语义。推理零图像依赖，不动现有 LiDAR 链路。

数据流两步走：**几何事实生成**（`gen_geo_facts.py`，从 GT 算出 `info['geo_facts']`，产物 pkl 后缀 `_geo`）
→ **VLM caption 生成**（`gen_vlm_caption.py`，VLM 看图 + 读 geo_facts 出中文 summary，
sidecar `*_vlmcap_summaries.json`、合并后 pkl 后缀 `_vlmcap`）。

## 0.2 环境矩阵（关键：两套 env，因 numpy/torch 不兼容必须分开）

| env | torch | numpy | transformers | 用途 |
|-----|-------|-------|--------------|------|
| `uniad_train` | 1.12.1+cu116 | 1.22.4 | 无 | 训练主环境；跑 几何事实生成、子集抽样、读写 KL pkl |
| `qwen_vl`（本方案新建） | 2.6.0+cu124 | 2.2.6 | 4.57.6 | 仅跑 VLM caption 的 Qwen2.5-VL 离线推理 |

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
写 `info['sync_info']['cameras']['CAM_FRONT']`（含 path/dt/valid）。容差 50ms（对齐建库 cfg）。
```bash
python tools/data_converter/add_cam_sync.py \
  --pkl-path data/kl_8/kl_infos_train.pkl data/kl_8/kl_infos_val.pkl
# 产物（保留副本，不覆盖原 pkl）: data/kl_8/kl_infos_{train,val}_with_cam.pkl
# 命中率: train 98.58% / val 98.19%（<50ms）; 缺帧标 valid=False
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
`gen_vlm_caption.py`：读 geo_facts + 渲染中文事实约束文本 + 前视图 → VLM → 一句中文 summary。
**关键产物是 JSON sidecar**（`*_summaries.json`，token→summary），pkl 输出可丢 /tmp。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/data_converter/gen_vlm_caption.py \
  --pkl-path data/kl_8/kl_infos_val_sub1k_geo.pkl \
  --model-path /mnt/disk1/models/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --out-path /tmp/sub1k_vlmcap.pkl
# 真正要用的产物: /tmp/sub1k_vlmcap_summaries.json
# 速度 ~1.4 帧/s（单卡 4090）；--limit N 抽样；--dry-run 只渲染 prompt 不加载模型
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

- **Step 1~5 流水线已全跑通**（val 1500 子集）：产物 `data/kl_8/kl_infos_val_sub1k_vlmcap.pkl`
  （1466/1500 带 summary，uniad_train 可读）。
- **VLM caption 全量未跑**：目前只在 val 1500 子集验证；train 43981 帧全量约 18h（可多卡并行）。
  train 侧需先跑 Step1(add_cam_sync)+Step2(gen_geo_facts) 得到 `kl_infos_train_with_cam_geo.pkl`。
- **LLMBridgeHead 未落地**：第 7 步，挂在 `UniADMotionLidar` 上、消费 `outs_motion['track_query']`。
  早期写过骨架已回退（git）；现数据已就绪，可开始重做。
- **第7步 Step0 兼容性验证已通过（2026-06-21）**：
  - uniad_train（python3.8 / torch1.12.1+cu116）装 `transformers==4.46.3 peft==0.13.2`
    可加载并训练 Qwen2.5-0.5B-Instruct（d_model=896，494M）；inputs_embeds 路径
    forward+LM loss+backward+LoRA 全通过。模型在 `/mnt/disk1/models/Qwen2.5-0.5B-Instruct`。
  - ⚠️ **必须用 bf16（或 fp32），不能 fp16**：fp16 下 LM loss=nan；bf16 正常（4090 支持）。
  - ⚠️ **pip 装 transformers 时 accelerate 会强升 torch→2.4，打断 mmcv.ops CUDA 扩展**！
    已用 `pip install --no-deps torch==1.12.1+cu116 torchvision==0.13.1+cu116` 恢复，
    mmcv.ops 验证 OK、训练链路完好。后续装包注意别再动 torch（用 --no-deps 或避开 accelerate 升级）。
- 文件清单（本方案新增）：
  `tools/data_converter/{add_cam_sync,gen_geo_facts,make_subset,gen_vlm_caption,merge_summaries}.py`、
  `tools/run_vlm_caption_train_7gpu.sh`（7 卡并行 VLM caption）、
  `projects/mmdet3d_plugin/uniad/dense_heads/llm_bridge_head.py`、
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm.py`、本文档。

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

### 3.0 定调（2026-06-19 讨论确定）

**一句话**：用离线大 VLM 看 6 路环视相机图，生成"图像才看得到的语义"作为监督；
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
- **决策**：先只用 **front** 单视角（VLM 算力省 6 倍，先验证价值再扩环视）。
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
  `task_loss_weight['llm']`；pipeline Collect 增加 `gt_caption`。
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

## 4. 下一步

1. ✅ LLM 底座：Qwen2.5-0.5B（d_llm=896）。
2. ✅ 方案定调：VLM 图像 老师 + 纯 LiDAR 推理（跨模态蒸馏）；几何模板互补。
3. ✅ token 边界 + caption schema（scene-level summary，生成方式 a）锁定。
4. ✅ **图像↔LiDAR 同步脚本**：`tools/data_converter/add_cam_sync.py`（后处理，模仿 add_sdc）。
   - 复用 converter 的 `CAM_NAME_MAP` / `make_sync_entry` 约定，写进 `info['sync_info']['cameras']`。
   - 核心匹配逻辑已离线自验：val 5191 帧 98.19% 命中（<50ms），median 35.9ms / p95 46.1ms，
     与 train 抽样一致；缺帧段正确标 `valid=False`。
   - 实际写 pkl 需在 docker（mmcv）里跑：
     `python tools/data_converter/add_cam_sync.py --pkl-path <train.pkl> <val.pkl> [--in-place]`
   - 默认 front、容差 0.05s（对齐建库 cfg 的 `camera_max_diff`）；`--views` 可扩环视。
   - **已运行**（uniad_train conda 环境，mmcv 1.5.0），保留副本不覆盖原 pkl：
     `data/kl_8/kl_infos_{train,val}_with_cam.pkl`（train 98.58% / val 98.19% 命中）。
     原 pkl 不动，几何/VLM/LLM config 显式指向 `_with_cam.pkl`；现有 LiDAR 训练不受影响。
5. ✅ **几何事实生成器**：`tools/data_converter/gen_geo_facts.py`（后处理，写 `info['geo_facts']`）。
   - 已验证 fut_traj 约定（相对自身当前位置的累积位移，LiDAR 帧）；冲突 = agent/ego 未来绝对轨迹最近距离。
   - per-agent：range/bearing(8扇区)/motion/heading/load(粗)/conflict+ttc/activity_gate；
     scene：agents_of_interest/ego_advice/congestion/crane_status。
   - **分布已 dump，activity 阈值数据化锁定**（val 83152 agent-instance）：
     `static_dur` 上限 3.5s（past traj 7 步），p50=1.0s；`crane_dist` p50=66m、p75=107m（多数 agent 远离吊机）。
     ⇒ 门控 `static≥2s & crane≤30m`：通过率 **3.0%**（宽松预筛但不滥发，VLM 看图终判）。
   - conflict=True 0.5%、ego_advice keep94%/yield4.8%/slow1%（合理）。
   - **congestion 修正**：原"30m内≥3 非移动"误把停车场当排队(66%)；
     改为"只数 moving_slow 且 ≥4"，降到 **1.5%**（真·缓行队列）。
   - 副本：`data/kl_8/kl_infos_{train,val}_with_cam_geo.pkl`（val 已跑；train 待跑）。
6. ▶ VLM caption：Qwen2.5-VL prompt 设计 + 批量生成 summary（吃 geo_facts 约束 + front 图）。
   - **环境**：新建独立 conda env `qwen_vl`（torch 2.6+cu124 / transformers 4.57.6 /
     qwen-vl-utils / torchvision 0.21），与 uniad_train(torch1.12) 隔离；VLM caption 是离线数据生成，不进训练/推理。
   - **模型**：Qwen2.5-VL-7B-Instruct，modelscope 下载到 `/mnt/disk1/models/`
     （⚠️ 系统盘 `/` 仅剩 46G，模型必须放 disk1，勿进 `~/.cache`）。单卡 24G 可跑。
   - **脚本**：`tools/data_converter/gen_vlm_caption.py`（后处理，写 `geo_facts['summary']`）。
     - facts→中文约束文本 + front 图 → VLM；system prompt 严令：几何事实不可改/编造，
       只补图像独有语义（装卸作业、吊机忙闲、满载/空载目视确认），输出一句≤60字中文 summary。
     - 只对 `agents_of_interest` 渲染细节；`activity_gate=True` 才提示"请结合图像确认作业"。
     - `--dry-run` 只渲染 prompt 不加载模型（已在 uniad_train 验证文本质量）；`--limit N` 抽样。
     - 已修：bearing 中文化（右前/正后…）、类名已含载货状态的类（空挂车等）不再叠加"空载/满载"。
   - 待模型下载完 → 单帧实测 VLM 输出质量 → 抽样调 prompt → 批量生成。
     - ✅ 模型已下载（16G，5 分片）；单帧实测通过：VLM 遵守几何事实、补出"吊机空闲/可能在作业"
       等图像语义，与 activity_gate 联动正确；summary 风格定为**简洁**（抓重点+图像语义，不复述清单）。
     - ✅ VLM caption 脚本去 mmcv 依赖（改 pickle 读写），可在纯净 qwen_vl 环境跑。
     - **子集策略**：`tools/data_converter/make_subset.py` 分层抽样——稀有场景（queue/conflict/
       activity_gate）全保留 + 普通帧补足。val 子集 1500 帧（1123 稀有 + 377 普通），
       `data/kl_8/kl_infos_val_sub1k_geo.pkl`，先在子集上跑通 LLMBridgeHead 再放大。
     - ▶ 子集批量生成中（GPU0，~1.5s/帧）。全量 train 43981 帧约 18h（待定，可多卡并行）。
     - ✅ 子集已生成（1500 帧，1468 有 summary / 32 无图跳过，~17.5min）。抽检：
       GATE/QUEUE/normal 质量好（方位+类别+作业状态+转向，mean 22.8 字）；
       **CONFLICT 初版退化成套话"前方有障碍物请减速"**（297 条几乎重复）。
     - ✅ 修 prompt：强令冲突场景必须点明目标(方位+类别)+TTC+建议，禁用"障碍物"泛称，加 few-shot。
       30 帧 conflict 重测通过：summary 具体化，TTC 越小建议从减速→让行，逻辑正确。
     - ⚠️ **几何 conflict 定义修正（2026-06-19，港口场景特性）**：初版 conflict=轨迹最近距离<4m，
       导致冲突目标多为**静止锥桶**，VLM 输出"向锥桶让行"的过度反应。
       **根因（用户澄清）**：港口锥桶用于路边圈定**禁入区**，不是路中央紧急避让物，ego 贴边正常驶过。
       **已修**：几何步骤把 Cone(cls=9) 加入 `_NO_CONFLICT_IDS`，conflict 计算跳过。
       效果：conflict 帧 5.8%→1.6%，冲突目标变为行人(70)/空载IGV(26) 等真·动态目标，锥桶清零。
     - ✅ numpy 跨版本交接：VLM caption 改为额外导出 `*_summaries.json`（token→summary），
       训练侧只读 JSON 合并，避免 qwen_vl(numpy2.x) 重写 pkl 致 uniad_train(numpy1.x) 读不了。
7. ✅ `LLMBridgeHead` + config + smoke test（2026-06-22 跑通）。
   - 新建 `dense_heads/llm_bridge_head.py`（`LLMBridgeHead`）：projector(256→896)+spatial_pe(3→896)
     + lazy 加载 Qwen2.5-0.5B(bf16,冻结+LoRA) + forward_train(LM loss, prompt/object 段 -100)
     + forward_test(generate)。**关键**：projector/spatial_pe 用自身(fp32) dtype 计算，
     输出再 cast 到 bf16 喂 LLM（否则 Linear dtype 不匹配报错）。
   - detector `uniad_motion_lidar.py`：加 `llm_head`/`with_llm_head`/`_current_caption`；
     forward_train 末尾接 llm loss(prefix='llm')；simple_test 接 forward_test 存 llm_caption。
   - **caption 注入**：`kl_dataset.py` 的 `KlTrackDataset._union2one`（line ~1963，**注意是子类
     的，不是父类 KlBEVFormerDataset 那个**）在 current frame 的 frame_meta 加 `gt_caption`；
     `_current_caption` 走整数键 metas_map[max(keys)] 取。`_extract_raw_meta` 带 summary。
   - config `base_e2e_lidar_occ_llm.py`：继承 occ base，加 llm_head + task_loss_weight['llm']=1.0，
     train/val 指向 `kl_infos_val_sub1k_vlmcap.pkl`，load_from 改 stage-1 ckpt（stage-2 ckpt 不存在）。
   - smoke 通过：caption 正确注入，llm.loss_llm≈5.2~5.7（有限），120 个 loss key（track/motion/
     occ/llm 共存不冲突），backward OK。
8. ▶ 正式训练：train 全量 VLM caption（被中断在 7.6%，需重跑，~9h）→ 全量训练 + val 评估。
   训练命令：`./tools/uniad_dist_train.sh projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm.py <GPUS>`
   （注意：当前 config 指向 val 子集；正式训练需把 train ann_file 换成 train 全量 c2 pkl）。
