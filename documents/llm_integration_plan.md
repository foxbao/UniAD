# LiDAR UniAD 端到端 + LLM 集成方案

> 讨论记录 / 2026-06-19。基线 config: `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ.py`。
> 当前主线 / 2026-06-23：Stage-1 纯探针 `projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe.py`。
> 旧 `base_e2e_lidar_occ_llm_train.py` 是 full-task 共训历史实验，不再作为当前首选训练入口。
> VLM teacher / 2026-06-25：后续新 caption 优先用 Qwen3-VL 系；`8B` 单卡快迭代，
> `32B` 走 2×4GPU 分片做高质量 teacher。既有 train pkl 是 `Qwen2.5-VL-7B-Instruct`
> 生成的历史 baseline，保留用于 A/B 和回退。
> 目标场景：自有数据集（港口 / 集装箱场），13 类（IGV、Crane、Forklift、Truck 等）。

---

# 第 0 部分：复现操作手册（给后续 agent / 复现者）

> 本节是可照着复现的操作记录。下面的"第 1~4 部分"是设计讨论与决策依据，先看本节能最快上手。

## 0.1 一句话目标

跨模态蒸馏：用离线大 VLM（当前新 teacher：Qwen3-VL-8B/32B；历史 baseline：Qwen2.5-VL-7B）
看**环视相机图（6 路，按目标方位自动路由 2-4 路）**，
结合 LiDAR 几何事实，生成中文场景 summary 作为监督；训练一个挂在 UniAD **LiDAR query** 上的小
LLM（默认 Qwen2.5-0.5B；Qwen3-0.6B 试验中），让它**推理时仅凭 LiDAR query** 就能说出图像老师才看得到的语义。推理零图像依赖，
不动现有 LiDAR 链路。（注：早期落地先只用 front，后因后方/侧方作业目标看不见导致 activity 召回偏低，
已改为 6 路环视，见 4.3-B。）

2026-06-23 收敛后的当前阶段是 **Stage-1 纯探针**：冻结 UniAD LiDAR 感知栈，只训练
`LLMBridgeHead`（Projector + LoRA），并把 LLM 输入 query/center detach，回答一个更干净的问题：
**冻结后的 LiDAR object query 里是否已经含有足够语义，让小 LLM 复现 VLM 老师 caption？**
这一步暂不追求反哺检测、预测、占据或规划。

数据流两步走：**几何事实生成**（`gen_geo_facts.py`，从 GT 算出 `info['geo_facts']`，产物 pkl 后缀 `_geo`）
→ **VLM caption 生成**（`gen_vlm_caption.py`，VLM 看图 + 读 geo_facts 出中文 summary，
sidecar `*_vlmcap_summaries.json`、合并后 pkl 后缀 `_vlmcap`）。

## 0.2 环境矩阵（关键：两套 env，因 numpy/torch 不兼容必须分开）

| env | torch | numpy | transformers | 用途 |
|-----|-------|-------|--------------|------|
| `uniad_train` | 1.12.1+cu116 | 1.22.4 | 4.46.3 (+peft 0.13.2) | 训练主环境；几何事实生成、子集抽样、读写 KL pkl、**LLMBridgeHead 训练**（默认 Qwen2.5-0.5B）|
| `/mnt/disk1/conda_envs/uniad_train_qwen3_py39` | 1.12.1+cu116 | 1.22.4 | 4.51.3 (+peft 0.13.2) | **Qwen3-0.6B student 隔离试验环境**；已验证 `base_e2e_lidar_llm_probe_qwen3_0p6b.py` 可 build + head loss smoke |
| `qwen_vl`（本方案新建） | 2.6.0+cu124 | 2.2.6 | 4.57.6 | 仅跑 VLM caption 离线推理；已验证可加载 Qwen2.5-VL-7B、Qwen3-VL-8B 与 Qwen3-VL-32B（`device_map=auto`） |

> uniad_train 的 transformers/peft 是 LLMBridgeHead 训练所需，**后装**（原始环境无）。
> ⚠️ 装时务必 `--no-deps` 锁 torch，否则 accelerate 会强升 torch→2.4 打断 mmcv.ops（见 4.2）。
> Qwen3-0.6B student 已下载到 `/mnt/disk1/models/Qwen3-0.6B`，但它的 `model_type=qwen3`
> 需要 transformers>=4.51；当前 `uniad_train` 的 4.46.3 会报 `KeyError: 'qwen3'`。已建立隔离环境
> `/mnt/disk1/conda_envs/uniad_train_qwen3_py39`，不要直接升级主训练环境。

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

Qwen3-0.6B student 隔离环境使用方法（已建好）：
```bash
conda activate /mnt/disk1/conda_envs/uniad_train_qwen3_py39
PYTHONPATH=$(pwd) python - <<'PY'
from mmcv import Config
import projects.mmdet3d_plugin
from third_party.uniad_mmdet3d.models.builder import build_model

cfg = Config.fromfile(
    'projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe_qwen3_0p6b.py')
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
print(type(model).__name__, type(model.llm_head._llm).__name__)
PY
```

该环境配置比较脆：Python3.9、torch1.12、transformers4.51、mmcv/spconv/custom CUDA ops、以及
`sitecustomize.py` shim 都要对齐。完整从零重建命令、验证命令和已知 `pip check` 警告见
`documents/qwen3_student_env_runbook.md`。

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
  里 `camera_processing_cfg.enable=False`）。早期只用 **front**，当前 VLM caption 已改为 6 路环视按目标方位路由。
- 13 类（post label_mapping）：0 行人 / 1 小车 / 2 满载IGV / 3 卡车 / 4 空挂车 / 5 满载挂车 /
  6 空载IGV / 7 正面吊（Crane） / 8 其他车辆 / 9 锥桶 / 10 集装箱叉车 / 11 叉车 / 12 轮胎吊。
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
2026-06-25 起增加 **teacher 深化开关 `--annotate-targets`**：利用每个 scene 的
`camera_extrinsics.json` / `intrinsics.json`，把相关 LiDAR 目标中心投影到相机图上，画出
`#track_id 类别 距离` 辅助标注，帮助 VLM 把几何事实中点名的目标和图像实体对齐。
同日增加 `--feedback-json`：把人工确认的 `motion_state` / `operation_state` 注入 prompt，
优先级高于 VLM 自行判断，用于修正港口设备“静止但正在作业”的典型误判。
`operation_state` 采用 `working/waiting/idle/unknown`，其中“等待作业”是 waiting，
不能因为含“作业”二字被评测成 working。
2026-06-25 追加相机兜底：若按目标方位路由的相机无效，回退到本帧任一有效相机；
若本帧完全没有有效相机，则走 text-only summary，只允许依据 geo_facts，不补作业/载货视觉判断，
并在 summary meta 标记 `no_valid_camera`。
同日根据人工抽检收紧状态口径：移动/行驶中的正面吊、轮胎吊、叉车、集装箱叉车只描述运动，
不能仅因吊具/吊臂/叉臂抬起就判 `working/waiting`；`waiting` 必须极保守，只有目标清楚可见且
明确处于作业位排队/等待装卸才使用；车辆/IGV/挂车/卡车只是停着、靠近箱区或靠近其他车辆，
一律按静止/空闲描述。正面吊例外：静止、吊具/吊臂抬起，并且吊具轴线/车身朝向与近距离
集装箱/挂车/集卡有明确空间对齐时，应判为正在装卸作业，不要求已经接触箱体；若只是邻近黄车/挂车、
吊臂抬起但没有明确对齐，不得写“正对”或判作业。不可见设备必须写看不清或不提。`crane_status` 只是几何诊断门控，
不再渲染到 prompt，避免诱导模型写图里没有的吊机。
**关键产物是 JSON sidecar**（`<pkl去后缀>_summaries.json`，token→summary），pkl 输出可丢 /tmp。
```bash
# 当前新 teacher（2026-06-25 后续新 caption 优先用）
CUDA_VISIBLE_DEVICES=0 python tools/data_converter/gen_vlm_caption.py \
  --pkl-path data/kl_8/kl_infos_val_sub6cam_geo.pkl \
  --model-path /mnt/disk1/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --out-path /tmp/val_sub6cam_qwen3vl8b_vlmcap.pkl
# 真正要用的产物: /tmp/val_sub6cam_qwen3vl8b_vlmcap_summaries.json

# teacher 深化版本：给 VLM 输入带 LiDAR 投影目标标注的相机图
CUDA_VISIBLE_DEVICES=0 python tools/data_converter/gen_vlm_caption.py \
  --pkl-path data/kl_8/kl_infos_val_sub6cam_geo.pkl \
  --model-path /mnt/disk1/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --out-path /tmp/val_sub6cam_qwen3vl8b_annotated_vlmcap.pkl \
  --annotate-targets --annotated-dir /tmp/val_sub6cam_qwen3_annotated_imgs

# 带人工反馈的 teacher：人工确认状态优先于 VLM 图像判断
CUDA_VISIBLE_DEVICES=0 python tools/data_converter/gen_vlm_caption.py \
  --pkl-path data/kl_8/kl_infos_val_sub6cam_geo.pkl \
  --model-path /mnt/disk1/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --out-path /tmp/val_sub6cam_qwen3vl8b_feedback_vlmcap.pkl \
  --annotate-targets --annotated-dir /tmp/val_sub6cam_qwen3_feedback_imgs \
  --feedback-json documents/llm_teacher_feedback_pilot.json
# 历史 baseline teacher: /mnt/disk1/models/Qwen2.5-VL-7B-Instruct
# Qwen2.5-VL-7B 速度约 1.4-1.8 帧/s（单卡 4090，多相机约 3.4 路/帧）；
# Qwen3-VL-8B 速度需重新实测；--limit N 抽样；--dry-run 只渲染 prompt 不加载模型
# 输入 pkl 须已含 6 路 sync_info（add_cam_sync 用 6 路 --views 生成）；无则只路由到 front。
# train 全量请用 7 卡分片：bash tools/run_vlm_caption_train_7gpu.sh（见第 8 步），
#   --num-shards N --shard-id i 把数据切 N 份并行，各写 *_summaries.shard{i}ofN.json

# 高质量 Qwen3-VL-32B teacher（8×4090：两个进程，每进程 4GPU 模型并行）
bash tools/run_vlm_caption_qwen32b_val_2x4gpu.sh
# val 已验证产物:
# data/kl_8/kl_infos_val_sub6cam_qwen3vl32b_annotated_vlmcap.pkl
# 805/805 帧有 summary；其中 3 帧无有效相机，退化为 text-only summary。
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

# 若把新 teacher 的 sidecar merge 到一个已经含旧 summary 的 pkl，
# 必须加 --clear-missing，避免未生成的 token 沿用旧 Qwen2.5 summary。
python tools/data_converter/merge_summaries.py \
  --pkl-path data/kl_8/kl_infos_val_sub6cam_v3_vlmcap.pkl \
  --json-path /tmp/val_sub6cam_v3_qwen3vl8b_annotated_vlmcap_summaries.json \
  --out-path /tmp/val_sub6cam_v3_qwen3vl8b_annotated_eval.pkl \
  --clear-missing
```

## 0.5 模型与磁盘

- **当前新 teacher**：Qwen3-VL-8B-Instruct，已下载并加载验证通过：
  **`/mnt/disk1/models/Qwen3-VL-8B-Instruct`**（17G，4 分片）。`qwen_vl` 环境中
  `AutoProcessor` → `Qwen3VLProcessor`、`AutoModelForImageTextToText` →
  `Qwen3VLForConditionalGeneration`，单卡 GPU0 `device_map={'': 0}` 可加载。
  下载命令：
  `modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /mnt/disk1/models/Qwen3-VL-8B-Instruct`
- **高质量 teacher**：Qwen3-VL-32B-Instruct，已下载到
  **`/mnt/disk1/models/Qwen3-VL-32B-Instruct`**（63G，14 分片）。`qwen_vl` 环境可用
  `AutoModelForImageTextToText + device_map=auto` 在 4 张 4090 上加载；8 卡全量生成时用
  `tools/run_vlm_caption_qwen32b_val_2x4gpu.sh` 启两个 4GPU shard。
  已完成 val 805 帧 annotated caption：
  `data/kl_8/kl_infos_val_sub6cam_qwen3vl32b_annotated_vlmcap.pkl`。
- **历史 baseline teacher**：Qwen2.5-VL-7B-Instruct，modelscope 下载到
  **`/mnt/disk1/models/Qwen2.5-VL-7B-Instruct`**（16G，5 分片）。不要立即删除；用于复现既有
  `*_vlmcap.pkl`、teacher 指标 A/B 和回退。
  ⚠️ 系统盘 `/` 仅剩 ~46G，模型与缓存**必须放 disk1**（`export MODELSCOPE_CACHE=/mnt/disk1/...`）。
- GPU：本机 8×RTX4090(24G)；Qwen2.5-VL-7B 与 Qwen3-VL-8B 单卡可加载，Qwen3-VL-32B
  需要多卡 `device_map=auto`。用空闲卡，勿占训练卡。

## 0.6 已知问题 / 待办

- **流水线 Step 1~5 + LLMBridgeHead 已全跑通**（截至 2026-06-23）：
  - 数据：6 路环视 val 子集 **v3** `data/kl_8/kl_infos_val_sub6cam_v3_vlmcap.pkl`（805 帧, 797 带 summary；
    全为稀有帧，gate 已排除非作业类）。v1/v2 子集与旧单 front `kl_infos_val_sub1k_vlmcap.pkl` 仅作历史对比。
    ⚠️ **子集帧数 ≠ 实际 eval queue 数**：make_subset 按稀有性挑帧，prev/next 可能落在子集外，
    时序 track eval 只能用能组成默认 5 帧 queue 的帧（v3 的 805 帧实际约 438 帧可 eval；
    旧 sub1k 1500→约 542）。teacher caption 评测是逐帧的、用全部 805；但跑 dist_eval 的 track/motion
    指标会少于子集帧数。caption 蒸馏训练本身不受影响（caption 是逐帧监督）。
  - 几何事实：**train/val 全量均已是 6 路 + gate 清理** `kl_infos_{train,val}_with_cam_geo.pkl`
    （train 43981 帧，6 路各 valid 98.45%~98.61%；2026-06-22 重做并排除非作业类 gate）。
  - **train 全量 caption 已完成**：7 卡分片跑 ~2.2h → merge → `kl_infos_train_vlmcap.pkl`
    （43747/43981=99.5% 带 summary）。注意：该 pkl 是 Qwen2.5-VL-7B 历史 teacher 生成。
    Qwen3-VL-8B 已下载验证，后续新 caption 应先在 val v3 做 A/B，再决定是否重做 train 全量。
  - **Qwen3-VL-32B val teacher 已完成**：2×4GPU 分片跑完 805 帧，并修复相机兜底后
    `missing_summary=0`（3 帧无有效相机，text-only）。当前指标：
    `advice=0.988`、`activity_addressed=0.895`、`activity_busy=0.084`、`halluc=0.991`。
    人工反馈已修正 7 个样本（移动设备不判等待/作业、不可见吊机不写入 summary、普通静止车辆/IGV
    不判等待作业），`waiting` 计数从 6 降到 0。结论：32B teacher 已可用，但 train 全量前仍建议先人工抽检作业/空闲状态。
  - 模型：`LLMBridgeHead` 已支持两种接线：
    1) 历史 full-task 分支 `base_e2e_lidar_occ_llm*.py`，与 track/map/motion/occ 共训；
    2) 当前 Stage-1 纯探针 `base_e2e_lidar_llm_probe.py`，继承 `base_e2e_lidar.py`，冻结非 LLM 模块，
    只训练 `llm_head.*`，并 `detach_inputs=True` 断开 LLM loss 到感知 query 的梯度。详见 3.3 / 3.4。
    Qwen3-0.6B 试验配置 `base_e2e_lidar_llm_probe_qwen3_0p6b.py` 已加入，等待训练环境兼容。
  - 评估：`eval_llm_caption.py` template/teacher 可跑；activity 拆为 addressed/busy（见 4.3-B）。
    6 路 + gate 入 prompt + 排除非作业类后 teacher activity_addressed **74.5%**（单 front 17.0%）。
- **训练状态 / 下一步**：
  - (0) ✅ train pkl 已重做 6 相机 + gate 清理（2026-06-22）。
  - (1) ✅ 7 卡分片 caption 已完成（~2.2h，43747 帧带 summary）。
  - (2) ✅ merge → `kl_infos_train_vlmcap.pkl`；训练/评测数据已就绪。
  - (2.5) ✅ Qwen3-VL-32B annotated teacher val 已完成：805/805 帧有 summary，
    `activity_addressed=0.895`、`halluc=0.991`。下一步不是立刻 train 全量，而是人工抽检
    作业/空闲状态；确认质量后再重做 train 全量 caption。
  - (3) ✅ 历史 full-task `base_e2e_lidar_occ_llm_train.py` 已跑 1 epoch，证明能输出中文 caption，
    但 `model≈shuffle`，没有证据表明它读到了同帧 query 语义；且主任务指标有退化。见 4.1 / 4.3-B。
  - (4) ✅ 当前 Stage-1 纯探针代码已落地并 smoke 通过：`base_e2e_lidar_llm_probe.py` 只训练
    `llm_head.*`，单 batch forward/backward OK。
  - (5) ⏳ **待跑当前主线训练**：`uniad_dist_train.sh base_e2e_lidar_llm_probe.py <GPUS>`。详见 4.3-A。
- 文件清单（本方案新增）：
  `tools/data_converter/{add_cam_sync,gen_geo_facts,make_subset,gen_vlm_caption,merge_summaries}.py`、
  `tools/run_vlm_caption_train_7gpu.sh`（7 卡并行 VLM caption）、
  `tools/run_vlm_caption_qwen32b_val_2x4gpu.sh`（Qwen3-VL-32B 两个 4GPU shard）、
  `tools/analysis_tools/eval_llm_caption.py`（caption 质量评测）、
  `projects/mmdet3d_plugin/uniad/dense_heads/llm_bridge_head.py`、
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm.py`（smoke）、
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train.py`（历史 full-task 训练）、
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe.py`（当前 Stage-1 纯探针）、本文档。

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
- **VLM 老师**：历史使用 Qwen2.5-VL-7B；2026-06-25 起后续新 caption 优先试 Qwen3-VL-8B。
  本地 8×RTX4090(24G)，7B/8B 单卡可跑，本地批量离线。
- **teacher 深化**：VLM 推理阶段可以选择带图像，也可以只读结构化事实；当前主线仍是
  **teacher 看图、student 推理不看图**。`--annotate-targets` 只改变离线 teacher 输入，不改变
  LLMBridgeHead / UniAD 推理输入。
- **人工反馈闭环**：港口设备状态拆成两层：
  `motion_state`（`moving/static/unknown`，几何可判）与
  `operation_state`（`working/waiting/idle/unknown`，图像/人工反馈判）。
  关键规则：**设备本体静止不等于空闲**。吊机、轮胎吊、叉车、集装箱叉车静止时仍可能正在装卸/取放货。
  人工反馈 JSON 是 `token -> track_id -> {motion_state, operation_state, note}`，生成时用
  `--feedback-json` 注入 prompt。
- **同步信息存储**：写进 pkl（增量加字段，dataset 直接可读；用户已确认可改 pkl）。

### 3.1 模型侧设计

新增一个 **LLM bridge head**，挂在 detector 上。当前保留两种接线：

1. **Stage-1 纯探针（当前主线）**：只用 `forward_track_train` 产出的
   `outs_track['track_query_embeddings']` + box center，冻结全部非 LLM 模块，只训练 `llm_head.*`。
2. **full-task 分支（历史实验）**：与 track/map/motion/occ 共训，用于保留对照，不再作为当前首选入口。

```
track_query_embeddings [num_tracks,256] ┐
(+ box 中心做空间编码)                   ├─► Projector (MLP, 256 → d_llm) ─► [N object tokens]
motion track_query（仅 fallback/历史分支）┘                                      │
                                                                               ▼
                          [prompt tokens] ⊕ [object tokens] ─► LLM ─► 文本(老师标签)
```

- **默认底座**：Qwen2.5-0.5B（d_llm=896），中文友好、单卡可 LoRA、最稳。
- **Qwen3-0.6B 试验分支**：`base_e2e_lidar_llm_probe_qwen3_0p6b.py`，`d_llm=1024`；
  已在隔离环境 `/mnt/disk1/conda_envs/uniad_train_qwen3_py39` 中验证：
  `UniADMotionLidar + LLMBridgeHead + PeftModelForCausalLM` 可 build，head loss smoke 有限，
  probe 模式可训练参数约 4.66M（Projector + LoRA）。
- **Projector**：小 MLP，256 → d_llm，每个 agent 一个 token。
- **空间编码**：box 中心 (x,y,z)（来自 `track_bbox_results[0][0].gravity_center`）→ MLP → 加到 token，给 LLM 空间先验。
- **训练**：冻结 LLM 主干，只训 Projector + LoRA。Stage-1 纯探针还会冻结 UniAD 感知栈并 detach 输入 query。
- **token 数**：agent 数不定，截断/padding 到 `max_agents`，截断要 log。


### 3.2 监督数据：几何模板 + VLM 图像（互补）

| 路线 | 来源 | 内容 | 前置成本 |
|------|------|------|----------|
| **几何模板** | 现有 GT 框/轨迹合成 | 确定性事实：方位/距离、运动状态、ego×agent 轨迹冲突/让行、是否在可行驶区 | 低，直接可做 |
| **VLM 图像**（主线） | Qwen3-VL-8B 看 6 路环视路由图（Qwen2.5-VL-7B 为历史 baseline） | 作业状态、满载/空载（粗）、吊机工作/空闲 | 中，需先做图像↔LiDAR 同步（已验证轻量） |

- 最终监督 = 几何精确空间/冲突事实 ⊕ VLM 场景语义，合并成每帧的 caption。
- VLM caption 生成时让模型同时吃几何步骤的结构化事实文本作为 prompt 约束，减少幻觉、对齐到 ego 视角。
- teacher 深化时可打开 `--annotate-targets`，把 GT LiDAR 目标中心投影到相机图上，作为 VLM 的
  grounding 辅助。它不是模型推理依赖，只用于离线生成更准的监督信号。
- 状态监督采用两层 schema，而不是单一“开车中/作业中/停止中”：
  - `motion_state`: `moving/static/unknown`
  - `operation_state`: `working/waiting/idle/unknown`
  对外 summary 可以说“行驶中/正在装卸作业/等待作业/空闲”，但内部评估和人工反馈按两层存。
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
- **生成方式 (a)**：几何模板拼几何事实 → Qwen3-VL-8B 看 6 路环视路由图 **改写润色 + 补作业/载货语义** →
  自然中文 summary。VLM 只负责"看图补语义 + 说人话"，几何事实由几何步骤保证准确，防幻觉。
  注：既有 `kl_infos_train_vlmcap.pkl` / v3 val caption 是 Qwen2.5-VL-7B 生成的历史版本。
- **teacher 深化版本**：在相机图上额外画 LiDAR 投影目标点与 `#ID 类别 距离` 标签，再送给 Qwen3-VL。
  目标是缓解多相机图像里“事实点名的目标”和“图中具体实体”对不上的问题，尤其是港口远处吊机、
  挂车、侧后方目标。2 帧 pilot 中，带标注 Qwen3 已能把原 plain 版本漏掉的“正前方 22.8m 空挂车”
  写进 summary；但是否系统性提升，必须看 val v3 全量 A/B 指标。
- **人工反馈版本**：`--feedback-json` 把人工确认的设备状态作为高优先级事实写进 prompt。
  当前 pilot `documents/llm_teacher_feedback_pilot.json` 包含 6 个标注目标：2 帧 #4 吊机
  `motion_state=static, operation_state=working`，4 帧 #6 集装箱叉车
  `motion_state=static, operation_state=waiting`。重跑后 Qwen3 summary 能把吊机改为
  “正在装卸作业”，并把集装箱叉车写成“等待作业”；
  `eval_llm_caption.py --feedback-json` 的 `operation_human_acc=1.000 (n=6)`。
- **scene-level 字段**（几何步骤先算，作为 VLM 的 prompt 约束 + summary 的事实底料）：
  `agents_of_interest`（对 ego 最相关的 id）、`ego_advice`（keep/yield/slow/stop，由冲突导出）、
  `congestion`、`crane_status`、以及每个关键 agent 的 pos/motion/heading/load/conflict。
  其中 `ego_advice` 是几何阶段已经算好的本车建议，teacher 只能照抄，不能按图像重新改写。
- **作业状态门控阈值**：**不预设，数据驱动**。
  - 门控只做"宽松预筛"——挡掉几何上绝无可能作业的 agent（如高速行驶中），
    最终"是否装卸"由 VLM 看图裁决。故门控应**故意放松**。
  - 几何生成器先 **dump 分布**（每个 agent 的静止时长、与最近 Crane 的距离），
    看真实直方图再定阈值；阈值写成 config 可调参数，不写死。
  - 宽松默认（待数据校准）：静止 >2s 且距最近 Crane <30m ⇒ "允许"标作业。


### 3.3 落点（代码）

- 新 detector head：`llm_bridge_head.py`（`LLMBridgeHead`，`@HEADS.register_module()`）。
- detector 接线：`uniad_motion_lidar.py` 的 `forward_train` / `simple_test`。head 内 `_agent_query`
  **优先用 `outs_track['track_query_embeddings']`**（与 box center 同源同序，见 3.3 末
  "query/center 对齐"），`outs_motion['track_query']` 仅作 fallback。注意 `simple_test` 里要在剥离
  `track_bbox_results`/`track_query_embeddings` 的 cleanup **之前** 调用。
- config 分两类：
  - **当前 Stage-1 纯探针**：`base_e2e_lidar_llm_probe.py` 继承 `base_e2e_lidar.py`，而不是
    `base_e2e_lidar_occ.py`。原因是当前问题只问"冻结 track query 是否含有可读语义"，LLMBridgeHead
    主输入是 `track_query_embeddings + box center`，不依赖 occ 输出。去掉 occ 可以降低耦合，避免把下游
    task 变化误判成 LLM 证据。关键开关：`llm_probe_only=True`、`freeze_non_llm=True`、
    `llm_head.detach_inputs=True`、`task_loss_weight=dict(track=0,map=0,motion=0,llm=1)`。
    `load_from` 指向 `base_e2e_lidar/latest.pth`，作为冻结 LiDAR query 特征提取器。
  - **历史 full-task 分支**：`base_e2e_lidar_occ_llm.py` / `_train.py` 继承 `base_e2e_lidar_occ`，
    与 track/map/motion/occ 共训。它验证了工程链路能跑通，但不是当前最干净的探针入口。
    旧说法"必须继承 occ 顶端才有料"已被修正：对 Stage-1 探针而言不成立。
- caption 不走 pipeline 的 Collect key，而是数据集（`kl_dataset.py` `_union2one`）把 `gt_caption`
  注入当前帧的 **img_metas**，detector（`uniad_motion_lidar.py` `_current_caption`）从 img_metas 取出喂 llm_head。
- 数据：图像同步脚本（补相机路径）+ 几何模板生成 + VLM caption 生成，产出每帧 caption，存入 pkl/sidecar。
- **query/center 对齐（2026-06-23 修）**：LLM 的 per-agent query 与 box center 必须 1:1 对应。
  曾用 `outs_motion['track_query']` 当 query、`outs_track['track_bbox_results']` 当 center——但
  `MotionHeadLidar` 把 track_query 过滤成 vehicle 类并剥掉 SDC slot，而 box 未过滤，导致 query[i]
  与 center[i] 描述不同物体、spatial_pe 加错位置。改用 `outs_track['track_query_embeddings']`：它与
  `track_bbox_results` 在 `select_active_track_query` 里同 topk bbox_index + 同 mask 生成，严格同序同长，
  且保留 caption 会提到的行人/锥桶等非 vehicle agent。motion query 仅作 fallback。

### 3.4 训练策略

- **Stage-1 纯探针（当前主线）**：冻结 UniAD LiDAR 感知栈，只训练 `llm_head.*`
  （Projector / spatial PE / LoRA）。这一步不反哺检测、map、motion、occ 或 planning。
- 三层隔离：
  1) `freeze_non_llm=True`：`UniADMotionLidar.train()` 每次切 train mode 后重新冻结所有非 `llm_head` 子模块；
  2) `llm_probe_only=True`：`forward_train` 跑完 track 取 query 后，直接只返回 `llm.loss_llm`；
  3) `detach_inputs=True`：`LLMBridgeHead._object_tokens` 对 query/center detach，断开 LLM loss 到感知 query 的梯度。
- loss：Stage-1 只优化 `llm.loss_llm`。full-task 历史分支仍可把 LM loss 经
  `loss_weighted_and_prefixed(prefix='llm')` 与现有 task loss 合并，但不作为当前结论依据。
- 验证：先看 LLM caption 指标的 `model > shuffle/noquery` 是否成立，再考虑结构化特征、更多 token 或任务反哺。

### 3.5 待定 / 风险

- **图像↔LiDAR 同步质量**：时间最近邻 + 标定，需抽检对齐误差；港口低速，容差相对宽松。
- **VLM 老师质量/成本**：大 VLM 选型、是否本地可跑、生成 ~5191 帧×6 视角的算力与时间。
- **幻觉**：VLM 对港口专有物体（IGV/吊具）可能识别不准；用几何事实约束 + 抽检。
- **部署**：推理纯 LiDAR，LLM 分支默认不上车；若要上车再议量化。
- **token 数**：agent 数不定，截断/padding。

## 4. 进度与下一步

### 4.1 已完成（细节见第 0 部分复现手册 / 第 3 节设计）

1. ✅ 默认 LLM 底座选定：Qwen2.5-0.5B（d_llm=896）。Qwen3-0.6B 试验底座已下载，
   对应 `d_llm=1024`，并已在隔离环境 `/mnt/disk1/conda_envs/uniad_train_qwen3_py39`
   中 build/smoke 通过。
2. ✅ 方案定调：VLM 图像老师 + 纯 LiDAR 推理（跨模态蒸馏）；几何模板互补。
3. ✅ token 边界 + caption schema（scene-level summary）锁定。见 3.2.1 / 3.2.2。
4. ✅ 图像↔LiDAR 同步：`add_cam_sync.py` → `kl_infos_{train,val}_with_cam.pkl`
   （front 命中 train 98.58% / val 98.19%）。**train/val 均已重做成 6 路**（2026-06-22）。见 0.4 Step 1 / 4.3-A。
5. ✅ 几何事实生成：`gen_geo_facts.py` → `..._with_cam_geo.pkl`（train/val 全量）。
   阈值数据化校准（见 0.4 Step 2 与 4.2 经验）。前视 gate 收紧的弯路已回退（见 4.3-B）。
6. ✅ VLM caption 多相机化：`gen_vlm_caption.py`，prompt 设计 + 按方位路由 2-4 路环视 +
   6 路 val 子集批量生成（805 帧，797 带 summary）。teacher activity_addressed 17%→74.5%（见 4.3-B）。
   见 0.4 Step 4 与 4.3-B。
7. ✅ `LLMBridgeHead` + config + smoke test（2026-06-22~23）：caption 注入正确、
   llm.loss_llm 有限；历史 full-task 分支可与 track/motion/occ 共存；当前 Stage-1 probe 分支只返回
   `llm.loss_llm` 且 backward OK。实现细节见 3.1 / 3.3；代码 `llm_bridge_head.py`、
   `uniad_motion_lidar.py`、`kl_dataset.py`、config `base_e2e_lidar_occ_llm.py` /
   `base_e2e_lidar_llm_probe.py`。
8. ✅ train 全量 caption（2026-06-22）：7 卡分片 ~2.2h（错开启动避 OOM，见 4.2）→ merge →
   `kl_infos_train_vlmcap.pkl`（43747/43981=99.5% 带 summary）。val/test 使用
   `kl_infos_val_sub6cam_v3_vlmcap.pkl`。
9. ✅ 历史 full-task `base_e2e_lidar_occ_llm_train.py` 已训练 1 epoch（2026-06-23）：
   `llm.loss_llm` 下降并能输出中文，但 caption 评测 `model≈shuffle`，没有证明模型使用了同帧 query；
   且主任务指标较 LiDAR baseline 退化。该分支保留为历史对照，不作为当前主线。
10. ✅ 当前 Stage-1 纯探针 `base_e2e_lidar_llm_probe.py` 已落地并验证（2026-06-23）：
    继承 `base_e2e_lidar.py`，`llm_probe_only=True`、`freeze_non_llm=True`、`detach_inputs=True`；
    构建检查只剩 `llm_head.*` 可训练，单 batch forward/backward OK。**下一步：起 probe 多卡训练**（4.3-A）。
11. ✅ Qwen3-VL-8B-Instruct 已下载并单卡加载验证（2026-06-25）：
    `/mnt/disk1/models/Qwen3-VL-8B-Instruct`，`qwen_vl` 环境 `transformers==4.57.6` 可加载。
    该模型作为后续新 teacher；Qwen2.5-VL-7B 暂保留为 baseline 和回退。
12. ✅ teacher 深化入口已落地（2026-06-25）：`gen_vlm_caption.py` 支持
    `AutoModelForImageTextToText` 加载 Qwen2.5/Qwen3，并支持 `--annotate-targets` 用相机内外参
    生成 LiDAR 投影目标辅助标注。2 帧 Qwen3 pilot 已生成 plain / annotated 对比和可视化，见
    `projects/work_dirs/stage2_e2e_lidar/qwen3_teacher_deepening_vis/`。
13. ✅ 人工反馈闭环已落地（2026-06-25）：`gen_vlm_caption.py --feedback-json` 支持
    `motion_state + operation_state`，并把人工确认状态注入 teacher prompt；`eval_llm_caption.py`
    支持 `--feedback-json` 输出按目标局部解析的 `operation_human_acc`。6 目标 pilot
    （2 个吊机 working + 4 个集装箱叉车 waiting）验证通过。
14. ✅ Qwen3-0.6B student 隔离环境已落地（2026-06-25）：`/mnt/disk1/conda_envs/uniad_train_qwen3_py39`
    使用 Python 3.9、torch1.12.1+cu116、transformers4.51.3、peft0.13.2、mmcv1.5.2；
    `LLMBridgeHead` 单独 loss smoke 通过，`base_e2e_lidar_llm_probe_qwen3_0p6b.py`
    经真实 `third_party.uniad_mmdet3d.models.builder.build_model` 验证可构建。
15. ✅ Qwen3-VL-32B teacher val 已跑通（2026-06-25）：模型下载到
    `/mnt/disk1/models/Qwen3-VL-32B-Instruct`，4GPU `device_map=auto` 加载验证，8GPU
    2×4 shard 完成 805 帧 annotated caption。相机兜底修复后 `missing_summary=0`，
    并根据人工抽检收紧 moving/waiting/不可见吊机规则；预览见
    `outputs/qwen32b_val_preview/overview.jpg` 和 `outputs/qwen32b_state_qa_preview/overview.jpg`。

### 4.2 经验教训（踩过的坑，复现必读）

- **bf16 强制**（训练）：Qwen0.5B 在 fp16 下 LM loss=nan，必须 bf16（4090 支持）或 fp32。
  projector/spatial_pe 在 fp32 算、输出再 cast 到 bf16 喂 LLM（否则 Linear dtype 报错）。
- **装包别动 torch**：pip 装 transformers 时 accelerate 会强升 torch→2.4，打断 mmcv.ops
  CUDA 扩展。装 LLM 依赖用 `transformers==4.46.3 peft==0.13.2`，并 `--no-deps` 锁
  `torch==1.12.1+cu116 torchvision==0.13.1+cu116`。
- **Qwen3 student 不进主 env**：Qwen3-0.6B 需要 transformers>=4.51，但该版本会 import
  torch2-only 符号。当前解法是独立 Python3.9 环境 + `sitecustomize.py` shim，且已验证
  `build_model` 和 head loss；不要直接把主 `uniad_train` 升到 transformers4.51+。
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

**A. Stage-1 纯探针训练（当前主线）**。数据准备已完成，当前要跑的是 probe config，而不是历史
`base_e2e_lidar_occ_llm_train.py`。
   ```bash
   # train pkl、val v3 pkl、caption sidecar 合并均已完成；只需启动当前主线训练。
   ./tools/uniad_dist_train.sh \
     projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe.py <GPUS>
   ```
   单卡调试可用：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python tools/train.py \
     projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_probe.py \
     --work-dir projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_llm_probe \
     --no-validate
   ```
   历史 full-task 命令
   `uniad_dist_train.sh projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train.py <GPUS>`
   仅作对照复现，不再作为当前建议入口。

**A0. Qwen3-VL-32B teacher QA / train 全量闸门**。32B annotated val 已跑完，现阶段不要直接
重做 train 全量；先确认作业/空闲状态质量。当前 val 指标：`activity_addressed=0.895`，
`activity_busy=0.084`，`halluc=0.991`。`activity_busy` 低不一定是坏事，可能是 32B 更常判空闲，
但必须人工抽检 Crane/Forklift/ContainerForkLift 的 working/waiting/idle。
   ```bash
   # 1) 指标复核
   conda activate uniad_train
   python tools/analysis_tools/eval_llm_caption.py \
     --pkl-path data/kl_8/kl_infos_val_sub6cam_qwen3vl32b_annotated_vlmcap.pkl \
     --mode teacher

   # 2) 人工看图抽检
   # 打开 outputs/qwen32b_val_preview/overview.jpg 和 captions.md。

   # 3) 质量确认后再跑 train 全量（估算约 40+ 小时，8GPU）
   PKL=data/kl_8/kl_infos_train_with_cam_geo.pkl \
   OUT_PREFIX=/mnt/disk1/tmp/train_qwen3vl32b_annotated_vlmcap \
   ANNOTATED_DIR=/mnt/disk1/tmp/train_qwen3vl32b_annotated_imgs \
   MERGED_OUT=data/kl_8/kl_infos_train_qwen3vl32b_annotated_vlmcap.pkl \
   bash tools/run_vlm_caption_qwen32b_val_2x4gpu.sh
   ```
   注意：`run_vlm_caption_qwen32b_val_2x4gpu.sh` 名字里带 val，但通过上面的环境变量可跑 train；
   它输出两个 shard sidecar，按脚本结尾提示用 `merge_summaries.py` 合并。

**B. 评估指标**（已落地 `tools/analysis_tools/eval_llm_caption.py`）：
   规则解析 caption → 关键语义命中率（conflict 类别/方位、TTC 桶、ego_advice、activity、幻觉），
   主对 geo_facts。baseline：`template`（几何直出，floor）/`teacher`（VLM summary，蒸馏上限）
   已可跑；`model`/`shuffle`/`noquery` **已实现**（`_run_model_captions`：建 detector+ckpt、
   跑 dataloader 时用 wrapper 抓每帧 (query,centres)，再按 mode 生成；shuffle 错位的是 query 不是 GT）。
   需 `--config --checkpoint` + `PYTHONPATH=$(pwd)`。历史 full-task ckpt 已跑过一次；当前 Stage-1
   probe ckpt 还待训练后正式评测。
   - ⚠️ **走过的弯路（已纠正，复现必读）**：曾以为 activity 召回低的根因是"门控把后方目标也
     标了、前视相机看不到"，于是把 activity_gate **收紧到只对前视扇区**（_FRONT_SECTORS +
     gate_visible_m）。方向错了——正解不是"剔掉后方目标"，而是**上 6 路环视让后方目标可见**。
     该收紧逻辑已从 `gen_geo_facts.py` **回退**，activity_gate 恢复全方位（static≥2s & crane≤30m）。
   - ⚠️ **评测指标 bug（已修，2026-06-22）**：旧 `activity` 指标用 `pred==gt['has_activity']`，
     而 `pred` 只匹配"作业/装卸"两词，**把 VLM 正确判出的"空闲/等待"当成漏报**。门控本意是
     三选一（装卸/等待/空闲），"空闲"是有效判断。旧 14.3%~15.6% 严重低估了真实效果。
     已拆成两个指标：`activity_addressed`（gated 帧是否给出三态之一，衡量"VLM 有没有看图回应
     门控"）/ `activity_busy`（其中判"装卸作业"的比例，是 rate 不是命中分）。
   - ⚠️ **人工反馈评测解析 bug（已修，2026-06-25）**：中文短语“等待作业”应是
     `waiting`，不能先被“作业”命中为 `working`；同帧可能同时出现吊机 working 与集装箱叉车
     waiting，因此 `operation_human_acc` 必须按 `track_id` 对应目标的类别/方位附近短语解析，
     不能只做整句场景级解析。另一个细节是类别名要按长词优先解析，避免“集装箱叉车”同时被误算成
     “叉车”并污染 hallucination 指标。
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
   - `shuffle` 对照（用错位帧的 query 生成、对本帧 GT 评分）是证伪试金石：若打乱后命中不掉 =
     LLM 没真用 LiDAR query、只靠语言先验/类别分布。**判决标准（按指标分层理解，不是看总分/loss）**：
     ① model 在 **activity_addressed 等非模板语义**上超过 template（template 没图像监督、这类天然弱）；
     ② model 在关键指标上**明显高于 shuffle/noquery**（这条最硬——证明真读了 query）；
     ③ geometry 指标（conflict_cls/bearing/ttc）不明显崩——注意 template 在这些上是**规则上限**，
     model 不必全面超过它，持平即可；④ halluc（无幻觉率）不明显恶化。
     若 `shuffle ≈ model` → 蒸馏没迁移，应收缩为结构化语义 head、LLM 只做自然语言表达。
   - ⚠️ **历史 full-task epoch1 结论（2026-06-23）**：`llm.loss_llm` 从约 2.43 降到约 0.66，
     但 loss 下降不等于 query 语义迁移。`--limit 50` caption 评测中，`model` 与 `shuffle`
     基本相同：`activity_addressed=0.680`、`advice=0.700`，而 `noquery` 退化为
     `activity_addressed=0.000`、`advice=1.000`。说明 object tokens 相比 noquery 有用，
     但没有证据表明模型使用了**同帧** query 语义。该 ckpt 的中文可视化在
     `projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train/llm_caption_vis/`，
     现象是能输出中文，但有重复、套话和空间事实偏差。
   - ⚠️ **历史 full-task 对主任务有扰动**：epoch1 val 指标为 `mAP=0.8033`、`NDS=0.8102`、
     `AMOTA=0.7238`、`motion_min_ade=0.4246`、`motion_mr=0.1364`；相比 LiDAR baseline
     （约 `mAP=0.8362`、`NDS=0.8449`、`AMOTA=0.8199`、`motion_min_ade=0.3505`、
     `motion_mr=0.0868`）明显退化。这是改成 Stage-1 冻结纯探针的重要原因。
   - ⚠️ **同帧对比**：model 系只跑 queue-able 子集（v3 805→438 帧，dataloader 序），与 template/teacher
     的全量 805 帧**分母不同**。下结论前必须给 template/teacher 加 `--queue-eval --config <cfg>`
     在同一批 438 帧上重算（实测 teacher activity_addressed 全量 0.745、queue-eval 0.778——上限是 0.778）。

**C. 优先改进**：
   - **(最高) Qwen3-VL-8B teacher A/B**：Qwen3 已下载验证，`--annotate-targets` teacher 深化入口已落地，
     但新 teacher 质量还未在本数据上全量量化。先跑 val v3 plain/annotated，不直接重做 train 全量。
   - **(最高) 跑 Stage-1 纯探针并做 model/shuffle/noquery 同帧评测**：这是当前能否成立的决定性实验。
     只有 `base_e2e_lidar_llm_probe.py` 的 `model` 明显高于 `shuffle/noquery`，才能说 frozen LiDAR query
     里有可被 LLM 读出的场景语义。
   - **(高) 提升 VLM activity 召回** — ✅ 已根治：6 路环视 + 评测指标修正 + gate 目标入 prompt
     + 排除非作业类，activity_addressed 17%→74.5%（见 B）。train pkl 已是 6 路+gate已清理，可直接跑全量。
   - ✅ **gate 误标非作业类（已做，2026-06-22）**：`_NO_ACTIVITY_IDS={0,1,9}` 排除锥桶/行人/小车
     （参照 `_NO_CONFLICT_IDS`），gate 目标 2481→1540。见 4.3-B。
   - (中) 显式特征入 head：当前只喂 track_query+box center；可拼 class logits / bbox 尺寸/yaw /
     velocity / conflict-advice 结构特征，提升可控性（评注第 2 条，训练后迭代）。
   - (低) 多卡 DDP 全量完整验证（smoke 只跑 3 iter）。
