# KL LiDAR Motion Turn-Aware 工作总结

更新时间：2026-06-29
主要相关提交：`dd632f7 feat(motion): add turn-aware training and analysis tooling`，
`d904ca8 feat(motion): add turnloss ablation configs and comparison tooling`，
`b63c4af feat(motion): add focused turn-aware comparison visualizers`

## 1. 背景

最近这轮工作围绕 `stage2_e2e_lidar` 的 motion prediction 展开。最初的问题是：

- `base_e2e_lidar.py` 训练出来的 motion 预测对直线效果较好；
- 对港口车辆的转弯、横向移动、作业式转向效果明显偏弱；
- 原始 KL motion anchor 主要被静止和直行样本主导，vehicle 组缺少真正的转弯 mode；
- 港口导航地图并不是所有车辆的强约束，尤其 Crane、Forklift、ContainerForklift 和遥控 IGV 经常不按地图走。

因此这轮工作不是只做一个单点改动，而是补了一套 motion 训练、评估、可视化和 failure mining 工具，方便后续系统性做消融。

## 2. 已经确认的现象

### 2.1 原始 anchor 的问题

原始 `data/others/motion_anchor_infos_kl.pkl` 的 vehicle 组基本是：

- 静止；
- 不同速度的直行；
- 小幅横向偏移；
- 缺少真正大角度转弯或作业式转向模板。

这会导致 MotionHead 虽然有 6 个 mode，但 mode 的初始意图覆盖不够。对直行车来说足够，对港口转弯车辆不够。

### 2.2 第一版 turn-aware anchor 的效果

我们先生成了 weighted-kmeans 版本：

```text
data/others/motion_anchor_infos_kl_turnaware_3grp.pkl
```

它的主要变化是：

- 车辆组增加左右转弯 anchor；
- Cone 单独分组，避免 Cone 被误分到 pedestrian/vehicle anchor；
- `num_anchor=6` 保持不变，便于消融。

epoch6 对比结果：

| 模型 | minADE | minFDE | MR | Recall |
|---|---:|---:|---:|---:|
| old anchor | 0.33683 | 0.59734 | 0.08120 | 0.92857 |
| turn-aware anchor | 0.33709 | 0.59262 | 0.08224 | 0.92712 |

整体 `minFDE` 小幅改善，但不是全面提升。

turn-bucket 结果更有信息量：

| Bucket | old minFDE | turn-aware minFDE | 变化 |
|---|---:|---:|---:|
| static_slow | 0.1481 | 0.1462 | -1.3% |
| straight | 1.7182 | 1.6998 | -1.1% |
| mild_turn | 2.9986 | 2.8581 | -4.7% |
| sharp_turn | 3.7036 | 3.7974 | +2.5% |

结论：

- turn-aware anchor 对 `mild_turn` 有明确帮助；
- `sharp_turn` 的 oracle minFDE 变差，但 top1FDE 变好；
- 说明问题不只是 anchor 覆盖，还包括 mode assignment、分类概率选择和 loss 对转弯样本的重视程度。

### 2.3 HDMap 2x2 epoch6 消融结果

2026-06-23 对以下 4 个已经训练 6 epoch 的配置做了统一对比：

```text
base_e2e_lidar
base_e2e_lidar_turnaware
base_e2e_lidar_HDMap_old_anchor
base_e2e_lidar_HDMap
```

这里的 `base_e2e_lidar` 就是旧 anchor 的无地图基线，工作目录已经改回这个标准名字。4 个配置构成一个 2x2 消融：

| 维度 | old anchor | turn-aware anchor |
|---|---|---|
| no map | `base_e2e_lidar` | `base_e2e_lidar_turnaware` |
| hard HDMap | `base_e2e_lidar_HDMap_old_anchor` | `base_e2e_lidar_HDMap` |

epoch6 validation 总体指标：

| 配置 | 地图 | anchor | minADE | minFDE | MR | Recall | mAP | AMOTA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `base_e2e_lidar` | 无 | old | **0.33683** | 0.59741 | 0.08124 | **0.92837** | 0.84523 | 0.80039 |
| `base_e2e_lidar_turnaware` | 无 | turn-aware | 0.33708 | **0.59263** | 0.08227 | 0.92705 | 0.84096 | 0.80758 |
| `base_e2e_lidar_HDMap_old_anchor` | hard HDMap | old | 0.33847 | 0.59873 | **0.07985** | 0.92375 | 0.83880 | 0.80574 |
| `base_e2e_lidar_HDMap` | hard HDMap | turn-aware | 0.33887 | 0.60617 | 0.08150 | 0.92650 | **0.84565** | **0.80917** |

按消融轴拆开看：

| 对比 | minFDE 变化 | 结论 |
|---|---:|---|
| no-map 下 old -> turn-aware | `0.59741 -> 0.59263`，-0.00478 | turn-aware anchor 对 no-map motion 有小幅收益 |
| HDMap 下 old -> turn-aware | `0.59873 -> 0.60617`，+0.00744 | 带地图时 turn-aware 没有转化为 motion 收益 |
| old anchor 下 no-map -> HDMap | `0.59741 -> 0.59873`，+0.00132 | hard HDMap 基本没有改善 motion |
| turn-aware anchor 下 no-map -> HDMap | `0.59263 -> 0.60617`，+0.01354 | hard HDMap 明显拉大平均终点误差 |

类别级结果也支持这个判断。以 turn-aware anchor 为例，加入 hard HDMap 后：

| 类别 | no-map turn-aware minFDE | HDMap turn-aware minFDE | 变化 |
|---|---:|---:|---:|
| Truck | 0.76540 | 0.78041 | 变差 |
| Trailer-Empty | 0.62171 | 0.65099 | 变差 |
| Trailer-Full | 0.81603 | 0.84742 | 变差 |
| Crane | 1.03974 | 1.06965 | 变差 |
| Forklift | 0.44427 | 0.38037 | 变好 |
| ContainerForklift | 2.04589 | 1.98250 | 变好 |

阶段性结论：

- 这组 2x2 消融不支持“hard HDMap 能显著改善 motion”的说法。
- `base_e2e_lidar_turnaware` 是 4 个配置里 motion minFDE 最好的。
- `base_e2e_lidar_HDMap` 的 mAP/AMOTA 最好，说明地图版本的 tracking/整体检测跟踪指标不差，但 motion 平均终点误差没有受益。
- hard HDMap 对 Forklift、ContainerForklift 这类小样本/低速作业类可能有局部帮助，但对 Truck、Trailer、Crane 等主力类别没有形成稳定正收益。
- 更合理的后续方向不是继续强化 hard HDMap，而是验证 weak-map gate，让地图作为可学习弱先验。

### 2.4 turnloss 与 turn-aware + turnloss epoch6 结果

2026-06-26 补充完成以下 4 个 no-map 配置的 6 epoch 对比：

```text
base_e2e_lidar
base_e2e_lidar_turnaware
base_e2e_lidar_turnloss
base_e2e_lidar_turnaware_turnloss
```

epoch6 validation 总体指标：

| 配置 | anchor | loss | minADE | minFDE | MR | Recall | mAP | AMOTA | NDS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base_e2e_lidar` | old 2grp | 原始 | 0.33683 | 0.59741 | 0.08124 | **0.92837** | 0.84523 | 0.80039 | 0.84994 |
| `base_e2e_lidar_turnaware` | turn-aware 3grp | 原始 | 0.33708 | 0.59263 | 0.08227 | 0.92705 | 0.84096 | 0.80758 | 0.84783 |
| `base_e2e_lidar_turnloss` | old 2grp | ADE+FDE + turn weighting | **0.32767** | **0.57616** | **0.07821** | 0.92605 | 0.84045 | 0.81009 | 0.84745 |
| `base_e2e_lidar_turnaware_turnloss` | turn-aware 3grp | ADE+FDE + turn weighting | 0.32822 | 0.58010 | 0.08124 | **0.92837** | **0.85473** | **0.81174** | **0.85593** |

结论：

- `turnloss` 是当前 motion 单项最强配置，`minFDE=0.57616`、`MR=0.07821` 最好。
- `turnaware_turnloss` 的 motion 只略弱于 `turnloss`，但 `mAP/AMOTA/NDS` 最好，是当前综合最均衡配置。
- 单看总体 `minADE/minFDE` 会低估问题，因为它们是 multi-mode 的 best-of-6 指标，不能说明实际推理时最高分轨迹是否选对。

### 2.5 epoch6 turn-bucket 与 top1/oracle 诊断

epoch6 加权 Overall turn-bucket 汇总：

| 配置 | minFDE | top1FDE | MR | top1MR | Recall |
|---|---:|---:|---:|---:|---:|
| `base_e2e_lidar` | 0.6092 | 1.2884 | 0.0831 | 0.1662 | 0.9297 |
| `base_e2e_lidar_turnaware` | 0.6032 | **1.2738** | 0.0838 | **0.1657** | 0.9284 |
| `base_e2e_lidar_turnloss` | **0.5875** | 1.3137 | **0.0801** | 0.1664 | 0.9269 |
| `base_e2e_lidar_turnaware_turnloss` | 0.5915 | 1.2929 | 0.0831 | 0.1698 | 0.9295 |

sharp turn 的 top1FDE：

| 范围 | base | turnaware | turnloss | turnaware_turnloss |
|---|---:|---:|---:|---:|
| Overall sharp_turn | 7.9891 | 7.6969 | 7.6445 | **7.6010** |
| CoreVehicle sharp_turn | 9.0630 | 8.6630 | 8.6627 | **8.6171** |

这里的 `oracle` 指从模型输出的多个 motion mode 中，事后根据 GT 选择误差最小的一条轨迹；它只用于诊断，不是实际推理结果。实际推理只能用模型自己打分最高的 `top1` mode。

因此：

| 现象 | 含义 | 优先处理 |
|---|---|---|
| top1FDE 差，minFDE 好 | 候选轨迹里有对的，但分类分数选错 | mode scoring / assignment |
| top1FDE 差，minFDE 也差 | 候选本身覆盖不足 | anchor / regression |
| missed track | motion head 没拿到正确目标 | tracking / detection |

当前最关键的问题是第一类：不少转弯样本 `minFDE` 明显好于 `top1FDE`，说明“轨迹有了，但没选中”。继续只改 anchor 的边际收益可能有限，下一步应优先改 mode scoring / assignment。

### 2.6 epoch6 可视化复查结论

已生成对比可视化：

```text
projects/work_dirs/stage2_e2e_lidar/visual_compare_epoch6_turnaware_turnloss/focus/index.html
projects/work_dirs/stage2_e2e_lidar/visual_compare_epoch6_turnaware_turnloss/full_bev/frames/index.html
projects/work_dirs/stage2_e2e_lidar/visual_compare_epoch6_turnaware_turnloss/full_bev/frames/pytorch_bev_vis.webm
```

代表样本中可以看到三类情况：

- `turnaware_turnloss` 在部分 sharp turn 上能补到其他模型 missed 的目标，或给出更接近 GT 的候选。
- 部分 sharp turn 中 `turnaware_turnloss` 的 `minFDE` 不差，但 `top1FDE` 很差，典型是 mode 选择错误。
- mild turn / straight 上存在回退样本，说明转弯加权不能继续无约束加大，否则会牺牲常规移动样本。

这支持一个更细的判断：`turnaware_turnloss` 是值得保留的综合候选，但当前主要瓶颈已经从“有没有转弯候选”转向“能不能把正确候选排到 top1”。

## 3. 本轮代码改动

### 3.1 TrajLoss 支持新的 mode assignment

修改文件：

```text
projects/mmdet3d_plugin/losses/traj_loss.py
```

新增参数：

```python
best_mode_metric='minade' | 'minfde' | 'ade_fde'
fde_weight=0.5
```

默认仍然是 `minade`，所以旧配置行为不变。

新实验配置使用：

```python
best_mode_metric='ade_fde'
fde_weight=0.5
```

含义是用：

```text
best_score = ADE + 0.5 * FDE
```

选择 best mode。这样分类 loss 不再只鼓励平均距离小的 mode，而是同时关注终点误差。这个改动主要针对转弯场景，因为转弯预测经常前半段还可以，但终点方向偏掉。

### 3.2 TrajLoss 支持 turn-bucket 样本加权

新增参数：

```python
turn_loss_weights=dict(
    static_slow=0.7,
    straight=1.0,
    mild_turn=1.5,
    sharp_turn=2.0)
```

训练时会根据 GT future trajectory 自动分 bucket：

- `static_slow`
- `straight`
- `mild_turn`
- `sharp_turn`

然后对对应样本加权。权重会按 batch mean 归一化，避免整体 loss 尺度被放大。

这个改动的目的：

- 减少静止/直行样本对训练目标的主导；
- 让模型在训练时真正更重视转弯样本；
- 配合 turn-aware anchor 使用，而不是只在 anchor 生成阶段加权。

### 3.3 TrajLoss 支持只作用于 mode classification 的 turn 权重

2026-06-26 进一步新增参数：

```python
turn_cls_loss_weights=dict(
    static_slow=1.0,
    straight=1.0,
    mild_turn=1.5,
    sharp_turn=2.5)
```

它和 `turn_loss_weights` 的区别：

| 参数 | 作用范围 | 目的 |
|---|---|---|
| `turn_loss_weights` | 分类、回归、minADE/minFDE 统计都会加权 | 让整个 motion loss 更重视转弯样本 |
| `turn_cls_loss_weights` | 只额外作用于 `l_class` mode classification | 让正确转弯候选更容易排到 top1 |

新增配置：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_turnaware_modescore.py
```

该配置继承 `base_e2e_lidar_turnaware_turnloss.py`，保持 turn-aware anchor、ADE+FDE best-mode 选择和基础 turn loss 不变，只额外加强转弯样本的 mode classification。这个实验专门验证 `top1FDE - minFDE` gap 是否能缩小。

### 3.4 新增 stratified K=6 anchor 生成

新增脚本：

```text
tools/generate_kl_motion_anchors_stratified.py
```

输出文件：

```text
data/others/motion_anchor_infos_kl_stratified_k6_3grp.pkl
```

注意：`data/others` 是 symlink，生成的 pkl 没有进入 git。换机器时需要重新生成。

stratified K=6 的核心思想是先规定每个 mode 的职责，再生成 anchor，而不是让 kmeans 自己决定 6 个中心。

vehicle 组 6 个 mode：

| mode | 含义 |
|---|---|
| 0 | static_slow |
| 1 | straight_short |
| 2 | straight_long |
| 3 | negative_lateral_turn |
| 4 | positive_lateral_turn |
| 5 | maneuver_sharp |

这样做的理由：

- weighted-kmeans 虽然补了转弯，但仍然可能被样本分布拉直；
- stratified K=6 保证 6 个 mode 中一定有左右转和 maneuver/sharp；
- `num_anchor=6` 不变，网络输出形状不变，消融更干净。

### 3.5 新增 weak-map 配置

新增配置：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_HDMap_weak.py
```

核心设置：

```python
map_agent_scope='all'
transformerlayers=dict(map_gate_init=-4.0)
```

含义：

- 仍然给所有 agent 看附近地图；
- 但 MotionFormer 初始更接近 no-map 分支；
- 地图从强约束变成弱先验，由网络自己学习什么时候用地图。

这个更符合港口场景：

- 自车或自动驾驶 IGV 可能更贴合地图；
- 其他车辆、吊车、叉车、作业车辆不一定受导航地图约束；
- 即使 IGV 也可能被遥控，不总是按地图走。

### 3.6 新增 planning-only HDMap fusion B 方案

2026-06-29 补充了一个更贴合港口规控场景的地图使用方式：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse.py
```

背景判断：

- 前面的 2x2 消融说明，导航 HDMap 对所有 agent 的 motion prediction 没有稳定正收益；
- 这不代表地图对自车 planning 没有价值，因为港口规控模块本来就会参考导航地图；
- 港口其他车辆、作业车辆、遥控 IGV 不一定按地图行驶，因此地图不适合继续作为所有 actor 的强 motion 约束；
- 更合理的拆法是：motion 默认不吃地图或只做弱先验，planning 单独融合地图作为自车路径先验。

新方案的核心设置：

```python
model = dict(
    map_lane_encoder=dict(
        map_path='data/kl_8/map/base_map.txt',
        num_lanes=64,
        num_points_per_lane=20,
    ),
    motion_head=dict(
        map_agent_scope='none',
    ),
    planning_head=dict(
        use_map_lane=True,
        map_local_k=16,
        map_attn_layers=1,
        map_gate_init=-2.0,
    ))
```

也就是说，地图仍然由 detector 里的 `MapLaneEncoder` 编码，但只给 `PlanningHeadSingleMode`
使用；`motion_head.map_agent_scope='none'` 显式关闭 motion 对地图的使用，避免把地图对 planning
的影响和 motion 分支混在一起。

融合流程：

```text
base_map.txt
  -> MapLaneEncoder
     -> 按当前帧 ego2global 转到自车坐标系
     -> 裁剪 pc_range 内最近 64 条 lane
     -> 每条 lane 重采样 20 个点
     -> outs_map:
        lane_query / lane_query_pos / lane_valid / lane_centroids

sdc_traj_query + sdc_track_query + command embedding
  -> 原始 plan_query
  -> 对最近 map_local_k=16 条 lane 做 cross-attention
  -> residual map delta
  -> 原 UniAD planning BEV cross-attention
  -> reg_branch
  -> 6-step SDC planning trajectory
```

代码入口：

| 文件 | 作用 |
|---|---|
| `projects/mmdet3d_plugin/uniad/detectors/uniad_motion_lidar.py` | `_build_outs_map()` 编码 HDMap，并在 train/test 中把 `outs_map` 传给 planning head |
| `projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py` | 新增 `use_map_lane`、`_lane_memory()` 和 `_apply_map_lane_attention()` |
| `projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse.py` | planning-only HDMap fusion 配置 |

实现上的保护：

- `PlanningHeadSingleMode` 默认 `use_map_lane=False`，旧配置不受影响；
- 新增 `map_delta_proj` 初始化为 0，所以从旧 no-map planner checkpoint warm-start 时，初始输出近似原 planner；
- `map_gate_init=-2.0` 让地图初始是弱影响，再由 planning loss 学会何时使用；
- 只选最近 `map_local_k=16` 条 lane，避免全局地图 lane 对 planning query 造成过多噪声。

当前 `base_e2e_lidar_plan.py` 的 no-map planning baseline 是 6 步、约 3 秒规划。`epoch_1`
评估结果记录如下，后续 map-fuse 需要按同 epoch 对齐比较：

| 指标 | no-map planner epoch1 |
|---|---:|
| L2 @ 1s | 0.562 m |
| L2 @ 2s | 1.239 m |
| L2 @ 3s | 2.180 m |
| avg.L2 | 1.327 m |
| avg.Collision | 5.47% |
| Left avg.L2 / Collision | 0.893 m / 0.66% |
| Right avg.L2 / Collision | 1.174 m / 1.55% |
| Straight avg.L2 / Collision | 1.365 m / 6.10% |

### 3.7 SDC planning GT 分布诊断

2026-06-29 增加了 planning GT 分布统计脚本：

```text
tools/analysis_tools/analyze_sdc_planning_distribution.py
```

运行命令：

```bash
conda activate uniad_train
python tools/analysis_tools/analyze_sdc_planning_distribution.py \
  --infos data/kl_8/kl_infos_train.pkl data/kl_8/kl_infos_val.pkl \
  --splits train val \
  --out-dir projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution
```

统计口径：

- `sdc_planning`: `[1, 6, 3]`，6 步约 3 秒；
- 有效步沿用 planning eval 口径：`sdc_planning_mask[..., 0] > 0`；
- `static`: 3 秒终点位移 `< 0.5 m`；
- `slow`: `0.5 m <= 3 秒终点位移 < 2.0 m`；
- `moving`: 3 秒终点位移 `>= 2.0 m`；
- `turning`: 终点位移不低速，且 heading/yaw 变化 `>= 15 deg` 或横向比例较大；
- command 沿用 `Right=0, Left=1, Straight=2`。

当前统计结果：

| split | N | static | slow | moving | turning | final disp p50 | avg speed p50 | Straight cmd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 41912 | 14.5% | 15.7% | 69.8% | 3.5% | 4.82 m | 1.70 m/s | 89.8% |
| val | 4963 | 16.3% | 24.1% | 59.6% | 2.9% | 3.54 m | 1.34 m/s | 87.4% |

更细的观察：

- `static + slow` 在 train 是 30.2%，在 val 是 40.4%，val 的低速偏置更重；
- 静止样本几乎全部属于 `Straight` command；
- 多个 scene 是 100% static，例如 `20251105_007/202511050952_record`、
  `20251108_007/202511080934_record`、`20251105_007/202511050951_record`；
- 这很像“自车停在固定位置采集检测/作业目标”的片段，对 planning 会形成保守低速偏置；
- 但 train 仍有约 69.8% moving，不能简单说数据集绝大多数都是停车，问题主要是平均指标会被静止/低速样本稀释。

输出文件：

```text
projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution/summary.csv
projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution/top_static_scenes.csv
projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution/train_samples.csv
projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution/val_samples.csv
projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution/train/*.png
projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution/val/*.png
```

对 planning 实验的影响：

- 不能只看 overall `avg.L2`，否则大量静止/低速样本会让指标看起来偏好；
- 需要额外报告 `static / slow / moving / turning` 分桶指标；
- 如果模型在 moving/turning 上明显偏慢或停住，即使 overall L2 不差，也说明 planning 没学好；
- 后续可考虑训练时对明显采集式 static scene 降权，或对 moving/turning 样本加权，但不要直接删除所有停车样本，因为港口等待停车也是真实场景。

### 3.8 planning final-displacement 诊断与 front-obstacle 切分

2026-07-01 进一步把 planning 诊断从 “L2 / collision” 扩展到 “终点位移是否偏短/偏长”。

修改文件：

```text
projects/mmdet3d_plugin/datasets/kl_dataset.py
tools/analysis_tools/diagnose_planning_disp.py
```

`KlTrackDataset._evaluate_planning()` 现在会额外输出：

```text
planning/pred_final_disp
planning/gt_final_disp
planning/final_disp_ratio
planning/<Command>/pred_final_disp
planning/<Command>/gt_final_disp
planning/<Command>/final_disp_ratio
planning/<Bucket>/pred_final_disp
planning/<Bucket>/gt_final_disp
planning/<Bucket>/final_disp_ratio
planning/FrontClear/*
planning/FrontObstacle/*
```

其中：

```text
final_disp_ratio = pred_final_disp / gt_final_disp
```

当 GT 终点位移小于 0.5 m 时不计算 ratio，避免静止样本分母过小导致比例失真。

`FrontObstacle` 是一个粗粒度诊断切分：检查未来占用 `segmentation` 在自车前方走廊内是否有占用。
默认走廊是：

```text
x: 0-30 m
y: -6 m 到 +6 m
```

这个切分不是正式指标，而是为了验证一个假设：planning 偏短到底是来自 loss reweighting，
还是来自 collision loss 在前方有障碍时把轨迹往回拉。它需要和 `GT>=3m / GT>=4m`
一起看，避免把“GT 本来就该停车”的样本误判成预测偏短。

新增离线诊断脚本：

```text
tools/analysis_tools/diagnose_planning_disp.py
```

它直接读取 `tools/uniad_dist_eval.sh --out` 保存的 pkl，不需要重新跑 GPU 推理。输出：

```text
planning_disp_diagnostics.csv
planning_disp_diagnostics.md
```

脚本会按以下维度打印 `N / mean / median / IQR`：

- Overall；
- Left / Right / Straight command；
- Static / Slow / MovingStraight / Turning；
- FrontClear / FrontObstacle；
- `GT>=2m / GT>=3m / GT>=4m`；
- `MovingStraight/Turning x FrontClear/FrontObstacle x GT>=thr`。

当前对 `base_e2e_lidar_plan_mapfuse_balanced` 的 epoch1-4 诊断结论：

| checkpoint | overall ratio | MovingStraight ratio | Turning ratio | 现象 |
|---|---:|---:|---:|---|
| epoch1 | 1.005 | 0.849 | 0.658 | 转弯明显偏短 |
| epoch2 | 1.061 | 0.614 | 0.872 | 直行明显偏短，overall 被 slow/static 掩盖 |
| epoch3 | 1.034 | 1.032 | 1.111 | 长度最均衡，L2 也最好 |
| epoch4 | 1.190 | 1.070 | 1.158 | 开始整体偏长 |

因此当前 `balanced` 实验不能只看最后 epoch。epoch3 是更合理的候选，epoch4 已经出现过冲趋势。

### 3.9 normalized balanced 配置

`base_e2e_lidar_plan_mapfuse_balanced.py` 使用的原始 planning loss 权重是：

```python
planning_motion_loss_weights=dict(
    static=0.3,
    slow=0.7,
    moving_straight=1.0,
    turning=1.5,
)
```

训练日志和数据统计显示，KL train split 下这个权重均值大约是 `0.869`，不是最初担心的
`0.4-0.5`。因此它不会把 planning 有效学习率砍半，但仍然会改变样本侧重点。为了让消融更干净，
新增 normalized 版本：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_balanced_norm.py
```

归一化后的权重：

```python
planning_motion_loss_weights=dict(
    static=0.345,
    slow=0.806,
    moving_straight=1.151,
    turning=1.727,
)
```

它保持相对偏好不变，但让训练集加权均值接近 1.0。这个配置用于区分：

| 可能因素 | 诊断方式 |
|---|---|
| 绝对 loss 尺度变化 | balanced vs balanced_norm |
| 相对样本偏好变化 | uniform mapfuse vs balanced_norm |
| collision loss 偏短压力 | 后续 collision-off / collision-down ablation |
| ADE 缺少末端约束 | 后续 FDE/后段加权 ablation |

### 3.10 planning 可视化右侧 drivable 背景

`visualize_lidar_e2e_motion.py` 新增参数：

```bash
--right-background drivable_gt
```

默认仍然是右侧画点云；开启后右侧不画点云，而是画验证 pipeline 生成的 `gt_lane_masks`
drivable space。这个 mask 是 drivable head 训练和算 map IoU 用的标签。

注意当前 `eval_epoch*_planning_results.pkl` 里没有保存逐帧预测 drivable mask，只保存了每帧
`ret_iou` 和整体验证 IoU。因此这个背景是 GT/label drivable space，不是模型预测 drivable。

实现时要注意坐标轴：

```text
gt_lane_masks: row = y, col = x
BEV display:  display_x = -y, display_y = x
```

所以绘制前必须转置 mask，并使用 `origin='lower'`。最初直接 `imshow` 会看起来像差了 90 度，
已经修正。

## 4. 新增配置矩阵

### 4.1 已有 baseline

| 配置 | 地图 | anchor | loss |
|---|---|---|---|
| `base_e2e_lidar.py` | 无 | old 2grp | 原始 |
| `base_e2e_lidar_turnaware.py` | 无 | turn-aware 3grp | 原始 |
| `base_e2e_lidar_HDMap_old_anchor.py` | hard map | old 2grp | 原始 |
| `base_e2e_lidar_HDMap.py` | hard map | turn-aware 3grp | 原始 |
| `base_e2e_lidar_HDMap_weak.py` | weak map | turn-aware 3grp | 原始 |

### 4.2 只改 loss 的新实验

| 配置 | 地图 | anchor | loss |
|---|---|---|---|
| `base_e2e_lidar_turnloss.py` | 无 | old 2grp | ADE+FDE + turn weighting |
| `base_e2e_lidar_turnaware_turnloss.py` | 无 | turn-aware 3grp | ADE+FDE + turn weighting |
| `base_e2e_lidar_turnaware_modescore.py` | 无 | turn-aware 3grp | ADE+FDE + turn weighting + turn mode scoring |
| `deferred/base_e2e_lidar_HDMap_turnloss.py` | hard map | turn-aware 3grp | ADE+FDE + turn weighting |
| `deferred/base_e2e_lidar_HDMap_weak_turnloss.py` | weak map | turn-aware 3grp | ADE+FDE + turn weighting |

### 4.3 loss + stratified K=6 anchor 的扩展实验（暂放 deferred）

| 配置 | 地图 | anchor | loss |
|---|---|---|---|
| `deferred/base_e2e_lidar_stratified_k6_turnloss.py` | 无 | stratified K=6 | ADE+FDE + turn weighting |
| `deferred/base_e2e_lidar_HDMap_stratified_k6_turnloss.py` | hard map | stratified K=6 | ADE+FDE + turn weighting |
| `deferred/base_e2e_lidar_HDMap_weak_stratified_k6_turnloss.py` | weak map | stratified K=6 | ADE+FDE + turn weighting |

### 4.4 planning 地图融合实验

| 配置 | planning 地图 | motion 地图 | 起点 |
|---|---|---|---|
| `base_e2e_lidar_plan.py` | 无 | 无 | `base_e2e_lidar/latest.pth` |
| `base_e2e_lidar_plan_mapfuse.py` | planning-only HDMap lane attention | 关闭，`map_agent_scope='none'` | `base_e2e_lidar_plan/latest.pth` |
| `base_e2e_lidar_plan_mapfuse_balanced.py` | planning-only HDMap lane attention + GT bucket reweight | 关闭，`map_agent_scope='none'` | `base_e2e_lidar_plan/latest.pth` |
| `base_e2e_lidar_plan_mapfuse_balanced_norm.py` | planning-only HDMap lane attention + 均值归一化 GT bucket reweight | 关闭，`map_agent_scope='none'` | `base_e2e_lidar_plan/latest.pth` |

这组实验不要和 motion HDMap 实验混在一起解读。它回答的是：导航 HDMap 作为自车规控先验，能否改善
SDC planning 的 L2 和 collision；不是回答“地图能否约束所有 agent 的 motion prediction”。

`base_e2e_lidar_plan_mapfuse_balanced.py` 是训练侧去偏实验：不删样本，只在 planning loss 上降低
采集式静止/低速样本权重，提高转弯样本权重。

`base_e2e_lidar_plan_mapfuse_balanced_norm.py` 是同一想法的更干净消融：保留相对权重，但让训练集
平均权重约等于 1.0，避免把“loss 总尺度变化”和“样本侧重点变化”混在一起。

## 5. 推荐训练顺序

为了保证消融清楚，建议按下面顺序跑：

### 5.1 第一组：验证 loss 是否有效

先跑：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_turnloss.py
```

对比：

```text
base_e2e_lidar.py
```

这样只看 loss 改动，不混入地图和 turn-aware anchor。

重点观察：

- Overall motion minFDE；
- `mild_turn` minFDE；
- `sharp_turn` minFDE；
- `wrong_mode` failure 数量；
- top1FDE 与 minFDE 的 gap 是否缩小。

如果这一步有效，再把同一套 loss 放到 turn-aware anchor 上，跑：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_turnaware_turnloss.py
```

这样才能区分“loss 有效”还是“old anchor 偶然更适配”。

### 5.2 第二组：验证 weak map 是否比 hard map 更合理

如果 no-map loss 有收益，再跑：

```text
deferred/base_e2e_lidar_HDMap_turnloss.py
deferred/base_e2e_lidar_HDMap_weak_turnloss.py
```

重点比较：

```text
base_e2e_lidar_turnloss.py
deferred/base_e2e_lidar_HDMap_turnloss.py
deferred/base_e2e_lidar_HDMap_weak_turnloss.py
```

这样可以判断地图到底是帮助 motion，还是对非地图约束车辆造成干扰。

### 5.3 第三组：验证 planning-only HDMap fusion

新增 B 方案后，建议先跑：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse.py
```

训练命令：

```bash
conda activate uniad_train
./tools/uniad_dist_train.sh projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse.py 4
```

对比：

```text
base_e2e_lidar_plan.py
base_e2e_lidar_plan_mapfuse.py
```

重点观察：

- L2 @ 1s / 2s / 3s；
- avg.L2；
- avg.Collision；
- Left / Right / Straight 分桶；
- Static / Slow / MovingStraight / Turning GT 运动分桶；
- 可视化中自车轨迹是否更贴近地图方向，尤其是中后段是否少发散。

2026-06-29 已把 GT 运动分桶接进 `KlTrackDataset._evaluate_planning()`，正常 eval 会额外输出：

```text
planning/Static/avg.L2
planning/Slow/avg.L2
planning/MovingStraight/avg.L2
planning/Turning/avg.L2
planning/<Bucket>/avg.Collision
planning/<Bucket>/N
```

分桶口径与 `analyze_sdc_planning_distribution.py` 对齐：`Static` 是 3 秒终点位移小于
0.5 m，`Slow` 是 0.5-2.0 m，`MovingStraight` 是移动但不明显转弯，`Turning` 是移动且
heading/yaw 变化或横向偏移明显。

这一步应按相同 epoch 对齐比较。如果从 `base_e2e_lidar_plan/latest.pth` warm-start，
则要单独记录“map-fuse fine-tune 了几轮”，不要直接和从零开始的 no-map planning epoch 数混淆。

如果 mapfuse 的分桶结果显示 `Static/Slow` 好看但 `MovingStraight/Turning` 不理想，再跑训练侧去偏配置：

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_balanced.py
projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_balanced_norm.py
```

核心权重：

```python
planning_motion_loss_weights=dict(
    static=0.3,
    slow=0.7,
    moving_straight=1.0,
    turning=1.5,
)
```

它只影响 `planning.loss_ade` 和 `planning.loss_collision_*`，不影响 track、map、motion、occ，也不改变数据
pipeline。该配置和 `base_e2e_lidar_plan_mapfuse.py` 一样从 `base_e2e_lidar_plan/latest.pth`
开始，方便做同起点消融：一个是纯 map-fuse，一个是 map-fuse + planning GT bucket reweight。

评估时必须同时看 `final_disp_ratio`，否则可能出现两种误判：

- avg.L2 下降，但轨迹系统性偏短或偏长；
- overall ratio 接近 1，但 MovingStraight / Turning 某一类明显偏短。

当前 `balanced` 的 epoch3 是长度最均衡的 checkpoint；epoch4 虽然已经训练更多，但
`final_disp_ratio` 显示整体偏长，不能简单认为“epoch 越靠后越好”。

### 5.4 第四组：验证 stratified K=6 anchor

如果 loss 方向正确，再跑：

```text
deferred/base_e2e_lidar_stratified_k6_turnloss.py
```

对比：

```text
base_e2e_lidar_turnloss.py
```

这样只看 stratified anchor 相对 weighted-kmeans anchor 是否继续提升。

## 6. 新增分析工具

### 6.1 turn-bucket 指标

新增脚本：

```text
tools/analysis_tools/compute_turn_bucket_metrics.py
```

它会把 GT future trajectory 分成：

- `static_slow`
- `straight`
- `mild_turn`
- `sharp_turn`

然后分别统计：

- GT 数量；
- matched 数量；
- recall；
- minADE；
- minFDE；
- MR；
- top1ADE；
- top1FDE；
- top1MR。

这比只看整体 minADE/minFDE 更有诊断价值。

### 6.2 failure mining

同一个脚本还会输出 failure CSV，默认文件名：

```text
<out-prefix>_failures.csv
```

failure 类型：

| 类型 | 含义 | 优先排查方向 |
|---|---|---|
| `missed_track` | GT 没有匹配到预测框 | tracking / matching |
| `poor_oracle` | 多模态中最好的 mode 也差 | anchor 覆盖 / 回归能力 |
| `wrong_mode` | oracle 好但 top1 差 | mode assignment / 分类 loss |
| `sharp_turn_fail` | sharp turn top1 明显失败 | turn anchor / turn loss |

这个工具的目的，是避免平均指标掩盖问题。比如：

- 如果 `wrong_mode` 很多，优先改 loss 的 mode assignment；
- 如果 `poor_oracle` 很多，优先改 anchor 覆盖；
- 如果 `missed_track` 很多，motion head 再怎么调也解决不了，需要看 tracking。

### 6.3 planning final-displacement 诊断

新增脚本：

```text
tools/analysis_tools/diagnose_planning_disp.py
```

典型用法：

```bash
conda activate uniad_train
python tools/analysis_tools/diagnose_planning_disp.py \
  --results \
    plan_e1=projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan/eval_epoch1_planning_results.pkl \
    balanced_e3=projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_balanced/eval_epoch3_planning_results.pkl \
  --out-dir projects/work_dirs/stage2_e2e_lidar/planning_disp_diagnostics/current
```

它用于回答三个问题：

| 问题 | 看什么 |
|---|---|
| 模型是不是整体偏短/偏长 | `overall final_disp_ratio` |
| 偏差是否集中在直行或转弯 | `MovingStraight / Turning` |
| collision 是否把轨迹往回拉 | `FrontClear / FrontObstacle`，尤其 `GT>=3m/4m` |

这个脚本只读取 eval pkl，不需要占 GPU。

## 7. 新增可视化和训练曲线工具

新增工具：

```text
tools/analysis_tools/plot_lidar_training_metrics.py
tools/analysis_tools/visualize_lidar_e2e_motion.py
tools/analysis_tools/visualize_hdmap_on_dom.py
tools/analysis_tools/visualize_track_drivable_eval.py
```

用途：

- 画训练曲线；
- 可视化 LiDAR E2E tracking + motion；
- 在 GT 面板上叠加 HDMap；
- 在预测面板上用 `--right-background drivable_gt` 替换点云背景，显示 drivable space；
- 检查 motion GT、预测轨迹、track ID、不同车辆颜色；
- 对 failure mining 输出的 sample index 做定点复查。

`--right-background drivable_gt` 使用 `gt_lane_masks`。绘制时已经按
`display_x=-y, display_y=x` 做了转置修正，避免 drivable mask 相对车辆/地图转 90 度。

## 8. 重要注意事项

### 8.1 baseline 行为保持不变

`TrajLoss` 新增参数都是 opt-in：

- 老配置不传 `best_mode_metric` 时仍然使用 `minADE`；
- 老配置不传 `turn_loss_weights` 时不做样本加权；
- 已训练过的 baseline 对比仍然有效。

### 8.2 stratified anchor pkl 没进 git

当前 pkl 路径：

```text
data/others/motion_anchor_infos_kl_stratified_k6_3grp.pkl
```

但 `data/others` 是 symlink，git 没有追踪这个 pkl。换环境时需要重新生成。

生成方式：

```bash
conda run -n uniad_train python tools/generate_kl_motion_anchors_stratified.py \
  --info data/kl_8/kl_infos_train.pkl \
  --out data/others/motion_anchor_infos_kl_stratified_k6_3grp.pkl
```

### 8.3 地图实验要避免混淆因素

不要直接用：

```text
base_e2e_lidar.py vs base_e2e_lidar_HDMap.py
```

当作纯地图对比，因为它们同时变了地图和 anchor。

更干净的比较是：

| 问题 | 对比 |
|---|---|
| old anchor 下地图有没有用 | `base_e2e_lidar.py` vs `base_e2e_lidar_HDMap_old_anchor.py` |
| turn-aware anchor 下地图有没有用 | `base_e2e_lidar_turnaware.py` vs `base_e2e_lidar_HDMap.py` |
| weak map 是否更合理 | `base_e2e_lidar_HDMap.py` vs `base_e2e_lidar_HDMap_weak.py` |
| 新 loss 下地图有没有用 | `base_e2e_lidar_turnloss.py` vs `deferred/base_e2e_lidar_HDMap_turnloss.py` vs `deferred/base_e2e_lidar_HDMap_weak_turnloss.py` |

## 9. 当前结论

目前可以比较稳地说：

1. 原始 anchor 对港口转弯车辆覆盖不足。
2. weighted turn-aware anchor 对 `mild_turn` 有明确收益，但对 `sharp_turn` 还不稳定。
3. 单纯换 anchor 不够，必须配合 loss 的 mode assignment、mode scoring 和 turn 样本加权。
4. 当前这版 hard HDMap 没有给 motion 带来稳定正收益，尤其在 turn-aware anchor 下还会拉大 minFDE；更合理的是把地图作为弱先验继续试。
5. 后续判断改动是否有效，不能只看整体 minADE/minFDE，必须结合 turn-bucket 和 failure mining。
6. `base_e2e_lidar_turnloss` 当前 motion 单项最好；`base_e2e_lidar_turnaware_turnloss` 当前综合指标最好，并且 sharp turn top1FDE 最好，但整体 top1FDE 仍未明显领先。
7. 可视化和 top1/oracle gap 说明下一步的主要问题不是单纯“缺转弯轨迹”，而是“正确轨迹没有被排到 top1”。
8. motion 上地图无稳定正收益，不等价于 planning 上地图无用；已新增 `base_e2e_lidar_plan_mapfuse.py`，把地图从 all-agent motion 约束拆成 planning-only 自车先验。
9. planning GT 存在明确静止/低速偏置，尤其 val 的 `static + slow` 达到 40.4%；后续 planning 评估必须分 static/slow/moving/turning，否则平均指标会掩盖低速保守倾向。
10. 训练侧去偏已作为独立配置保留：`base_e2e_lidar_plan_mapfuse_balanced.py`，降低 static/slow planning loss 权重，提高 turning 权重；它和 `base_e2e_lidar_plan_mapfuse.py` 同起点，适合直接消融。
11. planning 需要额外看 `final_disp_ratio`，否则 avg.L2 不能区分“轨迹方向错”和“轨迹长度偏短/偏长”。当前 `base_e2e_lidar_plan_mapfuse_balanced` 的 epoch3 长度最均衡，epoch4 已经有整体偏长趋势。
12. `balanced_norm` 是更干净的下一步消融，用于确认收益来自样本侧重点而不是 planning loss 总尺度变化。

## 10. 下一步建议

### 10.1 已完成实验的保留结论

已经完成并建议保留的 checkpoint：

```text
base_e2e_lidar_turnloss.py
base_e2e_lidar_turnaware_turnloss.py
```

使用建议：

| 目标 | 推荐 checkpoint |
|---|---|
| motion 单项最优 | `base_e2e_lidar_turnloss` |
| 综合检测/跟踪/motion 最均衡 | `base_e2e_lidar_turnaware_turnloss` |
| 原始 no-map baseline | `base_e2e_lidar` |
| anchor-only 消融 | `base_e2e_lidar_turnaware` |

### 10.2 下一轮优先方向：改 mode scoring / assignment

当前最值得做的不是继续盲目加大 turn loss，而是让模型更容易把正确候选排到 top1。

已新增配置：

```text
base_e2e_lidar_turnaware_modescore.py
```

核心思路：

- 保持 `turnaware_turnloss` 的 anchor 和基础 loss 不变；
- 只额外改 mode classification 的监督目标或权重；
- 重点减少 `top1FDE - minFDE` 的 gap；
- 不追求继续降低 oracle minFDE，而是提升实际 top1 轨迹。

可以考虑三种实现，从保守到激进：

| 方案 | 做法 | 风险 |
|---|---|---|
| A. 增大 best-mode 分类权重 | 对 `sharp_turn/mild_turn` 样本提高 mode classification loss 权重 | 最小，容易做消融 |
| B. FDE-aware soft label | 不只监督单个 best mode，而是按 FDE/ade_fde 给 6 个 mode 分软标签 | 中等，需要确认 loss 接口 |
| C. Top1 margin/ranking loss | 要求 oracle mode 分数高于错误 mode，尤其当 top1FDE 很差但 minFDE 好时 | 较大，可能影响稳定性 |

推荐先做 A，再做 B。C 适合作为后续增强，不建议第一步就上。

### 10.3 评估标准

新实验不要只看 overall `minFDE`，必须同时看：

| 指标 | 目标 |
|---|---|
| Overall top1FDE | 相比 `turnaware_turnloss` 下降 |
| CoreVehicle sharp_turn top1FDE | 优先下降 |
| `top1FDE - minFDE` gap | 明显缩小 |
| Overall minFDE/MR | 不能明显劣化 |
| straight / static_slow top1FDE | 不能明显回退 |

尤其要盯住这些对比：

```text
base_e2e_lidar_turnloss
base_e2e_lidar_turnaware_turnloss
base_e2e_lidar_turnaware_modescore
```

### 10.4 地图和 stratified anchor 放到下一阶段

motion 侧如果 mode scoring 改动有效，再继续看地图：

```text
deferred/base_e2e_lidar_HDMap_turnloss.py
deferred/base_e2e_lidar_HDMap_weak_turnloss.py
```

planning 侧已经新增更明确的地图实验：

```text
base_e2e_lidar_plan.py
base_e2e_lidar_plan_mapfuse.py
```

如果 `poor_oracle` 仍然很多，再训练：

```text
deferred/base_e2e_lidar_stratified_k6_turnloss.py
```

这样每一步都能回答一个明确问题：

- mode scoring 有没有解决 top1 选错；
- 地图对 all-agent motion 有没有用；
- weak map 是否比 hard map 更适合港口 motion；
- planning-only 地图融合是否改善自车规划；
- stratified anchor 是否比 weighted-kmeans anchor 更适合 sharp turn。
