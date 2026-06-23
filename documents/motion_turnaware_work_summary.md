# KL LiDAR Motion Turn-Aware 工作总结

更新时间：2026-06-23  
对应提交：`dd632f7 feat(motion): add turn-aware training and analysis tooling`

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

### 3.3 新增 stratified K=6 anchor 生成

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

### 3.4 新增 weak-map 配置

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
| `deferred/base_e2e_lidar_HDMap_turnloss.py` | hard map | turn-aware 3grp | ADE+FDE + turn weighting |
| `deferred/base_e2e_lidar_HDMap_weak_turnloss.py` | weak map | turn-aware 3grp | ADE+FDE + turn weighting |

### 4.3 loss + stratified K=6 anchor 的扩展实验（暂放 deferred）

| 配置 | 地图 | anchor | loss |
|---|---|---|---|
| `deferred/base_e2e_lidar_stratified_k6_turnloss.py` | 无 | stratified K=6 | ADE+FDE + turn weighting |
| `deferred/base_e2e_lidar_HDMap_stratified_k6_turnloss.py` | hard map | stratified K=6 | ADE+FDE + turn weighting |
| `deferred/base_e2e_lidar_HDMap_weak_stratified_k6_turnloss.py` | weak map | stratified K=6 | ADE+FDE + turn weighting |

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

### 5.3 第三组：验证 stratified K=6 anchor

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
- 检查 motion GT、预测轨迹、track ID、不同车辆颜色；
- 对 failure mining 输出的 sample index 做定点复查。

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
3. 单纯换 anchor 不够，必须配合 loss 的 mode assignment 和 turn 样本加权。
4. 当前这版 hard HDMap 没有给 motion 带来稳定正收益，尤其在 turn-aware anchor 下还会拉大 minFDE；更合理的是把地图作为弱先验继续试。
5. 后续判断改动是否有效，不能只看整体 minADE/minFDE，必须结合 turn-bucket 和 failure mining。

## 10. 下一步建议

优先训练：

```text
base_e2e_lidar_turnloss.py
```

如果 `wrong_mode` 明显减少、top1FDE 接近 minFDE，再训练组合版：

```text
base_e2e_lidar_turnaware_turnloss.py
```

如果组合版仍然有效，再继续看地图：

```text
deferred/base_e2e_lidar_HDMap_turnloss.py
deferred/base_e2e_lidar_HDMap_weak_turnloss.py
```

如果 `poor_oracle` 仍然很多，再训练：

```text
deferred/base_e2e_lidar_stratified_k6_turnloss.py
```

这样每一步都能回答一个明确问题：

- loss 有没有用；
- 地图有没有用；
- weak map 是否比 hard map 更适合港口；
- stratified anchor 是否比 weighted-kmeans anchor 更适合 sharp turn。
