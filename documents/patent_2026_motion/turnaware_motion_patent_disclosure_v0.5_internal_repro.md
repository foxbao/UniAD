# 转弯感知运动预测交底书内部复现记录

该文件仅用于内部复现、数据追溯和代理人答疑，不建议作为交底书正文提交。

## 1. 生成命令

```bash
python3 documents/patent_2026_motion/build_turnaware_patent_docx_v05.py
```

## 2. 文本来源

| 来源 | 用途 |
| --- | --- |
| 研发总结文档 | 最新实验结论、消融结果、问题归因。 |
| 上一版交底书草案 | 上一版交底书结构、背景、基础实施方式。 |
| 轨迹损失实现文件 | 平均位移误差/终点位移误差联合模式分配、转弯分桶权重、分类权重与运动桶分类实现。 |
| 锚点生成脚本 | 带权聚类的转弯感知锚点生成。 |
| 分层锚点生成脚本 | 分层保留式锚点生成备选实施例。 |
| 转弯分桶评估脚本 | 转弯分桶评估和最高置信度候选/最优候选诊断。 |
| 实验配置文件 | 实验配置和损失参数。 |

## 3. 实验数据源

| 文件 | 用途 |
| --- | --- |
| sources/v0.5_epoch6_overall_ablation_summary.csv | 交底书整体消融结果表。 |
| sources/v0.5_turn_bucket_epoch6_summary.csv | 交底书转弯分桶结果表。 |
| 原始转弯分桶评估结果 | 原始转弯分桶指标。 |
| 代表性可视化案例清单 | 代表可视化案例和最高置信度候选/最优候选诊断样本。 |

## 4. 图片生成/复用说明

| 图片 | 生成或复用方法 | 数据源 |
| --- | --- | --- |
| fig1_turnaware_anchor_generation_flow.png | 由本文档生成脚本使用 PIL 绘制。 | 算法流程来源：类别分组、局部坐标归一、运动分桶、分层锚点与转弯加权方案。 |
| fig2_anchor_compare.png | 由本文档生成脚本读取锚点 pkl 并使用 PIL 直接绘制中文图。 | 数据源：sources/motion_anchor_infos_kl.pkl 与 sources/motion_anchor_infos_kl_turnaware_3grp.pkl；绘图保留原始锚点坐标数据。 |
| fig3a_turn_focus_truck_1522.png、fig3b_turn_focus_crane_204.png、fig3c_turn_focus_crane_2409.png | 由 tools/analysis_tools/visualize_lidar_motion_focus_compare.py 使用 --chinese-labels、--hide-frame-title、--show-axis-labels、--patent-layout 与 --large-text 重新渲染样本索引 1522、204、2409 后复制。 | 数据源：base_e2e_lidar/eval_epoch6_results.pkl、base_e2e_lidar_turnaware_turnloss/eval_epoch6_results.pkl、data/kl_8/kl_infos_val.pkl 及对应点云；输出目录：sources/fig3_focus_patent_ttl_zh。 |
| fig4_turnaware_loss_mode_assignment_flow.png | 由本文档生成脚本使用 PIL 绘制。 | 算法流程和配置参数：轨迹损失实现文件、实验配置文件。 |
| fig5_candidate_confidence_diagnosis.png | 由本文档生成脚本使用 PIL 绘制。 | 诊断逻辑来源：转弯分桶评估脚本和研发总结文档。 |
| fig6_prediction_network_architecture.png | 由本文档生成脚本使用 PIL 绘制。 | 内容来源：本交底书中的通用预测网络架构描述，用于说明转弯感知锚点、轨迹解码、训练损失和推理输出的嵌入位置。 |

图3重生成命令：

```bash
PYTHONPATH="$PWD:$PYTHONPATH" /home/baojiali/anaconda3/envs/uniad_train/bin/python tools/analysis_tools/visualize_lidar_motion_focus_compare.py   --config projects/configs/stage2_e2e_lidar/base_e2e_lidar.py   --result base projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar/eval_epoch6_results.pkl   --result turnaware_turnloss projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_turnaware_turnloss/eval_epoch6_results.pkl   --case 1522:Truck:10:base_top1_8.06_vs_scheme_0.26   --case 204:Crane:5:base_top1_6.53_vs_scheme_0.81   --case 2409:Crane:4:base_top1_7.71_vs_scheme_1.18   --out-dir documents/patent_2026_motion/sources/fig3_focus_patent_ttl_zh   --split val   --score-thr 0.25   --point-stride 5   --zoom-margin 12   --chinese-labels   --hide-frame-title   --show-axis-labels   --patent-layout   --large-text
```

## 5. 完整复现 JSON

```text
documents/patent_2026_motion/sources/v0.5_reproducibility_record.json
```
