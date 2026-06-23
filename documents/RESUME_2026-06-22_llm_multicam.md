# 会话恢复上下文 (2026-06-22)

权威文档: documents/llm_integration_plan.md (先完整读它). 本文件是当前进度速记.

## 一句话目标
LiDAR UniAD 端到端 + 跨模态蒸馏: 离线 Qwen2.5-VL 看环视相机图生成中文 summary 作监督,
训练挂在 LiDAR query 上的 Qwen2.5-0.5B (LLMBridgeHead), 推理纯 LiDAR 说出图像语义.

## 环境 (两套 conda)
- uniad_train: torch1.12.1+cu116, numpy1.22, transformers4.46.3+peft0.13.2(后装). 训练/几何/子集/读写pkl.
- qwen_vl: torch2.6+cu124, numpy2.x, transformers4.57.6. 仅跑 Qwen2.5-VL 离线推理.
- 坑: 装包别动torch(accelerate会升torch->2.4打断mmcv.ops),用--no-deps. 模型放/mnt/disk1/models(系统盘满).
- 8xRTX4090, GPU0常空.

## 数据流水线 (tools/data_converter/, 增量后处理)
add_cam_sync(同步,写sync_info.cameras) -> gen_geo_facts(写info.geo_facts) -> make_subset(分层子集)
-> gen_vlm_caption(VLM出summary,产物token->summary JSON sidecar,支持--num-shards) -> merge_summaries(合并回pkl).
pkl后缀: _with_cam -> _geo -> _vlmcap. 字段名 geo_facts.

## 模型侧 (已落地,smoke通过)
- dense_heads/llm_bridge_head.py: projector(256->896)+spatial_pe(3->896)+Qwen0.5B(bf16,冻结+LoRA).
  LLM在__init__ eager构建(不能lazy否则LoRA进不了optimizer-真bug已修). bf16必须(fp16=nan). projector fp32算再cast bf16.
- detectors/uniad_motion_lidar.py: llm_head/with_llm_head/_current_caption; forward_train接llm loss; simple_test接forward_test.
- datasets/kl_dataset.py: KlTrackDataset._union2one(子类~1963)注入gt_caption到current frame; detector从整数键metas_map[max]取.
- config: base_e2e_lidar_occ_llm.py(smoke) + base_e2e_lidar_occ_llm_train.py(正式, train->train_vlmcap, val/test->v3子集). 旧_full.py已删.

## 评估 (tools/analysis_tools/eval_llm_caption.py)
规则解析caption->语义命中率,主对geo_facts. template/teacher可跑; model/shuffle/noquery待checkpoint.

## git: 分支 feat/lidar-llm-distill-data. 多相机改造未提交.

## 正在做: 多相机改造 (#8/#9/#10 已完成主体, 待 train 全量)
背景: 评估发现VLM teacher只15.6%该标帧说作业. 根因=单front看不到后方/侧方(正后方activity召回7%vs前视31-35%).
决策=上6路环视根治(已验证6路质量与front一致). 用户定: 多图一次喂VLM+算力控~2x(按需取图).

#8已做:
- gen_geo_facts.py 回退前视gate收紧(_FRONT_SECTORS/gate_visible_m已删),activity_gate恢复全方位(static>=2s&crane<=30m).
- add_cam_sync 跑6路 -> kl_infos_val_with_cam.pkl 含6路. gen_geo_facts重跑 -> kl_infos_val_with_cam_geo.pkl 已更新.

#9已完成 (gen_vlm_caption.py 多相机化):
- _BEARING_CAM/_CAM_ZH/_cam_path/_resolve_views: 按帧内AOI/conflict/gate目标方位选2-4路相机,每图标【方位相机】.
- _SYSTEM prompt 改多图环视+按方位对齐作业判断(rule5不再写死前视); docstring改环视. 语法OK. dry-run avg 3.4路/帧.

#10已完成 (val子集多相机caption + eval):
- make_subset 抽 kl_infos_val_sub6cam_geo.pkl (稀有帧911全保留). GPU0跑caption ~22min -> 903带summary.
- merge -> kl_infos_val_sub6cam_vlmcap.pkl.
- **eval_llm_caption.py 修了指标bug**: 旧activity只匹配"作业/装卸",把VLM正确判的"空闲/等待"当漏报.
  拆成 activity_addressed(给出三态之一)/activity_busy(其中判装卸的rate).
- **结果: 6cam teacher activity_addressed 37.3% vs 单front 17.0% 翻倍**(gated分母均777).
  conflict_cls 75->88, bearing 75->84, ttc 59->72 均涨. halluc(无幻觉)98->96持平.
- 文档已全面更新(0.1/0.4/0.6/4.1/4.3-B/C), memory已记 llm-multicam-activity-recall.

## 复核迭代 (另一AI复核6条建议, 全部核实属实并处理)
- train pkl 已重做6路(98.45-98.61%命中, gate 18.3%); 文档过期项已改(0.6/4.1/4.3-A).
- 新建正式config base_e2e_lidar_occ_llm_train.py(train->kl_infos_train_vlmcap, val/test->sub6cam, load_from stage2-occ ckpt). smoke config不动.
- §3 加banner标"front-only是历史阶段".
- **第4点(口径不一致, 已修)**: _resolve_views按全部gate目标选相机, 但_render_facts只渲染AOI->gate目标"给图不给题".
  改为渲染AOI∪conflict∪gate. 重跑v2: activity_addressed 37.3%->**69.4%**(同帧777). commit 2e23ebb.
- **gate误标锥桶(已修)**: activity_gate对静止近吊机的任意目标触发,含锥桶/行人/小车(val子集941个误标).
  加_NO_ACTIVITY_IDS={0,1,9}排除. gate目标2481->1540. commit dcc88f6.

## 锥桶排除后重生成数据 + v3验证 (已完成)
- gen_geo_facts重算 train+val _with_cam_geo.pkl (gate已排除锥桶等).
- 重抽val子集 (稀有帧911->805). v3 caption + eval: **activity_addressed 74.5%** (v2 69.4%, 单front 17%),
  advice 80->84.8 (清掉"向锥桶让行"误导帧), conflict/ttc/halluc不变. 产物 kl_infos_val_sub6cam_v3_vlmcap.pkl.

## train全量caption (已完成)
- 7卡分片 run_vlm_caption_train_7gpu.sh ~2.2h. 先遇OOM(7进程同时load 16G模型,分配器碰撞);
  修法=错开sleep启动+expandable_segments+每分片自检+前台tail. commit 8641695/b72acd3.
- 7片sidecar -> merge -> kl_infos_train_vlmcap.pkl (43747/43981=99.5%带summary).
- config base_e2e_lidar_occ_llm_train.py: train->该pkl, val/test->v3子集. 已验证三个pkl+load_from都存在. commit a249cc4.

## 待办: 只剩跑正式训练
./tools/uniad_dist_train.sh projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train.py <GPUS>
(前台DDP,实时打印loss). 跑出ckpt后可补 eval_llm_caption.py --mode model/shuffle (forward_test逐帧,当前NotImplementedError).


