_base_ = ['./base_e2e_lidar_occ_llm.py']

# Full-scale training config for the LiDAR e2e + LLMBridgeHead distillation.
# Inherits everything from the smoke config (model / llm_head / occ / etc.)
# and only redirects data: train on the FULL train set with VLM captions,
# evaluate on the val 1500-frame subset.
#
# Prerequisite: train-set captions generated + merged into
#   data/kl_8/kl_infos_train_vlmcap.pkl
# via tools/run_vlm_caption_train_7gpu.sh + merge_summaries.py
# (see documents/llm_integration_plan.md sec 4.3-A).

data = dict(
    train=dict(ann_file='kl_infos_train_vlmcap.pkl'),
    val=dict(ann_file='kl_infos_val_sub1k_vlmcap.pkl'),
    test=dict(ann_file='kl_infos_val_sub1k_vlmcap.pkl'))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_full'
