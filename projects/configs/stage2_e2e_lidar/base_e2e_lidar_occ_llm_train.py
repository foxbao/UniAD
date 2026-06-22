_base_ = ['./base_e2e_lidar_occ_llm.py']

# Stage-2 LiDAR e2e + LLMBridgeHead — FULL-SCALE training config.
# Inherits the llm_head/task_loss_weight from the smoke config, but points
# train at the full train-set VLM captions and val/test at the 6-cam val
# subset, so the smoke config (which trains/evals on the val subset only) stays
# untouched as a quick forward/loss check.
#
# Prerequisites (see documents/llm_integration_plan.md 4.3-A):
#   - data/kl_8/kl_infos_train_vlmcap.pkl  (7-gpu caption -> merge_summaries)
#   - data/kl_8/kl_infos_val_sub6cam_v3_vlmcap.pkl  (6-cam, gate cleaned, v3)
# Both must carry geo_facts.summary. train pkl is built from the 6-cam
# kl_infos_train_with_cam_geo.pkl (redone 2026-06-22, gate excludes
# pedestrian/car/cone).

data = dict(
    train=dict(ann_file='kl_infos_train_vlmcap.pkl'),
    val=dict(ann_file='kl_infos_val_sub6cam_v3_vlmcap.pkl'),
    test=dict(ann_file='kl_infos_val_sub6cam_v3_vlmcap.pkl'))

# The smoke config starts from the stage-1 drivable ckpt (a shortcut to test
# the llm forward/loss). For full training we want the trained stage-2 occ
# perception underneath the LLM head (BEV/track/motion/occ already converged,
# frozen per the occ base), so the LLM head learns on top of working queries.
load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/latest.pth'

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train'
