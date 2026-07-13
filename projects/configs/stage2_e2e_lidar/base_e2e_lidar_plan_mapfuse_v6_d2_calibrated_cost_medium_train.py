_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train.py']

# Deterministic scene-complete medium subset: 97 scenes / 10,238 frames.
# Start from D1.1 again so the cost head is trained fresh rather than inheriting
# the overconfident 2,010-frame pilot calibration.
data = dict(train=dict(
    ann_file='/tmp/kl_infos_d2_medium_10k_scenes.pkl'))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=125,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

log_config = dict(interval=25)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_medium_train')
