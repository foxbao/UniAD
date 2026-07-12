_base_ = ['./base_e2e_lidar_plan_mapfuse_v5_d12_utility_reg_train.py']

data = dict(train=dict(
    ann_file='/tmp/kl_infos_c23_balanced_pilot_scenes.pkl'))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=25,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

log_config = dict(interval=10)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v5_d12_utility_reg_pilot_train')
