_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_train.py']

# Scene-complete 10k control before committing to another 43,981-frame run.
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
    'base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_medium_train')
