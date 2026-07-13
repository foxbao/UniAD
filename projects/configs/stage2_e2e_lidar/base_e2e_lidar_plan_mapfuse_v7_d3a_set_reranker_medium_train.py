_base_ = ['./base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_train.py']

# Scene-complete 10k gate before any full-data D3-A run.
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
    'base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_medium_train')
