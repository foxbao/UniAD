_base_ = ['./base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_train.py']

# Short promotion pilot. Override data.train.ann_file on the command line with
# a deterministic scene-complete subset; keep outputs separate from full train.
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=25,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_pilot_train')
