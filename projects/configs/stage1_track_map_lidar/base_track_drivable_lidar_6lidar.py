"""Stage-1 drivable training on the six-sensor KL LiDAR variant."""

_base_ = ['./base_track_drivable_lidar.py']

data = dict(
    train=dict(
        data_root='data/kl_6/',
        ann_file='kl6_infos_train.pkl'),
    val=dict(
        data_root='data/kl_6/',
        ann_file='kl6_infos_val.pkl'),
    test=dict(
        data_root='data/kl_6/',
        ann_file='kl6_infos_val.pkl'))

# Stage 1 freezes the LiDAR backbone, so it must load the matching six-LiDAR
# BEVFormer pre-training checkpoint rather than the legacy eight-LiDAR one.
load_from = (
    './projects/work_dirs/bevformer_lidar/'
    'base_bevformer_lidar_6lidar/latest.pth')
resume_from = None

work_dir = './projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar_6lidar'
