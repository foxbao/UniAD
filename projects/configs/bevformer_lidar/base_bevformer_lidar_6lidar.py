"""Six-LiDAR BEVFormer pre-training on the rebuilt KL dataset."""

_base_ = ['./base_bevformer_lidar.py']

# The input is the newly fused six-sensor point cloud, not the legacy
# eight-sensor samples used by the base configuration.
data_root = 'data/kl_6/'

data = dict(
    train=dict(
        data_root=data_root,
        ann_file='kl6_infos_train.pkl'),
    val=dict(
        data_root=data_root,
        ann_file='kl6_infos_val.pkl'),
    test=dict(
        data_root=data_root,
        ann_file='kl6_infos_val.pkl'))

# Train from scratch: the legacy checkpoint learned from the eight-sensor
# fusion and is retained only as a historical baseline.
load_from = None
resume_from = None
work_dir = './projects/work_dirs/bevformer_lidar/base_bevformer_lidar_6lidar'
