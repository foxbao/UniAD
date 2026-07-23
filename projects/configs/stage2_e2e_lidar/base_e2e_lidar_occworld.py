_base_ = ['./base_e2e_lidar_occ.py']

# OccWorld keeps the trained LiDAR/TrackFormer/Motion/Occ stack and adds a
# dense semantic 3D world branch. The original query-based instance occupancy
# remains active as the dynamic-object branch.
model = dict(
    occ_head=dict(
        type='OccWorldHead',
        world_z_count=10,
        world_class_count=3,
        world_hidden_channels=64,
        world_ignore_index=255,
        world_loss_weight=1.0,
        world_current_loss_weight=1.0,
        world_future_loss_weight=1.0,
        world_visibility_loss_weight=1.0,
        world_valid_positive_weight=5.0,
        world_valid_prior=0.16,
        world_class_weights=None))

data = dict(
    train=dict(
        type='KlOccWorldDataset',
        occworld_label_root=(
            'outputs/patent_2026_occ/occworld_sequence_batch20'),
        occworld_expected_shape=(5, 10, 120, 160),
        occworld_ignore_index=255))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occ/latest.pth')
resume_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occworld'
