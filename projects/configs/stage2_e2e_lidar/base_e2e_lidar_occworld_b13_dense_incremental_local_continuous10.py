_base_ = [
    './base_e2e_lidar_occworld_b12_dense_incremental_local.py'
]

# B13 is the formal dense145 reproduction. It keeps the B12-selected
# incremental/local architecture, but initializes from the original LiDAR OCC
# checkpoint and trains for ten uninterrupted epochs under one cosine schedule.
load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occ/latest.pth')
resume_from = None

total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Retain every predeclared validation candidate. Test and blind splits remain
# sealed until one epoch is selected using the frozen validation split.
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b13_dense_incremental_local_continuous10')
