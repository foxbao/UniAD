_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_pilot.py'
]

# Continue the finite 5-epoch flow pilot with its optimizer state. Only the
# epoch-10 checkpoint is retained in this separate work directory.
total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
resume_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_pilot/'
    'epoch_5.pth')
checkpoint_config = dict(interval=5, max_keep_ckpts=1)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_continue10')
