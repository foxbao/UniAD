_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_pilot.py'
]

# Official BEV features, instance occupancy and gt_flow index H in native BEV
# order. OccWorld labels index H in image-aligned order, so all native inputs
# are converted at the world-head boundary. Train ten epochs from the original
# LiDAR OCC checkpoint with one continuous cosine schedule.
model = dict(
    occ_head=dict(
        world_align_bev_to_world_layout=True))

total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=5, max_keep_ckpts=1)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_aligned10')
