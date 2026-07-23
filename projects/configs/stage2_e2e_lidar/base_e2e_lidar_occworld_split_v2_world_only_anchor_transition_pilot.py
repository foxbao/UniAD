_base_ = ['./base_e2e_lidar_occworld_split_v2_world_only_anchor.py']

# B4 keeps the full persistence anchor and adds a small, separately reduced
# loss only where a future target differs from a currently known state.
model = dict(
    occ_head=dict(
        world_future_transition_loss_weight=0.1))

total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_transition_pilot')
