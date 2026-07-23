_base_ = ['./base_e2e_lidar_occworld_split_v2_world_only_anchor.py']

# B5 explicitly separates the sparse decision to change from the changed
# semantic class. The gate starts at 1%, preserving persistence at argmax.
model = dict(
    occ_head=dict(
        world_use_future_change_gate=True,
        world_future_change_prior=0.01,
        world_future_change_gate_loss_weight=0.1,
        world_future_changed_class_loss_weight=0.1,
        world_future_change_positive_weight=10.0))

total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_change_gate_pilot')
