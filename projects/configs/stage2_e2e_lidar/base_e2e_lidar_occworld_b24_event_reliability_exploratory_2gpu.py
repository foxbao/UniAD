_base_ = [
    './base_e2e_lidar_occworld_b24_event_reliability_exploratory.py'
]

# Two GPUs produce about three times as many optimizer updates per epoch as
# the six-GPU protocol. Scale the learning rate down by the same factor so
# the three-epoch optimization budget remains comparable.
optimizer = dict(type='AdamW', lr=1.0e-3 / 3.0, weight_decay=1e-4)
gpu_ids = range(2)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b24_event_reliability_exploratory_3epoch_2gpu')
