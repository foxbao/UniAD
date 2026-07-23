_base_ = [
    './base_e2e_lidar_occworld_b12_dense_cumulative_local.py'
]

# Full B12 candidate: current-source cumulative flow plus wide causal history.
model = dict(
    occ_head=dict(
        world_use_wide_history_context=True))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b12_dense_cumulative_wide')
