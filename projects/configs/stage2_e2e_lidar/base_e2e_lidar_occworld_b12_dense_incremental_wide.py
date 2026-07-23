_base_ = [
    './base_e2e_lidar_occworld_b12_dense_incremental_local.py'
]

# Add a zero-initialized 15x15 dilated history residual while preserving the
# incremental flow contract.
model = dict(
    occ_head=dict(
        world_use_wide_history_context=True))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b12_dense_incremental_wide')
