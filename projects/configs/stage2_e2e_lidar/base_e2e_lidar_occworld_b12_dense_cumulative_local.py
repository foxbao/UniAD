_base_ = [
    './base_e2e_lidar_occworld_b12_dense_incremental_local.py'
]

# The cumulative arm predicts each future displacement at the current source
# cell and warps the current instance volume independently per horizon.
model = dict(
    occ_head=dict(
        world_flow_parameterization='cumulative_current'))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'occworld_b12_initialization/aligned10_cumulative_current.pth')

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b12_dense_cumulative_local')
