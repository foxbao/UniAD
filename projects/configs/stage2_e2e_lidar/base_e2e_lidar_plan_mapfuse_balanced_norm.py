_base_ = ['./base_e2e_lidar_plan_mapfuse.py']

# Normalized variant of base_e2e_lidar_plan_mapfuse_balanced.py.
# It keeps the same relative preference (static < slow < straight < turning)
# but rescales the weights by the KL train split bucket mean (~0.869) so the
# planning loss scale stays close to the pure map-fuse control.
model = dict(
    planning_head=dict(
        planning_motion_loss_weights=dict(
            static=0.345,
            slow=0.806,
            moving_straight=1.151,
            turning=1.727,
        )))

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_balanced_norm'
