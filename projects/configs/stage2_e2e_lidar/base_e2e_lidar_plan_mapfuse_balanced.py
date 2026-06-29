_base_ = ['./base_e2e_lidar_plan_mapfuse.py']

# Keep the planning-only HDMap fusion path unchanged, but reduce the dominance
# of parked/low-speed SDC GT in the planning losses. This is an opt-in
# training-side debiasing ablation; base_e2e_lidar_plan_mapfuse.py remains the
# pure map-fuse control.
model = dict(
    planning_head=dict(
        planning_motion_loss_weights=dict(
            static=0.3,
            slow=0.7,
            moving_straight=1.0,
            turning=1.5,
        )))

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_balanced'
