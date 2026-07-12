_base_ = ['./base_e2e_lidar_plan_mapfuse_v4_c22_utility_gate_train.py']

# C2.3 aligns the gate target with the actual planning metric. It searches the
# base-to-map blend at 1s/2s/3s and falls back to base unless the best alpha
# improves average horizon L2 by at least 1 cm.
model = dict(
    planning_head=dict(
        lane_anchor_utility_target_mode='eval_grid',
        lane_anchor_utility_grid_size=21,
        lane_anchor_utility_min_improvement=0.01,
    ))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c23_eval_gate_train')
