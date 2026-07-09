_base_ = ['../base_e2e_lidar_plan_mapfuse_v2.py']

# v3 local-relative lane-anchor sanity eval: choose the HD-map centerline point
# closest to ego, sample a short forward segment using the base planner's
# per-step displacement profile, then convert the sampled centerline points to
# ego-relative displacements by subtracting the closest lane point. This avoids
# treating lateral/longitudinal offset to the lane centerline as future motion.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='replace',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.0,
        lane_anchor_sample_mode='local_forward',
        lane_anchor_reference='relative_start',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_local_relative_replace_eval'
