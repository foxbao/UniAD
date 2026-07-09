_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# v3 local lane-anchor sanity eval: choose the HD-map centerline closest to ego,
# then sample only the short forward segment matching the base planner's
# per-step cumulative displacement. This avoids the naive whole-lane failure
# mode where a full 20-point lane polyline became a 3s trajectory.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='replace',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.0,
        lane_anchor_sample_mode='local_forward',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_local_replace_eval'
