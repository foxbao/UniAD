_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# v3 best-endpoint lane-anchor sanity eval: consider multiple nearby HD-map
# centerlines, sample each as a local ego-relative short-horizon trajectory,
# test both lane point directions, then choose the anchor whose endpoint/shape
# best matches the base planner output. This keeps the test map-driven while
# avoiding the brittle "nearest lane only" failure mode.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='replace',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.0,
        lane_anchor_sample_mode='local_forward',
        lane_anchor_reference='relative_start',
        lane_anchor_select_mode='best_endpoint',
        lane_anchor_candidate_k=16,
        lane_anchor_direction_mode='bidirectional',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_bestendpoint_replace_eval'
