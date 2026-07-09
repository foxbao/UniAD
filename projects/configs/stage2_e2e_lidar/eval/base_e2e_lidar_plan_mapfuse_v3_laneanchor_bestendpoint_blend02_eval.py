_base_ = ['../base_e2e_lidar_plan_mapfuse_v2.py']

# v3 best-endpoint lane-anchor blend sanity eval: use the same multi-candidate
# bidirectional local-relative map anchor as the hard-replace test, but only
# move 20% from the learned planner trajectory toward the selected lane anchor.
# This tests whether map geometry is useful as a weak prior instead of a full
# trajectory replacement.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='blend',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.2,
        lane_anchor_sample_mode='local_forward',
        lane_anchor_reference='relative_start',
        lane_anchor_select_mode='best_endpoint',
        lane_anchor_candidate_k=16,
        lane_anchor_direction_mode='bidirectional',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_bestendpoint_blend02_eval'
