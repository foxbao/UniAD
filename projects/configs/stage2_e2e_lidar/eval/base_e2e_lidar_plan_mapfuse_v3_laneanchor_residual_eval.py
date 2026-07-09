_base_ = ['../base_e2e_lidar_plan_mapfuse_v2.py']

# v3 lane-anchor residual sanity eval: add the nearest HD-map centerline anchor
# to the learned planner output. This checks whether direct trajectory-space map
# conditioning can move outputs in a structured way without completely replacing
# the planner.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='residual',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.0,
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_residual_eval'
