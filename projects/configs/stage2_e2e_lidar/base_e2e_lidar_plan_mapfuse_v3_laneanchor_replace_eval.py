_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# v3 lane-anchor sanity eval: bypass query-space map residuals and use the
# nearest HD-map centerline as the ego trajectory anchor at the planning output.
# This is a diagnostic upper-bound test, not a production recipe. It answers:
# if planning is forced to follow a map-derived trajectory candidate, do the
# metrics improve on the buckets where map geometry should matter?
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='replace',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.0,
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_replace_eval'
