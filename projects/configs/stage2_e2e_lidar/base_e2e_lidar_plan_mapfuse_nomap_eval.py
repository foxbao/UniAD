_base_ = ['./base_e2e_lidar_plan_mapfuse.py']

# Ablation-eval config: reuse the map-trained checkpoint but turn OFF the
# planning-head map-lane attention at inference. This isolates how much the
# HD-map fusion branch actually moves the plan. `use_map_lane=False` makes
# PlanningHeadSingleMode._apply_map_lane_attention a pass-through (plan_query
# is returned untouched), so the map_attn/map_gate/map_delta modules are not
# built and the checkpoint's map_* weights are simply ignored on load.
# Everything else (BEV, motion, occ, planner regression) is bit-identical to
# base_e2e_lidar_plan_mapfuse, so any L2/Collision delta is attributable to
# the map branch alone.
model = dict(
    planning_head=dict(
        use_map_lane=False,
    ))

# Evaluate the exact checkpoint that was trained WITH map fusion.
load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_nomap_eval'
