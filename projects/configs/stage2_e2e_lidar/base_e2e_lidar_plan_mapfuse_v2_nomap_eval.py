_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# Ablation-eval companion for base_e2e_lidar_plan_mapfuse_v2. Reuse the v2
# map-trained checkpoint but turn OFF the planning-head map-lane attention at
# inference, isolating how much the HD-map fusion branch actually moves the
# plan AFTER the v2 recipe (occ start + map_gate_init=0.0 + delta_init='small')
# gave the branch a fair chance to be learned.
#
# `use_map_lane=False` makes PlanningHeadSingleMode._apply_map_lane_attention a
# pass-through (plan_query returned untouched); the map_attn/map_gate/map_delta
# modules are not built and the v2 checkpoint's map_* weights are ignored on a
# non-strict load. Everything else (BEV, motion, occ, planner regression) is
# bit-identical to base_e2e_lidar_plan_mapfuse_v2, so any L2/Collision delta
# between this and the map-on eval is attributable to the map branch alone.
#
# How to use: run this AND the plain v2 config against the SAME v2 checkpoint
# (pass the .pth on the trtexec/dist_eval command line, not via load_from).
# The map-on vs map-off delta is the definitive answer to "does the HDMap
# fusion add anything beyond BEV in this port ODD". See
# [[mapfuse-planning-fusion-is-dead]] for why v1 showed <1e-5 (branch was dead)
# and what a non-trivial v2 delta would mean.
model = dict(
    planning_head=dict(
        use_map_lane=False,
    ))

# Evaluate the exact v2 checkpoint that was trained WITH map fusion; the
# checkpoint is supplied on the eval command line.
load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2_nomap_eval'
