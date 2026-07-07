_base_ = ['./base_e2e_lidar_plan_mapfuse.py']

# v2: give the planning-only HDMap fusion branch a fair chance to actually be
# learned, instead of collapsing to a no-op like base_e2e_lidar_plan_mapfuse
# did (verified: map-on vs map-off differed by <1e-5 in every planning metric).
#
# Two coupled root causes are addressed here:
#   1. Starting point. The v1 config fine-tuned map on top of a fully-trained
#      planner (base_e2e_lidar_plan/latest, avg.L2~0.58). With the planner
#      already good, the map's marginal gradient is tiny and the optimizer
#      keeps the branch shut. v2 starts from base_e2e_lidar_occ instead, so the
#      planner and the map co-adapt while the planner is still forming.
#   2. Initialization. v1 used map_delta_proj=zeros + map_gate_init=-2.0
#      (sigmoid~0.12); both factors of `plan_query + gate*delta` start near 0,
#      so the branch gets ~0 gradient at step 0. v2 opens the gate
#      (map_gate_init=0.0 -> sigmoid 0.5) and seeds a small random delta
#      projection (map_delta_init='small'), giving a non-degenerate gradient
#      from the first step.
#
# Everything else (map_lane_encoder, motion map_agent_scope='none', losses,
# data) is inherited unchanged, so a clean map-on/map-off ablation on the
# resulting checkpoint isolates exactly the effect of these two levers.
#
# NB (deployment): opening the gate makes the map branch numerically active in
# the exported graph. The FP16 NaN issue previously seen for mapfuse dense
# engines must be re-verified after re-exporting ONNX from a v2 checkpoint.
# Training runs in fp32/float64 and is unaffected.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_gate_init=0.0,
        map_delta_init='small',
    ))

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2'
