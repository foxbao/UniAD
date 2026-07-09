_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# v3 sanity eval: force the planning-head map-attention context to perturb the
# plan query directly, bypassing the learned gate/delta projection that proved
# ineffective in v1/v2. This is intentionally not a production recipe; it tests
# whether the downstream BEV-attention + reg_branch path is sensitive to a
# strong map-conditioned query at all.
#
# Compare this config against base_e2e_lidar_plan_mapfuse_v2_nomap_eval.py on
# the SAME checkpoint. If x5 force-query still leaves planning unchanged, the
# bottleneck is after the map-attended plan query. If it moves the output, the
# current gated-residual training objective is too weak and v3 should move to
# lane-anchor / trajectory-level map conditioning.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='force_residual',
        map_force_scale=5.0,
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_forcequery_x5_eval'
