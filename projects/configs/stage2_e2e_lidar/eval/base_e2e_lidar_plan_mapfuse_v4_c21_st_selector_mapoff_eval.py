_base_ = ['../base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_train.py']

# Causal ablation for the trained C2.1 checkpoint. Keep the exact model
# structure so all weights load, but skip the explicit lane-anchor path during
# forward. Latent map fusion is already disabled by map_force_scale=0 in C1+.
model = dict(
    planning_head=dict(
        ablate_lane_anchor=True,
    ))

load_from = None
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_mapoff_eval')
