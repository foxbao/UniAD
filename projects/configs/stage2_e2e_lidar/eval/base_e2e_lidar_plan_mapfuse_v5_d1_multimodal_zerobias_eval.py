_base_ = ['./base_e2e_lidar_plan_mapfuse_v5_d1_multimodal_eval.py']

# Calibration diagnostic: remove the training-time safety margin and test
# whether learned map candidates already ranked immediately below fallback.
model = dict(planning_head=dict(
    multimodal_planner=dict(fallback_logit_bias=0.0)))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v5_d1_multimodal_zerobias_eval')
