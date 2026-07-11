_base_ = ['../base_e2e_lidar_plan_mapfuse_v4_c1_lane_selector_train.py']

# Keeps C1 inference unchanged while passing planning GT through the evaluator
# to report selector accuracy and oracle/predicted anchor score by motion bucket.
load_from = None
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'eval/base_e2e_lidar_plan_mapfuse_v4_c1_selector_diag_eval')
