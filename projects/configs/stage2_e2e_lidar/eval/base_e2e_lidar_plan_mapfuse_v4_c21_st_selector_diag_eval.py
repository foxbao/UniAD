_base_ = ['../base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_train.py']

# Hard-selector inference with planning GT passed only to diagnostic scoring.
load_from = None
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_diag_eval')
