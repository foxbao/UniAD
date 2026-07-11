_base_ = ['../base_e2e_lidar_plan_mapfuse_v4_c2_soft_selector_train.py']

# Evaluate the C2 soft-selection relaxation before any C2 optimization. The C1
# checkpoint supplies selector/gate/residual weights; only inference changes
# from hard argmax to a temperature-0.5 weighted candidate anchor.
load_from = None
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v4_c2_soft_selector_zeroshot_eval')
