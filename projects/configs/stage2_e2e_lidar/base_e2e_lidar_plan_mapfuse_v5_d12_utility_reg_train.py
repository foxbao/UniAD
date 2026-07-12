_base_ = ['./base_e2e_lidar_plan_mapfuse_v5_d11_utility_train.py']

# D1.2 keeps D1.1's trained map scorer/residual fixed and replaces the binary
# gate with direct regression of fallback_cost - predicted_map_cost.
model = dict(
    freeze_except_prefixes=[
        'planning_head.map_multimodal_planner.utility_regression_head',
    ],
    planning_head=dict(multimodal_planner=dict(
        utility_target_mode='improvement_regression',
        utility_regression_clip=2.0,
    )))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v5_d11_utility_pilot_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v5_d12_utility_reg_train')
