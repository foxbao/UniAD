_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_eval.py']

model = dict(planning_head=dict(
    multimodal_planner=dict(ablate_map=True)))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v6_d2_candidateoff_eval')
