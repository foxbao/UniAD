_base_ = ['./base_e2e_lidar_plan_mapfuse_v5_d1_multimodal_eval.py']

# Isolate D1's added value while preserving the exact C2.3 fallback path.
model = dict(
    map_lane_encoder=dict(planning_candidates=None),
    planning_head=dict(multimodal_planner=dict(ablate_map=True)))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v5_d1_candidateoff_eval')
