_base_ = ['./base_e2e_lidar_plan_mapfuse_v5_d1_candidateoff_eval.py']

# Remove both D1 candidates/lane attention and C2.3's lane-anchor fallback.
model = dict(planning_head=dict(ablate_lane_anchor=True))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v5_d1_mapoff_eval')
