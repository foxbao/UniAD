_base_ = ['./base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_eval.py']

# Remove map proposals while retaining the trained D3-A heads. The only set
# member is the byte-exact C2.3 fallback.
model = dict(planning_head=dict(
    multimodal_planner=dict(ablate_map=True)))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v7_d3a_candidateoff_eval')
