_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_eval.py']

# Keep the trained D2.1 weights but remove every map candidate. This leaves the
# exact C2.3 fallback as the only selectable trajectory.
model = dict(planning_head=dict(
    multimodal_planner=dict(ablate_map=True)))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v6_d21_candidateoff_eval')
