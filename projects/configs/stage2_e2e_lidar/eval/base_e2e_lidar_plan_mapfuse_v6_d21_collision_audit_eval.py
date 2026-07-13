_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_eval.py']

# Retain the best 16 map candidates and exact fallback so raw/refined
# trajectories can be compared against the selected D2.1 trajectory offline.
model = dict(planning_head=dict(multimodal_planner=dict(audit_topk=16)))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v6_d21_collision_audit_eval')
