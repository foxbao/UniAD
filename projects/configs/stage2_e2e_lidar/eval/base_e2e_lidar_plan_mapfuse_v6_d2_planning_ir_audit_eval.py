_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_eval.py']

# P0 audit only. Normal D2 eval keeps audit_topk=0 and does not retain the
# candidate payload. This config exports the best 16 map candidates plus the
# exact C2.3 fallback for offline Planning-IR experiments.
model = dict(planning_head=dict(multimodal_planner=dict(audit_topk=16)))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/eval/'
    'base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval')
