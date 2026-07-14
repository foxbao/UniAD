_base_ = ['./base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_train.py']

# D3-A.1 removes the representation/output mismatch measured in D3-A. Each
# shortlisted map path contributes both its raw and D2.1-refined trajectory.
# The cost head is evaluated on the actual variant, and a relative guard blocks
# variants predicted more collision-prone than the exact fallback.
model = dict(
    freeze_except_prefixes=[
        'planning_head.map_multimodal_planner.set_safety_encoder',
        'planning_head.map_multimodal_planner.set_cost_encoder',
        'planning_head.map_multimodal_planner.set_encoder',
        'planning_head.map_multimodal_planner.set_norm',
        'planning_head.map_multimodal_planner.set_cost_delta_head',
        'planning_head.map_multimodal_planner.set_collision_head',
        'planning_head.map_multimodal_planner.set_variant_embed',
    ],
    planning_head=dict(multimodal_planner=dict(
        set_candidate_variant_mode='raw_refined',
        set_collision_positive_weight=100.0,
        set_use_fallback_guard=True,
        set_fallback_guard_margin=0.05,
        set_fallback_guard_penalty=1000.0,
    )))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_medium_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v7_d3a1_dual_variant_guard_train')
