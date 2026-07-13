_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_train.py']

# D3-A keeps D2.1 as a frozen proposal/cost model, shortlists its best 16 map
# candidates, appends the exact fallback, then jointly reranks that set using
# online actor-motion clearance and explicit collision supervision.
model = dict(
    freeze_except_prefixes=[
        'planning_head.map_multimodal_planner.set_safety_encoder',
        'planning_head.map_multimodal_planner.set_cost_encoder',
        'planning_head.map_multimodal_planner.set_encoder',
        'planning_head.map_multimodal_planner.set_norm',
        'planning_head.map_multimodal_planner.set_cost_delta_head',
        'planning_head.map_multimodal_planner.set_collision_head',
    ],
    planning_head=dict(multimodal_planner=dict(
        score_loss_weight=0.0,
        residual_loss_weight=0.0,
        candidate_cost_loss_weight=0.0,
        candidate_cost_ranking_weight=0.0,
        use_set_reranker=True,
        set_topk=16,
        set_num_layers=2,
        set_num_heads=8,
        set_ffn_dims=512,
        set_dropout=0.1,
        # D2.1's residual caused two additional collision events. D3-A ranks
        # the raw map candidates and therefore removes that known regression.
        set_use_raw_candidates=True,
        set_cost_delta_scale=2.0,
        set_collision_cost_weight=2.0,
        set_cost_loss_weight=1.0,
        set_ranking_loss_weight=0.5,
        set_collision_loss_weight=1.0,
        set_collision_positive_weight=20.0,
        set_collision_target_penalty=2.0,
        set_actor_score_threshold=0.2,
        set_clearance_temperature=1.0,
        set_ego_width=3.0,
        set_ego_length=14.6,
    )))

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_medium_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_train')
