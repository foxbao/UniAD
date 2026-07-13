_base_ = ['./base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train.py']

# D2.1 jointly adapts the candidate representation, bounded map residual, and
# calibrated horizon-cost head. The exact fallback remains unchanged because
# MapMultimodalPlanner hard-zeros fallback residuals in candidate-cost mode.
model = dict(
    freeze_except_prefixes=[
        'planning_head.map_multimodal_planner.candidate_encoder',
        'planning_head.map_multimodal_planner.source_embed',
        'planning_head.map_multimodal_planner.map_attention',
        'planning_head.map_multimodal_planner.attention_norm',
        'planning_head.map_multimodal_planner.ffn',
        'planning_head.map_multimodal_planner.ffn_norm',
        'planning_head.map_multimodal_planner.residual_head',
        'planning_head.map_multimodal_planner.candidate_cost_head',
    ],
    planning_head=dict(multimodal_planner=dict(
        # The legacy D1 score head is frozen and unused by D2 selection. A
        # non-zero score loss would still backpropagate through shared features.
        score_loss_weight=0.0,
        residual_loss_weight=1.0,
    )))

# Shared representation moves conservatively; the already calibrated cost head
# receives a 4x rate so it can follow the changing candidate features.
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'planning_head.map_multimodal_planner.candidate_cost_head': dict(
            lr_mult=4.0),
    }))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_train')
