_base_ = ['./base_e2e_lidar_plan_mapfuse_v4_c23_eval_gate_train.py']

# D1 replaces scalar map blending with explicit topology-path x speed-profile
# candidate ranking. The proven C2.3 output remains a fallback candidate. All
# old modules are frozen; only the new scorer and bounded residual are trained.
model = dict(
    freeze_except_eval=True,
    freeze_except_prefixes=[
        'planning_head.map_multimodal_planner',
    ],
    map_lane_encoder=dict(
        planning_candidates=dict(
            profiles_path='data/others/planning_speed_profiles_d0_8.npz',
            planning_steps=6,
            num_points_per_lane=80,
            num_start_lanes=8,
            max_paths=16,
            path_margin=5.0,
            max_start_distance=12.0,
            max_heading_error_deg=80.0,
            max_depth=8,
            max_join_gap=6.0,
            lateral_offsets=[
                -2.0, -1.5, -1.0, -0.5, 0.0,
                0.5, 1.0, 1.5, 2.0,
            ],
        )),
    planning_head=dict(
        lane_anchor_static_gate_loss_weight=0.0,
        lane_anchor_selector_loss_weight=0.0,
        lane_anchor_utility_gate_loss_weight=0.0,
        multimodal_planner=dict(
            num_heads=8,
            dropout=0.1,
            coordinate_scale=20.0,
            residual_scale=1.5,
            fallback_logit_bias=2.0,
            score_temperature=0.25,
            score_loss_weight=1.0,
            residual_loss_weight=1.0,
            eval_horizon_indices=[1, 3, 5],
            oracle_recall_tolerance=0.01,
        )))

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c23_eval_gate_pilot_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v5_d1_multimodal_train')
