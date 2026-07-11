_base_ = ['./base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train.py']

# C1: explicit map-anchor selection. C0 oracle showed that surveyed lane
# geometry can reach avg.L2=0.2653 if the right local anchor is selected, while
# A's learned blend is 0.60495. This config keeps A's stable learned-blend
# output head, but replaces base-plan nearest-anchor selection with a trainable
# selector supervised by the GT-best candidate during training.
model = dict(
    freeze_except_prefixes=[
        'planning_head.lane_anchor_selector_head',
        'planning_head.lane_anchor_gate_head',
        'planning_head.lane_anchor_residual_head',
    ],
    planning_head=dict(
        use_map_lane=True,
        # Isolate C1 from the failed latent feature-fusion route. Map geometry
        # enters through explicit lane-anchor candidates only.
        map_fusion_mode='force_residual',
        map_force_scale=0.0,
        lane_anchor_mode='learned_blend',
        lane_anchor_sample_mode='local_forward',
        lane_anchor_reference='relative_start',
        lane_anchor_select_mode='learned_selector',
        lane_anchor_candidate_k=16,
        lane_anchor_direction_mode='bidirectional',
        lane_anchor_selector_loss_weight=0.5,
        lane_anchor_selector_teacher_force=True,
        lane_anchor_static_gate_loss_weight=0.1,
    ))

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c1_lane_selector_train')
