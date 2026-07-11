_base_ = ['./base_e2e_lidar_plan_mapfuse_v4_c1_lane_selector_train.py']

# C2 removes both C1 train/eval mismatches. Candidate anchors are always
# sampled at the frozen base planner's speed, and a differentiable softmax
# mixture replaces teacher-forced hard selection. The selector still receives
# GT-best candidate supervision, while planning ADE can now update it directly.
model = dict(
    freeze_except_prefixes=[
        'planning_head.lane_anchor_selector_head',
        'planning_head.lane_anchor_gate_head',
        'planning_head.lane_anchor_residual_head',
    ],
    planning_head=dict(
        lane_anchor_select_mode='soft_selector',
        lane_anchor_selector_teacher_force=False,
        lane_anchor_selector_sample_teacher=False,
        lane_anchor_selector_temperature=0.5,
        lane_anchor_selector_loss_weight=0.5,
    ))

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c1_lane_selector_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c2_soft_selector_train')
