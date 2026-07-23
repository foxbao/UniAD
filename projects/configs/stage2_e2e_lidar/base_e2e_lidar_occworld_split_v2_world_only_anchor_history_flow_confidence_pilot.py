_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_aligned10.py'
]

# B9 freezes the validated aligned10 model and learns only whether a physical
# prediction should override raw semantics at their supervised disagreements.
model = dict(
    freeze_except_prefixes=[
        'occ_head.world_decoder.physical_confidence_head',
    ],
    occ_head=dict(
        world_use_physical_confidence=True,
        world_physical_confidence_prior=0.8,
        world_physical_confidence_flow_threshold=0.5,
        world_physical_confidence_loss_weight=1.0,
        world_physical_confidence_positive_weight=1.0))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_aligned10/'
    'epoch_10.pth')
resume_from = None

total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=5, max_keep_ckpts=1)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_'
    'confidence_pilot')
