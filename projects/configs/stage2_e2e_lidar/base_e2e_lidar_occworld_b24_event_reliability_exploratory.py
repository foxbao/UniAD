_base_ = [
    './base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu.py'
]

# B24 is an exploratory train-scene experiment. The B17A decoder, flow,
# TrackFormer, MotionHead and original occupancy path remain frozen. Only the
# new causal local-context reliability head can update.
internal_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_b24_exploratory_internal_split_v1.json')
predicted_train_input_root = (
    'outputs/patent_2026_occ/'
    'occworld_online_inputs_full_predicted_history_train_v1')

data = dict(
    train=dict(
        occworld_manifest=internal_manifest,
        occworld_split='internal_train',
        occworld_online_input_root=predicted_train_input_root,
        occworld_online_input_probability=1.0))

model = dict(
    freeze_except_prefixes=[
        'occ_head.world_decoder.event_reliability_context_projection',
        'occ_head.world_decoder.event_reliability_head',
    ],
    freeze_except_eval=True,
    occ_head=dict(
        world_local_flow_overlay_threshold=None,
        world_physical_flow_fusion_threshold=None,
        world_use_event_reliability=True,
        world_event_reliability_prior=0.01,
        world_event_reliability_loss_weight=1.0,
        world_current_loss_weight=0.0,
        world_future_loss_weight=0.0,
        world_visibility_loss_weight=0.0,
        world_future_transition_loss_weight=0.0,
        world_future_stability_loss_weight=0.0,
        world_future_change_gate_loss_weight=0.0,
        world_future_changed_class_loss_weight=0.0,
        world_flow_loss_weight=0.0,
        world_physical_confidence_loss_weight=0.0))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu/'
    'epoch_3.pth')
resume_from = None

optimizer = dict(type='AdamW', lr=1e-3, weight_decay=1e-4)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

gpu_ids = range(2, 8)
total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b24_event_reliability_exploratory_3epoch_6gpu')
