_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10.py'
]

# B16 is a validation-gated pilot for the measured GT-to-TrackFormer anchor
# domain gap. Only the current anchor is replaced; the five-frame history is
# still B15's annotation replay, so this is not a deployment-equivalent run.
predicted_current_anchor_root = (
    'outputs/patent_2026_occ/'
    'occworld_predicted_current_anchors_b15_train_v1')

data = dict(
    train=dict(
        occworld_current_anchor_root=predicted_current_anchor_root))

# Fine-tune the already selected B15 decoder under the changed causal input.
# This is an explicit new schedule, not a continuation of B15's cosine curve.
load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b15_full_train_continuous10/epoch_7.pth')
resume_from = None

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b16_predicted_current_anchor_finetune3')
