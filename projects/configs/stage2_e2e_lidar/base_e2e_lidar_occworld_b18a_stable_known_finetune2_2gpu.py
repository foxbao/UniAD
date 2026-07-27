_base_ = [
    './base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu.py'
]

# B18A isolates one validation-derived hypothesis: improve raw predictions on
# currently observed voxels that stay semantically unchanged in future GT.
# Dynamic/change voxels are excluded from this auxiliary loss.
model = dict(
    occ_head=dict(
        world_future_stability_loss_weight=0.1))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu/'
    'epoch_3.pth')
resume_from = None

# Keep B17A's per-sample learning rate for a two-GPU validation experiment:
# 1.875e-5 / 3 * 2 = 1.25e-5.
optimizer = dict(type='AdamW', lr=1.25e-5, weight_decay=0.01)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

gpu_ids = range(2)
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b18a_stable_known_finetune2_2gpu')
