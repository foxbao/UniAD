_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10.py'
]

# B17A adapts the B15 decoder to the complete deployment-equivalent causal
# input: current plus all five history frames use sequential TrackFormer boxes.
predicted_train_input_root = (
    'outputs/patent_2026_occ/'
    'occworld_online_inputs_full_predicted_history_train_v1')

data = dict(
    train=dict(
        occworld_online_input_root=predicted_train_input_root,
        occworld_online_input_probability=1.0))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b15_full_train_continuous10/epoch_7.pth')
resume_from = None

# Three GPUs run B17A while the other three run B17B. Preserve B16's
# per-sample learning rate: 5e-5 / 8 * 3 = 1.875e-5.
optimizer = dict(type='AdamW', lr=1.875e-5, weight_decay=0.01)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

gpu_ids = range(3)
total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu')
