_base_ = [
    './base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu.py'
]

# B20 learns only a zero-initialized residual from the existing query-based
# future instance occupancy to 3D future instance logits. All B17A parameters,
# the flow branch and the local overlay remain frozen.
model = dict(
    freeze_except_prefixes=[
        'occ_head.world_decoder.query_occupancy_adapter',
    ],
    freeze_except_eval=True,
    occ_head=dict(
        world_use_query_occupancy_adapter=True,
        world_query_occupancy_adapter_scale=1.0))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu/'
    'epoch_3.pth')
resume_from = None

# The adapter has no bias and only 5 * (4 * 10) = 200 weights. A short,
# validation-only development run is enough to test whether the query signal
# adds information; it is not a full-model fine-tune.
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.0)
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
    'base_e2e_lidar_occworld_b20_query_residual_finetune2_2gpu')
