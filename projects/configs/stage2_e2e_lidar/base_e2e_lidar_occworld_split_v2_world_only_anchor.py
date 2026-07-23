_base_ = ['./base_e2e_lidar_occworld_split_v2.py']

# B2 is repeated without architecture changes on the expanded split. Class
# weights are inverse-sqrt frequencies computed from the 50 train scenes only.
model = dict(
    freeze_except_prefixes=['occ_head.world_decoder'],
    freeze_except_eval=True,
    occ_head=dict(
        world_class_weights=[
            0.382372,
            1.278013,
            1.339615,
        ],
        world_use_observation_anchor=True,
        world_observation_semantic_logit_scale=2.0,
        world_observation_valid_logit_scale=4.0))

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

data = dict(samples_per_gpu=1, workers_per_gpu=0)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor')
