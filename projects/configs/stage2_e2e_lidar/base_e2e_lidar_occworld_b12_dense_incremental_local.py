_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_aligned10.py'
]

# B12 uses the dense 145-reference train set while the frozen validation split
# remains unchanged. This arm retains the original incremental flow and local
# 5x5 history encoder as the factorial baseline.
dense_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_dense_train145')
dense_history_root = (
    'outputs/patent_2026_occ/occworld_history_dense_train145')
dense_manifest = (
    'documents/patent_2026_occ/kl_occworld_dense_train3_manifest_v1.json')

model = dict(
    occ_head=dict(
        world_flow_parameterization='incremental',
        world_use_wide_history_context=False))

data = dict(
    train=dict(
        occworld_label_root=dense_label_root,
        occworld_history_root=dense_history_root,
        occworld_manifest=dense_manifest,
        occworld_split='train'))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_'
    'aligned10/epoch_10.pth')
resume_from = None

total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=5, max_keep_ckpts=1)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b12_dense_incremental_local')
