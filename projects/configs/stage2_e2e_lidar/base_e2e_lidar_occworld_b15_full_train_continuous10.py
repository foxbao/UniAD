_base_ = [
    './base_e2e_lidar_occworld_b13_dense_incremental_local_continuous10.py'
]

# B15 keeps B13's incremental/local architecture, losses, initialization and
# uninterrupted ten-epoch schedule. Its formal launcher uses a larger global
# batch, so B15 is the full-data final model rather than a strict ablation.
full_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_full_train_v1')
full_history_root = (
    'outputs/patent_2026_occ/occworld_history_full_train_v1')
full_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_full_train3_final30_manifest_v1.json')

data = dict(
    train=dict(
        occworld_label_root=full_label_root,
        occworld_history_root=full_history_root,
        occworld_manifest=full_manifest,
        occworld_split='train'))

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occ/latest.pth')
resume_from = None

total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b15_full_train_continuous10')
