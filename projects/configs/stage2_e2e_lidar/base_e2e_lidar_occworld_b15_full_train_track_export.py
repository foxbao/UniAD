_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# Evaluation-pipeline view of the frozen B15 train split. This config exists
# only to export causal TrackFormer boxes for anchor-robust follow-up training;
# it must not be used for checkpoint selection or reported validation metrics.
full_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_full_train_v1')
full_history_root = (
    'outputs/patent_2026_occ/occworld_history_full_train_v1')
full_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_full_train3_final30_manifest_v1.json')

data = dict(
    val=dict(
        occworld_label_root=full_label_root,
        occworld_history_root=full_history_root,
        occworld_manifest=full_manifest,
        occworld_split='train'))
