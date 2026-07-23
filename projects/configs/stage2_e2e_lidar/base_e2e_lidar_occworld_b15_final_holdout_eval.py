_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# This config is only valid after validation has frozen epoch 7, visibility
# threshold 0.7 and B14 local-overlay threshold 0.9. It exposes no train,
# validation, old-test or blind20 labels through the evaluation loaders.
final_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_b15_final_holdout30_evaluation_v1.json')
final_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_b15_final_holdout30_v1')
final_history_root = (
    'outputs/patent_2026_occ/occworld_history_b15_final_holdout30_v1')

data = dict(
    val=dict(
        occworld_label_root=final_label_root,
        occworld_history_root=final_history_root,
        occworld_manifest=final_manifest,
        occworld_split='final_holdout'),
    test=dict(
        occworld_label_root=final_label_root,
        occworld_history_root=final_history_root,
        occworld_manifest=final_manifest,
        occworld_split='final_holdout'))
