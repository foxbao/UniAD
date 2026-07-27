_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# Frozen fresh-holdout view used only for sequential TrackFormer replay. It
# intentionally has no online-input override because this pass creates the
# predicted current/history inputs consumed by the model-evaluation config.
fresh_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_b17a_fresh_holdout_val65_evaluation_v1.json')
fresh_label_root = (
    'outputs/patent_2026_occ/'
    'occworld_sequence_b17_fresh_holdout_val65_v1')
fresh_history_root = (
    'outputs/patent_2026_occ/'
    'occworld_history_b17_fresh_holdout_val65_v1')

data = dict(
    val=dict(
        ann_file='kl_infos_val.pkl',
        occworld_label_root=fresh_label_root,
        occworld_history_root=fresh_history_root,
        occworld_manifest=fresh_manifest,
        occworld_split='fresh_holdout'),
    test=dict(
        ann_file='kl_infos_val.pkl',
        occworld_label_root=fresh_label_root,
        occworld_history_root=fresh_history_root,
        occworld_manifest=fresh_manifest,
        occworld_split='fresh_holdout'))
