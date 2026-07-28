_base_ = [
    './base_e2e_lidar_occworld_b23_raw_free_actor_arrival_eval.py'
]

# Single frozen B23 inference pass. Current state and all five history states
# come from sequential TrackFormer replay, never from future OccWorld labels.
fresh_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_b23_fresh_holdout_evaluation_v2.json')
fresh_label_root = (
    'outputs/patent_2026_occ/'
    'occworld_sequence_b23_fresh_holdout_remaining_val31_v1')
fresh_history_root = (
    'outputs/patent_2026_occ/'
    'occworld_history_b23_fresh_holdout_remaining_val31_v1')
fresh_online_input_root = (
    'outputs/patent_2026_occ/'
    'occworld_online_inputs_b23_fresh_holdout_remaining_val31_v1')

data = dict(
    val=dict(
        ann_file='kl_infos_val.pkl',
        occworld_label_root=fresh_label_root,
        occworld_history_root=fresh_history_root,
        occworld_manifest=fresh_manifest,
        occworld_split='fresh_holdout',
        occworld_online_input_root=fresh_online_input_root),
    test=dict(
        ann_file='kl_infos_val.pkl',
        occworld_label_root=fresh_label_root,
        occworld_history_root=fresh_history_root,
        occworld_manifest=fresh_manifest,
        occworld_split='fresh_holdout',
        occworld_online_input_root=fresh_online_input_root))
