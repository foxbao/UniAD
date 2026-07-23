_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_pilot_eval.py'
]

blind_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_blind20')
blind_history_root = (
    'outputs/patent_2026_occ/occworld_history_blind20')
blind_manifest = (
    'documents/patent_2026_occ/kl_occworld_blind_holdout_v1.json')

# The exporter maps the explicit ``blind`` output split to this val loader.
# Thresholds remain outside the model and are applied once by the evaluator.
data = dict(
    val=dict(
        occworld_label_root=blind_label_root,
        occworld_history_root=blind_history_root,
        occworld_manifest=blind_manifest,
        occworld_split='blind'))
