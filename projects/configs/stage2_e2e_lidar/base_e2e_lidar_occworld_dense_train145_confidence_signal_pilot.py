_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_confidence_signal_pilot.py'
]

# B11 repeats the B10 candidate-aware confidence experiment with up to three
# temporally separated references in each frozen train scene. Validation and
# all holdouts remain on the original scene-isolated manifests.
dense_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_dense_train145')
dense_history_root = (
    'outputs/patent_2026_occ/occworld_history_dense_train145')
dense_manifest = (
    'documents/patent_2026_occ/kl_occworld_dense_train3_manifest_v1.json')

data = dict(
    train=dict(
        occworld_label_root=dense_label_root,
        occworld_history_root=dense_history_root,
        occworld_manifest=dense_manifest,
        occworld_split='train'))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_dense_train145_confidence_signal_pilot')
