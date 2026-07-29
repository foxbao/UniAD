_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# Internal-dev is frozen before B24-head training, but remains an exploratory
# readout because B17A itself saw these original train scenes.
internal_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_b24_exploratory_internal_split_v1.json')
predicted_train_input_root = (
    'outputs/patent_2026_occ/'
    'occworld_online_inputs_full_predicted_history_train_v1')
full_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_full_train_v1')
full_history_root = (
    'outputs/patent_2026_occ/occworld_history_full_train_v1')

model = dict(
    occ_head=dict(
        world_local_flow_overlay_threshold=None,
        world_physical_flow_fusion_threshold=None,
        world_use_event_reliability=True,
        world_event_reliability_prior=0.01,
        world_event_reliability_loss_weight=0.0))

data = dict(
    val=dict(
        occworld_label_root=full_label_root,
        occworld_history_root=full_history_root,
        occworld_manifest=internal_manifest,
        occworld_split='internal_dev',
        occworld_online_input_root=predicted_train_input_root,
        occworld_online_input_probability=1.0),
    test=dict(
        occworld_label_root=full_label_root,
        occworld_history_root=full_history_root,
        occworld_manifest=internal_manifest,
        occworld_split='internal_dev',
        occworld_online_input_root=predicted_train_input_root,
        occworld_online_input_probability=1.0))
