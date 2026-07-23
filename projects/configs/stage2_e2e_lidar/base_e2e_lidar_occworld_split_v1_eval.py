_base_ = ['./base_e2e_lidar_occworld_split_v1_world_only.py']

occworld_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_batch20')
occworld_manifest = (
    'documents/patent_2026_occ/kl_occworld_scene_split_v1.json')

# The prototype labels were generated from kl_infos_train.pkl, so held-out
# validation/test here means scene-level manifest splits within that source
# annotation. The scenes remain disjoint from the ten training scenes.
data = dict(
    val=dict(
        type='KlOccWorldDataset',
        ann_file='kl_infos_train.pkl',
        test_mode=True,
        occworld_label_root=occworld_label_root,
        occworld_expected_shape=(5, 10, 120, 160),
        occworld_ignore_index=255,
        occworld_manifest=occworld_manifest,
        occworld_split='validation'),
    test=dict(
        type='KlOccWorldDataset',
        ann_file='kl_infos_train.pkl',
        test_mode=True,
        occworld_label_root=occworld_label_root,
        occworld_expected_shape=(5, 10, 120, 160),
        occworld_ignore_index=255,
        occworld_manifest=occworld_manifest,
        occworld_split='test'))
