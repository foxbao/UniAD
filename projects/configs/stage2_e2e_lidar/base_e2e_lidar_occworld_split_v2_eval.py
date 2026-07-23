_base_ = ['./base_e2e_lidar_occworld_split_v2.py']

occworld_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_expanded70')
occworld_manifest = (
    'documents/patent_2026_occ/kl_occworld_scene_split_v2.json')

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
