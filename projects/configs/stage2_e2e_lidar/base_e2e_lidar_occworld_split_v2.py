_base_ = ['./base_e2e_lidar_occworld.py']

occworld_label_root = (
    'outputs/patent_2026_occ/occworld_sequence_expanded70')
occworld_manifest = (
    'documents/patent_2026_occ/kl_occworld_scene_split_v2.json')

# V2 contains 70 scene-isolated references. The 15 scenes used during the V1
# study are restricted to train; all 20 validation/test scenes are fresh.
data = dict(
    train=dict(
        occworld_label_root=occworld_label_root,
        occworld_manifest=occworld_manifest,
        occworld_split='train'))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2')
