_base_ = ['./base_e2e_lidar_occworld.py']

# Formal experiments use the frozen scene-level split. Regenerate the
# ignored manifest with tools/analysis_tools/build_kl_occworld_scene_split.py.
data = dict(
    train=dict(
        occworld_manifest=(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v1.json'),
        occworld_split='train'))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v1')
