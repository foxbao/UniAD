_base_ = [
    './base_e2e_lidar_occworld_b14_local_overlay_eval.py'
]

# B15 deployed evaluation remains validation-only and retains B14's frozen
# local-overlay threshold. Raw checkpoint selection uses the sibling
# ``*_raw_eval.py`` config. The old test, consumed blind20 and newly frozen
# final holdout are not addressable through either config.
full_manifest = (
    'documents/patent_2026_occ/'
    'kl_occworld_full_train3_final30_manifest_v1.json')

data = dict(
    val=dict(
        occworld_manifest=full_manifest,
        occworld_split='validation'),
    test=dict(
        occworld_manifest=full_manifest,
        occworld_split='validation'))
