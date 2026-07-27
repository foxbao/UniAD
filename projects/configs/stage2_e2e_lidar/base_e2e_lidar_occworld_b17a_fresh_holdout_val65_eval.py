_base_ = [
    './base_e2e_lidar_occworld_b17a_fresh_holdout_val65_track_export.py'
]

# Complete online inputs are built from sequential TrackFormer replay. The
# inherited B15/B14 thresholds remain frozen and must not be retuned here.
fresh_online_input_root = (
    'outputs/patent_2026_occ/'
    'occworld_online_inputs_b17_fresh_holdout_val65_v1')

data = dict(
    val=dict(occworld_online_input_root=fresh_online_input_root),
    test=dict(occworld_online_input_root=fresh_online_input_root))
