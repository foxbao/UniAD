_base_ = [
    './base_e2e_lidar_occworld_b17a_full_predicted_history_finetune3_3gpu.py'
]

# B17B uses stochastic whole-sample mixing. Current and history always come
# from the same domain; fields are never mixed independently within a sample.
data = dict(
    train=dict(
        occworld_online_input_probability=0.5))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b17b_mixed_predicted_history_finetune3_3gpu')
