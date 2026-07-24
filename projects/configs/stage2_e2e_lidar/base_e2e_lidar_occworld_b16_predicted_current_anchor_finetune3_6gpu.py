_base_ = [
    './base_e2e_lidar_occworld_b16_predicted_current_anchor_finetune3.py'
]

# Resource-compatible B16 pilot for physical GPUs 2-7. With one sample per
# GPU, the global batch changes from 8 to 6, so scale the fine-tuning learning
# rate linearly from 5e-5 to 3.75e-5. The dataset still receives three complete
# passes: ceil(1305 / 6) = 218 updates/epoch, 654 updates in total.
optimizer = dict(lr=3.75e-5)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b16_predicted_current_anchor_finetune3_6gpu')
