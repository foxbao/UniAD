_base_ = ['./base_e2e_lidar_llm_qa_probe.py']

# Larger text-only QA probe. This keeps the same hard-GT QA data path and
# frozen UniAD LiDAR feature extractor as base_e2e_lidar_llm_qa_probe.py, but
# swaps the small Qwen2.5-0.5B backbone for Qwen3-4B-Instruct-2507.
#
# Qwen3-4B config:
#   model_type=qwen3, hidden_size=2560, 36 layers, bf16 weights.
# Keep a separate work_dir so 0.5B and 4B probe results are comparable.

model = dict(
    llm_head=dict(
        llm_name='/mnt/disk1/models/Qwen3-4B-Instruct-2507',
        d_llm=2560,
        max_text_len=160))

work_dir = (
    '/mnt/disk1/uniad_work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_llm_qa_probe_qwen3_4b')
