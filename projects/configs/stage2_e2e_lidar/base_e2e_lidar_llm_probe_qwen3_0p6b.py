_base_ = ['./base_e2e_lidar_llm_probe.py']

# Stage-1 LLM probe variant with Qwen3-0.6B as the student language model.
#
# Important environment note:
#   Qwen3 checkpoints use model_type='qwen3' and require transformers>=4.51.
#   The current uniad_train env has transformers==4.46.3, so this config is
#   intentionally a trial variant and will not build until that compatibility
#   issue is handled in a separate environment or controlled package upgrade.

model = dict(
    llm_head=dict(
        llm_name='/mnt/disk1/models/Qwen3-0.6B',
        d_llm=1024,
    ))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_llm_probe_qwen3_0p6b'
