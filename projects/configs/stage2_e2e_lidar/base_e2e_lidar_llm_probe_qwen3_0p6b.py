_base_ = ['./base_e2e_lidar_llm_probe.py']

# Stage-1 LLM probe variant with Qwen3-0.6B as the student language model.
#
# Important environment note:
#   Run this config in the isolated env:
#     conda activate /mnt/disk1/conda_envs/uniad_train_qwen3_py39
#   It keeps UniAD's torch1.12/mmcv stack, uses transformers==4.51.3 for
#   model_type='qwen3', and has a sitecustomize.py shim for torch1.12 symbols
#   imported by newer transformers. Do not upgrade the main uniad_train env.
#   Smoke verified: build_model -> UniADMotionLidar + LLMBridgeHead +
#   PeftModelForCausalLM, with only llm_head trainable in probe mode.

model = dict(
    llm_head=dict(
        llm_name='/mnt/disk1/models/Qwen3-0.6B',
        d_llm=1024,
    ))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_llm_probe_qwen3_0p6b'
