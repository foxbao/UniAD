_base_ = ['./base_e2e_lidar_occ.py']

# Stage-2 LiDAR e2e + LLMBridgeHead (cross-modal distillation).
# Smoke-test config: trains/evaluates on the val 1500-frame VLM-caption subset
# (kl_infos_val_sub1k_vlmcap.pkl, which carries geo_facts.summary). The goal here
# is to verify forward + llm.loss_llm decreases without disturbing the other
# task losses; full-scale training waits for train-set VLM-caption summaries.
# See documents/llm_integration_plan.md (step 7).

point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]

model = dict(
    task_loss_weight=dict(track=1.0, motion=1.0, occ=1.0, llm=1.0),
    llm_head=dict(
        type='LLMBridgeHead',
        llm_name='/mnt/disk1/models/Qwen2.5-0.5B-Instruct',
        d_llm=896,
        in_channels=256,
        max_agents=64,
        use_spatial_pe=True,
        pc_range=point_cloud_range,
        freeze_llm=True,
        use_lora=True,
        loss_weight=1.0))

# Point train/val/test at the VLM-caption subset (summaries merged in).
_sub = 'kl_infos_val_sub1k_vlmcap.pkl'
data = dict(
    train=dict(ann_file=_sub),
    val=dict(ann_file=_sub),
    test=dict(ann_file=_sub))

# The occ base's load_from points at a stage-2 ckpt that may not exist; the
# smoke test only checks the llm branch forward/loss, so start from the
# stage-1 drivable ckpt (present) instead of a missing stage-2 one.
load_from = './projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/latest.pth'

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ_llm'
