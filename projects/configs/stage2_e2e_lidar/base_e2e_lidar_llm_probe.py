_base_ = ['./base_e2e_lidar.py']

# Stage-1 LLM probe: keep the LiDAR perception stack fixed and train only the
# LLM bridge. This asks whether frozen UniAD LiDAR object queries contain
# enough scene semantics for a small LLM to reproduce the VLM teacher caption.
#
# This intentionally does NOT inherit base_e2e_lidar_occ.py: the probe consumes
# track_query_embeddings + box centres, not occ outputs. Keeping occ out reduces
# coupling and avoids interpreting downstream task changes as LLM evidence.

point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]

model = dict(
    llm_probe_only=True,
    freeze_non_llm=True,
    task_loss_weight=dict(track=0.0, map=0.0, motion=0.0, llm=1.0),
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
        detach_inputs=True,
        loss_weight=1.0))

data = dict(
    train=dict(ann_file='kl_infos_train_vlmcap.pkl'),
    val=dict(ann_file='kl_infos_val_sub6cam_v3_vlmcap.pkl'),
    test=dict(ann_file='kl_infos_val_sub6cam_v3_vlmcap.pkl'))

# Load a converged LiDAR e2e model as a frozen feature extractor. The probe
# trains only llm_head.* parameters on top of those fixed queries.
load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar/latest.pth'
resume_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_llm_probe'

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
find_unused_parameters = True
