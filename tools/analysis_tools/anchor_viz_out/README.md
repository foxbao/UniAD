# Motion anchor ablation: visualization & comparison

Tools and commands for comparing the turn-aware motion anchors against the
original (straight-only) anchors. See `tools/generate_kl_motion_anchors.py`
for how the turn-aware anchors are generated, and the project memory note
`kl-motion-anchor-turn-fix` for the root-cause analysis.

## Configs

- `base_e2e_lidar_HDMap.py` — experiment: turn-aware anchors
  (`motion_anchor_infos_kl_turnaware.pkl`).
- `base_e2e_lidar_HDMap_old_anchor.py` — control: original straight-only
  anchors (`motion_anchor_infos_kl.pkl`). Inherits HDMap, overrides ONLY the
  anchor path + work_dir, so the anchor set is the single variable.

## 1. Regenerate turn-aware anchors

```bash
conda activate uniad_train
python tools/generate_kl_motion_anchors.py \
  --info data/kl_8/kl_infos_train.pkl \
  --out  data/others/motion_anchor_infos_kl_turnaware.pkl \
  --k 6 --steps 12 --turn-weight 8.0
```
Prints a per-mode coverage report; the vehicle group should show modes with
|heading change| > 45deg (errors out if none appear).

## 2. Plot anchor shapes (old vs new)

Standalone snippet that reads both pkls and draws all 6 modes per group
(pedestrian / vehicle). Output: `anchor_viz_out/anchor_compare.png`.
Old = straight radial lines; new = includes left/right turning arcs.

## 3. Visualize motion predictions on turning frames

The val frames richest in turning vehicles (|heading|>45deg), found by scanning
`gt_fut_traj_locs` in `kl_infos_val.pkl`: frame 865 (5 turns, up to 135deg),
867, 1024, 1835, 179, 184-186, 874, 987 ...

Run the existing motion visualizer once per checkpoint, SAME frames. The
experiment uses its own config; the control reuses the HDMap config with a
`--cfg-options` override so only the anchor differs:

```bash
conda activate uniad_train
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):$PYTHONPATH

# NEW (turn-aware) anchors
python tools/analysis_tools/visualize_lidar_e2e_motion.py \
  --config projects/configs/stage2_e2e_lidar/base_e2e_lidar_HDMap.py \
  --checkpoint projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap/epoch_1.pth \
  --out-dir /tmp/turn_new --split val --start-index 862 --max-frames 7

# OLD (straight) anchors -- override anchor path to the original pkl
python tools/analysis_tools/visualize_lidar_e2e_motion.py \
  --config projects/configs/stage2_e2e_lidar/base_e2e_lidar_HDMap.py \
  --checkpoint projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap_old_anchor/epoch_1.pth \
  --out-dir /tmp/turn_old --split val --start-index 862 --max-frames 7 \
  --cfg-options model.motion_head.anchor_info_path=data/others/motion_anchor_infos_kl.pkl
```

Useful flags: `--scene-token` / `--token` to target a clip, `--score-thr`,
`--topk`, `--gt-map-overlay hdmap` to draw lanes on the GT panel. Each output
PNG is a GT-vs-prediction dual panel; same-named files across the two out-dirs
are the same frame.

Pair same-index frames (e.g. `003_000865_*.png`) side by side to read the
difference. Saved comparisons live in `tools/analysis_tools/anchor_viz_out/`.

## 4. Per-epoch metric comparison

Parse the val rows from each run's `*.log.json` and diff per-class
`*_motion_min_ade` / `min_fde`; the turn-aware gain concentrates on large
articulated classes (ContainerForklift, Trailer, Forklift, Truck).
