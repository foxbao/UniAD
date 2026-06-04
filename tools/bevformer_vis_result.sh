#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-projects/configs/bevformer_tiny/bevformer_tiny_imgx0.25.py}
PREDROOT=${PREDROOT:-projects/work_dirs/bevformer_tiny/bevformer_tiny_imgx0.25/eval_results.pkl}
OUT_DIR=${OUT_DIR:-projects/work_dirs/bevformer_tiny/bevformer_tiny_imgx0.25/vis_eval_results}
START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-24}
SCORE_THR=${SCORE_THR:-0.2}
TOPK=${TOPK:-80}
CAM_WIDTH=${CAM_WIDTH:-640}
VIDEO=${VIDEO:-epoch8_vis.webm}
FPS=${FPS:-4.0}

PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-}" \
python tools/analysis_tools/visualize_bevformer_detection.py \
  --config "$CONFIG" \
  --predroot "$PREDROOT" \
  --out-dir "$OUT_DIR" \
  --start-index "$START_INDEX" \
  --max-samples "$MAX_SAMPLES" \
  --score-thr "$SCORE_THR" \
  --topk "$TOPK" \
  --cam-width "$CAM_WIDTH" \
  --video "$VIDEO" \
  --fps "$FPS"
