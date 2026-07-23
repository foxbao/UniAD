#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${REPO_ROOT}/documents/patent_2026_occ/kl_occworld_full_train3_final30_manifest_v1.json"
CONFIG="projects/configs/stage2_e2e_lidar/base_e2e_lidar_occworld_b15_full_train_continuous10.py"
# B15's formal full-data run uses all eight local GPUs. This makes its global
# batch regime explicit and distinct from the earlier two-GPU B13 experiment.
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29535}"

python3 - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    manifest = json.load(source)
if manifest.get('status') != 'ready_after_full_generation_and_audit':
    raise SystemExit(
        'B15 data is not ready_after_full_generation_and_audit')
PY

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" OMP_NUM_THREADS=1 \
conda run --no-capture-output -n uniad_train env PYTHONPATH=. \
python -m torch.distributed.launch \
  --nproc_per_node="${#GPUS[@]}" \
  --master_port="${MASTER_PORT}" \
  tools/train.py "${CONFIG}" \
  --launcher pytorch --no-validate
