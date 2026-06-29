#!/usr/bin/env bash
set -o pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: WORK_DIR=/path/to/work_dir CUDA_VISIBLE_DEVICES=0,1 $0 <config> <gpus> [train.py args...]"
    echo "If WORK_DIR is not set, it falls back to the same config-derived path as tools/uniad_dist_train.sh."
    exit 1
fi

T=$(date +%m%d%H%M)

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1
GPUS=$2
shift 2
# -------------------------------------------------- #
GPUS_PER_NODE=$((GPUS < 8 ? GPUS : 8))
NNODES=$((GPUS / GPUS_PER_NODE))

MASTER_PORT=${MASTER_PORT:-28598}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}
TORCHRUN=${TORCHRUN:-torchrun}

DEFAULT_WORK_DIR=$(echo "${CFG%.*}" | sed -e "s/configs/work_dirs/g")/
WORK_DIR=${WORK_DIR:-$DEFAULT_WORK_DIR}
WORK_DIR=${WORK_DIR%/}
LOG_DIR=${WORK_DIR}/logs

# Intermediate files and logs will be saved to WORK_DIR.
mkdir -p "${LOG_DIR}"

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

PYTHONPATH="${SCRIPT_DIR}/..":$PYTHONPATH \
${TORCHRUN} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    --max_restarts=0 \
    "${SCRIPT_DIR}/train.py" \
    "${CFG}" \
    --launcher pytorch "$@" \
    --deterministic \
    --work-dir "${WORK_DIR}" \
    2>&1 | tee "${LOG_DIR}/train.${T}"
