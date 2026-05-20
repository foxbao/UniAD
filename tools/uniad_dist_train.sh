#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
if [ -z "${MASTER_PORT}" ]; then
    MASTER_PORT=$(python - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))
print(sock.getsockname()[1])
sock.close()
PY
)
fi
RANK=${RANK:-0}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

RESUME=0
EXTRA_ARGS=()
for ARG in "${@:3}"; do
    if [ "$ARG" = "--resume" ]; then
        RESUME=1
    else
        EXTRA_ARGS+=("$ARG")
    fi
done

if [ ${RESUME} -eq 1 ]; then
    RESUME_FROM="${WORK_DIR}latest.pth"
    if [ ! -f "${RESUME_FROM}" ]; then
        echo "Cannot resume: checkpoint not found at ${RESUME_FROM}"
        exit 1
    fi
    EXTRA_ARGS+=("--resume-from" "${RESUME_FROM}")
fi

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch "${EXTRA_ARGS[@]}" \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T
