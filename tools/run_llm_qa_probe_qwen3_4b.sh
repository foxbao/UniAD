#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
REPO_ROOT=$(dirname "${SCRIPT_DIR}")

CFG=${1:-${CFG:-projects/configs/stage2_e2e_lidar/base_e2e_lidar_llm_qa_probe_qwen3_4b.py}}
GPUS=${2:-${GPUS:-4}}
if [ "$#" -ge 1 ]; then
    shift
fi
if [ "$#" -ge 1 ]; then
    shift
fi

# Qwen3 training must avoid user-site TensorFlow imports and use the Qwen3
# compatible transformers environment.
export USE_TF=${USE_TF:-0}
export TRANSFORMERS_NO_TF=${TRANSFORMERS_NO_TF:-1}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export MASTER_PORT=${MASTER_PORT:-28648}
export WORK_DIR=${WORK_DIR:-/mnt/disk1/uniad_work_dirs/stage2_e2e_lidar/base_e2e_lidar_llm_qa_probe_qwen3_4b}
export TORCHRUN=${TORCHRUN:-$(command -v torchrun)}

cd "${REPO_ROOT}" || exit 1
exec "${SCRIPT_DIR}/uniad_dist_train_workdir.sh" "${CFG}" "${GPUS}" "$@"
