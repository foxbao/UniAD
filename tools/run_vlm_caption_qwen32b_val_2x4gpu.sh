#!/usr/bin/env bash
# Run Qwen3-VL-32B teacher caption generation on val with 2 data shards.
# Each shard uses one 4-GPU model-parallel process:
#   shard 0 -> physical GPUs 0,1,2,3
#   shard 1 -> physical GPUs 4,5,6,7
#
# Usage:
#   bash tools/run_vlm_caption_qwen32b_val_2x4gpu.sh
#
# Optional overrides:
#   PKL=data/kl_8/kl_infos_val_sub6cam_geo.pkl
#   MODEL=/mnt/disk1/models/Qwen3-VL-32B-Instruct
#   OUT_PREFIX=/tmp/val_sub6cam_qwen3vl32b_annotated_vlmcap
#   ANNOTATED_DIR=/mnt/disk1/tmp/val_sub6cam_qwen3vl32b_annotated_imgs
#   MAX_MEMORY=0:22GiB,1:22GiB,2:22GiB,3:22GiB
#   LIMIT=20

set -u

source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen_vl
cd "$(dirname "$0")/.."

PKL=${PKL:-data/kl_8/kl_infos_val_sub6cam_geo.pkl}
MODEL=${MODEL:-/mnt/disk1/models/Qwen3-VL-32B-Instruct}
OUT_PREFIX=${OUT_PREFIX:-/tmp/val_sub6cam_qwen3vl32b_annotated_vlmcap}
ANNOTATED_DIR=${ANNOTATED_DIR:-/mnt/disk1/tmp/val_sub6cam_qwen3vl32b_annotated_imgs}
MAX_MEMORY=${MAX_MEMORY:-0:22GiB,1:22GiB,2:22GiB,3:22GiB}
MERGED_OUT=${MERGED_OUT:-data/kl_8/kl_infos_val_sub6cam_qwen3vl32b_annotated_vlmcap.pkl}
LIMIT=${LIMIT:-}

GPU_GROUPS=("0,1,2,3" "4,5,6,7")
N=2
PIDS=()

EXTRA_ARGS=()
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

mkdir -p "$(dirname "$ANNOTATED_DIR")"

for i in 0 1; do
    gpus=${GPU_GROUPS[$i]}
    log="/tmp/qwen32b_val_shard${i}of${N}.log"
    shard_annotated_dir="${ANNOTATED_DIR}_shard${i}"
    CUDA_VISIBLE_DEVICES=$gpus \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup python tools/data_converter/gen_vlm_caption.py \
        --pkl-path "$PKL" \
        --model-path "$MODEL" \
        --device-map auto \
        --max-memory "$MAX_MEMORY" \
        --annotate-targets \
        --annotated-dir "$shard_annotated_dir" \
        --out-path "${OUT_PREFIX}.pkl" \
        --num-shards "$N" --shard-id "$i" \
        "${EXTRA_ARGS[@]}" \
        > "$log" 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "shard $i/$N -> GPUs $gpus (pid $pid, log $log)"
    sleep 20
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "!! shard $i died during startup. Last log lines:"
        tail -20 "$log"
        echo "!! Kill any survivor with: pkill -f 'gen_vlm_caption.py.*Qwen3-VL-32B'"
        exit 1
    fi
done

echo
echo "all shards launched. Watch progress in another terminal with:"
echo "  tail -f /tmp/qwen32b_val_shard0of2.log /tmp/qwen32b_val_shard1of2.log"
echo
echo "when both shards finish, merge with:"
echo "  conda activate uniad_train"
echo "  python tools/data_converter/merge_summaries.py \\"
echo "    --pkl-path $PKL \\"
echo "    --json-path ${OUT_PREFIX}_summaries.shard*of${N}.json \\"
echo "    --out-path $MERGED_OUT"
echo
status=0
for idx in "${!PIDS[@]}"; do
    pid=${PIDS[$idx]}
    if wait "$pid"; then
        echo "shard $idx finished."
    else
        rc=$?
        echo "!! shard $idx failed with exit code $rc. Last log lines:"
        tail -40 "/tmp/qwen32b_val_shard${idx}of${N}.log"
        status=$rc
    fi
done

if [ "$status" -eq 0 ]; then
    echo
    echo "all shards finished. Sidecars:"
    ls -lh "${OUT_PREFIX}_summaries".shard*of${N}.json
    echo
    echo "merge now with:"
    echo "  conda activate uniad_train"
    echo "  python tools/data_converter/merge_summaries.py \\"
    echo "    --pkl-path $PKL \\"
    echo "    --json-path ${OUT_PREFIX}_summaries.shard*of${N}.json \\"
    echo "    --out-path $MERGED_OUT"
fi

exit "$status"
