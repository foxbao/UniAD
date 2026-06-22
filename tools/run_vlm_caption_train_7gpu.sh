#!/usr/bin/env bash
# Run train-set VLM-caption generation data-parallel across GPUs 1-7 (7 shards).
# Each shard writes /tmp/train_vlmcap_summaries.shard{i}of7.json.
# After all finish: merge with tools/data_converter/merge_summaries.py.
#
# Usage:  bash tools/run_vlm_caption_train_7gpu.sh
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen_vl
cd "$(dirname "$0")/.."

PKL=data/kl_8/kl_infos_train_with_cam_geo.pkl
MODEL=/mnt/disk1/models/Qwen2.5-VL-7B-Instruct
N=7
GPUS=(1 2 3 4 5 6 7)   # leave GPU0 free
PIDS=()

for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]}
    # Stagger launches: 7 procs each loading a 16GB model at the exact same
    # instant spiked allocator pressure and OOM'd. A few seconds apart lets
    # each finish its load before the next starts. expandable_segments curbs
    # fragmentation (the OOM error explicitly recommended it).
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup python tools/data_converter/gen_vlm_caption.py \
        --pkl-path $PKL --model-path $MODEL --device cuda:0 \
        --out-path /tmp/train_vlmcap.pkl \
        --num-shards $N --shard-id $i \
        > /tmp/vlmcap_shard${i}.log 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "shard $i -> GPU $g (pid $pid)"
    # Self-check: wait for the model to load, then confirm the shard is still
    # alive (didn't OOM/crash) before launching the next. Catches failures
    # immediately instead of after all 7 are fired.
    sleep 12
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "!! shard $i (pid $pid) DIED during startup. Last log lines:"
        tail -6 /tmp/vlmcap_shard${i}.log | tr '\r' '\n' | grep -vE 'Loading checkpoint' | tail -6
        echo "!! Aborting launch. Kill any survivors with: pkill -9 -f gen_vlm_caption.py"
        exit 1
    fi
done

echo
echo "all $N shards launched and survived startup. GPU usage:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo
echo "when all done, merge with:"
echo "  conda activate uniad_train"
echo "  python tools/data_converter/merge_summaries.py \\"
echo "    --pkl-path $PKL \\"
echo "    --json-path /tmp/train_vlmcap_summaries.shard*of7.json \\"
echo "    --out-path data/kl_8/kl_infos_train_vlmcap.pkl"
echo
echo "===== live progress (Ctrl-C to stop watching; shards keep running) ====="
# Foreground tail so the terminal keeps scrolling all shards' progress; the
# shards run in the background, so Ctrl-C only stops watching, not the work.
tail -n 2 -f /tmp/vlmcap_shard*.log

