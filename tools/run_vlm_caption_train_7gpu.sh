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

for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$g nohup python tools/data_converter/gen_vlm_caption.py \
        --pkl-path $PKL --model-path $MODEL --device cuda:0 \
        --out-path /tmp/train_vlmcap.pkl \
        --num-shards $N --shard-id $i \
        > /tmp/vlmcap_shard${i}.log 2>&1 &
    echo "shard $i -> GPU $g (pid $!)"
done
echo "launched $N shards. watch: tail -f /tmp/vlmcap_shard0.log"
echo "when all done, merge with:"
echo "  conda activate uniad_train"
echo "  python tools/data_converter/merge_summaries.py \\"
echo "    --pkl-path $PKL \\"
echo "    --json-path /tmp/train_vlmcap_summaries.shard*of7.json \\"
echo "    --out-path data/kl_8/kl_infos_train_vlmcap.pkl"
