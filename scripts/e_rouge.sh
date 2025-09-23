#!/bin/bash

# Configuration for MergeKV experiments
model_li=("meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
budget_li=(0.05 0.1 0.2)
device='cuda'
shots=3
# cache_tail=0.5
# scale_factor=1.0
# shrink_factor=0.98
cache_dense=2
# metric="l2"
# score_update="sum"
# window_size=8
# kernel_size=5
# out_state=0

for model in "${model_li[@]}"
do
    for budget in "${budget_li[@]}"
    do
        python exam/exam_rouge.py \
            --merge \
            --method ZSMerge \
            --model_name $model \
            --device $device \
            --shots $shots \
            --cache_budget $budget \
            # --cache_tail $cache_tail \
            --cache_dense $cache_dense \
            # --scale_factor $scale_factor \
            # --shrink_factor $shrink_factor \
            # --metric $metric \
            # --score_update $score_update \
            # --window_size $window_size \
            # --kernel_size $kernel_size \
            # --out_state $out_state \
    done
done