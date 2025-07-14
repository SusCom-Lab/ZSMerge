#!/bin/bash

# meta-llama/Llama-2-13b-hf
model_path='Qwen/Qwen1.5-7B'
device='cuda'
shots=3
budget=0.05
cache_tail=0.4
scale_factor=0.6

python exam/exam_rouge.py \
    --merge \
    --method MKV \
    --model_name $model_path \
    --device $device \
    --shots $shots \
    --cache_budget $budget \
    --cache_tail $cache_tail \
    --scale_factor $scale_factor \
    --cache_dense 1
    