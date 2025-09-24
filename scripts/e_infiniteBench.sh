#!/bin/bash
echo hostname
hostname

# tmux new -s ibLM
# source venv/bin/activate
# tmux a -t ibLl


device='cuda:1'
method=full
n_sample=0
model='01-ai/Yi-1.5-9B'
model='meta-llama/Llama-3.1-8B-Instruct'

python exam/exam_infiniteBench.py \
    --model $model \
    --device $device \
    --method $method \
    --n_sample $n_sample

# device='cuda:0'
method=mergekv
cache_size=0.5

python exam/exam_infiniteBench.py \
    --model $model \
    --device $device \
    --method $method \
    --merge \
    --cache_size $cache_size \
    --n_sample $n_sample

