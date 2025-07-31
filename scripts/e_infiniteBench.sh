#!/bin/bash
echo hostname
hostname

# tmux new -s ibLM
# source venv/bin/activate
# tmux a -t ibLl


device='cuda:0'
method=full
n_sample=0
model='01-ai/Yi-1.5-9B'

python exam/exam_infiniteBench.py \
    --model $model \
    --device $device \
    --method $method \
    --n_sample $n_sample

device='cuda:0'
method=mergekv

python exam/exam_infiniteBench.py \
    --model $model \
    --device $device \
    --method $method \
    --merge \
    --cache_size 0.5 \
    --cache_tail 0.05 \
    --window_size 128 \
    --scale_factor 1.0 \
    --n_sample $n_sample

