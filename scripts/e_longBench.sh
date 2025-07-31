#!/bin/bash
echo hostname 
hostname


# tmux new -s ibLM
# source venv/bin/activate
# tmux a -t ibLl


device='cuda:1'
n_sample=100
model='01-ai/Yi-1.5-6B'

# python exam/exam_longbench_ds.py     \
#     --model_path $model     \
#     --save_dir ./results/longbench_ds \
#     --device $device \
#     --max_num_examples $n_sample    \
#     --max_capacity_prompts 1024 \
#     --cache_tail 0.1 \
#     --cache_dense 0.1 \
#     --scale_factor 1.0 \
#     --window_size 12 \
#     --merge


python exam/exam_longbench_ds.py     \
    --model_path $model     \
    --save_dir ./results/longbench_ds \
    --device $device \
    --max_num_examples $n_sample
