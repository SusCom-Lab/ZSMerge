#!/bin/bash
echo hostname
hostname

model_li=('meta-llama/Llama-3.1-8B-Instruct' 'meta-llama/Llama-3-8B-Instruct')
max_capacity_prompts_li=(1024 512)
device='cuda:0'
export CUDA_VISIBLE_DEVICES=2


for model in "${model_li[@]}"
do
    

    for max_capacity_prompts in "${max_capacity_prompts_li[@]}"
    do
    echo $model

        python exam/exam_longbench_ds.py     \
            --device $device \
            --model_path $model     \
            --max_capacity_prompts $max_capacity_prompts  \
            --window_size 8 \
            --window_pool maxpool \
            --merge \
            --save_dir ./results/longbench_ds
    done

    python exam/exam_longbench_ds.py     \
        --device $device \
        --model_path $model     \
        --max_capacity_prompts 0  \
        --window_size 8 \
        --window_pool maxpool \
        --save_dir ./results/longbench_ds

done

# python eval_ds.py --results_dir ./results/longbench_ds
