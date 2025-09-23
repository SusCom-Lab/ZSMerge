#!/bin/bash
echo hostname
hostname

max_capacity_prompts_li=(1024 512)
device='cuda:0'
# export CUDA_VISIBLE_DEVICES=2


for model in "${model_li[@]}"
do
    

    for max_capacity_prompts in "${max_capacity_prompts_li[@]}"
    do
    echo $model

        python exam/exam_longbench_ds.py     \
            --device $device \
            --model_path $model     \
            --max_capacity_prompts $max_capacity_prompts  \
            --merge \
            --save_dir ./results/longbench_ds
    done

    # python exam/exam_longbench_ds.py     \
    #     --device $device \
    #     --model_path $model     \
    #     --max_capacity_prompts 0  \
    #     --window_size 8 \
    #     --window_pool maxpool \
    #     --save_dir ./results/longbench_ds

done

# python exam/score_longbench_ds.py --results_dir ./results/longbench_ds_0922
