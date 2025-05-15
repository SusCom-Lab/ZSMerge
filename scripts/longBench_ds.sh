#!/bin/bash
echo hostname
hostname

model_li=('meta-llama/Llama-3.1-8B-Instruct' 'meta-llama/Llama-3-8B-Instruct')
max_capacity_prompts_li=(1024 512)
device='cuda:0'
export CUDA_VISIBLE_DEVICES=2

for max_capacity_prompts in "${max_capacity_prompts_li[@]}"
do
    for model in "${model_li[@]}"
    do
    echo $model
        python exam/exam_longbench_ds.py     \
            --device $device \
            --model_path $model     \
            --max_capacity_prompts $max_capacity_prompts  \
            --merge \
            --save_dir ./results/longbench_ds

        python exam/exam_longbench_ds.py     \
            --device $device \
            --model_path $model     \
            --max_capacity_prompts $max_capacity_prompts  \
            --save_dir ./results/longbench_ds

        # python exam/exam_longbench_ds.py     \
        #     --device $device \
        #     --max_num_examples 200    \
        #     --model_path $model     \
        #     --max_capacity_prompts 512  \
        #     --save_dir ./results/longbench_ds
    done
done

# python eval_ds.py --results_dir ./results/longbench_ds
