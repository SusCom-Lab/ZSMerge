#!/bin/bash
echo hostname
hostname

model_li=('meta-llama/Llama-2-7b-hf' 'mistralai/Mistral-7B-Instruct-v0.3')

device='cuda:0'
export CUDA_VISIBLE_DEVICES=0

for model in "${model_li[@]}"
do
echo $model
    python exam/exam_longbench_ds.py     \
        --device $device \
        --max_num_examples 200    \
        --model_path $model     \
        --max_capacity_prompts 512  \
        --merge \
        --save_dir ./results/longbench_ds

    # python exam/exam_longbench_ds.py     \
    #     --device $device \
    #     --max_num_examples 200    \
    #     --model_path $model     \
    #     --max_capacity_prompts 512  \
    #     --save_dir ./results/longbench_ds
done

# python eval_ds.py --results_dir ./results/longbench_ds
