#!/bin/bash

# Throughput Evaluation Script for MergeKV
# Compares baseline vs MergeKV performance across different configurations

# Configuration
model_li=("meta-llama/Llama-2-7b-hf" "Qwen/Qwen1.5-7B")
batch_sizes=(1 4 8 16)
prompt_lengths=(512 1024 2048)
generate_length=512
cache_ratios=(0.05 0.1 0.2)

# MergeKV parameters (optimized defaults)
cache_tail=0.5
cache_dense=0.02
scale_factor=1.0
shrink_factor=0.98
window_size=8
kernel_size=5
metric="l2"
score_update="sum"

echo "Starting Throughput Evaluation"
echo "Models: ${model_li[*]}"
echo "Batch sizes: ${batch_sizes[*]}"
echo "Prompt lengths: ${prompt_lengths[*]}"
echo "Generate length: $generate_length"
echo "Cache ratios: ${cache_ratios[*]}"
echo "=============================================="

for model in "${model_li[@]}"; do
    echo "Evaluating model: $model"

    for batch_size in "${batch_sizes[@]}"; do
        for prompt_length in "${prompt_lengths[@]}"; do

            echo "Configuration: batch_size=$batch_size, prompt_length=$prompt_length"

            # Baseline (no merge)
            echo "Running baseline (no merge)..."
            python exam/exam_throughput.py \
                --model_name "$model" \
                --batch_size $batch_size \
                --prompt_length $prompt_length \
                --generate_length $generate_length

            # MergeKV with different cache ratios
            for cache_ratio in "${cache_ratios[@]}"; do
                echo "Running MergeKV with cache_ratio=$cache_ratio..."
                python exam/exam_throughput.py \
                    --model_name "$model" \
                    --batch_size $batch_size \
                    --prompt_length $prompt_length \
                    --generate_length $generate_length \
                    --cache_ratio $cache_ratio \
                    --cache_tail $cache_tail \
                    --cache_dense $cache_dense \
                    --scale_factor $scale_factor \
                    --shrink_factor $shrink_factor \
                    --window_size $window_size \
                    --kernel_size $kernel_size \
                    --metric $metric \
                    --score_update $score_update \
                    --merge
            done

            echo "----------------------------------------"
        done
    done

    echo "Completed evaluation for $model"
    echo "=============================================="
done

echo "Throughput evaluation completed!"
echo "Check the output above for performance comparisons"