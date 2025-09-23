#!/bin/bash

# Hyperparameter Sensitivity Validation Script
# This script explores different values for each parameter while keeping others fixed at anchor values

model_path="meta-llama/Llama-2-7b-hf"
device='cuda'
shots=3
budget=0.05

# Anchor values (baseline configuration) - all AttForwardArgs parameters
anchor_cache_tail=0.5
anchor_cache_dense=0.02
anchor_scale_factor=1.0

echo "Starting Hyperparameter Sensitivity Validation"
echo "Model: $model_path"
echo "Anchor configuration:"
echo "  cache_tail=$anchor_cache_tail"
echo "  cache_dense=$anchor_cache_dense"
echo "  scale_factor=$anchor_scale_factor"
echo "=============================================="

# Function to run experiment with all parameters
run_experiment() {
    local prefix=$1
    local cache_tail=${2:-$anchor_cache_tail}
    local cache_dense=${3:-$anchor_cache_dense}
    local scale_factor=${4:-$anchor_scale_factor}

    python exam/exam_rouge.py \
        --merge \
        --method MKV \
        --model_name $model_path \
        --device $device \
        --shots $shots \
        --cache_budget $budget \
        --sample_num 200 \
        --prefix "$prefix"
}

# 1. Explore cache_tail values
echo "Exploring cache_tail parameter..."
for cache_tail in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "Running with cache_tail=$cache_tail"
    run_experiment "hp_cache_tail${cache_tail}" $cache_tail
done

# 2. Explore cache_dense values
echo "Exploring cache_dense parameter..."
for cache_dense in 0.01 0.02 0.03 0.04 0.05; do
    echo "Running with cache_dense=$cache_dense"
    run_experiment "hp_cache_dense${cache_dense}" $anchor_cache_tail $cache_dense
done

# 3. Explore scale_factor values
echo "Exploring scale_factor parameter..."
for scale_factor in 0.0 0.2 0.4 0.6 0.8 1.0; do
    echo "Running with scale_factor=$scale_factor"
    run_experiment "hp_scale_factor${scale_factor}" $anchor_cache_tail $anchor_cache_dense $scale_factor
done


echo "=============================================="
echo "Hyperparameter sensitivity validation completed!"