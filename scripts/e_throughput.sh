#!/bin/bash

  # meta-llama/Llama-2-13b-hf
  
  
python exam/exam_throughput.py \
  --model_name meta-llama/Llama-2-13b-hf \
  --batch_size 16 \
  --prompt_length 2048 \
  --generate_length 2048
  
python exam/exam_throughput.py \
  --model_name meta-llama/Llama-2-13b-hf \
  --batch_size 16 \
  --prompt_length 2048 \
  --generate_length 2048 \
  --cache_ratio 0.05 \
  --merge
