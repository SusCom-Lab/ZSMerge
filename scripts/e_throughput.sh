#!/bin/bash

  # meta-llama/Llama-2-13b-hf
  
  
python exam/exam_throughput.py \
  --model_name Qwen/Qwen1.5-7B \
  --batch_size 16 \
  --prompt_length 512 \
  --generate_length 512
  
python exam/exam_throughput.py \
  --model_name Qwen/Qwen1.5-7B \
  --batch_size 16 \
  --prompt_length 512 \
  --generate_length 512 \
  --cache_ratio 0.05 \
  --merge
