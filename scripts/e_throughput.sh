#!/bin/bash

python exam/exam_throughput.py \
  --model_name meta-llama/Llama-2-13b-hf \
  --batch_size 4 \
  --prompt_length 1024 \
  --generate_length 256 \
  --cache_ratio 0.05 \
  --merge