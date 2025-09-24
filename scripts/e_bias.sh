#!/bin/bash

# meta-llama/Llama-2-13b-hf
model_path='meta-llama/Llama-2-7b-hf'
device='cuda:1'
shots=3
budget=0.05
cache_tail=0.4
scale_factor=0.6


python exam/exam_wikitext_generate_mergekv.py --cache_size 20 --max_samples 10 --scale_factor 1
    