#!/bin/bash
echo hostname
hostname

python exam/exam_wikitext_generate_mergekv.py \
                --device cuda:1 \
                --cache_sizes 40,160 \
                --max_samples 10