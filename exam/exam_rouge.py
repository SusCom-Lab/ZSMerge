import os
import sys
sys.path.append(os.getcwd())

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
import json

from rouge import Rouge
import pandas as pd
import torch
import tqdm
import numpy as np
import argparse
import logging

from mergekv import AttentionForward as AF

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    

def model_load(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return config, tokenizer, model


def data_infer(model, tokenizer, requests, max_position_embeddings, max_tokens, out_file):
    results = []
    rouge = Rouge()

    seq_lens = []
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    skipped=0
    max_length = max_position_embeddings

    if Path(out_file).is_file():
        logger.info(f"skip! existed out_file: <{out_file}>")
        return
    else:
        logger.info(f"out_file: <{out_file}>")
    with torch.no_grad():
        for i, request in enumerate(tqdm.tqdm(requests)):

            stop = ['###']
            prompt = request['article']
            label = request['summary_gt']
            result = {}

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            context_length = len(input_ids[0])
            if context_length > max_length-max_tokens:
                skipped+=1
                logger.info(f'skipped:{skipped}')

            else:
                output_sequences = model.generate(
                    input_ids=input_ids,
                    output_attentions = False,
                    max_new_tokens=max_tokens,
                    num_beams=1,
                    do_sample=False,
                    top_p=None,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id]
                )

                generate_text = tokenizer.decode(output_sequences.squeeze(0)[context_length:])
                generate_text = generate_text[: generate_text.find(stop[0])]

                scores = rouge.get_scores(generate_text, label)[0]
                seq_lens.append(len(input_ids[0]))
                rouge1_score_list.append(scores['rouge-1']['f'])
                rouge2_score_list.append(scores['rouge-2']['f'])
                rougel_score_list.append(scores['rouge-l']['f'])

                result['result'] = {
                    "choices": [
                        {
                            "text": generate_text
                        }
                    ], 
                    "request_time": {
                        "batch_time": 0, 
                        "batch_size": 1},
                    "score":{
                        "rouge1": scores['rouge-1']['f'],
                        "rouge2": scores['rouge-2']['f'],
                        "rougel": scores['rouge-l']['f'],
                    }
                }
                
                results.append(result)
                logger.info('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
                with open(out_file, "a", encoding="utf-8") as f:
                    json.dump(result["result"], f, ensure_ascii=False)
                    f.write('\n')
    print("FINAL RESULTS")
    print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--method", type=str, default='merge')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dataset", type=str, default='xsum')
    parser.add_argument("--cache_budget", type=float, default=0.1)
    parser.add_argument("--cache_tail", type=float, default=0.4)
    parser.add_argument("--cache_dense", type=float, default=1)
    parser.add_argument("--scale_factor", type=float, default=0.6)
    parser.add_argument("--shrink_factor", type=float, default=0.98)
    parser.add_argument("--merge", action='store_true')
    parser.add_argument("--metric", type=str, default='dot_product')
    parser.add_argument("--score_update", type=str, default='max')
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # args
    args = parser.parse_args()
    logger.info(args)
    set_seed(args)
    model_name = args.model_name
    method = args.method
    merge = args.merge
    device = args.device
    cache_budget = args.cache_budget
    cache_dense = args.cache_dense
    cache_tail = args.cache_tail
    scale_factor = args.scale_factor
    shrink_factor = args.shrink_factor
    metric = args.metric
    score_update = args.score_update
    shots = args.shots
    max_tokens = 128
    dataset = args.dataset
    if dataset == 'cnn':
        dataset_path_ = f'./data/cnn_data/cnn_dailymail_{shots}shot.jsonl'
    elif dataset == 'xsum':
        dataset_path_ = f'./dataset/xsum_data/xsum_{shots}shot.jsonl'
    dataset_path = dataset_path_
    args_str = f"{method}{shots}_{merge}_{cache_budget}_{cache_tail}_{cache_dense}_{metric}_{score_update}_{scale_factor}"
    
    # if 'falcon' in model_name:
    #     logger.info("skip")
    #     return
    

    # model
    # config, tokenizer, model = model_load(model_name)
    tokenizer, model = AF.model_load(model_name=model_name, merge=False)
    config = model.config
    max_position_embeddings = config.max_position_embeddings
    cache_budget_ = int(cache_budget * max_position_embeddings)
    print(f"config.max_position_embeddings: {config.max_position_embeddings} -> {cache_budget_}")
    model.eval().to(device)
    # method
    AF.change_mode(merge, cache_budget=cache_budget_, cache_tail=cache_tail, cache_dense=cache_dense,
                             metric=metric, score_update=score_update, scale_factor=scale_factor
                            )


    # dataset
    requests = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    # infer
    if dataset == 'cnn':
        out_file = rf"./results/cnn/cnn_dm_{shots}shot_{model_name.split('/')[-1]}_{args_str}.jsonl"
    elif dataset == 'xsum':
        out_file = rf"./results/xsum/xsum_{shots}shot_{model_name.split('/')[-1]}_{args_str}.jsonl"
    if not (out_dir := Path(out_file).parent).exists():
        os.makedirs(out_dir)
    data_infer(model, tokenizer, requests, max_position_embeddings, max_tokens, out_file)

if __name__ == "__main__":
    main()
