'''
python exams/run_longbench_ds.py     \
    --max_num_examples 200    \
    --model_path mistralai/Mistral-7B-Instruct-v0.3     \
    --save_dir ./results/longbench_ds \
    --merge
'''
import os
import sys
sys.path.append(os.getcwd())

import logging
import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from mergekv import AttentionForward as AF


# 数据集和模型配置
datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}  # 数据集最大长度映射

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}  # 数据集到提示模板的映射

model2maxlen = {
    "Llama-2-7b-hf": 3500,
    "Mistral-7B-Instruct-v0.3": 31500,
    "Llama-3-8B-Instruct": 7950,
    "Meta-Llama-3-8B": 7950,
    "llama2-7b-chat-4k": 3500,
    "longchat-v1.5-7b-32k": 31500,
    "xgen-7b-8k": 7500,
    "internlm-7b-8k": 7500,
    "chatglm2-6b": 31500,
    "chatglm2-6b-32k": 31500,
    "chatglm3-6b-32k": 31500,
    "vicuna-v1.5-7b-16k": 15500
}

# 随机种子设置
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# 构建聊天格式提示
def build_chat(prompt):
    return f"[INST] {prompt} [/INST]"

def load_data(data_file, dataset, model_path, max_num_examples=None, sample_method="topk"):
    test_data = []
    with open(data_file) as fp:
        for line in fp:
            example = json.loads(line)
            template = model2prompt[dataset]
            prompt = template.format(**example)
            if "llama2" in model_path.lower():
                prompt = build_chat(prompt)
            example["prompt"] = prompt
            test_data.append(example)

    if max_num_examples and len(test_data) > max_num_examples:
        if sample_method == "random":
            test_data = random.sample(test_data, max_num_examples)
        elif sample_method == "topk":
            test_data = test_data[:max_num_examples]
    return test_data

# 模型初始化模块（添加方法配置）
def initialize_model(model_path, merge, cache_config=None, use_cache=True):
    """
    model_path: 模型路径
    merge: 处理方法
    cache_config: 包含缓存参数的字典（新增参数封装）
    use_cache: 是否使用缓存
    """
    config = AutoConfig.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=use_cache,
    ).to(device).eval()

    # 初始化模型...
    if merge:  # 集中处理缓存相关参数
        AF.change_mode(
            merge=cache_config.get('merge', False),
            cache_budget=cache_config['cache_budget'],
            cache_tail=cache_config['cache_tail'],
            cache_dense=cache_config['cache_dense'],
            metric=cache_config.get('metric', 'dot_product'),
            scale_factor=cache_config['scale_factor']
        )
        print(cache_config)
    return tokenizer, model

# 推理与生成模块
def generate_predictions(model, tokenizer, test_data, output_max_len, method, quant_method=None, **kwargs):
    # 创建结果文件路径
    if cache_config_str:= kwargs.get("cache_config_str"):
        method += cache_config_str
    result_file = os.path.join(kwargs["save_dir"],  f"{kwargs['model_name']}-{kwargs['max_capacity_prompts']}", f"{kwargs['dataset']}",  f"{method}.jsonl")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    # 加载已存在的ID
    existing_ids = set()
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data["_id"])
                except json.JSONDecodeError:
                    continue
    print(f"out_file ({len(existing_ids)}):<{result_file}>")
    
    # 过滤未处理的数据
    filtered_data = [ex for ex in test_data if ex["_id"] not in existing_ids]
    
    # 分批处理
    for i in tqdm(range(0, len(filtered_data), kwargs["eval_batch_size"]), desc=f"Processing {kwargs['dataset']}"):
        batch = filtered_data[i:i+kwargs["eval_batch_size"]]
        
        # 原有生成逻辑
        batch_prompts = [ex["prompt"] for ex in batch]
        tokenized = tokenizer(batch_prompts, padding="longest",  return_tensors="pt", add_special_tokens=True).to(device)
        
        # 处理超长输入
        exceed = False
        if tokenized.input_ids.shape[1] > kwargs["model_max_len"]:
            # continue # skip !!
            half = kwargs["model_max_len"] // 2
            tokenized.input_ids = torch.cat([
                tokenized.input_ids[:, :half],
                tokenized.input_ids[:, -half:]
            ], dim=1)
            exceed = True
        
        # 生成预测
        outputs = model.generate(
            **tokenized,
            max_new_tokens=output_max_len,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            eos_token_id=[tokenizer.eos_token_id],
            cache_implementation="quantized" if quant_method else None,
        )
        
        # 解码结果
        decoded = [tokenizer.decode(ids[len(tokenized.input_ids[0]):], skip_special_tokens=True) 
                  for ids in outputs]
        
        # 立即保存结果
        with open(result_file, 'a') as f:
            for ex, pred in zip(batch, decoded):
                record = {
                    "_id": ex["_id"],
                    # "prompt": ex["prompt"],
                    # "input": ex["input"],
                    # "context": ex["context"],
                    "answers": ex["answers"],
                    "all_classes": ex["all_classes"],
                    "language": ex["language"],
                    "pred": pred,
                    "exceed": exceed,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# 主流程模块
def main(args):
    
    set_seed(args.seed)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        
        # 数据加载
        data_file = f"dataset/LongBench/{dataset}.jsonl"
        test_data = load_data(data_file, dataset, args.model_path,  args.max_num_examples, args.sample_method)
        
        # 模型初始化
        # 初始化模型时仅传递必要参数
        cache_config={  # 封装缓存相关参数
            'merge': args.merge,
            'cache_budget': args.max_capacity_prompts,
            'cache_tail': args.cache_tail,
            'cache_dense': args.cache_dense,
            'scale_factor': args.scale_factor,
            'metric': 'dot_product'  # 固定值或可配置参数
        }
        tokenizer, model = initialize_model(
            model_path=args.model_path,
            merge=args.merge,
            cache_config=cache_config,
            use_cache=True
        )
        # 生成预测
        cache_config_str = "-".join(f"{k}-{v}" for k, v in cache_config.items())
        generate_predictions(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            output_max_len=dataset2maxlen[dataset],
            method=args.method,
            quant_method=None,
            # 新增关键参数
            save_dir=args.save_dir,
            model_name=model_name,
            dataset=dataset,
            # 原有参数
            eval_batch_size=1,
            model_max_len=model2maxlen[model_name], # model2maxlen.get(model_name, 2048),
            max_capacity_prompts=args.max_capacity_prompts,
            cache_config_str=cache_config_str
        )
        
        # 显存清理
        del model
        torch.cuda.empty_cache()

# 参数解析模块
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LongBench模型评估脚本",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 基础配置组
    base_group = parser.add_argument_group("基础配置")
    base_group.add_argument("--model_path", type=str, required=True,  help="模型路径（必填）")
    base_group.add_argument("--save_dir", type=str, required=True, help="结果保存目录（必填）")
    base_group.add_argument("--device", type=str, default="cuda", help="运行设备")
    base_group.add_argument("--seed", type=int, default=42, help="随机种子")

    # 缓存配置组（新增核心参数分组）
    cache_group = parser.add_argument_group("KV缓存配置")
    cache_group.add_argument("--method", type=str, required=False, )  # 明确可选方法 help="KV缓存处理方法")
    cache_group.add_argument("--max_capacity_prompts", type=int, default=512, help="最大提示词缓存容量")
    cache_group.add_argument("--cache_tail", type=float, default=0.2, help="尾部缓存保留比例")
    cache_group.add_argument("--cache_dense", type=float, default=2.0, help="密集缓存")
    cache_group.add_argument("--scale_factor", type=float, default=0.8, help="缓存缩放因子")
    cache_group.add_argument("--merge", action="store_true", help="是否启用缓存合并")

    # 生成配置组（原有参数归类）
    gen_group = parser.add_argument_group("生成配置")
    gen_group.add_argument("--eval_batch_size", type=int, default=1, help="评估批次大小")
    gen_group.add_argument("--max_num_examples", type=int, default=None, help="最大测试样本数")
    gen_group.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="样本采样方法")
    
    args = parser.parse_args()
    device = args.device
    # choices=["FullKV", "CacheMerge"],
    args.method = "CacheMerge" if args.merge else "FullKV"
    if args.method == "FullKV":
        args.max_capacity_prompts = 0  # 保持原有逻辑
    print(args)
    main(args)
