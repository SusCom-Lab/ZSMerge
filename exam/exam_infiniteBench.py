import sys
sys.path.append(r"/home/ldaphome/seanl/code/mergeKV")

import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Features, Value, Sequence
from mergekv import AttentionForward as AF

output_lengths = {
    "passkey": 2.0,
    "kv_retrieval": 22.7,
    "number_string": 4.0,
    "code_run": 12,
    # "code_debug": 12, # temp skip
    "math_find": 1.3,
    "math_calc": 43900.0,
    "longdialogue_qa_eng": 3.4,
    "longbook_qa_eng": 4800.0,
    "longbook_sum_eng": 1100.0,
    "longbook_choice_eng": 5.3,
    "longbook_qa_chn": 6.3,
}

def clean_model_name(model_name: str) -> str:
    """清理模型名中的特殊字符，适合用作文件夹名"""
    return re.sub(r"[^\w\-\.]", "_", model_name)


def get_mergekv_model(args):
    print(">>> Loading model using mergekv.AttentionForward...")
    tokenizer, model = AF.model_load(model_name=args.model, merge=False)
    config = model.config

    print(f"Max position embeddings: {config.max_position_embeddings}")
    print(f"Max position length: {args.max_length}")
    print(f"Cache budget set to: {args.cache_size} tokens")

    AF.change_mode(
        merge=args.merge,
        cache_budget=args.cache_size,
        cache_tail=args.cache_tail,
        window_size=args.window_size,
        cache_dense=args.cache_dense,
        # metric=args.metric,
        scale_factor=args.scale_factor,
        shrink_factor=args.shrink_factor,
        out_state=args.out_state,
    )

    model.eval().half().to(args.device) # , device_map="auto" in `model_load`
    return tokenizer, model


def evaluate_model(model, tokenizer, dataset, device, exist_ids,
                   max_length=131072, max_new_tokens=128, n_sample=0, save_file=None):
    """在数据集上逐条运行模型，收集生成结果，并逐行追加写入JSONL"""
    with open(save_file, "a", encoding="utf-8") as f:
        cnt = 0
        for item in tqdm(dataset, desc=f"Evaluating {save_file}"):
            cnt += 1
            if n_sample and n_sample < cnt:
                print(f"{n_sample = } acquired!")
                break
            
            if item["id"] in exist_ids:
                continue  # 跳过已完成
            context = item["context"]
            question = item["input"]
            prompt = context + "\n\nQuestion: " + question + "\nAnswer:"

            # inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
            #                    max_length=max_length).to(device)
            inputs_all = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs_all["input_ids"]
            if input_ids.size(1) > max_length:
                input_ids = torch.concat([input_ids[:, :max_length//2], input_ids[:, -max_length//2:]], dim=1)
                
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

            output_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            result = {
                "id": item["id"],
                "question": question,
                "prediction": output_text.strip(),
                "reference": item["answer"],
                "options": item.get("options", []),
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()  # 避免中途断掉丢数据


def load_existing_ids(save_file):
    """读取已有结果文件中的 ID，用于断点续跑"""
    exist_ids = set()
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        exist_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
    return exist_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark mergeKV on InfiniteBench")

    # 模型与硬件
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--method", type=str, default="mergekv",
                        help="Method name for saving results")

    # mergeKV 参数
    parser.add_argument("--merge", action="store_true", help="Enable merge mode")
    parser.add_argument("--cache_size", type=float, default=0.6)
    parser.add_argument("--cache_tail", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--shrink_factor", type=float, default=0.98)
    parser.add_argument("--out_state", type=int, default=0)
    parser.add_argument("--cache_dense", type=int, default=0.05)
    # parser.add_argument("--metric", type=str, default="l2")
    parser.add_argument("--scale_factor", type=float, default=0.9)

    # 数据集与运行
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma separated list of tasks")
    parser.add_argument("--save_dir", type=str, default="results/infiniteBench")
    parser.add_argument("--max_length", type=int, default=100000) # 131072
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--n_sample", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"{args=}")

    # 清理后的模型名
    clean_name = clean_model_name(args.model)

    # 加载模型
    tokenizer, model = get_mergekv_model(args)

    # 数据 schema
    ft = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string"))
    })

    # 多任务处理
    tasks=('longbook_sum_eng', 'longbook_qa_eng', 'longbook_choice_eng', 'longdialogue_qa_eng', 'longbook_qa_chn', 'code_debug', 'math_find', 'passkey', 'number_string', 'kv_retrieval')
    
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")] 
    else:
        tasks = output_lengths.keys()
    
    datasets = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)
    
    for task in tasks:
        max_new_tokens = int(output_lengths[task] * 1.2) + 2
        max_new_tokens = min(max_new_tokens, args.max_new_tokens)
        show_str = f"{task = }, {max_new_tokens = }"
        print("#" * (len(show_str) + 2))
        print(f"#{show_str}#")
        print("#" * (len(show_str) + 2))
        # 构造多级目录：save_dir / model / method / task-cache_size.jsonl
        task_dir = os.path.join(args.save_dir, clean_name, f"{args.method}-{args.cache_size}")
        os.makedirs(task_dir, exist_ok=True)
        save_file = os.path.join(task_dir, f"{task}-{args.cache_size}.jsonl")

        # 读取已有ID（断点续跑）
        exist_ids = load_existing_ids(save_file)
        print(f">>> Task {task}: skipping {len(exist_ids)} already processed items")

        # 加载数据集
        dataset = datasets[task]

        # 评估并写结果
        evaluate_model(model, tokenizer, dataset, args.device,
                       exist_ids, max_length=args.max_length,
                       max_new_tokens=max_new_tokens,
                       n_sample=args.n_sample,
                       save_file=save_file)

        print(f">>> Finished Task {task}! Results saved to {save_file}")


if __name__ == "__main__":
    main()
