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
    "code_debug": 12,
    "math_find": 1.3,
    "math_calc": 43900.0,
    "longdialogue_qa_eng": 3.4,
    "longbook_qa_eng": 4800.0,
    "longbook_sum_eng": 1100.0,
    "longbook_choice_eng": 5.3,
    "longbook_qa_chn": 6.3,
}

def clean_model_name(model_name: str) -> str:
    """Clean special characters in model name to make it suitable for folder names"""
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
        scale_factor=args.scale_factor,
        shrink_factor=args.shrink_factor,
        out_state=args.out_state,
    )

    model.eval().half().to(args.device)
    return tokenizer, model


def evaluate_model(model, tokenizer, dataset, device, exist_ids,
                   max_length=131072, max_new_tokens=128, n_sample=0, save_file=None):
    """Run model on dataset item by item, collect generation results, and append to JSONL file line by line"""
    with open(save_file, "a", encoding="utf-8") as f:
        cnt = 0
        for item in tqdm(dataset, desc=f"Evaluating {save_file}"):
            cnt += 1
            if n_sample and n_sample < cnt:
                print(f"{n_sample = } acquired!")
                break
            
            if item["id"] in exist_ids:
                continue  # Skip already completed items
            context = item["context"]
            question = item["input"]
            prompt = context + "\n\nQuestion: " + question + "\nAnswer:"

            inputs_all = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs_all["input_ids"]
            if input_ids.size(1) > max_length:
                input_ids = torch.cat([input_ids[:, :max_length//2], input_ids[:, -max_length//2:]], dim=1)
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
            f.flush()  # Flush to avoid data loss if interrupted


def load_existing_ids(save_file):
    """Load existing IDs from result file for resuming evaluation"""
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

    # Model and hardware settings
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--method", type=str, default="mergekv",
                        help="Method name for saving results")

    # MergeKV parameters
    parser.add_argument("--merge", action="store_true", help="Enable merge mode")
    parser.add_argument("--cache_size", type=float, default=0.5, help="Cache budget size")
    parser.add_argument("--cache_tail", type=float, default=0.02, help="Cache tail fraction")
    parser.add_argument("--window_size", type=int, default=8, help="Attention window size")
    parser.add_argument("--shrink_factor", type=float, default=0.98, help="Shrink factor for scores")
    parser.add_argument("--out_state", type=int, default=0, help="Output state mode")
    parser.add_argument("--cache_dense", type=float, default=0.5, help="Dense cache fraction")
    parser.add_argument("--scale_factor", type=float, default=1, help="Scale factor for attention")

    # Dataset and execution parameters
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma separated list of tasks")
    parser.add_argument("--save_dir", type=str, default="results/infiniteBench",
                        help="Directory to save results")
    parser.add_argument("--max_length", type=int, default=100000,
                        help="Maximum input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--n_sample", type=int, default=0,
                        help="Number of samples to process (0 for all)")

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"{args=}")

    # Clean model name for file paths
    clean_name = clean_model_name(args.model)

    # Load model
    tokenizer, model = get_mergekv_model(args)

    # Dataset schema
    ft = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string"))
    })

    # Task selection
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = output_lengths.keys()
    
    datasets = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)
    
    for task in tasks:
        max_new_tokens = int(output_lengths[task] * 1.2) + 2
        max_new_tokens = min(max_new_tokens, args.max_new_tokens)

        show_str = f"task = {task}, max_new_tokens = {max_new_tokens}"
        print("#" * (len(show_str) + 2))
        print(f"#{show_str}#")
        print("#" * (len(show_str) + 2))

        # Construct multi-level directory: save_dir / model / method / task-cache_size.jsonl
        task_dir = os.path.join(args.save_dir, clean_name, f"{args.method}-{args.cache_size}")
        os.makedirs(task_dir, exist_ok=True)
        save_file = os.path.join(task_dir, f"{task}-{args.cache_size}.jsonl")

        # Load existing IDs for resuming evaluation
        exist_ids = load_existing_ids(save_file)
        print(f">>> Task {task}: skipping {len(exist_ids)} already processed items")

        # Load dataset
        dataset = datasets[task]

        # Evaluate and write results
        evaluate_model(model, tokenizer, dataset, args.device,
                       exist_ids, max_length=args.max_length,
                       max_new_tokens=max_new_tokens,
                       n_sample=args.n_sample,
                       save_file=save_file)

        print(f">>> Finished Task {task}! Results saved to {save_file}")


if __name__ == "__main__":
    main()
