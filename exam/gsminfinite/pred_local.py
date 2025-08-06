# from model_handler import ModelHandler
# from no_rag_pipeline import NoRAGPipeline
from tqdm import tqdm
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, fields

import sys
sys.path.append(r"/home/ldaphome/seanl/code/mergeKV/")
from mergekv import AttentionForward as AF
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def dump_dict_to_json(data, filename):
    import os
    import json
    """Dumps a Python dictionary to a JSON file, creating the directory if needed.

    Args:
        data: The Python dictionary to be dumped.
        filename: The name of the JSON file to be created (e.g., "data/output.json").
    """
    try:
        # Extract the directory path from the filename
        directory = os.path.dirname(filename)

        # Create the directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            print(f"Successfully dumped dictionary to {filename}")
    except (TypeError, OSError) as e:
          print(f"Error dumping dictionary to JSON: {e}")

@dataclass
class LocalArgs:
    model_path: str = "Qwen/Qwen2-7B-Instruct"
    method: str = "FullKV"
    device: str = "cuda:0"
    cache_size: int = 2048        # KVCache 最大容量
    cache_dense: float = 0.05     # dense cache 比例
    cache_tail: float = 0.1       # 尾部保留比例
    scale_factor: float = 0.9     # 缩放因子
    merge: bool = True            # 是否启用合并
    floor: float = 0.05           # 剪枝阈值
    window_size: int = 16         # 滑动窗口大小
    shrink_factor: float = 0.8    # 缩减比例
    out_state: bool = False       # 是否输出 KV 状态


def get_local_args_from_parser(args):
    """从 argparse 的 args 中提取 LocalArgs 所需字段"""
    local_arg_fields = {f.name for f in fields(LocalArgs)}
    filtered_args = {k: v for k, v in vars(args).items() if k in local_arg_fields}
    return LocalArgs(**filtered_args)

def get_mergekv_model(args):
    print(">>> Loading model using mergekv.AttentionForward...")
    tokenizer, model = AF.model_load(model_name=args.model_path, merge=False)
    config = model.config

    print(f"Max position embeddings: {config.max_position_embeddings}")
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

    model.eval().to(args.device)
    return tokenizer, model


def get_unique_id(sample):
    
    op = sample.get("op", "OP_N/A")
    id_field = sample.get("id", "ID_N/A")
    unique_id = f"{op}-{id_field}"
    return unique_id

def process_and_save_jsonl(unprocessed_dataset, tokenizer, model, args):
    save_dir = f"datasets/{args.save_dataset}-{args.save_name}_{args.length}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.method}-{args.cache_size}.jsonl")

    # 如果已有文件，加载已完成的 id
    existing_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if "id" in record:
                        existing_ids.add(get_unique_id(record))
                        
                except json.JSONDecodeError:
                    continue
        print(f"Resuming from {len(existing_ids)} existing records")

    # 以追加模式写入
    print(f"{save_path = }")
    with open(save_path, 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(unprocessed_dataset)), desc="Processing queries"):
            # record_id = unprocessed_dataset[i].get("id", i)

            # # 跳过已有 id
            # if record_id in existing_ids:
            #     continue

            # ========================= 核心修改点 =========================
            # 1. 正确地构造输入
            # `apply_chat_template` 会处理好多轮对话的格式，包括 system, user, assistant 角色
            record = unprocessed_dataset[i]
            unique_id = get_unique_id(record)
            try:
                messages = record['messages']
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # 确保在最后添加生成提示 (如 <|im_start|>assistant)
                )
            except Exception as e:
                print(f"Error applying chat template for record {unique_id}: {e}")
                raise e
                continue
            
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)


            # with torch.no_grad():
            #     outputs = model.generate(
            #         **inputs,
            #         max_new_tokens=args.max_tokens,
            #         temperature=args.temperature,
            #         do_sample=False
            #     )
            # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # answer = generated_text.strip()

            
            print(f"{unique_id = }  {inputs.input_ids.shape[1] = }")
            # 跳过已有 id
            if unique_id in existing_ids:
                print(f"{unique_id = }  skip！！！！")
                continue
            # 2. 模型生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    do_sample=True if args.temperature > 0 else False,
                    pad_token_id=tokenizer.eos_token_id # 避免 pad_token_id not set 的警告
                )
            
            # 3. 正确地解码输出 (只解码新生成的部分)
            input_token_len = inputs.input_ids.shape[1]
            generated_ids = outputs[0, input_token_len:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 构建记录
            newline = dict(unprocessed_dataset[i])
            newline["replies"] = [answer.strip()] # 保存为列表以保持格式一致
            newline.pop("problem", "")
            newline.pop("question", "")
            newline.pop("messages", "")

            # 立即写入 JSONL
            f.write(json.dumps(newline, ensure_ascii=False) + "\n")
            f.flush()  # 确保实时写入磁盘
            
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample with command line arguments."
    )
    # 保存相关
    parser.add_argument('--save-name', type=str, help="Save model name", default="base")
    parser.add_argument('--save-dataset', type=str, help="Save dataset name", default="base")
    parser.add_argument('--dataset-name', type=str, help="The name of the dataset for organizing the folders")

    # 必填
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name or path of the model (e.g., Qwen/Qwen2-7B-Instruct)'
    )

    # 基础配置
    parser.add_argument(
        '--backend-type',
        type=str,
        default="openai",
        help="Backend type in ['openai', 'anthropic', 'gemini']"
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate per example.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7).'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=3072,
        help='Maximum number of tokens for generation (default: 3072).'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        help='Batch size (default: 200).'
    )
    parser.add_argument(
        '--length',
        type=str,
        default="0",
        help='Noise context length (e.g., 0, 8000, 16000)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help="Maximum number of examples per op"
    )
    parser.add_argument(
        '--filter-config',
        type=json.loads,
        help='Filter configuration as a JSON string.'
    )
    parser.add_argument(
        '--op-range',
        type=str,
        help='Operating range, can be an integer, or a comma-separated list of integers.'
    )
    # 设备与方法配置
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="Device to run the model (e.g., cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        '--method',
        type=str,
        default="FullKV",
        choices=["FullKV", "ZSMerge"],
        help="KV cache method to use"
    )

    # KVCache相关
    parser.add_argument(
        '--cache-size',
        type=int,
        default=2048,
        help="KVCache maximum prompt capacity per layer"
    )
    parser.add_argument(
        '--cache-dense',
        type=float,
        default=0.05,
        help="Dense cache ratio (default: 0.05)"
    )
    parser.add_argument(
        '--cache-tail',
        type=float,
        default=0.1,
        help="Tail cache ratio (default: 0.1)"
    )
    parser.add_argument(
        '--scale-factor',
        type=float,
        default=0.9,
        help="Scale factor for KVCache eviction"
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        default=True,
        help="Whether to enable KV merging"
    )
    parser.add_argument(
        '--floor',
        type=float,
        default=0.05,
        help="Pruning threshold for KVCache"
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=16,
        help="Sliding window size for KV retention"
    )
    parser.add_argument(
        '--shrink-factor',
        type=float,
        default=0.8,
        help="Factor for shrinking KVCache size when pruning"
    )
    parser.add_argument(
        '--out-state',
        action='store_true',
        default=False,
        help="Whether to output KV states for debugging"
    )

    args = parser.parse_args()
    return args

# print(get_payload(100, 2))
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
    import json

    # parser = argparse.ArgumentParser(description="Run benchmark tests and organize results")
    # parser.add_argument('--model-name', type=str, help="The name of the model for organizing the folders")
    import argparse
    args = parse_args()
    args.cache_size = (1024 * 9) if args.save_dataset == "symbolic" else args.cache_size
    args.merge = args.method != "FullKV"

    print(f"{args=}")

    if args.op_range:
        try:
            # Attempt to parse as a single integer
            args.op_range = [int(args.op_range)]
        except ValueError:
            # If not a single integer, split by comma and convert to integers
            try:
                args.op_range = [int(x.strip()) for x in args.op_range.split(',')]
            except ValueError:
                raise ValueError("Invalid input for --op-range. Please provide an integer or a comma-separated list of integers.")


    subsets = [f"ops_{x}" for x in args.op_range]
    use_full_query = True

    # 自动构建 LocalArgs
    local_args = get_local_args_from_parser(args)
    tokenizer, model = get_mergekv_model(
        args=local_args
    )
    use_full_query = True

    
    # for length in [0, 8000, 16000, 32000, 64000, 128000]:
    length = args.length
    try:
        # dataset = load_from_disk("/home/ldaphome/seanl/.cache/huggingface/datasets/InfiniAILab___gsm_infinite_symbolic_8k")
        # opset = set(args.op_range)
        # unprocessed_dataset = unprocessed_dataset.filter(lambda example: example["op"] in opset)
        print("dataset", f"{args.dataset_name}_{length}")
        full_dataset = load_dataset(
            f"{args.dataset_name}_{length}",
            download_mode="reuse_dataset_if_exists"  # 优先复用本地缓存
            )
        filter_config = args.filter_config
        if filter_config:
            filtered_datasets = []
            for split in subsets:
                dataset_split = full_dataset[split]
                total_samples = min(args.limit, len(dataset_split))
                filtered_data = []
                for config in filter_config:
                    num_to_add = int(total_samples * config["percentage"])
                    current_filter = {key: value for key, value in config.items() if key not in ["percentage"]}
                    filtered_subset = dataset_split.filter(lambda example: all(example[key] == value for key, value in current_filter.items()))
                    filtered_data.extend(filtered_subset.select(range(min(num_to_add, len(filtered_subset)))))
                filtered_datasets.append(Dataset.from_list(filtered_data))
            unprocessed_dataset = concatenate_datasets(filtered_datasets)
        else:
            unprocessed_dataset = concatenate_datasets([full_dataset[split].select(range(min(args.limit, len(full_dataset[split])))) for split in subsets])
        # ========================================================================
        process_and_save_jsonl(unprocessed_dataset, tokenizer, model, args)
    except Exception as e:
        print(e)
        raise
