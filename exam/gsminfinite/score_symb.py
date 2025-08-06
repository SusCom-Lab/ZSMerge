import json
import re
import argparse
from typing import List, Dict, Set, Any
# (之前的导入保持不变)
from pathlib import Path # 确保导入 pathlib

def evaluate_sample_with_regex(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用正则表达式评估单个样本。
    使用 op 和 id 的组合作为唯一标识符。
    """
    # 2. case中的id字段不足以表示唯一的样本，使用 op\id联合值作为样本索引
    op = sample.get("op", "OP_N/A")
    id_field = sample.get("id", "ID_N/A")
    unique_id = f"{op}-{id_field}"

    replies: List[str] = sample["replies"]
    answer_list: List[str] = sample["answer_list"]

    # 1. replies中多个元素直接拼接，作为一个答案
    full_reply_text = " ".join(replies)

    # 将正确答案列表转换为集合，以便进行集合运算
    # 同时处理可能存在的 "None" 字符串
    ground_truth_set: Set[str] = set(answer_list)
    if "None" in ground_truth_set:
        ground_truth_set.remove("None")
        
    found_vars_set: Set[str]

    # 特殊情况：如果正确答案是空的 (即不应该找到任何变量)
    if not ground_truth_set:
        # 查找文本中所有 'V' + 数字格式的变量
        all_found_vars = re.findall(r'\bV\d+\b', full_reply_text)
        found_vars_set = set(all_found_vars)
        # 如果没有找到任何变量，得分为1.0 (正确)；否则为0.0 (错误)
        score = 1.0 if not found_vars_set else 0.0
    else:
        # 2. 从answer_list出发构建正则规则
        # 按长度降序排序，防止'V1'优先于'V10'被匹配
        sorted_vars = sorted(list(ground_truth_set), key=len, reverse=True)
        # 构建形如 \b(V10|V3|V1)\b 的正则表达式
        pattern = r'\b(' + '|'.join(re.escape(var) for var in sorted_vars) + r')\b'

        # 去replies中匹配满足规则的字段
        matches = re.findall(pattern, full_reply_text)
        found_vars_set = set(matches)
        
        # 3. 按照集合中匹配到的比例计算单个样本的得分 (召回率)
        # 计算找到的正确变量数量
        correctly_found_count = len(ground_truth_set.intersection(found_vars_set))
        
        # 计算得分
        score = correctly_found_count / len(ground_truth_set)

    return {
        "id": unique_id,
        "score": score,
        # "score": 1 if score >= 1 else 0,
        "expected": sorted(list(ground_truth_set)) if ground_truth_set else ["None"],
        "found": sorted(list(found_vars_set))
    }

    
def process_file(file_path: Path):
    """
    读取并评估单个 .jsonl 文件。
    """
    print(f"\n{'='*25} Processing File: {file_path.name} {'='*25}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 逐行读取文件，并将每行解析为一个JSON对象
            dataset = [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse JSON in {file_path.name}: {e}")
        return
    except Exception as e:
        print(f"  [ERROR] Could not read file {file_path.name}: {e}")
        return

    if not dataset:
        print("  File is empty or contains no valid data.")
        return

    all_results = []
    total_score = 0.0

    for sample in dataset:
        if not all(k in sample for k in ["op", "id", "replies", "answer_list"]):
            print(f"  [WARNING] Skipping a malformed sample in {file_path.name}: {str(sample)[:100]}...")
            continue
        
        result = evaluate_sample_with_regex(sample)
        all_results.append(result)
        total_score += result['score']

    # 1. 结果每个case的得分格式化的返回
    print("--- Individual Sample Scores ---")
    # 动态调整ID列的宽度以适应更长的复合ID
    max_id_len = max(len(res['id']) for res in all_results) if all_results else 20

    for res in all_results:
        print(
            f"  ID: {res['id']:<{max_id_len}} | "
            f"Score: {res['score']:.2f} | "
            f"Expected: {res['expected']} | "
            f"Found: {res['found']}"
        )

    if not all_results:
        print("\n  No valid samples found to evaluate.")
        return

    average_score = total_score / len(all_results)
    print("\n--- File Summary ---")
    print(f"  Total Samples Evaluated: {len(all_results)}")
    print(f"  Average Score for this file: {average_score:.4f}")
    print(f"{'='* (62 + len(file_path.name))}")
    return all_results

def score_fold(input_dir):
    """
    主函数：查找目录中的所有 .jsonl 文件并逐一处理。
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        print(f"Error: Directory not found at '{input_dir}'")
        return

    # 使用 rglob 递归查找所有 .jsonl 文件
    jsonl_files = sorted(list(input_dir.rglob('*.jsonl')))

    if not jsonl_files:
        print(f"No '.jsonl' files found in directory '{input_dir}' or its subdirectories.")
        return

    print(f"Found {len(jsonl_files)} '.jsonl' file(s) to process.\n")
    fold_results = {}
    for file_path in jsonl_files:
        file_results = process_file(file_path)
        fold_results[file_path] = file_results
        
    return fold_results

if __name__ == '__main__':
    # (此部分的其他导入保持不变)
    
    parser = argparse.ArgumentParser(
        description="使用正则表达式评估目录中所有.jsonl文件的模型结果。"
    )
    # 命令行参数从 --input-file 更改为 --input-dir
    parser.add_argument(
        '--input-dir', 
        type=str, 
        # default="/home/ldaphome/seanl/code/mergeKV/gsm_infinite/gsm-infinite/datasets/symbolic-qwen2‑7B‑instruct-2025‑08‑04_8k",
        # default="/home/ldaphome/seanl/code/KVCache-Factory/gsm_infinite/gsm-infinite/datasets/symbolic-qwen2‑7B‑instruct-2025‑08‑04_8k",
        # default="/home/ldaphome/seanl/code/mergeKV/gsm_infinite/gsm-infinite/datasets/symbolic-qwen-2.5-7b-instruct_8k",
        default="/home/ldaphome/seanl/code/KVCache-Factory/gsm_infinite/gsm-infinite/datasets/symbolic-qwen-2.5-7b-instruct_8k",
        help="包含评估数据的.jsonl文件的目录路径。"
    )

    args = parser.parse_args()
    fold_results = score_fold(args.input_dir)