import re
import  os
import json
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple, Optional

# ============================================
# 1. 辅助与ID生成函数 (Helpers & ID Generation)
# ============================================

def get_unique_id(sample: Dict[str, Any]) -> str:
    """为每个样本生成一个唯一的ID。"""
    op = sample.get("op", "OP_N/A")
    id_field = sample.get("id", "ID_N/A")
    unique_id = f"{op}-{id_field}"
    return unique_id

def is_integer(s: Optional[str]) -> bool:
    """检查字符串是否可以转换为整数。"""
    if s is None:
        return False
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

# ============================================
# 2. 核心答案提取函数 (Core Answer Extraction)
# ============================================

def extract_ground_truth_answer(solution: str) -> Optional[int]:
    """从标准答案字符串中提取整数答案。"""
    try:
        # 查找 "Answer: " 后的数字
        match = re.search(r"Answer:\s*(-?\d+)", solution, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except (ValueError, AttributeError):
        pass
    return None

def extract_model_answer(generated_text: str) -> Optional[int]:
    """从模型生成的文本中提取整数答案，尝试多种模式。"""
    # 预处理：转小写，处理特殊字符（如果需要）
    text = generated_text.lower()
    text = re.sub(r'.\x08', '', text) # 替换退格符为空

    # 定义一系列正则表达式模式来捕获答案
    # 模式越严格，越靠前
    patterns = [
        r"answer is[:\s]*(-?\d+)",
        r"answer:\s*(-?\d+)",
        r"solution:\s*(-?\d+)",
        r"is\s+(-?\d+)\.",      # a is 123.
        r"oxed\{(-?\d+)\}",
        r"final answer is\s*(-?\d+)",
        r"result is\s*(-?\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # 找到一个匹配后，就用它并返回
            answer_str = match.group(1)
            if is_integer(answer_str):
                return int(answer_str)
    
    # 如果上面的模式都失败了，尝试找文本末尾的独立数字
    # 这是一种比较宽松的后备策略
    final_number_match = re.search(r"(-?\d+)(?!.*\d)", text)
    if final_number_match:
        answer_str = final_number_match.group(1)
        if is_integer(answer_str):
            return int(answer_str)

    return None

# ============================================
# 3. 评估与计分函数 (Evaluation & Scoring)
# ============================================

def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    评估单个案例。
    一个案例可以有多个模型回复（replies）。
    返回一个包含评估结果的字典。
    """
    unique_id = get_unique_id(case)
    ground_truth = extract_ground_truth_answer(case.get("solution", ""))
    replies = case.get("replies", [])
    
    if ground_truth is None:
        return {
            "id": unique_id,
            "score": 0.0,
            "is_correct": False,
            "reason": "Could not parse ground truth answer.",
            "ground_truth_answer": None,
            "model_answers": [],
            "correct_replies_count": 0,
            "total_replies": len(replies)
        }
        
    correct_replies_count = 0
    model_answers = []
    
    for reply in replies:
        model_answer = extract_model_answer(reply)
        model_answers.append(model_answer)
        if model_answer is not None and model_answer == ground_truth:
            correct_replies_count += 1
            
    # pass@k 逻辑：只要有一个回复正确，就算整个case通过
    is_correct = correct_replies_count > 0
    score = 1.0 if is_correct else 0.0
    
    return {
        "id": unique_id,
        "score": score, # 0.0 or 1.0
        "is_correct": is_correct,
        "reason": "Correct" if is_correct else "Incorrect or parsing failed",
        "ground_truth_answer": ground_truth,
        "model_answers": model_answers, # 提取出的所有模型答案
        "correct_replies_count": correct_replies_count,
        "total_replies": len(replies)
    }

# ============================================
# 4. 主流程与报告函数 (Main Flow & Reporting)
# ============================================

def load_dataset_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """从jsonl文件中加载数据集。"""
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return []
    return dataset

# run_evaluation 函数保持不变，或者稍作调整以控制打印
def run_evaluation(file_name: str, print_details: bool = True) -> Dict[str, Any]:
    """
    运行单个文件的评估流程。
    返回该文件的评估摘要。
    """
    if print_details:
        print(f"\n--- Starting evaluation for: {os.path.basename(file_name)} ---")
    
    dataset = load_dataset_from_jsonl(file_name)
    if not dataset:
        if print_details:
            print("Evaluation stopped as no data was loaded.")
        return {"file": os.path.basename(file_name), "accuracy": 0.0, "total_cases": 0, "correct_cases": 0}

    results = [evaluate_case(case) for case in dataset]
    
    total_cases = len(results)
    total_correct = sum(res["score"] for res in results)
    
    if print_details:
        print("\n--- Detailed Results per Case ---")
        for res in results:
            status = "✅ CORRECT" if res["is_correct"] else "❌ INCORRECT"
            print(
                f"ID: {res['id']:<20} | Status: {status:<15} | "
                f"Ground Truth: {res['ground_truth_answer']:<5} | "
                f"Correct Replies: {res['correct_replies_count']}/{res['total_replies']}"
            )
        print("\n--- Evaluation Finished for this file ---\n")

    accuracy = (total_correct / total_cases) * 100 if total_cases > 0 else 0.0
    
    return {
        "file": os.path.basename(file_name),
        "accuracy": accuracy,
        "total_cases": total_cases,
        "correct_cases": int(total_correct),
        "results": results
    }
    
def evaluate_directory(directory_path: str):
    """
    评估指定目录下的所有 .jsonl 文件。
    """
    # 查找所有 .jsonl 文件
    input_dir = Path(directory_path)
    if not input_dir.is_dir():
        print(f"Error: Directory not found at '{directory_path}'")
    # 使用 rglob 递归查找所有 .jsonl 文件
    file_paths = sorted(list(input_dir.rglob('*.jsonl')))
    
    
    if not file_paths:
        print(f"No .jsonl files found in the directory: {directory_path}")
        return

    print(f"Found {len(file_paths)} .jsonl files to evaluate in '{directory_path}'.")
    print("="*50)
    
    all_summaries = {}
    for file_path in sorted(file_paths): # 对文件排序以保证顺序一致
        # 调用单个文件的评估函数，但只获取摘要，不在循环中打印细节
        # summary = run_evaluation(file_path, print_details=False) 
        summary = run_evaluation(file_path, print_details=True) 
        all_summaries[file_path] = summary

    # 循环结束后，打印所有文件的总结报告
    print("\n--- Overall Evaluation Summary ---")
    print(f"{'File Name':<30} | {'Accuracy':<12} | {'Correct/Total Cases':<20}")
    print("-" * 70)
    
    total_correct_all_files = 0
    total_cases_all_files = 0

    for summary in all_summaries.values():
        print(
            f"{summary['file']:<30} | {summary['accuracy']:.2f}%{'':<6} | "
            f"{summary['correct_cases']}/{summary['total_cases']}"
        )
        total_correct_all_files += summary['correct_cases']
        total_cases_all_files += summary['total_cases']
    
    print("-" * 70)
    
    # 计算并打印总体平均准确率
    overall_avg_accuracy = (total_correct_all_files / total_cases_all_files) * 100 if total_cases_all_files > 0 else 0.0
    print(f"{'Overall Average':<30} | {overall_avg_accuracy:.2f}%{'':<6} | "
          f"{total_correct_all_files}/{total_cases_all_files}")
    print("="*50)
    return all_summaries
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluates all .jsonl files in a given directory for an integer-answer task."
    )
    parser.add_argument(
        '--directory_path', 
        type=str, 
        help="The path to the directory containing .jsonl files.",
        # default="/home/ldaphome/seanl/code/mergeKV/gsm_infinite/gsm-infinite/datasets/medium-qwen2‑7B‑instruct-2025‑08‑04_8k",
        # default="/home/ldaphome/seanl/code/mergeKV/gsm_infinite/gsm-infinite/datasets/medium-qwen-2.5-7b-instruct_8k",
        # default="/home/ldaphome/seanl/code/KVCache-Factory/gsm_infinite/gsm-infinite/datasets/medium-qwen-2.5-7b-instruct_8k",
        # default="/home/ldaphome/seanl/code/KVCache-Factory/gsm_infinite/gsm-infinite/datasets/hard-qwen-2.5-7b-instruct_8k",
        default="/home/ldaphome/seanl/code/mergeKV/gsm_infinite/gsm-infinite/datasets/hard-qwen-2.5-7b-instruct_8k",
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory_path):
        print(f"Error: The provided path '{args.directory_path}' is not a valid directory.")
    else:
        evaluate_directory(args.directory_path)
    
    # 你可以在这里对 evaluation_results 进行进一步处理，比如保存到文件
    # if evaluation_results:
    #     with open("evaluation_results.json", "w") as f:
    #         json.dump(evaluation_results, f, indent=2)
    #     print("Detailed results saved to evaluation_results.json")