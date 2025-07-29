# simple_ruler_runner.py

import os
import json
import torch
from tqdm import tqdm
from functools import wraps

from mergekv import AttentionForward

# RULER 数据读取的辅助函数 (来自你的脚本)
def read_manifest(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # --- 1. 配置实验参数 ---
    model_name = "meta-llama/Llama-2-7b-hf"
    # 选择一个具体的RULER任务进行测试
    task_name = "niah_single_1"
    seq_length = "4096"
    
    # 你的自定义Attention的参数
    attention_config = {"cache_budget": 16384} # 比如你的 cache_budget 参数

    # 设置保存目录
    save_dir = f"./results/{model_name.split('/')[-1]}/ruler_simple_test"
    os.makedirs(save_dir, exist_ok=True)
    pred_file = os.path.join(save_dir, f"{task_name}_{seq_length}.jsonl")

    # --- 2. 加载模型和分词器 ---
    print("Loading model and tokenizer...")
    tokenizer, model = AttentionForward.model_load(model_name=model_name)
    model.eval().to(AttentionForward.device)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 3. 应用你的Attention变体 ---
    print("Applying custom attention variant...")
    AttentionForward.change_mode(merge=True, **attention_config)

    # --- 4. 加载RULER数据 ---
    # 假设你的RULER数据在 'experiments/benchmark/ruler/data' 目录下
    # 请根据你的实际路径修改
    data_path = f"experiments/benchmark/ruler/data/llama/{seq_length}/{task_name}/validation.jsonl"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please make sure you have downloaded the RULER dataset and placed it correctly.")
        return
        
    print(f"Loading data from: {data_path}")
    test_data = read_manifest(data_path)

    # --- 5. 运行推理并保存结果 ---
    print(f"Starting inference on {len(test_data)} samples...")
    with open(pred_file, "w", encoding="utf-8") as f_out:
        for sample in tqdm(test_data, desc=f"Running {task_name}-{seq_length}"):
            input_text = sample["input"]
            
            # 准备输入
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(AttentionForward.device)
            
            # 使用 model.generate 进行推理
            output = model.generate(
                input_ids,
                max_new_tokens=256, # RULER 任务通常需要生成最多256个token
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            # 解码并提取生成的部分
            generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            # 准备要保存的结果
            result_to_save = {
                "index": sample["index"],
                "pred": generated_text,
                # 你可以根据需要添加更多RULER评估需要的字段
                "input": sample["input"],
                "outputs": sample["outputs"],
                "length": sample["length"],
            }
            
            f_out.write(json.dumps(result_to_save) + "\n")

    print(f"Inference complete. Results saved to: {pred_file}")

    # --- 6. (可选) 切换回原始模式并验证 (可选) ---
    print("\nSwitching back to original attention for a quick comparison...")
    AttentionForward.change_mode(merge=False)
    # 你可以在这里再跑一个样本，看看行为是否恢复正常
    # ...

if __name__ == "__main__":
    main()