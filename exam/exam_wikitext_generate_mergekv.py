#!/usr/bin/env python3
"""
exam_wikitext_generate_mergekv.py.py

该脚本使用 'mergekv' 方法修改 Llama-2-7b-hf 模型，
然后在 wikitext 数据集上运行生成任务，并保存 attention forward 的输出结果。

示例:
python exam_wikitext_generate_mergekv.py --cache_size 200 --max_samples 10
"""
import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import torch
import datasets
import argparse
import logging
from collections import OrderedDict

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

# 导入 mergekv 的核心库
from mergekv import AttentionForward as AF

# -------------------- 全局变量，用于从 mergekv 中捕获状态 --------------------
layer_out_flag = True
# ---------------------------------------------------------------------------


def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def get_mergekv_model(args):
    """
    使用 mergekv 的方法加载、修改并准备模型。
    """
    logger.info("Loading model using mergekv.AttentionForward...")
    # 1. 使用 AF.model_load 加载模型和分词器
    tokenizer, model = AF.model_load(model_name=args.model_name, merge=False)
    config = model.config
    
    # 计算实际的缓存大小
    # 注意：您的原始 mergekv 脚本使用 config.max_position_embeddings * cache_budget
    # 这里我们为了与前一个脚本统一，直接使用 cache_size 作为预算
    cache_budget_tokens = int(args.cache_size)
    logger.info(f"Max position embeddings: {config.max_position_embeddings}")
    logger.info(f"Cache budget set to: {cache_budget_tokens} tokens")

    logger.info("Applying mergekv modification via AF.change_mode...")
    # 2. 使用 AF.change_mode 应用模型修改
    AF.change_mode(
        merge=True, # 始终启用
        cache_budget=cache_budget_tokens, 
        cache_tail=args.cache_tail, 
        window_size=args.window_size,
        cache_dense=args.cache_dense,
        metric=args.metric, 
        score_update=args.score_update, 
        scale_factor=args.scale_factor,
        shrink_factor=1,
        out_state=1
    )

    # 3. 将模型移动到目标设备
    logger.info(f"Moving model to device: {args.device}")
    model.eval().half().to(args.device)
    
    return config, tokenizer, model


def collect_out_states(model):
    """从模型的各个层收集 'out_state'。"""
    out_dict = OrderedDict()
    for name, module in model.named_modules():
        # 我们只关心修改后的 LlamaAttentionLESS 模块
        if isinstance(module, LlamaAttention) and hasattr(module, "out_state") and module.out_state is not None:
            try:
                # 从模块名中解析层索引
                layer_idx = int(name.split(".")[2])
                out_dict[layer_idx] = module.out_state.detach().cpu()
                # print(f"Collected out_state from layer {layer_idx}, shape: {out_dict[layer_idx].shape}")
            except (IndexError, ValueError):
                print(f"Warning: Could not parse layer index from module name: {name}")
    return out_dict

def save_out_states(model, args, sample_idx):
    """
    保存收集到的 'out_state' 到文件。
    这个版本直接接收一个列表，而不是去模型里查找。
    """
    save_dir = args.save_root
    os.makedirs(save_dir, exist_ok=True)
    # 使用与之前脚本一致的目录结构
    base_dir = Path(save_dir) / f"{args.arch}_{args.method}_{args.cache_size}"
    os.makedirs(base_dir, exist_ok=True)
    
    states_to_save = collect_out_states(model)
    if not states_to_save:
        logger.warning("Warning: no 'out_state' found in the global list. The hook might have failed.")
        return
        
    path = os.path.join(base_dir, f"sample_{sample_idx:05d}.pt")
    # 直接保存这个列表
    torch.save(states_to_save, path)
    # logger.info(f"[{sample_idx}] Saved {len(states_to_save)} items (likely layers) -> {path}")


# -------------------- 主流程 (来自 wikitext_generate.py) --------------------


def run(args):
    # 1. 数据
    logger.info("Loading wikitext dataset...")
    # wikitext = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    cache_dir = "/home/ldaphome/seanl/.cache/huggingface/datasets"   # 替换成你查到的

    wikitext = datasets.load_dataset(
        "wikitext", "wikitext-2-raw-v1",
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists",
        trust_remote_code=True
    )
    dataset = wikitext[args.split]

    # 2. 模型
    config, tokenizer, model = get_mergekv_model(args)

    # 3. 滑动拼接 + 生成
    buffer, token_len = [], 0
    count = 0
    for example in dataset:
        if count >= args.max_samples:
            break

        text = example["text"].strip()
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.append(text)
        token_len += len(ids)

        # 累计超过100个token时触发一次生成
        if token_len >= 100:
            concat_text = " ".join(buffer)
            inputs = tokenizer(concat_text, return_tensors="pt").to(args.device)
            
            # logger.info(f"\n--- Generating for sample {count} ---")
            logger.info(f"Input token length: {inputs.input_ids.shape[1]}")
            
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            save_out_states(model, args, count)
            
            # 

            count += 1
            buffer.clear()
            token_len = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用mergekv方法修改Llama-2模型，并在wikitext上运行以保存中间状态。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 通用参数 (来自 wikitext 脚本) ---
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Hugging Face 上的模型名称。")
    parser.add_argument("--arch", type=str, default="llama2", choices=["llama2"],
                        help="模型架构名称，用于保存目录。")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="要处理的最大样本数量。")
    parser.add_argument("--max_new_tokens", type=int, default=1,
                        help="每次生成调用的新token数量。")
    parser.add_argument("--save_root", type=str, default="./bias_out_states_mergekv",
                        help="保存out_state的根目录。")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"],
                        help="使用的数据集划分。")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备的名称 (例如 'cuda', 'cpu')。")

    # --- mergekv 特定参数 ---
    parser.add_argument("--method", type=str, default='mergekv',
                        help="使用的修改方法，用于保存目录。")
    parser.add_argument("--cache_sizes", type=str, default='10',
                        help="KV缓存的大小（以token数量计）。")
    parser.add_argument("--cache_size", type=int, default=10,
                        help="KV缓存的大小（以token数量计）。")
    parser.add_argument("--window_size", type=int, default=None,)
    parser.add_argument("--cache_tail", type=float, default=0.4,
                        help="mergekv参数: cache_tail。")
    parser.add_argument("--cache_dense", type=float, default=1.0,
                        help="mergekv参数: cache_dense。")
    parser.add_argument("--scale_factor", type=float, default=0.6,
                        help="mergekv参数: scale_factor。")
    parser.add_argument("--metric", type=str, default='dot_product',
                        help="mergekv参数: metric。")
    parser.add_argument("--score_update", type=str, default='max',
                        help="mergekv参数: score_update。")

    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        args.device = "cpu"
    
    logger.info("Starting run with the following configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 40)
    
    if "," in args.cache_sizes:
        for cache_size in args.cache_sizes.split(","):
            args.cache_size = int(cache_size)
            print(f"set cache_size to {args.cache_size:_^30}")
            run(args)
    else:
        run(args)