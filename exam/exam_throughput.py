
import os
import sys
sys.path.append(os.getcwd())

import torch
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mergekv import AttentionForward as AF



def benchmark_kv_cache(model, tokenizer, batch_size, prompt_length, generate_length):
    """
    Benchmark model performance with full KV cache
    Args:
        model: Loaded pretrained model
        tokenizer: Corresponding tokenizer
        batch_size: Number of parallel sequences
        prompt_length: Token length of input prompts
        generate_length: Number of tokens to generate
    """
    device = "cuda"  # Hardcoded for GPU benchmarking
    # Create batch prompts with dummy content
    prompts = (["a " * (prompt_length-1)] * batch_size)
    
    # Tokenize and move to GPU
    input_ids = tokenizer(
        prompts, 
        add_special_tokens=False,
        return_tensors="pt",
        padding=True
    ).input_ids.to(device)
    input_len = input_ids.size(-1)
    assert input_len == prompt_length, f"input_len<{input_len}> != prompt_length<{prompt_length}>"
    # Warmup GPU (optional)
    # _ = model.generate(input_ids, max_new_tokens=1)
    
    # Performance measurement
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=generate_length,
            min_new_tokens=generate_length,
            pad_token_id=tokenizer.eos_token_id
        )
        output_len = outputs.size(-1) - input_ids.size(-1)
        assert output_len == generate_length, f"output_len<{output_len}> != generate_length<{generate_length}>"
    latency = time.time() - start_time
    
    # Calculate metrics
    throughput = batch_size * generate_length / latency
    
    # Print formatted results
    print(f"\n[Benchmark Results]")
    print(f"Model: {args.model_name}")
    print(f"Batch Size: {batch_size} | Prompt Length: {prompt_length} | Generate Length: {generate_length}")
    print(f"Latency: {latency:.2f}s | Throughput: {throughput:.2f} tokens/s")


def load_model(model_name, torch_dtype=torch.float16):
    """
    Load model and tokenizer with safety checks
    Returns:
        config: Model configuration
        tokenizer: Text tokenizer
        model: Loaded LLM
    """
    # Load model components
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to avoid warnings
    
    # Load model with automatic device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype
    ).to("cuda")  # Explicit GPU placement for benchmarking
    
    return config, tokenizer, model

if __name__ == "__main__":
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Tool")
    
    # Required parameters
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name/path (e.g. 'meta-llama/Llama-2-13b-hf')")
    
    # Optimization parameters
    parser.add_argument("--merge", action="store_true",
                        help="Enable model merging (placeholder for custom implementation)")
    parser.add_argument("--cache_ratio", type=float, default=0.05,
                        help="KV cache ratio (0.0-1.0)", metavar="[0-1]")
    
    # Benchmark parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of parallel sequences", metavar="N")
    parser.add_argument("--prompt_length", type=int, default=512,
                        help="Input context length in tokens", metavar="L")
    parser.add_argument("--generate_length", type=int, default=128,
                        help="Number of tokens to generate", metavar="K")
    
    args = parser.parse_args()
    
    # Model loading
    print(f"\nLoading {args.model_name}...")
    config, tokenizer, model = load_model(args.model_name)
    
    # Placeholder for merge functionality
    if args.merge:
        print("Applying model merge...")
        AF.change_mode(
            merge=True, 
            cache_budget=int(args.prompt_length * args.cache_ratio)
        )
    
    # Run benchmark
    print("Starting benchmark...")
    benchmark_kv_cache(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        generate_length=args.generate_length
    )