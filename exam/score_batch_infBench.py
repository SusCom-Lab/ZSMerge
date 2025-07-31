#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A batch scoring script to run 'exam/score_infiniteBench.py' on all .jsonl files
in a given results directory using multiple processes for acceleration.

It automatically infers the model name, method name, and task name from the 
file path, collects all scores, and prints a final summary in JSON format.

The expected directory structure is:
<results_dir>/<model_name>/<method_name>/*.jsonl

Usage:
    python exam/score_batch_infBench.py --results_dir results/infiniteBench
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

# A list of all valid tasks, copied from your scoring script.
ALL_TASKS = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longdialogue_qa_eng",
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longbook_qa_chn",
    "math_find",
    "math_calc",
    "code_run",
    "code_debug",
]

# The path to your scoring script.
SCORING_SCRIPT_PATH = "exam/score_infiniteBench.py"

def find_task_from_filename(filename: str) -> str | None:
    """Identifies the task name from the beginning of a filename."""
    for task in ALL_TASKS:
        if filename.startswith(task):
            return task
    return None

def process_single_file(jsonl_file: Path) -> dict | None:
    """
    Processes a single .jsonl file. This function is designed to be run
    in a separate process.
    It returns a dictionary with the results or None if processing fails.
    """
    # 1. Infer task name
    task_name = find_task_from_filename(jsonl_file.name)
    if not task_name:
        print(f"⚠️  Skipping file: Could not determine task for '{jsonl_file.name}'.")
        return None

    # 2. Infer model and method names
    try:
        method_name = jsonl_file.parent.name
        model_name = jsonl_file.parent.parent.name
    except IndexError:
        print(f"⚠️  Skipping file: Could not determine model/method for '{jsonl_file}'.")
        return None

    # 3. Construct and run the scoring command
    command = [
        "python",
        SCORING_SCRIPT_PATH,
        "--pred_file",
        str(jsonl_file),
        "--task",
        task_name,
        "--model_name",
        model_name
    ]

    try:
        process_output = subprocess.check_output(
            command, text=True, stderr=subprocess.PIPE
        )
        
        # 4. Parse the score
        match = re.search(r"Score: (\d+\.\d+)", process_output)
        if match:
            score = float(match.group(1))
            return {
                "model": model_name,
                "method": method_name,
                "task": task_name,
                "score": score
            }
        else:
            print(f"❌ Error parsing score for {jsonl_file.name}:\n{process_output}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Scoring script failed for {jsonl_file.name}:\n{e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch score all .jsonl files using multiple processes."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="The root directory containing model prediction results."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help=f"Number of parallel processes to use. Defaults to CPU count ({cpu_count()})."
    )
    args = parser.parse_args()
    print(f"{args=}")

    if not args.results_dir.is_dir():
        print(f"Error: Directory not found at '{args.results_dir}'")
        return

    if not Path(SCORING_SCRIPT_PATH).exists():
        print(f"Error: Scoring script not found at '{SCORING_SCRIPT_PATH}'")
        return

    summary_results = {}
    
    print(f"🔍 Searching for result files in: {args.results_dir.resolve()}")
    result_files = sorted(list(args.results_dir.rglob("*.jsonl")))
    
    if not result_files:
        print("No .jsonl files found. Exiting.")
        return

    print(f"Found {len(result_files)} result files. Starting evaluation with {args.num_workers} worker(s).\n")

    # Create a multiprocessing Pool
    with Pool(processes=args.num_workers) as pool:
        # Use imap_unordered for efficient parallel processing
        # and tqdm for a progress bar.
        results_iterator = pool.imap_unordered(process_single_file, result_files)
        
        for result in tqdm(results_iterator, total=len(result_files), desc="Overall Progress"):
            if result:
                # Aggregate results into the summary dictionary
                model = result["model"]
                method = result["method"]
                task = result["task"]
                score = result["score"]
                
                if model not in summary_results:
                    summary_results[model] = {}
                if method not in summary_results[model]:
                    summary_results[model][method] = {}
                
                summary_results[model][method][task] = score

    # Print the final summary
    print("\n==================================================")
    print("📊 Final Scoring Summary")
    print("==================================================")
    if summary_results:
        # Sort keys for consistent output
        sorted_summary = {k: v for k, v in sorted(summary_results.items())}
        print(json.dumps(sorted_summary, indent=2))
    else:
        print("No results were successfully processed.")


if __name__ == "__main__":
    main()