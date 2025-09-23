"""
LongBench Dataset Scoring Script

This script evaluates LongBench dataset results by calculating scores for different metrics
and generating comprehensive reports.
"""
import os
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List

from metrics_longbench import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer(dataset, predictions, answers, all_classes):
    """Calculate score for a dataset using the appropriate metric."""
    total_score = 0.0
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Score LongBench evaluation results")
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Main directory containing all experiment results')
    return parser.parse_args()

def get_result_files(root_dir: str) -> List[Dict]:
    """Traverse directory to get all result file information."""
    files = []
    for model_dir in Path(root_dir).iterdir():
        if not model_dir.is_dir():
            continue

        # Parse model name and capacity
        model_name, capacity = model_dir.name.rsplit('-', 1)

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset = dataset_dir.name
            for result_file in dataset_dir.glob('*.jsonl'):
                method = result_file.stem
                files.append({
                    "model": model_name,
                    "capacity": int(capacity),
                    "dataset": dataset,
                    "method": method,
                    "path": str(result_file)
                })
    return files

def calculate_score(file_info: Dict) -> Dict:
    """Calculate score for a single result file."""
    try:
        with open(file_info['path'], 'r', encoding='utf-8') as f:
            predictions, answers, all_classes = [], [], []
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]

        score = scorer(
            file_info['dataset'],
            predictions,
            answers,
            all_classes
        )

        return {
            **file_info,
            "score": score,
            "n_case": len(answers),
            "status": "success"
        }
    except Exception as e:
        print(f"Error processing {file_info['path']}: {str(e)}")
        return {
            **file_info,
            "score": -1,
            "n_case": 0,
            "status": str(e)
        }

def save_results(results: List[Dict], output_dir: str):
    """Save scoring results to pickle and CSV files."""
    # Save as pickle
    pickle_path = os.path.join(output_dir, 'scores.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    # Generate CSV report
    csv_path = os.path.join(output_dir, 'report.csv')
    datasets = sorted({r['dataset'] for r in results})
    methods = sorted({r['method'] for r in results})

    with open(csv_path, 'w') as f:
        # Header
        f.write("Method," + ",".join(datasets) + "\n")

        # Data rows
        for method in methods:
            row = [method]
            for dataset in datasets:
                scores = [r['score'] for r in results
                         if r['method'] == method and r['dataset'] == dataset]
                row.append(str(max(scores)) if scores else "N/A")
            f.write(",".join(row) + "\n")

def main():
    """Main function to orchestrate the scoring process."""
    args = parse_args()

    # 1. Get all result files
    result_files = get_result_files(args.results_dir)
    print(f"Found {len(result_files)} result files")

    # 2. Calculate scores
    scores = []
    for f in result_files:
        print(f"Processing {f['path']}...")
        scores.append(calculate_score(f))
        # print(scores[-1])

    # 3. Save results
    save_results(scores, args.results_dir)
    print(f"Evaluation completed, results saved to {args.results_dir}")
    print(f"Generated files: scores.pkl, report.csv")

if __name__ == '__main__':
    main()