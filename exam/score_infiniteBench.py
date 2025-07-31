"""exam/score_infiniteBench.py
python exam/score_infiniteBench.py \
  --pred_file results/infiniteBench/meta-llama_Llama-3.1-8B-Instruct/full/kv_retrieval-0.6.jsonl \
  --task kv_retrieval

"""

from pathlib import Path
import json
import re
import string
from collections import Counter

from tqdm import tqdm
import evaluate

# Assuming args.py is in the same directory or accessible
# from args import parse_args
# For standalone running, we can mock the args
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=Path, required=True, help="Path to the prediction file.")
    parser.add_argument("--task", type=str, required=True, help="Task name.")
    parser.add_argument("--model_name", type=str, default="unknown_model", help="Model name for scoring logic.")
    parser.add_argument("--output_dir", type=Path, default="results", help="Base output directory.")
    return parser.parse_args()


ROUGE_SCORER = evaluate.load("rouge")


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Chinese version. Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."  # noqa
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth) -> tuple[float, float, float]:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths) -> float:
    """Computes the F1, recall, and precision."""
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def qa_f1_score_zh(pred: str, ground_truths: list[str]) -> float:
    """
    QA F1 score for chinese.
    """
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        norm_pred = normalize_zh_answer(pred)
        norm_label = normalize_zh_answer(ground_truth)

        # One character one token.
        pred_tokens = list(norm_pred)
        label_tokens = list(norm_label)
        scores = f1_score(pred_tokens, label_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def load_json(fname):
    return json.load(open(fname))


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r", encoding="utf8") as fin:
        for line in fin:
            if line.strip() == "":  # Skip empty lines
                continue
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def first_int_match(prediction):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def split_retrieval_answer(pred: str):
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    return words


def get_score_one_kv_retrieval(pred, label, model_name: str) -> bool:
    if isinstance(label, list):
        label = label[0]
    for c in ['\n', ':', '\"', '\'', '.', ',', '?', '!', '{', '}']:
        pred = pred.replace(c, ' ')
    words = pred.split()
    return label in words


def get_score_one_passkey(pred, label, model_name: str) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label, model_name: str) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_code_run(pred, label, model_name: str) -> bool:
    """
    Returns the score of one example in Code.Run.
    """
    if isinstance(label, list):
        label = label[0]
    pred = pred.strip()
    for c in ["\n", ".", "`", "'", '"', ":"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if len(words) == 0:
        return False
    try:
        pred = int(words[-1])
        return label == pred
    except Exception:
        return False


def get_score_one_code_debug(pred, label, model_name: str) -> bool:
    """
    Returns the score of one example in Code.Debug.
    """
    # Safety check: If the label doesn't have two elements, it's malformed.
    if not isinstance(label, list) or len(label) < 2:
        return False

    pred = pred.strip()
    label_c = label[1]  # The correct option letter (e.g., 'B')
    fn_name = label[0]  # The correct function name (e.g., 'repack_carchive')
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred == label_c:
            return True
    ans_prefixes = [
        "answer is:",
        "is:",
        "answer:",
        "correct option is:"
    ]
    pred = pred.strip()
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    if pred.startswith(label_c) or pred.startswith(fn_name):
        return True
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        pred = pred[idx + len(prefix) + 1 :]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                return True
        return False
    return False

def get_score_one_math_find(pred, label, model_name: str) -> bool:
    if isinstance(label, list):
        # In math_find, there is always only one label.
        label = label[0]

    # Handle the case where the label is a string like "88" instead of an int 88.
    if isinstance(label, str):
        try:
            # Try to convert to int first, as it's more common.
            label = int(label)
        except ValueError:
            try:
                # If int conversion fails, try float.
                label = float(label)
            except ValueError:
                # If both fail, the label is not a valid number.
                return False

    if isinstance(label, int):
        # Find first int or float in the prediction
        first_num_match = re.search(r"-?\d+(\.\d+)?", pred.replace(",", ""))
        if first_num_match is None:
            return False
        first_num_str = first_num_match.group(0).strip()
        try:
            # We need to handle if the found number is a float like "88.0"
            return int(float(first_num_str)) == label
        except ValueError:
            return False
    elif isinstance(label, float):
        # Find first float or int in the prediction
        first_num_match = re.search(r"-?\d+(\.\d+)?", pred.replace(",", ""))
        if first_num_match is None:
            return False
        first_num_str = first_num_match.group(0).strip()
        try:
            return float(first_num_str) == label
        except ValueError:
            return False
    else:
        # This part should now be much harder to reach.
        raise TypeError(f"Expected int or float, but got {type(label)} after conversion attempts.")


def get_score_one_longdialogue_qa_eng(pred, label, model_name: str) -> bool:
    pred = pred.strip()
    pred = pred.upper()
    for item in label:
        if item.upper() in pred:
            return 1
    return 0

# ========================================================================
# DEPRECATED/REMOVED FUNCTION: This function is incorrect for the task.
# We will use qa_f1_score instead, which is wrapped by get_score_one_longbook_qa_eng.
# ========================================================================
# def get_score_one_longbook_choice_eng(pred, label, model_name: str) -> bool:
#    ... (old code removed)


def get_score_one_longbook_qa_eng(pred, label, model_name: str) -> float:
    return qa_f1_score(pred, label)


def get_score_one_longbook_sum_eng(
    pred: str, label: str, model_name: str
) -> float:
    score = ROUGE_SCORER.compute(
        predictions=[pred], references=[label], use_aggregator=False
    )
    return score["rougeLsum"][0]


def get_score_one_longbook_qa_chn(pred, label, model_name: str) -> float:
    return qa_f1_score_zh(pred, label)


def get_score_one_math_calc(pred, label, model_name: str) -> float:
    assert isinstance(label, list), f"Expected list, got {type(label)}"
    if isinstance(label[0], list):
        label = label[0]
    pred_nums = []
    pred_list = re.split("[^0-9]", pred)
    for item in pred_list:
        if item != "":
            pred_nums.append(int(item))

    if model_name == "gpt4":
        pred_nums = pred_nums[1:]

    cnt = 0
    for i in range(len(label)):
        if i >= len(pred_nums):
            break
        if label[i] == pred_nums[i]:
            cnt += 1
        else:
            break
    return cnt / len(label)


def get_score_one(
    pred: str, label: str, task_name: str, model_name: str
) -> float:
    # ========================================================================
    # MODIFIED LOGIC: Mapped 'longbook_choice_eng' to the correct scoring function.
    # ========================================================================
    NAME_TO_SCORE_GETTER = {
        # Retrieve
        "kv_retrieval": get_score_one_kv_retrieval,
        "passkey": get_score_one_passkey,
        "number_string": get_score_one_number_string,
        # Code
        "code_run": get_score_one_code_run,
        "code_debug": get_score_one_code_debug,
        # Longbook
        "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
        "longbook_qa_eng": get_score_one_longbook_qa_eng,
        "longbook_sum_eng": get_score_one_longbook_sum_eng,
        "longbook_choice_eng": get_score_one_longbook_qa_eng, # Use QA F1 score
        "longbook_qa_chn": get_score_one_longbook_qa_chn,
        # Math
        "math_find": get_score_one_math_find,
        "math_calc": get_score_one_math_calc,
    }
    # This part handles legacy task names like kv_retrieval_prefix
    for key, val in list(NAME_TO_SCORE_GETTER.items()):
        if task_name.startswith(key):
            task_name = key
            break

    assert task_name in NAME_TO_SCORE_GETTER, f"Invalid task name: {task_name}"
    score = NAME_TO_SCORE_GETTER[task_name](pred, label, model_name)
    return float(score)

def get_labels(preds: list, data_name: str) -> list:
    """
    Extracts labels from the prediction records.
    For 'code_debug', it dynamically constructs the label [function_name, option_char].
    """
    possible_label_keys = ["ground_truth", "label", "reference"]
    label_key = None
    if not preds:
        return []
    for key in possible_label_keys:
        if key in preds[0]:
            label_key = key
            break
    
    if not label_key:
        raise ValueError(f"Cannot find a valid label key in the first record: {preds[0]}")

    # Special handling for code_debug to construct the label correctly
    if data_name == "code_debug":
        constructed_labels = []
        for p in preds:
            if "options" not in p or not isinstance(p.get("options"), list):
                raise ValueError(f"Missing or invalid 'options' field for code_debug task in record: {p}")
            
            fn_name_list = p.get(label_key)
            if not fn_name_list:
                print(f"Warning: Empty reference in record {p.get('id', 'N/A')}. This will be scored as 0.")
                constructed_labels.append([]) # Append empty list to fail safety check
                continue

            fn_name = fn_name_list[0]
            options = p["options"]
            
            try:
                # Find the index of the correct function name in the options list
                idx = options.index(fn_name)
                # Convert index (0, 1, 2...) to character ('A', 'B', 'C'...)
                option_char = chr(ord('A') + idx)
                # Append the constructed label
                constructed_labels.append([fn_name, option_char])
            except ValueError:
                # This happens if the reference function name is not in the options list.
                print(f"Warning: Reference '{fn_name}' not found in options for record {p.get('id', 'N/A')}. This will be scored as 0.")
                constructed_labels.append([fn_name])
        return constructed_labels

    # Original logic for all other tasks
    return [x.get(label_key, "XXXXXXXXXX") for x in preds]


def get_preds(preds: list, data_name: str) -> list[str]:
    pred_strings = []
    if not preds:
        return []
    possible_pred_keys = ["prediction", "pred"]
    for pred in preds:
        this_pred = "NO PREDICTION"
        for pred_key in possible_pred_keys:
            if pred_key in pred:
                this_pred = pred[pred_key]
                break
        else:
            raise ValueError(f"Cannot find prediction in {pred}")
        pred_strings.append(this_pred)
    return pred_strings


def get_score(
    labels: list, preds: list, data_name: str, model_name: str
) -> float:
    """
    Computes the average score for a task.
    """
    if not labels:
        return 0.0
    assert len(labels) == len(preds)
    scores = []
    for label, pred in tqdm(zip(labels, preds), total=len(labels), desc="Scoring"):
        score = get_score_one(pred, label, data_name, model_name)
        scores.append(score)
    return sum(scores) / len(scores)


def compute_scores(preds_path, data_name: str, model_name: str):
    print("Loading prediction results from", preds_path)
    preds = list(iter_jsonl(preds_path))
    if not preds:
        print("Warning: Prediction file is empty. Score is 0.")
        print(f"Task: {data_name}, Model: {model_name}, Score: {0.0:.4f}")
        return
        
    labels = get_labels(preds, data_name)
    preds_str = get_preds(preds, data_name)

    acc = get_score(labels, preds_str, data_name, model_name)
    print(f"Task: {data_name}, Model: {model_name}, Score: {acc:.4f}")


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

if __name__ == "__main__":
    args = parse_args()
    if args.task == "all":
        tasks = ALL_TASKS
    else:
        # Allow matching prefixes, e.g., 'kv_retrieval' matches 'kv_retrieval-0.6'
        tasks = [t for t in ALL_TASKS if args.task.startswith(t)]
        if not tasks:
            tasks = [args.task]

    for task in tasks:
        pred_file = Path(args.pred_file)
        if not pred_file.exists():
             print(f"Predictions not found in: {pred_file}, skipping.")
             continue
        
        model_name = args.model_name
        # Try to infer model name from path if not provided
        if model_name == "unknown_model" and len(pred_file.parts) > 2:
            model_name = pred_file.parent.parent.name
        
        compute_scores(pred_file, task, model_name)