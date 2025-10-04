#!/usr/bin/env python3

import json
import argparse
import re
from collections import Counter
from typing import Any, Dict, List, Tuple
import unicodedata


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def normalize_answer(s: Any) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower()
    # Remove punctuation and symbols using Unicode categories (P*, S*)
    s = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in s)
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_results_list(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "results" in obj:
        return obj["results"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported results JSON format")


def extract_inputs_list(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        return obj["results"]
    raise ValueError("Unsupported inputs JSON format")


def get_prediction_text(entry: Dict[str, Any], keys_priority: List[str]) -> str:
    for key in keys_priority:
        if key in entry and isinstance(entry[key], str):
            return entry[key]
    # Sometimes nested
    if "prediction" in entry and isinstance(entry["prediction"], dict):
        inner = entry["prediction"]
        for key in keys_priority:
            if key in inner and isinstance(inner[key], str):
                return inner[key]
    return ""


def get_gold_answers(item: Dict[str, Any], gold_keys: List[str]) -> List[str]:
    for k in gold_keys:
        if k in item:
            v = item[k]
            if isinstance(v, list):
                return [str(x) for x in v]
            if isinstance(v, str):
                return [v]
    return []


def compute_f1_for_files(results_path: str, inputs_path: str, pred_keys: List[str], gold_keys: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    results_raw = load_json(results_path)
    results_list = extract_results_list(results_raw)

    inputs_raw = load_json(inputs_path)
    inputs_list = extract_inputs_list(inputs_raw)

    # Build index mapping using sample_index when available
    idx_to_gold: Dict[int, List[str]] = {}
    for idx, item in enumerate(inputs_list):
        idx_to_gold[idx] = get_gold_answers(item, gold_keys)

    per_sample = []
    f1_sum = 0.0
    count = 0

    for default_idx, res in enumerate(results_list):
        # Prefer explicit sample_index
        idx = res.get("sample_index", default_idx)
        gold_answers = idx_to_gold.get(idx, [])
        prediction = get_prediction_text(res, pred_keys)

        if not gold_answers:
            continue

        best_f1 = max((f1_score(prediction, g) for g in gold_answers), default=0.0)

        per_sample.append({
            "index": idx,
            "prediction": prediction,
            "gold_answers": gold_answers,
            "f1": best_f1,
        })
        f1_sum += best_f1
        count += 1

    avg_f1 = f1_sum / count if count else 0.0
    return avg_f1, per_sample


def main():
    ap = argparse.ArgumentParser("Compute F1 score between predictions and gold answers")
    ap.add_argument("--results", required=True, help="Path to optimized_results.json")
    ap.add_argument("--inputs", required=True, help="Path to input dataset JSON (for gold answers)")
    ap.add_argument("--print_samples", action="store_true", help="Print per-sample F1 details")
    args = ap.parse_args()

    # Support multiple possible keys for predictions and golds
    prediction_keys = [
        "answer",
    ]
    gold_keys = [
        "answers",
    ]

    avg_f1, per_sample = compute_f1_for_files(
        results_path=args.results,
        inputs_path=args.inputs,
        pred_keys=prediction_keys,
        gold_keys=gold_keys,
    )

    print(f"Samples evaluated: {len(per_sample)}")
    print(f"Average F1: {avg_f1:.4f}")


if __name__ == "__main__":
    main()


