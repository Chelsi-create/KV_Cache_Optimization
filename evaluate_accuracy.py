#!/usr/bin/env python3
"""
Accuracy Evaluation Script for CacheBlend Pipeline
Uses OpenAI API to judge semantic correctness of generated answers
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import time

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Install with: pip install openai")


JUDGE_PROMPT = """You are an expert evaluator judging the correctness of question-answering systems.

Task: Determine if the Generated Answer is semantically correct compared to the Gold Answer.

Rules:
1. If the Generated Answer conveys the same core information as the Gold Answer, mark it as CORRECT
2. Minor differences in wording, order, or additional context are acceptable
3. If the Generated Answer is partially correct but missing key information, mark it as INCORRECT
4. If the Generated Answer is completely wrong or unrelated, mark it as INCORRECT

Examples:

Gold: "Paris"
Generated: "Paris, France"
Judgment: CORRECT (same meaning, added context is fine)

Gold: "Exeter College, Oxford"
Generated: "Oxford"
Judgment: INCORRECT (missing specific college name)

Gold: "1945"
Generated: "The year 1945"
Judgment: CORRECT (same information, different phrasing)

Gold: "William Shakespeare"
Generated: "Shakespeare"
Judgment: CORRECT (partial name is acceptable for famous people)

Gold: "Columbia University"
Generated: "Harvard University"
Judgment: INCORRECT (completely different information)

Now evaluate:

Gold Answer: {gold_answer}
Generated Answer: {generated_answer}

Respond with ONLY one word: either "CORRECT" or "INCORRECT"
"""


def load_json_file(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_gold_answers(input_file: str) -> Dict[str, str]:
    """Extract gold answers from input file"""
    data = load_json_file(input_file)
    
    # Handle different JSON structures
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and "results" in data:
        samples = data["results"]
    elif isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    else:
        samples = [data]
    
    gold_answers = {}
    for sample in samples:
        sample_id = sample.get("id", sample.get("question", ""))
        # Try different field names for the gold answer
        answer = (sample.get("answer") or 
                 sample.get("answers") or 
                 sample.get("gold_answer") or 
                 sample.get("target") or
                 "")
        
        # Handle list of answers (take first one)
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        
        gold_answers[str(sample_id)] = str(answer).strip()
    
    return gold_answers


def extract_generated_answers(results_file: str) -> Dict[str, str]:
    """Extract generated answers from results file"""
    data = load_json_file(results_file)
    
    # Handle different JSON structures
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        results = [data]
    
    generated_answers = {}
    for result in results:
        sample_id = result.get("sample_id", result.get("id", result.get("question", "")))
        answer = result.get("answer", "")
        generated_answers[str(sample_id)] = str(answer).strip()
    
    return generated_answers


def judge_with_openai(
    gold_answer: str, 
    generated_answer: str, 
    client: OpenAI,
    model: str = "gpt-4o-mini"
) -> tuple[bool, str]:
    """
    Use OpenAI API to judge if generated answer is correct
    Returns: (is_correct, reasoning)
    """
    
    prompt = JUDGE_PROMPT.format(
        gold_answer=gold_answer,
        generated_answer=generated_answer
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        judgment = response.choices[0].message.content.strip().upper()
        is_correct = "CORRECT" in judgment
        
        return is_correct, judgment
        
    except Exception as e:
        print(f"\nError calling OpenAI API: {e}")
        return False, f"ERROR: {str(e)}"


def calculate_accuracy(
    input_file: str,
    results_file: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate accuracy using OpenAI API
    
    Args:
        input_file: Path to original input file with gold answers
        results_file: Path to results file with generated answers
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: OpenAI model to use for judging
        output_file: Optional path to save detailed results
        verbose: Print per-sample results
    
    Returns:
        Dictionary with accuracy metrics
    """
    
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package required. Install with: pip install openai")
    
    # Initialize OpenAI client
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass --api_key")
    
    client = OpenAI(api_key=api_key)
    
    print("="*80)
    print("ACCURACY EVALUATION")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Results file: {results_file}")
    print(f"Judge model: {model}")
    print("="*80)
    
    # Load gold and generated answers
    print("\nLoading data...")
    gold_answers = extract_gold_answers(input_file)
    generated_answers = extract_generated_answers(results_file)
    
    print(f"Gold answers: {len(gold_answers)}")
    print(f"Generated answers: {len(generated_answers)}")
    
    # Match samples
    common_ids = set(gold_answers.keys()) & set(generated_answers.keys())
    if not common_ids:
        print("\nWarning: No matching sample IDs found!")
        print(f"Gold IDs sample: {list(gold_answers.keys())[:5]}")
        print(f"Generated IDs sample: {list(generated_answers.keys())[:5]}")
        return {"error": "No matching samples"}
    
    print(f"Matched samples: {len(common_ids)}")
    
    # Evaluate each sample
    results = []
    correct_count = 0
    
    print("\nEvaluating answers...")
    for sample_id in tqdm(sorted(common_ids), desc="Judging"):
        gold = gold_answers[sample_id]
        generated = generated_answers[sample_id]
        
        if not gold:
            print(f"\nWarning: Sample {sample_id} has no gold answer, skipping")
            continue
        
        # Call OpenAI API
        is_correct, judgment = judge_with_openai(gold, generated, client, model)
        
        if is_correct:
            correct_count += 1
        
        result = {
            "sample_id": sample_id,
            "gold_answer": gold,
            "generated_answer": generated,
            "correct": is_correct,
            "judgment": judgment
        }
        results.append(result)
        
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Sample {sample_id}")
            print(f"  Gold: {gold}")
            print(f"  Generated: {generated}")
            print(f"  Judgment: {judgment}")
        
        # Rate limiting - be nice to OpenAI API
        time.sleep(0.1)
    
    # Calculate metrics
    total = len(results)
    accuracy = (correct_count / total * 100) if total > 0 else 0.0
    
    evaluation_results = {
        "input_file": input_file,
        "results_file": results_file,
        "judge_model": model,
        "total_samples": total,
        "correct": correct_count,
        "incorrect": total - correct_count,
        "accuracy": accuracy,
        "per_sample_results": results
    }
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total samples: {total}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total - correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*80)
    
    # Save detailed results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
    
    return evaluation_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate answer accuracy using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires OPENAI_API_KEY env var)
  python evaluate_accuracy.py \\
    --input inputs/musique_s.json \\
    --results results_optimized_v6/results_optimized_v6_sdpa.json
  
  # With API key and output file
  python evaluate_accuracy.py \\
    --input inputs/musique_s.json \\
    --results results_optimized_v6/results_optimized_v6_sdpa.json \\
    --api_key sk-... \\
    --output evaluation_results.json
  
  # Use different model
  python evaluate_accuracy.py \\
    --input inputs/musique_s.json \\
    --results results_optimized_v6/results_optimized_v6_sdpa.json \\
    --model gpt-4o
        """
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to input file with gold answers"
    )
    parser.add_argument(
        "--results", 
        required=True,
        help="Path to results file with generated answers"
    )
    parser.add_argument(
        "--api_key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for judging (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output",
        help="Path to save detailed evaluation results JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample output"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    calculate_accuracy(
        input_file=args.input,
        results_file=args.results,
        api_key=args.api_key,
        model=args.model,
        output_file=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

