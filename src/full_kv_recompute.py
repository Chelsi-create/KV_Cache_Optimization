#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import yaml  # type: ignore
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer  # type: ignore

# Local modules
from rag_retrieval import RetrievalConfig, ColbertRetrieval  # type: ignore
from build_kv_v2 import extract_texts  # type: ignore

# For semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using token-level F1 only.")


# --------------------------- F1 Score Calculation ---------------------------

def normalize_answer(s: str) -> str:
    """Lower case and remove punctuation, articles and extra whitespace."""
    import string, re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    import collections
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_max_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute max F1 score across multiple ground truth answers."""
    if not ground_truths:
        return 0.0
    return max(compute_f1(prediction, gt) for gt in ground_truths)


def compute_semantic_similarity(prediction: str, ground_truth: str, model) -> float:
    """
    Compute semantic similarity using sentence-transformers.
    Returns cosine similarity score between 0 and 1.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE or model is None:
        return 0.0
    
    # Encode both texts
    pred_embedding = model.encode(prediction, convert_to_tensor=True)
    truth_embedding = model.encode(ground_truth, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = st_util.cos_sim(pred_embedding, truth_embedding).item()
    return max(0.0, similarity)  # Ensure non-negative


def compute_max_semantic_similarity(prediction: str, ground_truths: List[str], model) -> float:
    """Compute max semantic similarity across multiple ground truth answers."""
    if not ground_truths or model is None:
        return 0.0
    return max(compute_semantic_similarity(prediction, gt, model) for gt in ground_truths)
    
# --------------------------- I/O helpers ---------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]  # type: ignore
    if isinstance(data, dict) and "results" in data:
        return data["results"]  # type: ignore
    return [data]  # type: ignore


# --------------------------- Retrieval ---------------------------

def run_retrieval(samples: List[Dict[str, Any]], cfg: Dict[str, Any], top_k: int) -> None:
    retrieval_cfg = RetrievalConfig(**cfg.get("retrieval", {}))
    if not getattr(retrieval_cfg, "checkpoint", None):
        retrieval_cfg.checkpoint = getattr(retrieval_cfg, "model_id", "colbert-ir/colbertv2.0")
    retrieval = ColbertRetrieval(retrieval_cfg)
    retrieval.prepare(samples)
    retrieval.retrieve(samples, top_k=top_k)


# --------------------------- Prompt building ---------------------------

def build_prompt_from_topk(sample: Dict[str, Any], top_k: int) -> Tuple[str, str]:
    text_key_pairs: List[Tuple[int, str]] = extract_texts(sample)  # [(idx, text), ...]
    idx2text = {i: t for i, t in text_key_pairs}
    retrieved_indices: List[int] = [int(i) for i in sample.get("retrieved_indices", [])]
    sel = retrieved_indices[:top_k] if retrieved_indices else []

    chunks = [idx2text[i] for i in sel if i in idx2text]
    context = "\n\n".join(chunks) if chunks else ""
    question = (sample.get("question") or "").strip()
    return context, question


def encode_input(tokenizer, context: str, question: str):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msg_content = f"Use the following context to answer.\n\n{context}\n\nQuestion: {question} DO NOT REPEAT THE QUESTION IN THE ANSWER."
        messages = [{"role": "user", "content": msg_content}]
        return tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
    # Fallback non-chat prompt
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids


# --------------------------- Decoding (full recompute) ---------------------------

def decode_full_recompute(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Dict[str, Any]:
        """
        Generate by fully prefilling on the input (no past_key_values).
        Uses a background thread to consume TextIteratorStreamer so TTFT is correct.
        """
        # Ensure PAD exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Record prompt length
        prompt_tokens = input_ids.shape[1]

        # --- PLACE INPUTS ON THE RIGHT DEVICE ---
        # If the model is sharded (device_map set), keep on CPU (Accelerate will scatter).
        # If it's a single-device model, move to that device.
        if getattr(model, "hf_device_map", None) is None:
            input_ids = input_ids.to(next(model.parameters()).device, non_blocking=True)

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        }

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer

        import threading
        first_token_time: Optional[float] = None
        start = time.perf_counter()

        def _run():
            with torch.inference_mode():
                model.generate(**generation_kwargs)

        th = threading.Thread(target=_run, daemon=True)
        th.start()

        pieces: List[str] = []
        try:
            for ch in streamer:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                pieces.append(ch)
        finally:
            th.join()

        end = time.perf_counter()
        text = "".join(pieces).strip()
        try:
            gen_ids = tokenizer.encode(text, add_special_tokens=False)
            num_tokens = len(gen_ids)
        except Exception:
            num_tokens = max(1, len(pieces))

        # Timing metrics
        ttft = (first_token_time - start) if first_token_time else 0.0
        e2e_latency = end - start
        
        # FIXED: TPOT should only measure decode time (after first token)
        decode_time = (end - first_token_time) if first_token_time else e2e_latency
        tpot = (decode_time / num_tokens) if num_tokens > 0 else 0.0
        
        # Throughput: output tokens per second
        throughput = (num_tokens / e2e_latency) if e2e_latency > 0 else 0.0

        return {
            "answer": text,
            "ttft": ttft,
            "e2e_latency": e2e_latency,
            "throughput": throughput,
            "tpot": tpot,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": num_tokens,
        }


# --------------------------- Main ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full recompute baseline (no cache reuse), timing only")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default="inputs/musique_s.json", help="Path to input dataset JSON")
    parser.add_argument("--output", type=str, default="results/full_kv_recompute_results", help="Directory to write results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of passages to include as context")
    parser.add_argument("--retrieval_json", type=str, default="retrieval_topk.json", help="Retrieval JSON filename")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cfg = load_config(args.config)
    samples = load_samples(args.input)

    model_name = cfg.get("model", {}).get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
    device_name = cfg.get("model", {}).get("device", "cuda:0")
    top_k = cfg.get("retrieval", {}).get("top_k", args.top_k)
    # prefer a 'generation' block; fall back to your old field if present
    max_new_tokens = cfg.get("generation", {}).get("max_new_tokens",
                        cfg.get("prefill", {}).get("query_prompt", {}).get("max_new_tokens", 32))

    # --- model & tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep the model on a single explicit device to avoid PKV/device surprises
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map=None,   # single device
    ).to(device).eval()
    
    # --- Load semantic similarity model ---
    semantic_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            print(f"\nLoading semantic similarity model...", flush=True)
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and accurate
            print(f"Semantic similarity model loaded successfully!", flush=True)
        except Exception as e:
            print(f"Warning: Could not load semantic model: {e}", flush=True)
            print(f"Falling back to token-level F1 only.", flush=True)

    # --- Setup retrieval ---
    retrieval_cfg = RetrievalConfig(**cfg.get("retrieval", {}))
    if not getattr(retrieval_cfg, "checkpoint", None):
        retrieval_cfg.checkpoint = getattr(retrieval_cfg, "model_id", "colbert-ir/colbertv2.0")
    retrieval = ColbertRetrieval(retrieval_cfg)
    
    # Prepare retrieval index once (loads corpus)
    print(f"\n{'='*80}", flush=True)
    print(f"Preparing retrieval index...", flush=True)
    print(f"{'='*80}\n", flush=True)
    retrieval.prepare(samples)
    
    # --- Per-sample retrieval + decode (full recompute) ---
    results: List[Dict[str, Any]] = []
    retrieval_results: List[Dict[str, Any]] = []
    print(f"\n{'='*80}", flush=True)
    print(f"Processing {len(samples)} samples with retrieval + full KV recompute...", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    for idx, sample in enumerate(samples):
        sid = sample.get("id", str(idx))
        try:
            # Retrieve for this sample
            retrieval.retrieve([sample], top_k=top_k)
            
            # Save retrieval results
            retrieval_results.append({
                "id": sample.get("id"),
                "retrieved_indices": sample.get("retrieved_indices", []),
                "retrieved_scores": sample.get("retrieved_scores", []),
            })
            
            # Build prompt and decode
            context, question = build_prompt_from_topk(sample, top_k)
            input_ids = encode_input(tokenizer, context, question)  # CPU tensor
            decode_result = decode_full_recompute(model, tokenizer, input_ids, max_new_tokens)
            
            # Compute metrics
            ground_truths = sample.get("answers", sample.get("answer", []))
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths]
            
            # Token-level F1
            f1_score = compute_max_f1(decode_result["answer"], ground_truths)
            
            # Semantic similarity (model-based)
            semantic_score = compute_max_semantic_similarity(
                decode_result["answer"], ground_truths, semantic_model
            )
            
            decode_result.update({
                "sample_id": sid,
                "f1_token": f1_score,  # Token-based F1
                "f1_semantic": semantic_score,  # Semantic similarity (0-1)
                "f1": semantic_score if semantic_model else f1_score,  # Primary metric
                "question": question,
                "ground_truth": ground_truths,
            })
            results.append(decode_result)
            
            # Print progress and answer
            print(f"[{idx+1}/{len(samples)}] Sample ID: {sid}", flush=True)
            print(f"  Retrieved: {len(sample.get('retrieved_indices', []))} documents (top-{top_k} used)", flush=True)
            print(f"  Question: {question[:100]}{'...' if len(question) > 100 else ''}", flush=True)
            print(f"  Ground Truth: {ground_truths}", flush=True)
            print(f"  Generated Answer: {decode_result['answer']}", flush=True)
            if semantic_model:
                print(f"  F1 (Semantic): {semantic_score:.4f} | F1 (Token): {f1_score:.4f}", flush=True)
            else:
                print(f"  F1 (Token): {f1_score:.4f}", flush=True)
            print(f"  TTFT: {decode_result['ttft']:.3f}s | TPOT: {decode_result['tpot']:.4f}s | E2E: {decode_result['e2e_latency']:.3f}s", flush=True)
            print(f"  Tokens: {decode_result['prompt_tokens']} prompt + {decode_result['generation_tokens']} generated", flush=True)
            print(f"{'-'*80}\n", flush=True)
        except Exception as e:
            print(f"[{idx+1}/{len(samples)}] Sample ID: {sid}", flush=True)
            print(f"  ❌ ERROR: {str(e)}", flush=True)
            print(f"{'-'*80}\n", flush=True)
            results.append(
                {
                    "sample_id": sid,
                    "answer": f"Error: {str(e)}",
                    "ttft": 0.0,
                    "e2e_latency": 0.0,
                    "throughput": 0.0,
                    "tpot": 0.0,
                    "prompt_tokens": 0,
                    "generation_tokens": 0,
                    "f1": 0.0,
                    "f1_token": 0.0,
                    "f1_semantic": 0.0,
                }
            )

    # --- summary row ---
    if results:
        n = len(results)
        def avg(key: str) -> float:
            return sum(r.get(key, 0.0) for r in results) / max(1, n)
        results.append({
            "sample_id": "average",
            "answer": "average_metrics",
            "ttft": avg("ttft"),
            "e2e_latency": avg("e2e_latency"),
            "throughput": avg("throughput"),
            "tpot": avg("tpot"),
            "prompt_tokens": avg("prompt_tokens"),
            "generation_tokens": avg("generation_tokens"),
            "f1": avg("f1"),
            "f1_token": avg("f1_token"),
            "f1_semantic": avg("f1_semantic"),
        })

    # Save results
    results_path = os.path.join(args.output, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save retrieval results
    retrieval_json_path = os.path.join(args.output, args.retrieval_json)
    with open(retrieval_json_path, "w") as f:
        json.dump(retrieval_results, f, indent=2)

    processed = len(results) - (1 if results and results[-1].get("sample_id") == "average" else 0)
    print(f"\n{'='*60}", flush=True)
    print(f"Completed processing {processed} samples.", flush=True)
    print(f"Results saved to {results_path}", flush=True)
    
    if results and results[-1].get("sample_id") == "average":
        avg_metrics = results[-1]
        print(f"\n{'='*60}", flush=True)
        print("AVERAGE METRICS:", flush=True)
        print(f"{'='*60}", flush=True)
        if semantic_model:
            print(f"  F1 (Semantic):     {avg_metrics['f1_semantic']:.4f}", flush=True)
            print(f"  F1 (Token):        {avg_metrics['f1_token']:.4f}", flush=True)
        else:
            print(f"  F1 (Token):        {avg_metrics['f1']:.4f}", flush=True)
        print(f"  TTFT:              {avg_metrics['ttft']:.4f} seconds", flush=True)
        print(f"  TPOT:              {avg_metrics['tpot']:.4f} seconds", flush=True)
        print(f"  E2E Latency:       {avg_metrics['e2e_latency']:.4f} seconds", flush=True)
        print(f"  Throughput:        {avg_metrics['throughput']:.2f} tokens/sec", flush=True)
        print(f"  Prompt Tokens:     {avg_metrics['prompt_tokens']:.0f}", flush=True)
        print(f"  Generation Tokens: {avg_metrics['generation_tokens']:.0f}", flush=True)
        print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
