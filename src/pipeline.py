#!/usr/bin/env python3
import os
import json
import time
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
from threading import Thread
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# --- local modules ---
from rag_retrieval import RetrievalConfig, ColbertRetrieval
from build_kv_cache import KVCachesBuilder, _tokenize_chunk, _prefill_get_past
from scheduler import TangoScheduler


# ---------------- utilities ----------------

def read_json(path: str) -> Any:
    """Read a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    """Write an object to JSON, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_samples(path: str) -> List[Dict[str, Any]]:
    """Load dataset samples from JSON supporting several common shapes."""
    data = read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return [data]


def abs_path(base_dir: str, maybe_path: str) -> str:
    """Resolve a possibly relative path against a base directory."""
    if not maybe_path:
        return ""
    return maybe_path if os.path.isabs(maybe_path) else os.path.join(base_dir, maybe_path)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure module logging and return the pipeline logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("pipeline_full_kv_reuse_fixed")


def load_existing_retrieval(retrieval_json_path: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """If an existing retrieval JSON exists, load and return its samples."""
    if not retrieval_json_path or not os.path.exists(retrieval_json_path):
        return None
    try:
        data = read_json(retrieval_json_path)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "results" in data:
            samples = data["results"]
        else:
            samples = [data]
        if not isinstance(samples, list) or not samples:
            return None
        logger.info(f"[Retrieval] Found existing retrieval file. Skipping retrieval: {retrieval_json_path}")
        logger.info(f"[Retrieval] Loaded {len(samples)} samples from existing retrieval JSON")
        return samples
    except Exception as e:
        logger.warning(f"[Retrieval] Failed to read existing retrieval JSON ({retrieval_json_path}): {e}")
        return None

def topk_indices_from_sample(sample: Dict[str, Any], cfg: Dict[str, Any]) -> List[int]:
    """Return the numeric indices for the top‑k passages we want on GPU initially."""
    rkey = cfg["retrieval"]["retrieved_key"]
    retrieved = sample.get(rkey, []) or []
    max_gpu = int(cfg["cache"]["max_gpu_chunks"])
    return [int(i) for i in (retrieved[:max_gpu] if isinstance(retrieved, list) else [])]

# ---------------- main pipeline ----------------

def main():
    """CORRECTED: Main pipeline with proper retrieval integration"""
    ap = argparse.ArgumentParser("Full KV Reuse Pipeline WITH speculative scheduler (fixed)")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    ap.add_argument("--input", default="inputs/musique_s.json", help="Path to input dataset JSON")
    ap.add_argument("--output", default="results/pipeline_results", help="Output directory")
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    args = ap.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logging(args.log_level)
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_id = cfg["model"]["model_name"]
    device = cfg["model"].get("device", "cuda:0")
    
    logger.info(f"[Config] Model: {model_id} | Device: {device}")
    logger.info(f"[Paths] Output root: {args.output}")
    
    # --- Load dataset ---
    logger.info(f"[Input] Loading dataset: {args.input}")
    samples = load_samples(args.input)
    logger.info(f"[Input] Loaded {len(samples)} samples")
    
    # --- Output paths ---
    retrieval_json_path = abs_path(args.output, cfg["paths"]["retrieval_json"])
    summary_path = os.path.join(args.output, "summary.json")
    per_step_debug_path = os.path.join(args.output, "per_step_debug.json")
    sched_out_path = abs_path(args.output, cfg["scheduler"].get("out_path", "scheduler/decoder_trace.json"))
    
    # --- CRITICAL: Check for existing retrieval first ---
    maybe_existing = load_existing_retrieval(retrieval_json_path, logger)
    if maybe_existing is not None:
        samples = maybe_existing
    
    # --- Retrieval (skip if JSON already exists) ---
    retriever = None
    if maybe_existing is None:
        r_conf = RetrievalConfig(
            model_id=cfg["retrieval"]["model_id"],
            dataset_name=cfg["retrieval"]["dataset_name"],
            r_text_index_key=cfg["retrieval"]["r_text_index_key"],
            doc_key=cfg["retrieval"]["doc_key"],
            question_key=cfg["retrieval"]["question_key"],
            retrieved_key=cfg["retrieval"]["retrieved_key"],
            page_id_key=cfg["retrieval"]["page_id_key"],
            top_k=int(cfg["retrieval"]["top_k"])
        )
        retriever = ColbertRetrieval(r_conf)
    
    # --- Initialize shared components ---
    scheduler = TangoScheduler()
    shared_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if shared_tokenizer.pad_token is None and getattr(shared_tokenizer, 'eos_token', None) is not None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token
    shared_model = AutoModelForCausalLM.from_pretrained(model_id).to(torch.device(device)).eval()
    
    logger.info("[Pipeline] Running full_kv_reuse + speculative + scheduler (fixed)")
    
    final_answers = []
    per_step_debug_all = []
    sched_trace_all = []
    
    for si in tqdm(range(1), desc="Samples", unit="sample"):  # Process 1 sample for now
        sample = samples[si]
        sample_id = str(sample.get("id", f"sample_{si}"))
        
        # --- 1. RETRIEVAL (if not pre-loaded) ---
        if maybe_existing is None:
            logger.info(f"[Sample {si}] Running retrieval...")
            enriched = retriever.prepare([sample])
            enriched = retriever.retrieve(enriched)
            sample = enriched[0]
            samples[si] = sample
            
            # Flush retrieval results after each sample (optional - for incremental saving)
            write_json(retrieval_json_path, samples)
        
        # --- 2. Get initial GPU indices from retrieval results ---
        topk_indices = topk_indices_from_sample(sample, cfg)
        gpu_indices_initial = sorted(set(int(i) for i in topk_indices))
        
        logger.info(f"[Sample {si}] Initial GPU chunks from retrieval: {gpu_indices_initial}")
        
        # --- 3. FIXED: Per-step decoding with proper retrieval data ---
        per_step_result = scheduler.run_per_step_decode(
            retr_samples=[sample],  # Pass the sample WITH retrieval results
            model_id=model_id,
            device=device,
            dtype=str(cfg["model"].get("dtype", "auto")),
            max_gpu=int(cfg["scheduler"]["max_gpu"]),
            max_samples=1,
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 10),
            scheduler_interval=int(cfg["scheduler"].get("scheduler_interval", 5)),
            provided_tokenizer=shared_tokenizer,
            provided_model=shared_model,
            promote_per_step=int(cfg["scheduler"].get("promote_per_step", 2)),
            initial_gpu_indices=gpu_indices_initial,
            use_original_decode=False
        )
        
        # --- 4. Extract and log results ---
        if per_step_result.get("results") and len(per_step_result["results"]) > 0:
            decode_metrics_final = per_step_result["results"][0]
            
            logger.info(f"[Sample {si}] Per-step decoding completed:")
            logger.info(f"  - Question: {sample.get('question', '')}")
            logger.info(f"  - Answer: {decode_metrics_final.get('answer', '')}")
            logger.info(f"  - Generated tokens: {decode_metrics_final.get('decoded_tokens', 0)}")
            logger.info(f"  - TTFT: {decode_metrics_final.get('ttft', 0):.3f}s")
            logger.info(f"  - E2E latency: {decode_metrics_final.get('e2e_latency', 0):.3f}s")
            logger.info(f"  - TPOT: {decode_metrics_final.get('tpot', 0):.3f}s")
            logger.info(f"  - Throughput: {decode_metrics_final.get('throughput', 0):.2f} tokens/s")
            
            # Collect debug information
            if "trace" in decode_metrics_final:
                sample_debug = {
                    "sample_index": si,
                    "sample_id": sample_id,
                    "question": sample.get("question", ""),
                    "answer": decode_metrics_final.get("answer", ""),
                    "metrics": {
                        "ttft": decode_metrics_final.get("ttft", 0),
                        "e2e_latency": decode_metrics_final.get("e2e_latency", 0),
                        "throughput": decode_metrics_final.get("throughput", 0),
                        "decoded_tokens": decode_metrics_final.get("decoded_tokens", 0)
                    },
                    "per_token_trace": decode_metrics_final["trace"]
                }
                per_step_debug_all.append(sample_debug)
                
                # Update trace with sample index
                for row in decode_metrics_final["trace"]:
                    row["sample_index"] = si
                sched_trace_all.extend(decode_metrics_final["trace"])
            
            final_answers.append({
                "sample_index": si,
                "question": sample.get("question", ""),
                "mode": "full_kv_reuse_per_step",
                "gpu_indices_initial": gpu_indices_initial,
                **decode_metrics_final
            })
        else:
            logger.warning(f"[Sample {si}] Per-step decoding failed")
            # Could add fallback decoding here if needed
    
    # --- 5. Final output and summary ---
    if maybe_existing is None:
        write_json(retrieval_json_path, samples)
        logger.info(f"[Output] Retrieval JSON: {retrieval_json_path}")
    
    # Calculate averages
    ttfts = [x["ttft"] for x in final_answers if x.get("ttft") is not None]
    e2es = [x["e2e_latency"] for x in final_answers if x.get("e2e_latency") is not None]
    thr = [x["throughput"] for x in final_answers if x.get("throughput") is not None]
    tpot = [x["tpot"] for x in final_answers if x.get("tpot") is not None]
    
    write_json(summary_path, {
        "status": "completed",
        "model": model_id,
        "device": device,
        "count": len(final_answers),
        "averages": {
            "ttft": sum(ttfts) / len(ttfts) if ttfts else None,
            "e2e_latency": sum(e2es) / len(e2es) if e2es else None,
            "throughput": sum(thr) / len(thr) if thr else None,
            "tpot": sum(tpot) / len(tpot) if tpot else None,
        },
        "results": final_answers,
        "timestamps": {
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    })
    
    if per_step_debug_all:
        write_json(per_step_debug_path, {"per_step_debug": per_step_debug_all})
        logger.info(f"[pipeline] Per-step debug info saved to: {per_step_debug_path}")
    
    if sched_trace_all:
        write_json(sched_out_path, {"trace": sched_trace_all})
        logger.info(f"[pipeline] Scheduler trace saved to: {sched_out_path}")
    
    logger.info(f"[pipeline] Done. Summary: {summary_path}")

if __name__ == "__main__":
    main()
