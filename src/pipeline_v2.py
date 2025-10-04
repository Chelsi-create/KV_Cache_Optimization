#!/usr/bin/env python3

"""
Minimalistic Pipeline for RAG with KV Cache Optimization
Uses build_kv_v2.py for chunk caching and per-step decoding with chunk prediction
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np

# Local imports
from rag_retrieval import RetrievalConfig, ColbertRetrieval
from build_kv_v2 import build_chunk_kv_caches, build_qa_prompt, QUERY_PROMPT
from scheduler_v2 import BanditScheduler

def convert_to_serializable(obj):
    """Convert NumPy/PyTorch types to JSON-serializable Python types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif hasattr(obj, 'item'):  # PyTorch scalars
        return obj.item()
    else:
        return obj

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging and return logger"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("pipeline")

def load_samples(path: str) -> List[Dict[str, Any]]:
    """Load dataset samples from JSON"""
    with open(path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "results" in data:
        return data["results"]
    return data

def concatenate_chunk_kv_caches(gpu_chunks: Dict[int, Any], selected_chunks: List[int]) -> Optional[Any]:
    """
    Concatenate selected chunk KV caches into single past_key_values
    Args:
        gpu_chunks: {chunk_idx: past_key_values}
        selected_chunks: List of chunk indices to concatenate
    Returns:
        Combined past_key_values or None
    """
    if not selected_chunks or not gpu_chunks:
        return None
    
    # Collect valid KV caches
    kv_entries = []
    for chunk_idx in selected_chunks:
        if chunk_idx in gpu_chunks:
            kv_entries.append(gpu_chunks[chunk_idx])
    
    if not kv_entries:
        return None
    
    # Get number of layers from first entry
    num_layers = len(kv_entries[0])
    
    # Concatenate layer by layer along sequence dimension
    combined_kv = []
    for layer_idx in range(num_layers):
        layer_keys = []
        layer_values = []
        
        for kv_cache in kv_entries:
            k, v = kv_cache[layer_idx]
            
            # Ensure [batch, heads, seq, dim] format
            if k.dim() == 3:  # [heads, seq, dim] -> [batch, heads, seq, dim]
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            
            layer_keys.append(k)
            layer_values.append(v)
        
        # Concatenate along sequence dimension (dim=2)
        merged_k = torch.cat(layer_keys, dim=2)
        merged_v = torch.cat(layer_values, dim=2)
        
        combined_kv.append((merged_k, merged_v))
    
    return tuple(combined_kv)

def _compute_chunk_boundaries_from_kv_lengths(
    gpu_chunks: Dict[int, Any], selected_chunks: List[int]
) -> List[tuple]:
    """
    Compute (start, end) token boundaries for each selected chunk when their
    per-chunk KV caches are concatenated along the sequence dimension.
    """
    boundaries: List[tuple] = []
    current = 0
    for chunk_idx in selected_chunks:
        if chunk_idx not in gpu_chunks:
            continue
        kv_cache = gpu_chunks[chunk_idx]
        try:
            key_tensor = kv_cache[0][0]
            chunk_len = int(key_tensor.shape[2])
        except Exception:
            chunk_len = 0
        start = current
        end = current + chunk_len
        boundaries.append((start, end))
        current = end
    return boundaries

def create_sparse_kv_cache_from_chunks(full_kv_cache, chunk_boundaries, decode_chunk_indices, prefill_chunk_indices):
    if not decode_chunk_indices:
        return full_kv_cache
    
    # Calculate which token positions to keep based on chunk boundaries
    keep_positions = []
    for decode_chunk_idx in decode_chunk_indices:
        if decode_chunk_idx < len(chunk_boundaries):
            start, end = chunk_boundaries[decode_chunk_idx]
            keep_positions.extend(range(start, end))
    
    if not keep_positions:
        return full_kv_cache
    
    # Sort positions to maintain order
    keep_positions = sorted(keep_positions)
    
    # Create sparse KV cache by selecting only the keep positions
    # CRITICAL: We keep the original positions to preserve RoPE encoding
    sparse_kv_cache = []
    for layer_idx, layer_kv in enumerate(full_kv_cache):
        keys, values = layer_kv
        
        # Check if keep_positions are within bounds
        max_seq_len = values.shape[2]
        invalid_positions = [pos for pos in keep_positions if pos >= max_seq_len]
        if invalid_positions:
            # Filter out invalid positions
            keep_positions = [pos for pos in keep_positions if pos < max_seq_len]
        
        if not keep_positions:
            return full_kv_cache
        
        # Select only the positions we want to keep
        sparse_keys = keys[:, :, keep_positions, :]
        sparse_values = values[:, :, keep_positions, :]
        sparse_kv_cache.append((sparse_keys, sparse_values))
    
    # Return the same type as the original cache to maintain compatibility
    if hasattr(full_kv_cache, '__class__'):
        # Try to create the same type of object
        try:
            return full_kv_cache.__class__(sparse_kv_cache)
        except:
            # Fallback to tuple if we can't recreate the original type
            return tuple(sparse_kv_cache)
    else:
        return tuple(sparse_kv_cache)


def generate_with_kv_optimization(
    sample: Dict[str, Any],
    gpu_chunks: Dict[int, Any],
    cpu_chunks: Dict[int, str],
    scheduler: BanditScheduler,  # Use the fixed scheduler
    model: Any,
    tokenizer: Any,
    device: str,
    max_tokens: int = 20
) -> Dict[str, Any]:

    # Metrics timers
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    step_durations: List[float] = []
    
    print("=== STEP 1: INITIAL CHUNK SELECTION ===")
    
    # Initialize scheduler
    scheduler.initialize(
        sample=sample,
        gpu_chunks=gpu_chunks,
        cpu_chunks=cpu_chunks,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_gpu=len(gpu_chunks)  # Same as initial
    )
    
    initial_chunks = list(gpu_chunks.keys())
    print(f"Initial GPU chunks: {initial_chunks}")
    
    print("\\n=== STEP 2: BUILD INITIAL COMBINED KV CACHE ===")
    
    # Build initial combined KV cache (no sparse pruning before first token)
    combined_kv = concatenate_chunk_kv_caches(gpu_chunks, initial_chunks)
    
    if combined_kv is None:
        print("No valid KV caches available")
        return {"error": "No KV caches"}
    
    chunk_seq_len = combined_kv[0][0].shape[2]
    print(f"Initial combined KV cache sequence length: {chunk_seq_len}")
    
    print("\\n=== STEP 3: PREPARE QUESTION ===")
    
    question = sample.get("question", "")
    formatted_question = QUERY_PROMPT + question
    question_ids = tokenizer.encode(formatted_question, add_special_tokens=False)
    question_input = torch.tensor([question_ids], device=device)
    question_length = len(question_ids)
    
    print(f"Question: {question}")
    print(f"Question tokens: {question_length}")
    
    print("\\n=== STEP 4: GENERATE FIRST TOKEN ===")
    
    # Create initial attention mask
    total_context_len = chunk_seq_len + question_length
    attention_mask = torch.ones(1, total_context_len, device=device)
    position_ids = torch.arange(chunk_seq_len, chunk_seq_len + question_length, device=device).unsqueeze(0)
    
    generated_tokens = []
    trace = []
    current_gpu_chunks = initial_chunks.copy()
    
    with torch.no_grad():
        # First forward pass (wrap legacy tuple KV as DynamicCache)
        initial_cache = DynamicCache.from_legacy_cache(combined_kv)
        outputs = model(
            input_ids=question_input,
            past_key_values=initial_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        
        past_key_values = outputs.past_key_values
        
        # Sample first token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.multinomial(
            torch.softmax(next_token_logits / 0.7, dim=-1),
            num_samples=1
        ).squeeze(-1)
        
        token_id = next_token.item()
        # Record TTFT as soon as the first token (including EOS) is determined
        first_token_time = time.perf_counter()
        if token_id != tokenizer.eos_token_id:
            generated_tokens.append(token_id)
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=True)
            print(f"First token: {token_id} = '{decoded_token}'")
        
        trace.append({
            "step": 0,
            "token": token_id,
            "decoded": tokenizer.decode([token_id], skip_special_tokens=True),
            "chunks_used": current_gpu_chunks,
            "context_length": total_context_len
        })
        
        print("\\n=== STEP 5: PER-STEP DECODING WITH DYNAMIC CHUNK UPDATES ===")
        
        for step in range(1, max_tokens):
            if token_id == tokenizer.eos_token_id:
                print(f"Step {step}: EOS token, stopping")
                break
            
            # STEP 5A: Predict new chunks using bandit
            predicted_chunks = scheduler.predict(step, generated_tokens)
            print(f"[Trace] T{step} predicted: {predicted_chunks}")
            
            # STEP 5B: Check if heavy scheduler should run
            print("Scheduler finally running")
            should_reschedule = scheduler.current_step % scheduler.get_scheduler_interval() == 0
            
            # Start per-token timer (includes scheduling overhead and model forward)
            step_start = time.perf_counter()
            
            if should_reschedule:
                print(f"\\n--- HEAVY SCHEDULING AT STEP {step} ---")
                
                # Get updated GPU chunks from scheduler
                new_gpu_order = scheduler.schedule_to_gpu()
                updated_gpu_chunks = scheduler.get_gpu_chunks()
                
                print(f"[Trace] T{step} promotions: {scheduler.last_promoted}, demotions: {scheduler.last_demoted}")
                
                # FIXED: Rebuild combined KV cache if GPU chunks changed
                if new_gpu_order != current_gpu_chunks:
                    print(f"GPU chunks changed: {current_gpu_chunks} -> {new_gpu_order}")
                    
                    try:
                        # Build new combined KV cache with updated chunks
                        new_combined_kv = concatenate_chunk_kv_caches(updated_gpu_chunks, new_gpu_order)
                        
                        if new_combined_kv is not None:
                            # Apply sparse pruning using predicted chunks while preserving positions
                            full_len = new_combined_kv[0][0].shape[2]
                            try:
                                boundaries = _compute_chunk_boundaries_from_kv_lengths(updated_gpu_chunks, new_gpu_order)
                                decode_set = set(predicted_chunks) if predicted_chunks else set(new_gpu_order)
                                decode_indices = [ci for ci in new_gpu_order if ci in decode_set]
                                if not decode_indices:
                                    decode_indices = new_gpu_order
                                pruned_kv = create_sparse_kv_cache_from_chunks(
                                    new_combined_kv,
                                    boundaries,
                                    decode_indices,
                                    new_gpu_order,
                                )
                                combined_kv = pruned_kv
                                pruned_len = combined_kv[0][0].shape[2]
                            except Exception as e:
                                print(f"Sparse pruning failed, using full KV. Error: {e}")
                                combined_kv = new_combined_kv
                                pruned_len = combined_kv[0][0].shape[2]

                            # Keep rope base on full_len even if pruned
                            chunk_seq_len = full_len
                            current_gpu_chunks = new_gpu_order
                            
                            print(f"Rebuilt combined KV cache, full seq len: {full_len}, pruned len: {pruned_len}")
                            
                            # FIXED: Update past_key_values to use new combined cache
                            # This is crucial - we need to restart the attention from the new combined cache
                            
                            # Prepare question input again for the new context
                            # Attention mask must reflect pruned cache length + question tokens
                            new_total_context_mask = pruned_len + question_length
                            attention_mask = torch.ones(1, new_total_context_mask, device=device)
                            position_ids = torch.arange(chunk_seq_len, chunk_seq_len + question_length, device=device).unsqueeze(0)
                            
                            # Re-process question with new combined KV cache (wrap as DynamicCache)
                            rebuilt_cache = DynamicCache.from_legacy_cache(combined_kv)
                            question_outputs = model(
                                input_ids=question_input,
                                past_key_values=rebuilt_cache,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=True,
                            return_dict=True)
                            
                            # FIXED: Generate all previous tokens again with new context
                            # This ensures consistency with the new chunk combination
                            if generated_tokens:
                                prev_tokens_input = torch.tensor([generated_tokens], device=device)
                                prev_outputs = model(
                                    input_ids=prev_tokens_input,
                                    past_key_values=question_outputs.past_key_values,
                                    use_cache=True,
                                    return_dict=True
                                )
                                past_key_values = prev_outputs.past_key_values
                            else:
                                past_key_values = question_outputs.past_key_values
                                
                            print(f"Updated past_key_values for new chunk combination")
                            
                        else:
                            print("Failed to rebuild combined KV cache, keeping old cache")
                            
                    except Exception as e:
                        print(f"Error rebuilding KV cache: {e}")
                        print("Continuing with old cache")
                else:
                    print("No GPU chunk changes, keeping current cache")
            
            # STEP 5C: Generate next token (with potentially updated context)
            input_token = next_token.unsqueeze(-1)
            
            outputs = model(
                input_ids=input_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            
            # Sample next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(
                torch.softmax(next_token_logits / 0.7, dim=-1),
                num_samples=1
            ).squeeze(-1)
            
            token_id = next_token.item()
            
            if token_id == tokenizer.eos_token_id:
                print(f"Step {step}: EOS token, stopping")
                break
            
            generated_tokens.append(token_id)
            # Stop per-token timer once a non-EOS token is produced
            step_end = time.perf_counter()
            step_durations.append(step_end - step_start)
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=True)
            print(f"Step {step}: Token {token_id} = '{decoded_token}', Chunks: {current_gpu_chunks}")
            
            # STEP 5D: Update rewards based on generation quality
            # Simple reward: successful token generation = 1.0
            reward = 1.0 if token_id != tokenizer.eos_token_id else 0.5
            scheduler.update_rewards(current_gpu_chunks, reward)
            
            trace.append({
                "step": step,
                "token": token_id,
                "decoded": decoded_token,
                "chunks_used": current_gpu_chunks.copy(),
                "context_length": past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else chunk_seq_len + len(generated_tokens),
                "predicted_chunks": predicted_chunks,
                "promotions": scheduler.last_promoted.copy() if should_reschedule else [],
                "demotions": scheduler.last_demoted.copy() if should_reschedule else []
            })
    
    # Cleanup
    scheduler.shutdown()
    
    # Final results
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    end_time = time.perf_counter()
    
    # Compute metrics
    ttft_ms: Optional[float] = None
    ttft_s: Optional[float] = None
    if first_token_time is not None:
        ttft_s = (first_token_time - start_time)
        ttft_ms = ttft_s * 1000.0
    decode_time_s = sum(step_durations) if step_durations else 0.0
    tpot_ms: Optional[float] = None
    tpot_s: Optional[float] = None
    if step_durations:
        tpot_s = (decode_time_s / len(step_durations))
        tpot_ms = tpot_s * 1000.0
    e2e_s = (end_time - start_time)
    e2e_ms = e2e_s * 1000.0
    throughput_toks_per_s: Optional[float] = None
    if decode_time_s > 0.0:
        throughput_toks_per_s = len(step_durations) / decode_time_s
    
    print(f"\\n=== FINAL RESULTS ===")
    print(f"Generated text: '{generated_text}'")
    print(f"Total tokens generated: {len(generated_tokens)}")
    print(f"Final GPU chunks: {current_gpu_chunks}")
    if ttft_s is not None:
        print(f"TTFT (s): {ttft_s:.4f}")
    else:
        print("TTFT (s): n/a")
    if tpot_s is not None:
        print(f"TPOT avg (s/token): {tpot_s:.4f}")
    else:
        print("TPOT avg (s/token): n/a")
    print(f"E2E latency (s): {e2e_s:.4f}")
    if throughput_toks_per_s is not None:
        print(f"Throughput (tok/s): {throughput_toks_per_s:.2f}")
    else:
        print("Throughput (tok/s): n/a")
    
    return {
        "question": question,
        "answer": generated_text,
        "generated_tokens": generated_tokens,
        "trace": trace,
        "final_gpu_chunks": current_gpu_chunks,
        "total_promotions": len([t for t in trace if t.get("promotions")]),
        "total_demotions": len([t for t in trace if t.get("demotions")]),
        "metrics": {
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "e2e_s": e2e_s,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "throughput_toks_per_s": throughput_toks_per_s,
            "num_output_tokens": len(generated_tokens)
        }
    }

def run_pipeline(
    input_file: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "results",
    top_k: int = 5,
    max_tokens: int = 20,
    device: str = "cuda:0"
):
    
    logger = setup_logging()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== PIPELINE START ===")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Top-k chunks: {top_k}")
    
    # 1. Load dataset
    logger.info(f"Loading dataset: {input_file}")
    samples = load_samples(input_file)
    logger.info(f"Loaded {len(samples)} samples")
    
    # 2. Setup retrieval (per-sample to enable sample-by-sample execution)
    retrieval_config = RetrievalConfig()
    retriever = ColbertRetrieval(retrieval_config)

    # 3. Initialize shared tokenizer/model once
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and getattr(tokenizer, 'eos_token', None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(torch.device(device)).eval()

    # 4. Setup scheduler
    scheduler = BanditScheduler()

    results: List[Dict[str, Any]] = []
    for si, sample in enumerate(tqdm(samples, desc="Samples", unit="sample")):
        # --- Retrieval per-sample ---
        prepared_list = retriever.prepare([sample])
        retrieved_list = retriever.retrieve(prepared_list, top_k=top_k)
        sample = retrieved_list[0] if retrieved_list else sample
        samples[si] = sample

        # Build per-sample KV caches using provided tokenizer/model (avoids reloads)
        kv = build_chunk_kv_caches(
            samples=[sample],
            model_id=model_id,
            top_k=top_k,
            device=device,
            provided_tokenizer=tokenizer,
            provided_model=model,
        )
        gpu_chunks = kv.get("gpu_chunks", {})
        cpu_chunks = kv.get("cpu_chunks", {})

        if not gpu_chunks and not cpu_chunks:
            logger.warning(f"[Sample {si}] No chunks available after KV build; skipping")
            continue

        out = generate_with_kv_optimization(
            sample=sample,
            gpu_chunks=gpu_chunks,
            cpu_chunks=cpu_chunks,
            scheduler=scheduler,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_tokens=max_tokens,
        )
        sample_id = str(sample.get("id", f"sample_{si}"))
        out["sample_index"] = si
        out["sample_id"] = sample_id
        results.append(out)

    # 5. Save results
    out_path = os.path.join(output_dir, "result.json")
    with open(out_path, "w") as f:
        serializable_results = convert_to_serializable({"results": results})
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved results to {out_path}")

    return {"output_path": out_path, "count": len(results)}


def main():
    import argparse
    ap = argparse.ArgumentParser("Minimal RAG + KV pipeline v2")
    ap.add_argument("--input", required=True, help="Path to input dataset JSON")
    ap.add_argument("--output_dir", default="results", help="Output directory")
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.2", help="HF model id")
    ap.add_argument("--top_k", type=int, default=5, help="Top-k passages to cache on GPU")
    ap.add_argument("--max_tokens", type=int, default=20, help="Max new tokens to generate")
    ap.add_argument("--device", default="cuda:0", help="Device like cuda:0 or cpu")
    args = ap.parse_args()

    run_pipeline(
        input_file=args.input,
        model_id=args.model_id,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        device=args.device,
    )

if __name__ == "__main__":
    main()