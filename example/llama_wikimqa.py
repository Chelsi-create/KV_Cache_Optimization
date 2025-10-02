from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
import uuid
import time

# ===== Configuration and Initialization =====
eval_dataset = load_dataset("inputs/hotpotqa.json")
print("Field names of the first dataset sample:", eval_dataset[0].keys())  # Verify field structure

model_name = "meta-llama/Llama-3.1-70B"
output_path = Path("results/cache/hotpotqa_Llama-3.1-70B_compatible.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Initialize model
llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.5,
    max_num_seqs=32,
    enable_prefix_caching=True  # Explicitly enable prefix caching
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== Prompt Templates =====
prefix_template = (
    "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n{passages}"
)
query_template = "\n\nQuestion: {question}\nAnswer:"

results = {"blend": [], "full": []}

# ===== Core Logic: Utilize vllm's automatic caching for repeated prefixes =====
for ex_idx, ex in enumerate(eval_dataset):
    print(f"Processing example {ex_idx + 1}/{len(eval_dataset)}")
    answers = ex["answers"]  # Answer field
    question = normalize_question(ex["question"])  # Question field
    
    # Extract passages from ctxs
    passages_list = [ctx["text"] for ctx in ex["ctxs"]]
    passages = "\n".join([f"- {p}" for p in passages_list])

    # raw_passages = "\n".join([f"- {p}" for p in passages_list])
    # max_passage_tokens = 6000 
    # tokenized_passages = tokenizer(
    #     raw_passages,
    #     truncation=True,
    #     max_length=max_passage_tokens,
    #     return_tensors="pt"
    # )
    # passages = tokenizer.decode(tokenized_passages["input_ids"][0], skip_special_tokens=True)

    # Build prompts
    prefix_prompt = prefix_template.format(passages=passages)  # Prefix (document section)
    suffix_prompt = query_template.format(question=question)   # Suffix (question section)
    full_prompt = prefix_prompt + suffix_prompt               # Complete prompt

    # --------------------------
    # First run Full mode (no caching)
    # --------------------------
    # Clear possible cache
    dummy_prompt = "Ignore this: " + str(uuid.uuid4())
    llm.generate(dummy_prompt, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=None))
    
    # Full computation (no cache reuse)
    full_sampling = SamplingParams(temperature=0, max_tokens=32, prompt_logprobs=None)
    full_outputs = llm.generate(full_prompt, full_sampling)
    full_res = full_outputs[0].outputs[0].text.strip()

    # Calculate Full mode metrics
    full_metrics = full_outputs[0].metrics
    full_ttft = full_metrics.first_token_time - full_metrics.first_scheduled_time
    full_num_tokens = len(full_outputs[0].outputs[0].token_ids)
    full_tpot = (full_metrics.finished_time - full_metrics.first_token_time) / max(1, full_num_tokens)
    full_latency = full_metrics.finished_time - full_metrics.first_scheduled_time
    full_throughput = full_num_tokens / max(1e-6, full_latency)
    full_f1 = max([compute_f1(full_res, ans, tokenizer) for ans in answers])
    print("TTFT of full mode: ", full_ttft)

    # --------------------------
    # Then run Blend mode (with caching)
    # --------------------------
    # Warm up prefix: generate 1 token to trigger vllm's prefix KV cache
    warmup_sampling = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=None)
    llm.generate(prefix_prompt, warmup_sampling)  # Only for caching prefix, ignore result
    
    # Give engine a little time to process cache
    time.sleep(0.01)
    
    # Generate answer with cache reuse
    blend_sampling = SamplingParams(temperature=0, max_tokens=32, prompt_logprobs=None)
    blend_outputs = llm.generate(full_prompt, blend_sampling)
    blend_res = blend_outputs[0].outputs[0].text.strip()

    # Calculate Blend mode metrics
    blend_metrics = blend_outputs[0].metrics
    blend_ttft = blend_metrics.first_token_time - blend_metrics.first_scheduled_time
    blend_num_tokens = len(blend_outputs[0].outputs[0].token_ids)
    blend_tpot = (blend_metrics.finished_time - blend_metrics.first_token_time) / max(1, blend_num_tokens)
    blend_latency = blend_metrics.finished_time - blend_metrics.first_scheduled_time
    blend_throughput = blend_num_tokens / max(1e-6, blend_latency)
    blend_f1 = max([compute_f1(blend_res, ans, tokenizer) for ans in answers])
    print("TTFT of blend mode: ", blend_ttft)

    # Save results
    results["blend"].append({
        "answer": blend_res, "ttft": blend_ttft, "tpot": blend_tpot,
        "latency": blend_latency, "throughput": blend_throughput, "f1": blend_f1
    })
    results["full"].append({
        "answer": full_res, "ttft": full_ttft, "tpot": full_tpot,
        "latency": full_latency, "throughput": full_throughput, "f1": full_f1
    })

# Calculate averages and save
summary = {}
for mode in ["blend", "full"]:
    metrics = {k: [r[k] for r in results[mode]] for k in ["ttft", "tpot", "latency", "throughput", "f1"]}
    summary[mode] = {f"mean_{k}": float(np.mean(v)) for k, v in metrics.items()}

with open(output_path, "w") as f:
    json.dump({"per_sample": results, "summary": summary}, f, indent=2)

print(f"Results saved to {output_path}")
