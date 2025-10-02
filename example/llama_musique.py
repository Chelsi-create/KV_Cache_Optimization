from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
import uuid
import time

# ===== 配置与初始化 =====
eval_dataset = load_dataset("inputs/hotpotqa.json")
print("数据集第一个样本的字段名:", eval_dataset[0].keys())  # 确认字段结构

model_name = "meta-llama/Llama-3.1-70B"
output_path = Path("results/cache/hotpotqa_Llama-3.1-70B_compatible.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# 初始化模型
llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.5,
    max_num_seqs=32,
    enable_prefix_caching=True  # 显式启用前缀缓存
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== Prompt模板 =====
prefix_template = (
    "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n{passages}"
)
query_template = "\n\nQuestion: {question}\nAnswer:"

results = {"blend": [], "full": []}

# ===== 核心逻辑：利用vllm自动缓存重复前缀 =====
for ex_idx, ex in enumerate(eval_dataset):
    print(f"Processing example {ex_idx + 1}/{len(eval_dataset)}")
    answers = ex["answers"]  # 答案字段
    question = normalize_question(ex["question"])  # 问题字段
    
    # 从ctxs提取段落
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

    # 构建提示词
    prefix_prompt = prefix_template.format(passages=passages)  # 前缀（文档部分）
    suffix_prompt = query_template.format(question=question)   # 后缀（问题部分）
    full_prompt = prefix_prompt + suffix_prompt               # 完整提示词

    # --------------------------
    # 先运行Full模式（无缓存）
    # --------------------------
    # 清除可能的缓存
    dummy_prompt = "Ignore this: " + str(uuid.uuid4())
    llm.generate(dummy_prompt, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=None))
    
    # 全量计算（无缓存复用）
    full_sampling = SamplingParams(temperature=0, max_tokens=32, prompt_logprobs=None)
    full_outputs = llm.generate(full_prompt, full_sampling)
    full_res = full_outputs[0].outputs[0].text.strip()

    # 计算Full模式指标
    full_metrics = full_outputs[0].metrics
    full_ttft = full_metrics.first_token_time - full_metrics.first_scheduled_time
    full_num_tokens = len(full_outputs[0].outputs[0].token_ids)
    full_tpot = (full_metrics.finished_time - full_metrics.first_token_time) / max(1, full_num_tokens)
    full_latency = full_metrics.finished_time - full_metrics.first_scheduled_time
    full_throughput = full_num_tokens / max(1e-6, full_latency)
    full_f1 = max([compute_f1(full_res, ans, tokenizer) for ans in answers])
    print("ttft of full : ", full_ttft)

    # --------------------------
    # 再运行Blend模式（使用缓存）
    # --------------------------
    # 预热前缀：生成1个token触发vllm缓存前缀KV对
    warmup_sampling = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=None)
    llm.generate(prefix_prompt, warmup_sampling)  # 仅用于缓存前缀，结果忽略
    
    # 给引擎一点时间处理缓存
    time.sleep(0.01)
    
    # 复用缓存生成答案
    blend_sampling = SamplingParams(temperature=0, max_tokens=32, prompt_logprobs=None)
    blend_outputs = llm.generate(full_prompt, blend_sampling)
    blend_res = blend_outputs[0].outputs[0].text.strip()

    # 计算Blend模式指标
    blend_metrics = blend_outputs[0].metrics
    blend_ttft = blend_metrics.first_token_time - blend_metrics.first_scheduled_time
    blend_num_tokens = len(blend_outputs[0].outputs[0].token_ids)
    blend_tpot = (blend_metrics.finished_time - blend_metrics.first_token_time) / max(1, blend_num_tokens)
    blend_latency = blend_metrics.finished_time - blend_metrics.first_scheduled_time
    blend_throughput = blend_num_tokens / max(1e-6, blend_latency)
    blend_f1 = max([compute_f1(blend_res, ans, tokenizer) for ans in answers])
    print("ttft of blend : ", blend_ttft)

    # 保存结果
    results["blend"].append({
        "answer": blend_res, "ttft": blend_ttft, "tpot": blend_tpot,
        "latency": blend_latency, "throughput": blend_throughput, "f1": blend_f1
    })
    results["full"].append({
        "answer": full_res, "ttft": full_ttft, "tpot": full_tpot,
        "latency": full_latency, "throughput": full_throughput, "f1": full_f1
    })

# 计算平均值并保存
summary = {}
for mode in ["blend", "full"]:
    metrics = {k: [r[k] for r in results[mode]] for k in ["ttft", "tpot", "latency", "throughput", "f1"]}
    summary[mode] = {f"mean_{k}": float(np.mean(v)) for k, v in metrics.items()}

with open(output_path, "w") as f:
    json.dump({"per_sample": results, "summary": summary}, f, indent=2)

print(f"Results saved to {output_path}")