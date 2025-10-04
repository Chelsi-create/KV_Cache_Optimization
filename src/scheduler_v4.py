#!/usr/bin/env python3

"""
OPTIMIZED Scheduler v4 - Fixed Performance Issues

Key optimizations:
1. Removed inefficient KV cache processing overhead
2. Simplified sparse attention logic
3. Reduced memory transfer frequency
4. Improved UCB algorithm efficiency
5. Eliminated redundant computations
"""

import time
import math
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Tuple
import yaml
import torch
import numpy as np

from build_kv_v2 import build_chunk_sequence

class OptimizedBanditScheduler:
    """
    OPTIMIZED scheduler with fixed performance issues:
    1. Efficient sparse attention handling
    2. Reduced transfer overhead
    3. Simplified prediction logic
    """
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        promote_per_step: Optional[int] = None,
        scheduler_interval: Optional[int] = None,
        exploration_c: float = 1.0,
        max_candidates: int = 15,
        sparsity_ratio: float = 1.0,  # 1.0 = full attention
        epsilon: float = 0.2,
    ):
        self.config_path = config_path
        self.promote_per_step = promote_per_step or 1
        self.scheduler_interval = scheduler_interval or 20  # Less frequent scheduling
        self.exploration_c = exploration_c
        self.max_candidates = max_candidates
        self.sparsity_ratio = sparsity_ratio
        self.epsilon = epsilon
        
        # Core state
        self.sample: Optional[Dict[str, Any]] = None
        self.device: Optional[torch.device] = None
        self.model = None
        self.tokenizer = None
        
        # OPTIMIZED: Simplified data structures
        self.gpu_chunks: Dict[int, Any] = {}
        self.cpu_chunks: Dict[int, str] = {}
        self.ready_kv: Dict[int, Any] = {}
        
        # OPTIMIZED: Simplified bandit state
        self.rewards: Dict[int, float] = {}
        self.counts: Dict[int, int] = {}
        self.last_used: Dict[int, int] = {}
        self.current_step: int = 0
        
        # GPU management
        self.max_gpu: int = 0
        self.current_gpu_order: List[int] = []
        self.last_promoted: List[int] = []
        self.last_demoted: List[int] = []
        
        # Load config defaults
        self._load_config_defaults()
    
    def get_sparsity_config(self) -> Dict[str, Any]:
        """Get current sparsity configuration"""
        return {
            'ratio': self.sparsity_ratio,
            'enabled': self.sparsity_ratio < 1.0
        }
    
    def initialize(
        self,
        sample: Dict[str, Any],
        gpu_chunks: Dict[int, Any],
        cpu_chunks: Dict[int, str],
        tokenizer: Any,
        model: Any,
        device: str,
        max_gpu: Optional[int] = None,
    ) -> None:
        self.sample = sample
        self.gpu_chunks = dict(gpu_chunks)
        self.cpu_chunks = dict(cpu_chunks)
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device)
        self.max_gpu = max_gpu if max_gpu is not None else len(gpu_chunks)
        self.current_gpu_order = list(gpu_chunks.keys())[:self.max_gpu]
        
        # Initialize rewards and counts with simple baseline
        for idx in set(gpu_chunks.keys()) | set(cpu_chunks.keys()):
            if idx in gpu_chunks:
                self.rewards[idx] = 1.0  # Higher baseline for GPU chunks
                self.counts[idx] = 1
            else:
                self.rewards[idx] = 0.5  # Lower baseline for CPU chunks
                self.counts[idx] = 1
            self.last_used[idx] = 0
        
        # OPTIMIZED: Pre-compute CPU chunks efficiently
        self._precompute_cpu_chunks_batch()
    
    def _precompute_cpu_chunks_batch(self):
        """Pre-compute CPU chunk KV caches in batches for efficiency"""
        print(f"Pre-computing KV caches for {len(self.cpu_chunks)} CPU chunks...")
        start_time = time.perf_counter()
        
        chunk_indices = [idx for idx in self.cpu_chunks.keys() if idx not in self.gpu_chunks]
        
        with torch.inference_mode():
            for idx in chunk_indices:
                text = self.cpu_chunks.get(idx, "")
                if text:
                    try:
                        input_ids = build_chunk_sequence(text, self.tokenizer)
                        current_input = torch.tensor([input_ids], device=self.device)
                        outputs = self.model(current_input, use_cache=True, return_dict=True)
                        kv = outputs.past_key_values
                        
                        if hasattr(kv, 'to_legacy_cache'):
                            kv = kv.to_legacy_cache()
                        
                        # Keep KV on CPU for later promotion
                        cpu_kv = []
                        for (k, v) in kv:
                            cpu_k = k.detach().cpu()
                            cpu_v = v.detach().cpu()
                            cpu_kv.append((cpu_k, cpu_v))
                        self.ready_kv[idx] = tuple(cpu_kv)
                    except Exception:
                        pass  # Skip failed chunks
        
        precompute_time = time.perf_counter() - start_time
        print(f"Pre-computation completed in {precompute_time:.3f}s, {len(self.ready_kv)} chunks ready")
    
    def predict(self, step: int, generated_tokens: List[int]) -> List[int]:
        """OPTIMIZED: Simplified prediction logic"""
        self.current_step = step - 5  # Adjust for lookahead
        
        if step < 8:  # Don't predict too early
            return []
        
        candidates = self._get_candidates()
        if not candidates:
            return []
        
        # Simple scoring based on rewards and recency
        scores = {}
        for idx in candidates:
            base_score = self.rewards.get(idx, 0.0)
            recency = math.exp(-0.1 * max(0, self.current_step - self.last_used.get(idx, -100)))
            scores[idx] = base_score + 0.2 * recency
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in ranked[:self.promote_per_step]]
        return selected
    
    def schedule_to_gpu(self) -> List[int]:
        """OPTIMIZED: Simplified scheduling with actual transfers"""
        if not self.ready_kv:
            return self.current_gpu_order
        
        # Calculate priorities
        ready_priorities = []
        current_priorities = {}
        
        for idx in self.ready_kv.keys():
            if idx not in self.current_gpu_order:
                priority = self._calculate_priority(idx)
                ready_priorities.append((idx, priority))
        
        for idx in self.current_gpu_order:
            current_priorities[idx] = self._calculate_priority(idx)
        
        if not ready_priorities:
            return self.current_gpu_order
        
        ready_priorities.sort(key=lambda x: x[1], reverse=True)
        
        promotions = []
        demotions = []
        new_gpu_order = list(self.current_gpu_order)
        
        # Implement transfers with epsilon-greedy exploration
        for idx, priority in ready_priorities[:self.promote_per_step]:
            if idx in self.ready_kv:
                force_explore = (np.random.random() < self.epsilon)
                
                # Find space by evicting lowest priority chunk if needed
                if len(new_gpu_order) >= self.max_gpu:
                    lowest_idx = min(new_gpu_order, key=lambda x: current_priorities.get(x, 0))
                    lowest_priority = current_priorities.get(lowest_idx, 0)
                    
                    should_swap = (priority > lowest_priority) or force_explore
                    
                    if should_swap:
                        # Actual demotion: GPU -> CPU
                        if self._demote_chunk_to_cpu(lowest_idx):
                            new_gpu_order.remove(lowest_idx)
                            demotions.append(lowest_idx)
                    else:
                        continue  # Don't promote if not beneficial
                
                # Actual promotion: CPU -> GPU
                if self._promote_chunk_to_gpu(idx):
                    new_gpu_order.append(idx)
                    promotions.append(idx)
        
        # Remove duplicates while preserving order
        seen = set()
        deduped_order = []
        for idx in new_gpu_order:
            if idx not in seen:
                seen.add(idx)
                deduped_order.append(idx)
        
        self.current_gpu_order = deduped_order
        self.last_promoted = promotions
        self.last_demoted = demotions
        
        if promotions or demotions:
            print(f"[Scheduler] promotions: {promotions}, demotions: {demotions}")
        
        return self.current_gpu_order
    
    def _promote_chunk_to_gpu(self, idx: int) -> bool:
        """Actual CPU->GPU tensor transfer"""
        try:
            if idx not in self.ready_kv:
                return False
            
            cpu_kv = self.ready_kv[idx]
            gpu_kv = []
            
            # Transfer to GPU efficiently
            with torch.cuda.device(self.device):
                for (k, v) in cpu_kv:
                    gpu_k = k.to(self.device, non_blocking=True)
                    gpu_v = v.to(self.device, non_blocking=True)
                    gpu_kv.append((gpu_k, gpu_v))
                
                torch.cuda.synchronize()
            
            self.gpu_chunks[idx] = tuple(gpu_kv)
            return True
            
        except Exception as e:
            print(f"[Transfer] Failed to promote chunk {idx}: {e}")
            return False
    
    def _demote_chunk_to_cpu(self, idx: int) -> bool:
        """Actual GPU->CPU tensor transfer"""
        try:
            if idx not in self.gpu_chunks:
                return False
            
            gpu_kv = self.gpu_chunks[idx]
            cpu_kv = []
            
            # Transfer to CPU
            for (k, v) in gpu_kv:
                cpu_k = k.detach().cpu()
                cpu_v = v.detach().cpu()
                cpu_kv.append((cpu_k, cpu_v))
            
            self.ready_kv[idx] = tuple(cpu_kv)
            del self.gpu_chunks[idx]
            
            torch.cuda.empty_cache()
            return True
            
        except Exception as e:
            print(f"[Transfer] Failed to demote chunk {idx}: {e}")
            return False
    
    def update_rewards(self, used_chunks: List[int], reward: float = 1.0) -> None:
        """OPTIMIZED: Simple reward updates"""
        current_time = self.current_step
        
        for idx in used_chunks:
            self.last_used[idx] = current_time
            
            # Simple moving average update
            old_reward = self.rewards.get(idx, 0.0)
            old_count = self.counts.get(idx, 0)
            
            self.counts[idx] = old_count + 1
            alpha = 1.0 / self.counts[idx]  # Decreasing learning rate
            self.rewards[idx] = (1 - alpha) * old_reward + alpha * reward
    
    def get_gpu_chunks(self) -> Dict[int, Any]:
        return self.gpu_chunks
    
    def get_scheduler_interval(self) -> int:
        return self.scheduler_interval
    
    def shutdown(self) -> None:
        """No-op for performance"""
        pass
    
    # OPTIMIZED: Simplified private methods
    def _load_config_defaults(self) -> None:
        """Load config with optimized defaults"""
        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                sch = cfg.get("scheduler", {})
                if "promote_per_step" in sch:
                    self.promote_per_step = int(sch["promote_per_step"])
                if "scheduler_interval" in sch:
                    self.scheduler_interval = int(sch["scheduler_interval"])
        except:
            pass  # Use defaults
    
    def _get_candidates(self) -> List[int]:
        """Get candidate chunks for scheduling"""
        candidates = list(self.ready_kv.keys())
        if len(candidates) > self.max_candidates:
            candidates = np.random.choice(candidates, self.max_candidates, replace=False).tolist()
        return candidates
    
    def _calculate_priority(self, idx: int) -> float:
        """OPTIMIZED: Simplified priority calculation"""
        base_reward = self.rewards.get(idx, 0.0)
        
        # Recency bonus
        current_time = self.current_step
        last_used = self.last_used.get(idx, -100)
        recency_factor = math.exp(-0.1 * max(0, current_time - last_used))
        
        # Exploration bonus
        exploration_bonus = 0.1 if idx not in self.gpu_chunks else 0.0
        
        return base_reward + 0.2 * recency_factor + exploration_bonus


class FastKVCacheManager:
    """OPTIMIZED: Fast KV cache operations"""
    
    @staticmethod
    def fast_concatenate_chunks(gpu_chunks: Dict[int, Any], selected_chunks: List[int]) -> Optional[Any]:
        """Fast KV cache concatenation with minimal overhead"""
        if not selected_chunks or not gpu_chunks:
            return None
        
        valid_caches = []
        for chunk_idx in selected_chunks:
            if chunk_idx in gpu_chunks:
                valid_caches.append(gpu_chunks[chunk_idx])
        
        if not valid_caches:
            return None
        
        num_layers = len(valid_caches[0])
        combined_kv = []
        
        with torch.inference_mode():
            for layer_idx in range(num_layers):
                keys_to_concat = []
                values_to_concat = []
                
                for kv_cache in valid_caches:
                    k, v = kv_cache[layer_idx]
                    if k.dim() == 3:
                        k = k.unsqueeze(0)
                        v = v.unsqueeze(0)
                    keys_to_concat.append(k)
                    values_to_concat.append(v)
                
                # Efficient concatenation
                merged_k = torch.cat(keys_to_concat, dim=2)
                merged_v = torch.cat(values_to_concat, dim=2)
                combined_kv.append((merged_k, merged_v))
        
        return tuple(combined_kv)