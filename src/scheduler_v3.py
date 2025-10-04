#!/usr/bin/env python3

"""
FIXED HIGH-PERFORMANCE OPTIMIZED Scheduler for Minimal Latency
Addresses all critical performance bottlenecks + the 3 major issues
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

class HighPerformanceBanditScheduler:
    """
    FIXED OPTIMIZED scheduler addressing:
    1. 5-step lookahead prediction
    2. Actual CPU<->GPU transfers
    3. Improved UCB algorithm
    """
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        promote_per_step: Optional[int] = None,
        scheduler_interval: Optional[int] = None,
        exploration_c: float = 1.0,
        max_candidates: int = 20,
        enable_background: bool = False,
        sparsity_ratio: float = 0.3,
        enable_sparsity: bool = True,
        sparsity_strategy: str = "priority",
        epsilon: float = 0.3,  # Epsilon-greedy exploration rate
    ):
        self.config_path = config_path
        self.promote_per_step = promote_per_step
        self.scheduler_interval = scheduler_interval
        self.exploration_c = exploration_c
        self.max_candidates = max_candidates
        self.enable_background = enable_background

        # Core state
        self.sample: Optional[Dict[str, Any]] = None
        self.device: Optional[torch.device] = None
        self.model = None
        self.tokenizer = None

        # OPTIMIZED: Pre-allocated data structures
        self.gpu_chunks: Dict[int, Any] = {}
        self.cpu_chunks: Dict[int, str] = {}
        self.ready_kv: Dict[int, Any] = {}  # Pre-computed KV caches

        # FIXED: Improved bandit state with consistent time scales
        self.rewards: Dict[int, float] = {}
        self.counts: Dict[int, float] = {}  # Changed to float for EMA
        self.last_used: Dict[int, int] = {}  # Track when chunk was last used
        self.time_weights: Dict[int, float] = {}  # Exponential decay weights
        self.current_step: int = 0

        # EMA parameters for consistent time scales
        self.reward_ema_alpha: float = 0.2
        self.count_ema_alpha: float = 0.1

        # OPTIMIZED: Fixed GPU management
        self.max_gpu: int = 0
        self.current_gpu_order: List[int] = []
        self.last_promoted: List[int] = []
        self.last_demoted: List[int] = []

        # OPTIMIZED: Disable background threading for minimal latency
        self._stop_background_work = True

        # Config load
        self._load_config_defaults()

        self.sparsity_ratio = sparsity_ratio
        self.enable_sparsity = enable_sparsity
        self.sparsity_strategy = sparsity_strategy
        self.epsilon = epsilon  # Exploration rate

    def get_sparsity_config(self) -> Dict[str, Any]:
        """Get current sparsity configuration"""
        return {
            'ratio': self.sparsity_ratio,
            'enabled': self.enable_sparsity,
            'strategy': self.sparsity_strategy
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

        # FIXED: Initialize rewards and counts with proper baseline
        for idx in set(gpu_chunks.keys()) | set(cpu_chunks.keys()):
            if idx in gpu_chunks:
                self.rewards[idx] = 0.5  # Neutral baseline for GPU chunks
                self.counts[idx] = 1.0   # Start with EMA count
            else:
                self.rewards[idx] = 0.1  # Lower baseline for CPU chunks
                self.counts[idx] = 0.5   # Lower initial count for exploration
            
            self.last_used[idx] = 0
            self.time_weights[idx] = 1.0

        # OPTIMIZED: Pre-compute ALL CPU chunk KV caches at initialization
        if not self.enable_background:
            self._precompute_all_cpu_chunks()

    def _precompute_all_cpu_chunks(self):
        """Pre-compute all CPU chunk KV caches upfront"""
        print(f"Pre-computing KV caches for {len(self.cpu_chunks)} CPU chunks...")
        start_time = time.perf_counter()
        
        chunk_indices = list(self.cpu_chunks.keys())
        with torch.inference_mode():
            for i in range(0, len(chunk_indices), 4):  # Process in batches
                batch_indices = chunk_indices[i:i+4]
                for idx in batch_indices:
                    if idx not in self.gpu_chunks:  # Skip already on GPU
                        text = self.cpu_chunks.get(idx, "")
                        if text:
                            try:
                                input_ids = build_chunk_sequence(text, self.tokenizer)
                                current_input = torch.tensor([input_ids], device=self.device)
                                outputs = self.model(current_input, use_cache=True, return_dict=True)
                                kv = outputs.past_key_values
                                
                                if hasattr(kv, 'to_legacy_cache'):
                                    kv = kv.to_legacy_cache()
                                
                                # Move KV tensors to CPU for later promotion
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
        """FIXED: Predict chunks needed for FUTURE step (5 steps ahead)"""
        self.current_step = step - 5  # Adjust for lookahead (step represents future step)
        
        # Don't predict too early
        if step < 8:  # 3 + 5 step lookahead
            return []
        
        candidates = self._fast_candidate_pool()
        if not candidates:
            return []
        
        # FIXED: Use predictive scoring for future step
        scores = self._predictive_ucb_scores(candidates, step)
        ranked = sorted(scores.items(), key=lambda x: x, reverse=True)
        selected = [idx for idx, _ in ranked[:self.promote_per_step]]
        
        return selected

    def prepare_chunks_async(self, chunk_indices: List[int]) -> None:
        """Prepare predicted chunks for future use"""
        for idx in chunk_indices:
            if idx in self.ready_kv and idx not in self.gpu_chunks:
                # Mark as high priority for next scheduling cycle
                self.rewards[idx] = self.rewards.get(idx, 0.0) + 0.2  # Prediction bonus

    def schedule_to_gpu(self) -> List[int]:
        """FIXED: Implement actual CPU<->GPU tensor transfers"""
        
        print(f"[Scheduler] schedule_to_gpu called: ready_kv has {len(self.ready_kv)} chunks, current_gpu has {len(self.current_gpu_order)} chunks")
        
        if not self.ready_kv:
            print("[Scheduler] schedule_to_gpu: no ready_kv; keeping current order")
            return self.current_gpu_order

        # Calculate priorities for both ready and current GPU chunks
        ready_priorities = []
        current_gpu_priorities = {}
        
        for idx in self.ready_kv.keys():
            if idx not in self.current_gpu_order:
                priority = self._fast_priority(idx)
                ready_priorities.append((idx, priority))
        
        for idx in self.current_gpu_order:
            current_gpu_priorities[idx] = self._fast_priority(idx)
        
        if not ready_priorities:
            print(f"[Scheduler] schedule_to_gpu: no eligible CPU candidates (all {len(self.ready_kv)} chunks already on GPU); keeping current order")
            return self.current_gpu_order

        ready_priorities.sort(key=lambda x: x[1], reverse=True)
        print(f"[Scheduler] Found {len(ready_priorities)} CPU candidates with priorities")
        
        promotions = []
        demotions = []
        new_gpu_order = list(self.current_gpu_order)
        
        print(f"[Scheduler] Current GPU: {self.current_gpu_order}, max_gpu: {self.max_gpu}")
        print(f"[Scheduler] Top {self.promote_per_step} CPU candidates: {ready_priorities[:self.promote_per_step]}")
        print(f"[Scheduler] Current GPU priorities: {[(idx, f'{current_gpu_priorities[idx]:.3f}') for idx in self.current_gpu_order[:3]]}")
        
        # FIXED: Implement actual CPU->GPU transfers with intelligent eviction
        for idx, priority in ready_priorities[:self.promote_per_step]:
            if idx in self.ready_kv:
                # FIXED: Epsilon-greedy exploration - sometimes force exploration
                force_explore = (np.random.random() < self.epsilon)
                
                # Find space by evicting lowest priority chunk if needed
                if len(new_gpu_order) >= self.max_gpu:
                    lowest_idx = min(new_gpu_order, key=lambda x: current_gpu_priorities.get(x, 0))
                    lowest_priority = current_gpu_priorities.get(lowest_idx, 0)
                    
                    # Swap if: incoming priority is higher OR we're forcing exploration
                    should_swap = (priority > lowest_priority) or force_explore
                    
                    if should_swap:
                        # ACTUAL DEMOTION: GPU -> CPU
                        if self._demote_chunk_to_cpu(lowest_idx):
                            new_gpu_order.remove(lowest_idx)
                            demotions.append(lowest_idx)
                            reason = "exploration" if force_explore else "priority"
                            print(f"[Transfer] Demoted chunk {lowest_idx} (GPU->CPU, {reason}), priority={lowest_priority:.3f}")
                    else:
                        print(f"[Transfer] Skipping swap: incoming priority {priority:.3f} < GPU priority {lowest_priority:.3f}")
                        continue  # Don't promote if incoming priority is lower
                
                # ACTUAL PROMOTION: CPU -> GPU  
                if self._promote_chunk_to_gpu(idx):
                    new_gpu_order.append(idx)
                    promotions.append(idx)
                    print(f"[Transfer] Promoted chunk {idx} (CPU->GPU), priority={priority:.3f}")
        
        # Deduplicate while preserving order
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
            print(f"[Scheduler] new GPU order: {self.current_gpu_order}")
        
        return self.current_gpu_order

    def _promote_chunk_to_gpu(self, idx: int) -> bool:
        """FIXED: Actual CPU->GPU tensor transfer"""
        try:
            if idx not in self.ready_kv:
                return False
            
            cpu_kv = self.ready_kv[idx]
            gpu_kv = []
            
            # Transfer each layer's KV tensors to GPU
            with torch.cuda.device(self.device):
                for (k, v) in cpu_kv:
                    gpu_k = k.to(self.device, non_blocking=True)
                    gpu_v = v.to(self.device, non_blocking=True)
                    gpu_kv.append((gpu_k, gpu_v))
            
            # Synchronize to ensure transfer completion
            torch.cuda.synchronize()
            
            # Update GPU chunks dictionary
            self.gpu_chunks[idx] = tuple(gpu_kv)
            
            return True
            
        except Exception as e:
            print(f"[Transfer] Failed to promote chunk {idx}: {e}")
            return False

    def _demote_chunk_to_cpu(self, idx: int) -> bool:
        """FIXED: Actual GPU->CPU tensor transfer"""
        try:
            if idx not in self.gpu_chunks:
                return False
            
            gpu_kv = self.gpu_chunks[idx]
            cpu_kv = []
            
            # Transfer each layer's KV tensors to CPU
            for (k, v) in gpu_kv:
                cpu_k = k.detach().cpu()
                cpu_v = v.detach().cpu()
                cpu_kv.append((cpu_k, cpu_v))
            
            # Update ready_kv and remove from GPU
            self.ready_kv[idx] = tuple(cpu_kv)
            del self.gpu_chunks[idx]
            
            # Free GPU memory
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"[Transfer] Failed to demote chunk {idx}: {e}")
            return False

    def update_rewards(self, used_chunks: List[int], reward: float = 1.0) -> None:
        """FIXED: Consistent EMA updates with recency tracking"""
        current_time = self.current_step
        
        for idx in used_chunks:
            # Update last used time
            self.last_used[idx] = current_time
            
            # FIXED: Use EMA for both rewards AND counts for consistency
            old_reward = self.rewards.get(idx, 0.0)
            old_count = self.counts.get(idx, 0.0)
            
            # EMA update for rewards
            self.rewards[idx] = (1 - self.reward_ema_alpha) * old_reward + self.reward_ema_alpha * reward
            
            # EMA update for counts (represents "effective" count)
            self.counts[idx] = (1 - self.count_ema_alpha) * old_count + self.count_ema_alpha * 1.0
            
            # Update time-based weight
            self.time_weights[idx] = 1.0

    def get_gpu_chunks(self) -> Dict[int, Any]:
        return self.gpu_chunks

    def get_scheduler_interval(self) -> int:
        return max(10, int(self.scheduler_interval))

    def shutdown(self) -> None:
        """OPTIMIZED: No-op shutdown for performance"""
        pass

    # FIXED: Improved private methods

    def _load_config_defaults(self) -> None:
        """Load config with performance-optimized defaults"""
        if self.promote_per_step is None:
            self.promote_per_step = 1
        if self.scheduler_interval is None:
            self.scheduler_interval = 10
        
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

    def _fast_candidate_pool(self) -> List[int]:
        """Select candidate chunks for consideration"""
        candidates = list(self.ready_kv.keys())
        
        if len(candidates) > self.max_candidates:
            candidates = np.random.choice(candidates, self.max_candidates, replace=False).tolist()
        
        return candidates

    def _fast_priority(self, idx: int) -> float:
        """FIXED: Improved priority calculation with recency"""
        current_time = self.current_step
        
        # Base reward (already EMA)
        base_reward = self.rewards.get(idx, 0.0)
        
        # Recency bonus
        last_used = self.last_used.get(idx, -100)
        recency_factor = math.exp(-0.1 * max(0, current_time - last_used))
        
        # Exploration bonus
        novelty = 0.1 if idx not in self.gpu_chunks else 0.0
        
        return base_reward + 0.2 * recency_factor + novelty

    def _fast_ucb_scores(self, candidates: List[int]) -> Dict[int, float]:
        """FIXED: Improved UCB with consistent time scales and recency"""
        current_time = self.current_step
        total_effective_count = sum(self.counts.values()) or 1.0
        scores = {}
        
        for idx in candidates:
            # Get EMA-based values (both rewards and counts use EMA now)
            ema_reward = self.rewards.get(idx, 0.0)
            ema_count = max(0.1, self.counts.get(idx, 0.1))
            
            # FIXED: Proper UCB confidence interval
            confidence = math.sqrt(2 * math.log(total_effective_count + 1) / ema_count)
            
            # FIXED: Recency bonus (decay over time)
            last_used = self.last_used.get(idx, -100)
            recency_factor = math.exp(-0.1 * max(0, current_time - last_used))
            
            # FIXED: Dynamic exploration bonus
            if idx not in self.gpu_chunks:
                exploration_bonus = 0.2 * (1.0 + recency_factor)
            else:
                exploration_bonus = 0.05 * recency_factor
            
            ucb_score = (ema_reward + 
                        self.exploration_c * confidence + 
                        exploration_bonus + 
                        0.1 * recency_factor)
            
            scores[idx] = ucb_score
        
        return scores

    def _predictive_ucb_scores(self, candidates: List[int], future_step: int) -> Dict[int, float]:
        """FIXED: UCB scoring for future step prediction"""
        total_effective_count = sum(self.counts.values()) or 1.0
        scores = {}
        
        for idx in candidates:
            ema_reward = self.rewards.get(idx, 0.0)
            ema_count = max(0.1, self.counts.get(idx, 0.1))
            
            # Confidence interval for future step
            confidence = math.sqrt(2 * math.log(total_effective_count + future_step) / ema_count)
            
            # Predict future relevance
            last_used = self.last_used.get(idx, -100)
            time_since_use = max(0, self.current_step - last_used)
            future_relevance = math.exp(-0.05 * time_since_use)
            
            # Exploration bonus for prediction
            if idx not in self.gpu_chunks:
                prediction_bonus = 0.3 * future_relevance
            else:
                prediction_bonus = 0.1 * future_relevance
            
            ucb_score = (ema_reward * future_relevance + 
                        self.exploration_c * confidence + 
                        prediction_bonus)
            
            scores[idx] = ucb_score
        
        return scores


class FastKVCacheManager:
    """OPTIMIZED: Fast KV cache operations for minimal latency"""
    
    @staticmethod
    def fast_concatenate_chunks(gpu_chunks: Dict[int, Any], selected_chunks: List[int]) -> Optional[Any]:
        """Fast KV cache concatenation with pre-allocation"""
        if not selected_chunks or not gpu_chunks:
            return None

        valid_caches = []
        for chunk_idx in selected_chunks:
            if chunk_idx in gpu_chunks:
                valid_caches.append(gpu_chunks[chunk_idx])

        if not valid_caches:
            return None

        num_layers = len(valid_caches[0])  # Get number of layers from first cache
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

                merged_k = torch.cat(keys_to_concat, dim=2)
                merged_v = torch.cat(values_to_concat, dim=2)
                combined_kv.append((merged_k, merged_v))

        return tuple(combined_kv)
