#!/usr/bin/env python3
"""
ASYNC PREFETCH SCHEDULER v6 - True background prefetch with CUDA streams
This scheduler implements true overlapping of chunk transfers with generation
Key improvements:
- Background prefetch of full KV caches using CUDA streams
- Non-blocking transfer completion checks
- Pointer swapping instead of cache rebuilding
"""

import time
import math
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Tuple
import yaml
import re
import torch
import numpy as np
from build_kv_v2 import build_chunk_sequence
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False
    print("[Warning] scikit-learn not available, falling back to text similarity")

class AsyncPrefetchScheduler:
    def __init__(
        self,
        device: str,
        config_path: str = "configs/config.yaml",
        promote_per_step: Optional[int] = None,
        scheduler_interval: Optional[int] = None,
        exploration_c: float = 1.0,
        max_candidates: int = 10,
        sparsity_ratio: float = 1.0,
        epsilon: float = 0.15,
    ):
        self.device = torch.device(device)
        self.config_path = config_path
        self.promote_per_step = promote_per_step or 1
        self.scheduler_interval = scheduler_interval or 3  # Less frequent to reduce overhead
        self.exploration_c = exploration_c
        self.max_candidates = max_candidates
        self.sparsity_ratio = sparsity_ratio
        self.epsilon = epsilon
        self.use_tfidf = False
        self.tfidf_vectorizer = None
        self.chunk_vectors = None
        self.chunk_index_map = {}  # Maps chunk_idx to position in vectors
        # Prefetch throttling and memory safety
        self.max_prefetched_gpu = max(2, (self.promote_per_step or 1))  # cap concurrent prefetched GPU caches
        self.tfidf_weight = 0.7  # weight for TF-IDF vs. bandit priority in hybrid scoring

        # Core state
        self.sample: Optional[Dict[str, Any]] = None
        self.model = None
        self.tokenizer = None

        # Chunk management
        self.gpu_chunks: Dict[int, Any] = {}
        self.cpu_chunks: Dict[int, str] = {}
        self.ready_kv: Dict[int, Any] = {}

        # Bandit state - simplified
        self.rewards: Dict[int, float] = {}
        self.counts: Dict[int, int] = {}
        self.last_used: Dict[int, int] = {}
        self.current_step: int = 0

        # GPU management
        self.max_gpu: int = 0
        self.current_gpu_order: List[int] = []

        # ASYNC PREFETCH INFRASTRUCTURE
        self.transfer_stream = torch.cuda.Stream(self.device)  # Dedicated stream for transfers
        self.prefetch_lock = threading.Lock()
        self.prefetched: Dict[int, Tuple[tuple, torch.cuda.Event]] = {}  # chunk_idx -> (gpu_kv, event)
        self.next_prefetch_steps = set()  # Steps when to trigger prefetch
        self.pending_prefetch = set()  # Chunks waiting for precomputation to finish
        self.protected_gpu_chunks: List[int] = []  # Chunks promoted to GPU that should not be evicted
        
        # Background processing state
        self.background_precompute = True
        self.precompute_started = False
        
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
        """
        Fast initialization - no pre-computation during TTFT
        """
        print(f"[AsyncScheduler] Fast initialization starting...")
        init_start = time.perf_counter()
        
        self.sample = sample
        self.gpu_chunks = dict(gpu_chunks)
        self.cpu_chunks = dict(cpu_chunks)
        self.tokenizer = tokenizer
        self.model = model
        self.max_gpu = max_gpu if max_gpu is not None else len(gpu_chunks)
        self.current_gpu_order = list(gpu_chunks.keys())[:self.max_gpu]

        # Initialize rewards: CPU chunks start higher to encourage exploration
        for idx in set(gpu_chunks.keys()) | set(cpu_chunks.keys()):
            if idx in gpu_chunks:
                self.rewards[idx] = 1.0
                self.counts[idx] = 1
            else:
                # CPU chunks start at 0.8 (not 0.5) to be competitive
                self.rewards[idx] = 0.3
                self.counts[idx] = 1
            self.last_used[idx] = 0

        if self.use_tfidf and TFIDF_AVAILABLE:
            self._initialize_tfidf()
            print("[AsyncScheduler] Tfidf initialized successfully")
        else:
            print("[AsyncScheduler] Tfidf not available, falling back to text similarity")

        # Initialize prefetch schedule - first prediction at step 3
        self.next_prefetch_steps = {3}
        
        init_time = time.perf_counter() - init_start
        print(f"[AsyncScheduler] Initialization completed in {init_time*1000:.2f}ms")
        print(f"[AsyncScheduler] First prediction scheduled at step 5, then every {self.scheduler_interval} steps")

        # Start background pre-computation if needed
        if self.cpu_chunks and self.background_precompute:
            threading.Thread(target=self._background_precompute, daemon=True).start()


    def _initialize_tfidf(self):
        """Initialize TF-IDF for chunk selection"""
        try:
            # Gather CPU chunk texts (we only rank CPU candidates)
            all_texts = []
            all_indices = []

            for idx in sorted(self.cpu_chunks.keys()):
                text = self.cpu_chunks.get(idx, "")
                if text:
                    all_texts.append(text)
                    all_indices.append(idx)
            
            if len(all_texts) < 2:
                print("[TF-IDF] Not enough chunks for TF-IDF, falling back")
                self.use_tfidf = False
                return
            
            # Initialize vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=True,
                strip_accents='unicode',
                norm='l2'
            )
            
            # Fit and transform
            self.chunk_vectors = self.tfidf_vectorizer.fit_transform(all_texts)
            self.chunk_index_map = {idx: i for i, idx in enumerate(all_indices)}
            
            print(f"[TF-IDF] Initialized with {len(all_indices)} chunks")
            
        except Exception as e:
            print(f"[TF-IDF] Initialization failed: {e}")
            self.use_tfidf = False
    
    def _background_precompute(self):
        """
        Background pre-computation that doesn't affect TTFT
        Precomputes KV caches for CPU chunks
        """
        if self.precompute_started:
            return
        self.precompute_started = True
        
        print(f"[Background] Starting pre-computation for {len(self.cpu_chunks)} CPU chunks...")
        time.sleep(0.01)  # Minimal delay to ensure first token is generated first
        
        start_time = time.perf_counter()
        chunk_indices = [idx for idx in self.cpu_chunks.keys() if idx not in self.gpu_chunks]
        # Prioritize pending first, then by similarity to the question (dataset-agnostic)
        question_text = (self.sample.get("question", "") if self.sample else "")
        def _sim(i: int) -> float:
            try:
                text = self.cpu_chunks.get(i, "")
                return self._text_similarity(text, question_text)
            except Exception:
                return 0.0
        try:
            chunk_indices.sort(key=lambda i: (0 if i in self.pending_prefetch else 1, -_sim(i)))
        except Exception:
            pass
        
        with torch.inference_mode():
            for i, idx in enumerate(chunk_indices):
                try:
                    text = self.cpu_chunks.get(idx, "")
                    if text:
                        input_ids = build_chunk_sequence(text, self.tokenizer)
                        current_input = torch.tensor([input_ids], device=self.device)
                        outputs = self.model(current_input, use_cache=True, return_dict=True)
                        kv = outputs.past_key_values
                        
                        if hasattr(kv, 'to_legacy_cache'):
                            kv = kv.to_legacy_cache()
                        
                        # Keep KV on CPU initially
                        cpu_kv = []
                        for (k, v) in kv:
                            cpu_k = k.detach().cpu()
                            cpu_v = v.detach().cpu()
                            cpu_kv.append((cpu_k, cpu_v))
                        
                        self.ready_kv[idx] = tuple(cpu_kv)
                        # Progress update every 2 chunks
                        # if (i + 1) % 2 == 0:
                            # print(f"[Background] Precomputed {i + 1}/{len(chunk_indices)} chunks...")

                except Exception:
                    pass  # Skip failed chunks
        
        precompute_time = time.perf_counter() - start_time
        print(f"[Background] Pre-computation completed in {precompute_time:.3f}s, {len(self.ready_kv)} chunks ready")

    def _calculate_priority(self, idx: int) -> float:
        """
        Calculate priority with exploration bonus for CPU chunks
        CPU chunks need higher bonus to overcome GPU's higher initial reward
        """
        base_reward = self.rewards.get(idx, 0.0)
        current_time = self.current_step
        last_used = self.last_used.get(idx, -100)
        recency_factor = math.exp(-0.1 * max(0, current_time - last_used))
        
        # EXPLORATION BONUS: CPU chunks get +0.35 bonus to compete with GPU chunks
        exploration_bonus = 0.35 if idx not in self.gpu_chunks else 0.0

        # TEXT SIMILARITY BONUS: favor chunks whose text overlaps with the question
        question_text = self.sample.get("question", "") if self.sample else ""
        chunk_text = self.cpu_chunks.get(idx, "")
        similarity = self._text_similarity(chunk_text, question_text)
        similarity_bonus = 0.6 * similarity  # Weight similarity highly!

        return base_reward + 0.2 * recency_factor + exploration_bonus + similarity_bonus
    
    def predict_chunks(self, step: int, generated_tokens: List[int]) -> List[int]:
        """
        Predict which chunks will be needed for future steps
        Uses epsilon-exploration + exploration bonus for CPU chunks
        """
        self.current_step = step - 5
        
        if step < 8:
            return []
        
        candidates = self._get_candidates()
        if not candidates:
            print(f"[Scheduler] WARNING: No candidates available at step {step}")
            return []
        
        # Debug: show candidates
        ready_count = len([c for c in candidates if c in self.ready_kv])
        not_ready_count = len(candidates) - ready_count
        if not_ready_count > 0:
            print(f"[Scheduler] Candidates: {len(candidates)} total ({ready_count} precomputed, {not_ready_count} pending)")

        # EPSILON-EXPLORATION: occasionally explore random chunks
        if np.random.random() < self.epsilon:
            # Exploration: randomly sample chunks to try
            num_to_select = min(self.promote_per_step, len(candidates))
            selected = np.random.choice(candidates, num_to_select, replace=False).tolist()
            print(f"[Scheduler] EXPLORATION: Randomly selected chunks {selected}")
            return selected
        
        # EXPLOITATION: use priority-based scoring with exploration bonus
        scores = {}
        for idx in candidates:
            scores[idx] = self._calculate_priority(idx)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in ranked[:self.promote_per_step]]
        
        return selected


    ####TFIDF- Predict method
    # def predict_chunks(self, step: int, generated_tokens: List[int]) -> List[int]:
    #     """
    #     Predict chunks using TF-IDF similarity (preferred) or fallback to UCB
    #     """
    #     self.current_step = step - 5
        
    #     if step < 8:
    #         return []
        
    #     candidates = self._get_candidates()
        
    #     if not candidates:
    #         print(f"[Scheduler] WARNING: No candidates available at step {step}")
    #         return []
        
    #     # Debug
    #     ready_count = len([c for c in candidates if c in self.ready_kv])
    #     not_ready_count = len(candidates) - ready_count
    #     if not_ready_count > 0:
    #         print(f"[Scheduler] Candidates: {len(candidates)} total ({ready_count} precomputed, {not_ready_count} pending)")
        
    #     # === USE TF-IDF IF AVAILABLE ===
    #     if self.use_tfidf and self.tfidf_vectorizer is not None:
    #         selected = self._tfidf_select(candidates, generated_tokens)
    #         if selected:
    #             print(f"[TF-IDF] Selected chunks {selected} from {len(candidates)} candidates")
    #             return selected

    #     return []


    def _tfidf_select(self, candidates: List[int], generated_tokens: List[int]) -> List[int]:
        """Select chunks using TF-IDF similarity, hybridized with bandit priority."""
        try:
            # Build query: question + recent generation
            question = self.sample.get("question", "") if self.sample else ""
            
            # Add recent generation context (last 50 tokens)
            if generated_tokens and self.tokenizer:
                recent_text = self.tokenizer.decode(generated_tokens[-50:], skip_special_tokens=True)
                query = question + " " + recent_text
            else:
                query = question
            
            if not query:
                return []

            valid_candidates = [c for c in candidates 
                   if c in self.chunk_index_map and c not in self.gpu_chunks]

            if not valid_candidates:
                print("[TF-IDF] No valid CPU candidates available")
                return []
            
            # Transform query
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Get candidate positions
            candidate_positions = [self.chunk_index_map[c] for c in valid_candidates]
            candidate_vectors = self.chunk_vectors[candidate_positions]
            
            # Compute similarities
            similarities = cosine_similarity(query_vec, candidate_vectors)[0]

            # Hybrid with bandit priority
            # Normalize similarities to [0,1]
            if len(similarities) > 0:
                sim_min, sim_max = float(similarities.min()), float(similarities.max())
                if sim_max > sim_min:
                    sims_norm = (similarities - sim_min) / (sim_max - sim_min)
                else:
                    sims_norm = np.zeros_like(similarities)
            else:
                sims_norm = similarities

            # Compute priorities for same candidates, normalize to [0,1]
            prios = np.array([self._calculate_priority(c) for c in valid_candidates], dtype=float)
            if len(prios) > 0:
                p_min, p_max = float(prios.min()), float(prios.max())
                if p_max > p_min:
                    prios_norm = (prios - p_min) / (p_max - p_min)
                else:
                    prios_norm = np.zeros_like(prios)
            else:
                prios_norm = prios

            combined = self.tfidf_weight * sims_norm + (1.0 - self.tfidf_weight) * prios_norm

            # Select top chunks by combined score
            num_to_select = min(self.promote_per_step, len(valid_candidates))
            top_indices = combined.argsort()[-num_to_select:][::-1]
            selected = [valid_candidates[i] for i in top_indices]

            return selected
            
        except Exception as e:
            print(f"[TF-IDF] Selection error: {e}")
            return []


    def _text_similarity(self, text: str, question: str) -> float:
        """Compute a lightweight, dataset-agnostic similarity between a chunk and the question."""
        if not text or not question:
            return 0.0
        # Tokenize to words, lowercase
        tokens_a = set(re.findall(r"\w+", text.lower()))
        tokens_b = set(re.findall(r"\w+", question.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        inter = len(tokens_a & tokens_b)
        # Cosine-like score with binary weights
        return inter / max(1.0, (len(tokens_a) * len(tokens_b)) ** 0.5)

    def prefetch_chunks_async(self, chunk_idx: int):
        """
        ASYNC PREFETCH: Transfer CPU KV cache to GPU in background
        This runs in a separate thread and uses a dedicated CUDA stream
        """
        # Throttle: avoid too many prefetched GPU tensors
        with self.prefetch_lock:
            if len(self.prefetched) >= self.max_prefetched_gpu:
                # Drop lowest-priority prefetched to make room
                try:
                    def _prio(c: int) -> float:
                        return self._calculate_priority(c)
                    # Keep the best, drop one worst
                    worst = min(self.prefetched.keys(), key=_prio)
                    del self.prefetched[worst]
                except Exception:
                    # If priority fails, drop an arbitrary entry (FIFO-ish)
                    try:
                        old_key = next(iter(self.prefetched.keys()))
                        del self.prefetched[old_key]
                    except StopIteration:
                        pass
        # If already prefetched or prefetching, skip
        with self.prefetch_lock:
            if chunk_idx in self.prefetched:
                return

        def _transfer_worker():
            try:
                # Ensure CPU KV exists; compute on-demand if needed
                if chunk_idx not in self.ready_kv:
                    self.pending_prefetch.add(chunk_idx)
                    text = self.cpu_chunks.get(chunk_idx, "")
                    if not text:
                        return
                    with torch.inference_mode():
                        input_ids = build_chunk_sequence(text, self.tokenizer)
                        current_input = torch.tensor([input_ids], device=self.device)
                        outputs = self.model(current_input, use_cache=True, return_dict=True)
                        kv = outputs.past_key_values
                        if hasattr(kv, 'to_legacy_cache'):
                            kv = kv.to_legacy_cache()
                        cpu_kv_tmp = []
                        for (k, v) in kv:
                            cpu_k = k.detach().cpu()
                            cpu_v = v.detach().cpu()
                            cpu_kv_tmp.append((cpu_k, cpu_v))
                        self.ready_kv[chunk_idx] = tuple(cpu_kv_tmp)

                cpu_kv = self.ready_kv[chunk_idx]
                
                # Use dedicated transfer stream for non-blocking transfer
                with torch.cuda.stream(self.transfer_stream):
                    gpu_kv = []
                    for k, v in cpu_kv:
                        gpu_k = k.to(self.device, non_blocking=True)
                        gpu_v = v.to(self.device, non_blocking=True)
                        gpu_kv.append((gpu_k, gpu_v))
                    
                    # Record event when transfer is complete
                    event = torch.cuda.Event()
                    event.record(self.transfer_stream)
                
                # Store the transferred KV cache with completion event
                with self.prefetch_lock:
                    self.prefetched[chunk_idx] = (tuple(gpu_kv), event)
                    print(f"[AsyncPrefetch] Chunk {chunk_idx} transferred to GPU (background)")
                    if chunk_idx in self.pending_prefetch:
                        self.pending_prefetch.discard(chunk_idx)
                
            except Exception as e:
                print(f"[AsyncPrefetch] Transfer failed for chunk {chunk_idx}: {e}")

        # Launch transfer in background thread
        threading.Thread(target=_transfer_worker, daemon=True).start()
    
    def retry_pending_prefetch(self):
        """
        Retry prefetching chunks that were pending (waiting for precomputation)
        Call this periodically to check if pending chunks are now ready
        """
        if not self.pending_prefetch:
            return
        
        # Check which pending chunks are now ready
        ready_to_prefetch = []
        for chunk_idx in list(self.pending_prefetch):
            if chunk_idx in self.ready_kv:
                ready_to_prefetch.append(chunk_idx)
                self.pending_prefetch.discard(chunk_idx)
        
        # Prefetch them
        if ready_to_prefetch:
            print(f"[AsyncPrefetch] Retrying {len(ready_to_prefetch)} pending chunks: {ready_to_prefetch}")
            for chunk_idx in ready_to_prefetch:
                self.prefetch_chunks_async(chunk_idx)

    def schedule_if_needed(self, step: int, generated_tokens: List[int]):
        """
        Check if it's time to predict and prefetch next chunks
        This is called every generation step but only acts at intervals
        """
        if step in self.next_prefetch_steps:
            # Predict chunks needed for future steps
            future_step = step + self.scheduler_interval
            predicted_chunks = self.predict_chunks(future_step, generated_tokens)
            
            # Launch async prefetch for predicted chunks
            # Note: prefetch_chunks_async handles both ready and not-yet-ready chunks
            for chunk_idx in predicted_chunks:
                if chunk_idx not in self.prefetched and chunk_idx not in self.gpu_chunks:
                    self.prefetch_chunks_async(chunk_idx)
            
            # Schedule next prefetch
            self.next_prefetch_steps = {step + self.scheduler_interval}

    def get_ready_kv(self, chunk_idx: int) -> Optional[tuple]:
        """
        NON-BLOCKING check: return GPU KV if transfer is complete
        This allows generation to continue while transfers happen in background
        """
        with self.prefetch_lock:
            if chunk_idx in self.prefetched:
                gpu_kv, event = self.prefetched[chunk_idx]
                # Non-blocking check if transfer is complete
                if event.query():  # Returns True if event is complete
                    del self.prefetched[chunk_idx]
                    return gpu_kv
        return None

    def get_available_chunks(self) -> List[int]:
        """
        Get list of chunks that are ready to use (either on GPU or transferred)
        """
        available = list(self.gpu_chunks.keys())
        
        with self.prefetch_lock:
            for chunk_idx in self.prefetched:
                gpu_kv, event = self.prefetched[chunk_idx]
                if event.query():  # Transfer complete
                    available.append(chunk_idx)
        
        return available

    def force_sync_chunk(self, chunk_idx: int) -> Optional[tuple]:
        """
        BLOCKING wait for a specific chunk if needed
        Only use this when absolutely necessary
        """
        with self.prefetch_lock:
            if chunk_idx in self.prefetched:
                gpu_kv, event = self.prefetched[chunk_idx]
                event.synchronize()  # Wait for transfer to complete
                del self.prefetched[chunk_idx]
                return gpu_kv
        return None

    def update_rewards(self, used_chunks: List[int], reward: float = 1.0) -> None:
        """
        Reward updates using moving average (like scheduler_v4)
        Moving average naturally prevents rewards from getting stuck at 1.0
        """
        current_time = self.current_step
        
        # Update rewards for used chunks using moving average
        for idx in used_chunks:
            self.last_used[idx] = current_time
            old_reward = self.rewards.get(idx, 0.0)
            old_count = self.counts.get(idx, 0)
            self.counts[idx] = old_count + 1
            alpha = 1.0 / self.counts[idx]  # Decreases over time (moving average)
            self.rewards[idx] = (1 - alpha) * old_reward + alpha * reward

    def get_gpu_chunks(self) -> Dict[int, Any]:
        """Get current GPU chunks including recently transferred ones"""
        current_gpu = dict(self.gpu_chunks)
        
        # Add completed transfers
        with self.prefetch_lock:
            for chunk_idx in list(self.prefetched.keys()):
                gpu_kv, event = self.prefetched[chunk_idx]
                if event.query():
                    current_gpu[chunk_idx] = gpu_kv
                    del self.prefetched[chunk_idx]
        
        return current_gpu

    def get_scheduler_interval(self) -> int:
        return self.scheduler_interval

    def commit_gpu_order(self, new_order: List[int]) -> None:
        """Commit a new GPU chunk order and prune evicted chunks to free memory."""
        # First, incorporate any completed transfers into gpu_chunks
        with self.prefetch_lock:
            for chunk_idx in list(self.prefetched.keys()):
                gpu_kv, event = self.prefetched[chunk_idx]
                if event.query():
                    self.gpu_chunks[chunk_idx] = gpu_kv
                    del self.prefetched[chunk_idx]

        # Prune evicted chunks from GPU mapping to release memory
        keep_set = set(new_order)
        for idx in list(self.gpu_chunks.keys()):
            if idx not in keep_set:
                del self.gpu_chunks[idx]

        self.current_gpu_order = list(new_order)

        # Keep protected only for chunks that remain on GPU
        keep_set = set(new_order)
        self.protected_gpu_chunks = [idx for idx in self.protected_gpu_chunks if idx in keep_set]

        # Also prune prefetched GPU caches aggressively to avoid OOM
        # Keep at most max_prefetched_gpu best by priority
        with self.prefetch_lock:
            if len(self.prefetched) > self.max_prefetched_gpu:
                try:
                    # Rank prefetched by priority descending, keep top-K
                    ranked = sorted(self.prefetched.keys(), key=lambda c: self._calculate_priority(c), reverse=True)
                    keep = set(ranked[: self.max_prefetched_gpu])
                    for c in list(self.prefetched.keys()):
                        if c not in keep:
                            del self.prefetched[c]
                except Exception:
                    # Fallback: trim arbitrarily
                    while len(self.prefetched) > self.max_prefetched_gpu:
                        try:
                            k = next(iter(self.prefetched.keys()))
                            del self.prefetched[k]
                        except StopIteration:
                            break

    def protect_chunks(self, chunks: List[int]) -> None:
        """Record predicted chunks that are now resident on GPU to prevent eviction."""
        for chunk_idx in chunks:
            if chunk_idx not in self.protected_gpu_chunks:
                self.protected_gpu_chunks.append(chunk_idx)

    def get_protected_chunks(self) -> List[int]:
        """Return chunks that should not be evicted during scheduling."""
        return list(self.protected_gpu_chunks)

    def shutdown(self) -> None:
        """Clean shutdown - wait for pending transfers"""
        print("[AsyncScheduler] Shutting down...")
        
        # Wait for any pending transfers
        with self.prefetch_lock:
            for chunk_idx, (gpu_kv, event) in self.prefetched.items():
                event.synchronize()
        
        # Clear state
        self.prefetched.clear()
        print("[AsyncScheduler] Shutdown complete")

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
            pass

    def _get_candidates(self) -> List[int]:
        """
        Get candidate chunks for scheduling
        Includes both ready_kv (precomputed) and cpu_chunks (not yet precomputed)
        """
        # Include both precomputed chunks AND chunks we know about from CPU
        candidates = list(set(self.ready_kv.keys()) | set(self.cpu_chunks.keys()))
        
        if len(candidates) > self.max_candidates:
            candidates = np.random.choice(candidates, self.max_candidates, replace=False).tolist()
        return candidates

# Maintain backward compatibility with original scheduler name
BanditScheduler = AsyncPrefetchScheduler

class FastKVCacheManager:
    """Fast KV cache operations with async support"""

    @staticmethod
    def fast_concatenate_chunks(gpu_chunks: Dict[int, Any], selected_chunks: List[int]) -> Optional[Any]:
        """Fast KV cache concatenation"""
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

                merged_k = torch.cat(keys_to_concat, dim=2)
                merged_v = torch.cat(values_to_concat, dim=2)
                combined_kv.append((merged_k, merged_v))

        return tuple(combined_kv)