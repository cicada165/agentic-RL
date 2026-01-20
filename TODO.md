# TODO.md - SEARCH-R1 Development Tasks

## Immediate Tasks (Orchestrator Roadmap)

### Task 1: Core Structures
- [x] **CODER**: Implement `TokenStep` dataclass in `src/core/trajectory.py`
  - Fields: `token_id`, `token_text`, `log_prob`, `position`
  - Type hints: `int`, `str`, `float`, `int`
  - Docstring explaining role in RL pipeline
  - Immutable after creation
  - **Status**: Basic structure exists in `trajectory.py` (root), needs relocation to `src/core/`

- [ ] **CODER**: Relocate and enhance `Trajectory` dataclass to `src/core/trajectory.py`
  - Move existing `trajectory.py` to `src/core/trajectory.py` (create directory structure)
  - Fields: `query`, `token_steps: List[TokenStep]`, `generated_text`, `reward`, `final_answer`, `full_input_ids`, `generated_positions`, `advantage`
  - Add methods: `add_step()`, `compute_log_prob_sum()` for step tracking and log probability computation
  - Enhance docstrings explaining role in GRPO pipeline
  - Ensure all type hints are explicit (not just comments)
  - Retain `generated_text` for debugging hallucinations
  - Create `src/core/__init__.py` to export classes

### Task 2: Environment & Rewards
- [ ] **CODER**: Implement `SearchEngine` mock in `src/agent/search_engine.py`
  - Create abstract base class `SearchEngine` with `search(query: str) -> str` method
  - Implement `MockSearchEngine` with school regulations knowledge base
  - Keyword matching (case-insensitive)
  - Return information string or "No information found for: {query}"
  - Type hints and docstrings
  - Make swappable for real API integration (Google/Bing)

- [ ] **CODER**: Implement format reward in `src/rewards/format_reward.py`
  - Function: `check_format_correctness(generated_text: str) -> float`
  - Validation: Exactly one `<answer>...</answer>` at end
  - Validation: If `<search>` appears, must have matching `</search>` and `<information>...</information>`
  - Validation: Search and information tags must appear in pairs
  - Returns: -1.0 if incorrect, 0.5 if correct
  - Type hints and docstrings

- [ ] **CODER**: Implement answer correctness in `src/rewards/correctness.py`
  - Function: `check_answer_correctness(final_answer: str, ground_truth: str) -> float`
  - Exact match → 2.0
  - Special case "未找到相关内容" → 0.5
  - Fuzzy matching using SequenceMatcher (similarity >= 0.5 → 1.0, else → 0.0)
  - Type hints and docstrings

- [ ] **CODER**: Create composite reward function
  - Function: `compute_reward(trajectory: Trajectory, ground_truth: str) -> float`
  - Combines format_reward + answer_reward
  - Returns total reward in range [-1.0, 2.5]
  - Type hints and docstrings

### Task 3: GRPO Trainer
- [ ] **CODER**: Implement Qwen3 model wrapper in `src/agent/model.py`
  - Initialize Qwen3-4B or larger (configurable)
  - Device handling (CUDA if available, else CPU)
  - Precision: float16 on CUDA, float32 on CPU
  - ChatML format prompt template with XML tag instructions
  - Type hints and docstrings

- [ ] **CODER**: Implement trajectory generation in `src/core/grpo.py`
  - Method: `generate_trajectory(query: str, max_tokens: int) -> Trajectory`
  - Token-by-token generation with log_prob tracking
  - Detect `<search>` tags and trigger `search_engine.search()`
  - Insert `<information>` tags with search results
  - Stop at `</answer>` tag or max_tokens
  - Extract `final_answer` from `<answer>` tags
  - Track all token steps with positions
  - Type hints and docstrings

- [ ] **CODER**: Implement advantage calculation in `src/core/grpo.py`
  - Method: `compute_advantages(rewards: List[float]) -> torch.Tensor`
  - Formula: $A_i = (r_i - \mu) / \sigma$ where μ = mean, σ = std + ε
  - Edge case: If `group_size == 1`, return [0.0]
  - Edge case: Zero std → use ε=1e-8 to prevent division by zero
  - Type hints and docstrings

- [ ] **CODER**: Implement K3 KL divergence in `src/core/grpo.py`
  - Method: `compute_kl_divergence(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor`
  - Formula: `ratio = exp(old_log_probs - new_log_probs)`, `k3 = ratio - log(ratio) - 1`
  - Return mean KL divergence across tokens
  - Type hints and docstrings

- [ ] **CODER**: Implement PPO-clip policy loss in `src/core/grpo.py`
  - Method: `update_policy(trajectories: List[Trajectory]) -> Dict[str, float]`
  - Extract old_log_probs from trajectory.token_steps
  - For each update iteration:
    - Recompute new_log_probs using current model
    - Calculate ratio: `ratio = exp(new_log_probs - old_log_probs)`
    - Apply PPO clipping: `min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`
    - Policy loss: `-mean(clipped_objective)`
    - KL penalty: `beta * mean(K3_kl_divergence)`
    - Total loss: `policy_loss + kl_penalty`
    - Backpropagate, clip gradients, update parameters
  - Keep "hand-rolled" math visible (ratio, surr1, surr2 calculations)
  - Type hints and docstrings

- [ ] **CODER**: Implement training step in `src/core/grpo.py`
  - Method: `train_step(queries: List[str], ground_truths: List[str], group_size: int) -> Dict[str, float]`
  - For each (query, ground_truth) pair:
    - Generate `group_size` trajectories
    - Compute rewards for each trajectory
    - Calculate group statistics (mean, std)
    - Compute relative advantages: $(r_i - \mu) / \sigma$
    - Store advantage in trajectory.advantage
  - Collect all trajectories
  - Call `update_policy(trajectories)`
  - Return metrics (loss, kl_div, avg_reward, avg_tokens, search_trajectories)
  - Type hints and docstrings

### Task 4: Optimization
- [ ] **CODER**: Integrate memory management
  - Add `torch.cuda.empty_cache()` after policy updates
  - Implement gradient clipping (norm 0.5) in optimizer step
  - Handle device placement explicitly (`.to(self.device)`) for all new tensors
  - Add error handling for CUDA OOM (fallback to CPU or reduce batch size)

- [ ] **CODER**: Create configuration system in `src/utils/config.py`
  - Define `SearchR1Config` dataclass
  - Fields: model_name_or_path, device, max_tokens, group_size, update_times, clip_epsilon, beta, learning_rate, batch_size, num_epochs, max_grad_norm
  - Load from config file/object (no hardcoded hyperparameters)
  - Validation: group_size >= 2
  - Type hints and docstrings

- [ ] **CODER**: Create data loader in `src/data/dataset.py`
  - Function: `create_training_data() -> Tuple[List[str], List[str]]`
  - Returns (queries, ground_truths) pairs
  - Can be extended to load from files (JSON, CSV)
  - Type hints and docstrings

- [ ] **CODER**: Create main training script in `src/run.py`
  - Initialize GRPOTrainer with config
  - Load training data
  - Implement training loop with logging
  - Integrate SearchEngine, Trajectory, Reward Functions, GRPOTrainer
  - Log metrics at intervals (config.log_interval)

## Testing & Validation
- [ ] **CODER**: Write unit tests in `tests/test_rewards.py`
  - Test format correctness validation
  - Test answer correctness (exact match, fuzzy match, edge cases)
  - Test composite reward computation

- [ ] **CODER**: Write unit tests in `tests/test_grpo_math.py`
  - Test advantage calculation (mean/std normalization)
  - Test K3 KL divergence approximation
  - Test PPO clipping logic
  - Test edge cases (single trajectory, zero variance)

- [ ] **REVIEWER**: Code audit
  - Security review (no hardcoded secrets, safe file operations)
  - Logic review (correctness of GRPO math, advantage calculation)
  - Check for "hallucinations" (incorrect implementations)
  - Verify type hints and docstrings on all functions
  - Verify config-driven hyperparameters (no hardcoding)
  - Verify tensor device placement
  - Verify "hand-rolled" math visibility

## Documentation
- [ ] **CODER**: Update README.md with usage instructions
- [ ] **CODER**: Create math derivations doc in `docs/math_derivations.md`
- [ ] **CODER**: Add inline code documentation where needed

---

## Future Enhancements

### Phase 2: Advanced Features
- [ ] **CODER**: Implement dense rewards (process supervision per step, not just final)
- [ ] **CODER**: Add multi-turn dialogue support (conversation history in trajectories)
- [ ] **CODER**: Implement evaluation metrics (BLEU, ROUGE, semantic similarity)
- [ ] **CODER**: Add checkpointing system (save/load model and training state)
- [ ] **CODER**: Implement resume from checkpoint capability

### Phase 3: Scalability
- [ ] **CODER**: Add batch inference for parallel trajectory generation
- [ ] **CODER**: Implement async/threading for search engine calls
- [ ] **CODER**: Add caching for repeated search queries
- [ ] **CODER**: Implement gradient accumulation for effective larger batches
- [ ] **CODER**: Add streaming data loaders for large datasets
- [ ] **CODER**: Implement multi-GPU support via device_map="auto"

### Phase 4: Production Integration
- [ ] **CODER**: Implement `WebSearchEngine` class for real API integration (Google/Bing)
- [ ] **CODER**: Add rate limiting and retry logic for API calls
- [ ] **CODER**: Implement config file loading (YAML/JSON)
- [ ] **CODER**: Add environment variable support for sensitive values
- [ ] **CODER**: Create plugin architecture for different search providers

### Phase 5: Advanced RL
- [ ] **CODER**: Implement active learning (query selection strategies)
- [ ] **CODER**: Add transfer learning (pre-training on search tasks)
- [ ] **CODER**: Implement multi-modal search (image/document search capabilities)
- [ ] **CODER**: Add distributed training support (DDP for multi-node)

---

**Current Status**: Task 1 Partially Complete - Relocate trajectory.py to src/core/ and add methods
**Next Agent**: @CODER
**Next Task**: Relocate `trajectory.py` to `src/core/trajectory.py`, create directory structure, add helper methods, and enhance docstrings
