# PROJECT_CONTEXT.md
search-r1-grpo/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── model.py          # Qwen3 wrapper with generation logic
│   │   └── search_engine.py  # Mock environment & rag tools
│   ├── core/
│   │   ├── __init__.py
│   │   ├── trajectory.py     # @dataclass for Trajectory/TokenStep
│   │   └── grpo.py           # Advantage calc, K3 KL approx, PPO-clip loss
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── format_reward.py  # XML tag structure validation
│   │   └── correctness.py    # Exact match & fuzzy logic
│   ├── data/
│   │   └── dataset.py        # Query/GroundTruth loader
│   ├── utils/
│   │   └── config.py         # Hyperparams (group_size, beta, lr)
│   └── run.py                # Main training entry point
├── tests/
│   ├── test_rewards.py
│   └── test_grpo_math.py
├── docs/
│   └── math_derivations.md
├── requirements.txt
├── README.md
└── TODO.md
## @Project_Goal
We are engineering **SEARCH-R1**, an Agentic Reinforcement Learning system that solves the "passive execution" failure of traditional RAG. By utilizing **Group Relative Policy Optimization (GRPO)** without a Critic network, we train a Qwen3-based model to autonomously decide *when* to search and *how* to reason using a strict `<think> -> <search> -> <answer>` XML flow.

## @Orchestrator_Tasks
1.  **Core Structures:** Implement the `Trajectory` and `TokenStep` dataclasses to track generated tokens, log probabilities, and reward signals across `group_size` samples.
2.  **Environment & Rewards:** Build the `SearchEngine` mock (school regulations) and the composite reward function (Format Reward + Exact Match Correctness).
3.  **GRPO Trainer:** Implement the custom training loop:
    * **Sampling:** Generate `G` trajectories per query.
    * **Advantage:** Compute $A_i = (r_i - \mu) / \sigma$.
    * **Loss:** Implement PPO-clip policy loss + K3 KL divergence penalty.
4.  **Optimization:** Integrate explicit `torch.cuda.empty_cache()` calls and gradient clipping (norm 0.5) to manage memory stability on consumer hardware.

## @Architect_Rules
* **Algorithm Constraint:** DO NOT use a Value Network (Critic). Use group-based advantage normalization.
* **Model Selection:** Default to `Qwen3-4B` or larger. The 0.6B variant is unstable for instruction following and format adherence.
* **Tag Protocol:** The model MUST output strict XML tags:
    * `<think>` for reasoning.
    * `<search>` (paired with `<information>`) for tool use.
    * `<answer>` for the final result.
* **Math Constraints:**
    * **KL Divergence:** Use the "K3" approximation: $k3 = r - \log(r) - 1$.
    * **Advantage:** If `group_size == 1`, force advantage to 0.0 (prevent training noise).
    * **Reward:** Binary (0/1) or discrete step (0.5) rewards only; avoid complex continuous shaping for now.

## @Coder_Standards
* **Type Hints:** STRICTLY REQUIRED for all function signatures (e.g., `def train_step(self, trajectories: List[Trajectory]) -> Dict[str, float]:`).
* **Tensor Safety:** Explicitly handle `device` placement (`.to(self.device)`) for all new tensors created during loss calculation.
* **Code Clarity:** Keep the "hand-rolled" math visible. Do not bury the GRPO formula in abstract classes; we want to see the `ratio`, `surr1`, and `surr2` calculations in `loss.py`.
* **Debugging:** Retain `generated_text` in the `Trajectory` object to visually inspect model hallucinations during debugging.