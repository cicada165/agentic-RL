# SEARCH-R1 Technical Specifications

## System Overview
SEARCH-R1 is an Agentic Reinforcement Learning system implementing **Group Relative Policy Optimization (GRPO)**. It trains a Qwen3-based language model to autonomously decide when and how to search for information.

## detailed Documentation

The technical specifications have been modularized into the following documents:

- **[GRPO Algorithm](docs/architecture/GRPO_ALGORITHM.md)**: Details the mathematical implementation of the GRPO algorithm, including advantage calculation and Policy/KL loss functions.
- **[Trajectory Management](docs/architecture/TRAJECTORY_MANAGEMENT.md)**: Explains the `TokenStep` and `Trajectory` data structures and how log probabilities are recomputed.
- **[Reward Design](docs/architecture/REWARD_DESIGN.md)**: Describes the result-oriented reward system, covering format correctness and answer accuracy schemas.
- **[Search Integration](docs/architecture/SEARCH_INTEGRATION.md)**: Documents the `SearchEngine` interface and how external search providers are integrated into the reasoning loop.

## Key Features
1. **Dynamic Search**: The model learns to emit `<search>` tags when internal knowledge is insufficient.
2. **Stable Training**: GRPO removes the need for a critic network, reducing memory usage and training instability.
3. **Structured Reasoning**: Enforces a strict XML-based thought process (`<think>`, `<search>`, `<information>`, `<answer>`).
