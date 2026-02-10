# Reward Design

This document explains the reward functions used to train the agent. The system uses a result-oriented reward approach.

## Structure

The total reward is the sum of:
1. **Format Reward**: For following the XML tag protocol.
2. **Answer Reward**: For correctness of the final answer.

```python
Total Reward = Format Reward + Answer Reward
```

## 1. Format Correctness
Validates the structure of the generated response.

**Rules**:
- Must contain exactly one `<answer>` pair.
- `<answer>` must be at the end.
- `<search>` and `<information>` tags must be paired correctly.

**Scoring**:
- **+0.5**: Format is correct.
- **-1.0**: Format is incorrect (invalid nesting, missing tags).

## 2. Answer Correctness
Validates the content of the answer against the ground truth.

**Scoring**:
- **+2.0**: Exact match with ground truth.
- **+1.0**: Fuzzy match similarity ≥ 0.5.
- **+0.5**: Correctly identified "No information found" (when appropriate).
- **0.0**: Incorrect answer.

## Customization
The reward functions are modular and can be extended to include:
- **Process Rewards**: Rewarding valid search queries.
- **Efficiency Rewards**: Penalizing excessive token usage.
- **Utility Rewards**: Checking if `<information>` was actually used in `<answer>`.
