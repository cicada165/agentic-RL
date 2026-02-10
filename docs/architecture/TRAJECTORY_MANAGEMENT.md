# Trajectory Management

This document outlines the data structures used to track generation steps and trajectories in Search-R1.

## Data Structures

### TokenStep
Represents a single token generation step.

```python
@dataclass
class TokenStep:
    token_id: int          # Token ID from vocabulary
    token_text: str        # Decoded token text
    log_prob: float        # Log probability under generating policy
    position: int          # Absolute position in sequence
```
**Constraints**:
- `log_prob` is immutable after generation.
- `position` must account for inserted tokens (e.g., from search results).

### Trajectory
Encapsulates a complete generation path for a query.

```python
@dataclass
class Trajectory:
    query: str                                    # Input query
    token_steps: List[TokenStep]                 # generated tokens
    generated_text: str                           # Full text (including tags)
    reward: float                                 # Total computed reward
    final_answer: str                             # Extracted <answer> content
    full_input_ids: List[int]                    # Full sequence IDs
    generated_positions: List[int]                # Indices of generated tokens
    advantage: float                              # Computed group-relative advantage
```

## Log Probability Recomputation
During PPO updates, we need to calculate `new_log_probs` for the *same* actions under the *new* policy.

**Algorithm**:
1. Reconstruct the full input sequence (prompt + generated + injected info).
2. Perform a forward pass with the current model.
3. Extract logits corresponding to the positions of the originally generated tokens.
4. Compute `log_softmax` to get current log probabilities.

This allows us to compute the probability ratio `π_new / π_old` required for the PPO objective.
