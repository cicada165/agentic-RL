# GRPO Algorithm Implementation

This document details the implementation of the Group Relative Policy Optimization (GRPO) algorithm for Search-R1.

## Overview
GRPO uses group-relative advantages to optimize the policy without requiring a separate critic network.

## Core Components

### 1. Architecture Principles
- **No Critic Network**: Uses group-relative advantage calculation with mean/std normalization.
- **K3 KL Approximation**: Efficient KL divergence calculation.
- **PPO-style Clipping**: Ensures stable updates.

### 2. GRPOTrainer Logic

#### Advantage Calculation
We compute advantages relative to a group of trajectories for the same query.
```python
A_i = (r_i - μ) / (σ + ε)
```
Where:
- `r_i` is the reward for trajectory `i`.
- `μ` is the mean reward of the group.
- `σ` is the standard deviation of the group.
- `ε` is a small constant (1e-8) for stability.

#### KL Divergence (K3 Approximation)
We use the K3 approximation for KL divergence:
```python
ratio = exp(old_log_probs - new_log_probs)
k3 = ratio - log(ratio) - 1
kl_div = mean(k3)
```
This approximation: `KL(π_old || π_new) ≈ (π_old/π_new) - log(π_old/π_new) - 1`

#### Policy Update
The update step minimizes the following loss:
```python
Loss = PolicyLoss + β * KL_Penalty
```
Where `PolicyLoss` uses PPO clipping:
```python
ratio = exp(new_log_probs - old_log_probs)
surr1 = ratio * A
surr2 = clamp(ratio, 1-ε, 1+ε) * A
PolicyLoss = -mean(min(surr1, surr2))
```

### 3. Training Loop
1. **Generate**: Create `group_size` trajectories for each query in the batch.
2. **Reward**: Calculate rewards for each trajectory.
3. **Advantage**: Compute normalized advantages within each group.
4. **Update**:
   - Recompute log probabilities of the generated tokens under the current policy.
   - Calculate loss and gradients.
   - Update model parameters using AdamW.
