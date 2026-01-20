# SEARCH-R1 Technical Specifications

## System Overview

SEARCH-R1 is an Agentic Reinforcement Learning system implementing Group Relative Policy Optimization (GRPO) to train a Qwen3-based language model to autonomously decide when and how to search for information using XML-structured tokens before answering queries.

## Architecture Principles

1. **No Critic Network**: Uses group-relative advantage calculation with mean/std normalization
2. **Token Protocol**: Strict XML structure (`<think>`, `<search>`, `<information>`, `<answer>`)
3. **K3 KL Approximation**: Efficient KL divergence calculation for GRPO
4. **Config-Driven**: All hyperparameters externalized, no hardcoding
5. **Modular Design**: SearchEngine is swappable for real API integration

---

## Data Structures

### TokenStep Schema

```python
@dataclass
class TokenStep:
    """Represents a single token generation step in a trajectory"""
    token_id: int          # Token ID from vocabulary
    token_text: str        # Decoded token text
    log_prob: float        # Log probability of this token under current policy
    position: int          # Absolute position in the full sequence (prompt + generated + information)
```

**Purpose**: Tracks individual token decisions during generation for policy gradient computation.

**Constraints**:
- `log_prob` must be computed from model's log_softmax output
- `position` must account for inserted `<information>` tags from search results
- All fields are immutable after creation

---

### Trajectory Schema

```python
@dataclass
class Trajectory:
    """Represents a complete generation trajectory for one query"""
    query: str                                    # Input query string
    token_steps: List[TokenStep]                 # Ordered list of generated token steps
    generated_text: str                           # Complete generated text (concatenated tokens)
    reward: float                                 # Total reward (format + answer correctness)
    final_answer: str                             # Extracted answer from <answer> tags
    full_input_ids: List[int]                    # Full tokenized sequence (prompt + generated + information)
    generated_positions: List[int]                # Positions of generated tokens in full_input_ids
    advantage: float                              # Group-relative advantage (computed post-generation)
```

**Purpose**: Encapsulates a complete generation path for GRPO training.

**Constraints**:
- `token_steps` must be in generation order
- `generated_text` must include all XML tags and search results
- `full_input_ids` must include prompt, generated tokens, and inserted information tokens
- `advantage` is computed after group sampling, not during generation
- `reward` is computed after generation using reward functions

**Scalability Considerations**:
- Trajectories are in-memory during training; batch processing recommended for large datasets
- Consider trajectory serialization/deserialization for distributed training
- Memory-efficient storage: only store token_ids and recompute log_probs on-demand if needed

---

## SearchEngine Interface

### Abstract Interface

```python
class SearchEngine(ABC):
    """Abstract interface for search functionality"""
    
    @abstractmethod
    def search(self, query: str) -> str:
        """
        Search for information given a query string.
        
        Args:
            query: Search query extracted from <search> tags
            
        Returns:
            Retrieved information string to be inserted into <information> tags
            
        Raises:
            SearchError: If search fails (optional, can return empty string)
        """
        pass
```

### Current Implementation: MockSearchEngine

**Data Structure**:
- `knowledge_base: Dict[str, str]` - Keyword-to-information mapping

**Behavior**:
- Keyword matching (case-insensitive)
- Returns information string or "No information found for: {query}"

### Future Scalability

**API Integration Pattern**:
```python
class WebSearchEngine(SearchEngine):
    """Production search engine using web APIs"""
    
    def __init__(self, api_key: str, provider: str = "google"):
        # Initialize API client
        pass
    
    def search(self, query: str) -> str:
        # Call external API
        # Handle rate limiting, retries, timeouts
        # Return formatted information
        pass
```

**Scalability Requirements**:
- Rate limiting and caching for API calls
- Async/parallel search execution for batch queries
- Error handling and fallback mechanisms
- Configurable timeout and retry policies

---

## Reward Functions

### Format Correctness Check

**Signature**:
```python
def check_format_correctness(generated_text: str) -> float:
    """
    Validates XML tag structure and nesting.
    
    Args:
        generated_text: Complete generated text with XML tags
        
    Returns:
        -1.0 if format is incorrect (invalid structure)
        0.5 if format is correct (proper tag nesting and closure)
    """
```

**Validation Rules**:
1. Must contain exactly one `<answer>` and one `</answer>` tag
2. `<answer>` must appear at the end of the text (after all other content)
3. If `<search>` appears, must have matching `</search> and `<information>...</information>`
4. Search and information tags must appear in pairs (both or neither)
5. `<think>` tags are optional and removed before validation

**Scalability**: O(n) where n is text length. Uses regex matching.

---

### Answer Correctness Check

**Signature**:
```python
def check_answer_correctness(final_answer: str, ground_truth: str) -> float:
    """
    Validates answer correctness using fuzzy matching.
    
    Args:
        final_answer: Extracted answer from <answer> tags
        ground_truth: Expected correct answer
        
    Returns:
        0.0 if incorrect (similarity < 0.5 or empty)
        1.0 if similar (0.5 <= similarity < 1.0)
        2.0 if exact match
        0.5 if answer is "未找到相关内容" (no content found)
    """
```

**Matching Algorithm**:
1. Exact string match → 2.0
2. Special case: "未找到相关内容" → 0.5
3. Preprocess: remove whitespace, lowercase
4. SequenceMatcher similarity ratio
5. Threshold: similarity >= 0.5 → 1.0, else → 0.0

**Scalability**: O(n*m) for SequenceMatcher where n, m are string lengths. Consider approximate matching for very long strings.

---

### Total Reward Computation

**Signature**:
```python
def compute_reward(trajectory: Trajectory, ground_truth: str) -> float:
    """
    Computes total reward for a trajectory.
    
    Args:
        trajectory: Trajectory object with generated_text and final_answer
        ground_truth: Expected correct answer
        
    Returns:
        Total reward = format_reward + answer_reward
        Range: [-1.0, 2.5] (format: -1.0 or 0.5, answer: 0.0, 0.5, 1.0, or 2.0)
    """
```

**Formula**: `reward = check_format_correctness(text) + check_answer_correctness(answer, truth)`

---

## GRPOTrainer Class Structure

### Class Definition

```python
class GRPOTrainer:
    """Main trainer implementing GRPO algorithm for Search-R1"""
    
    def __init__(self, config: SearchR1Config):
        """
        Initialize trainer with model, tokenizer, optimizer, and search engine.
        
        Args:
            config: Configuration object with all hyperparameters
        """
```

### Core Components

**Model Initialization**:
- Model: Qwen3-4B or larger (Qwen3-0.6B only for unit tests)
- Device: CUDA if available, else CPU
- Precision: float16 on CUDA, float32 on CPU
- Device mapping: "auto" for multi-GPU

**Optimizer**:
- Type: AdamW
- Learning rate: From config (default: 1e-5)
- Parameters: All model parameters

**Search Engine**:
- Type: SearchEngine instance (swappable)
- Initialized in constructor

**Prompt Template**:
- System prompt with XML tag instructions
- Format: ChatML format for Qwen models
- Includes examples of search and non-search scenarios

---

### API Signatures

#### Generation

```python
def generate_trajectory(self, query: str, max_tokens: int = None) -> Trajectory:
    """
    Generate a single trajectory for a query.
    
    Args:
        query: Input query string
        max_tokens: Maximum tokens to generate (defaults to config.max_tokens)
        
    Returns:
        Trajectory object with token_steps, generated_text, final_answer
        
    Behavior:
        - Token-by-token generation with log_prob tracking
        - Detects <search> tags and triggers search_engine.search()
        - Inserts <information> tags with search results
        - Stops at </answer> tag or max_tokens
        - Extracts final_answer from <answer> tags
    """
```

**Scalability**:
- Sequential generation (can be parallelized for batch generation)
- Memory: Stores full sequence in memory
- Consider streaming generation for very long sequences

---

#### Advantage Calculation

```python
def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
    """
    Calculate group-relative advantages using mean/std normalization.
    
    Args:
        rewards: List of reward values for trajectories in a group
        
    Returns:
        Tensor of advantages: A_i = (r_i - μ) / σ
        
    Formula:
        μ = mean(rewards)
        σ = std(rewards) + ε (population std, ε=1e-8)
        A_i = (r_i - μ) / σ
        
    Edge Cases:
        - Single trajectory: returns [0.0] (no advantage)
        - Zero std: uses ε to prevent division by zero
    """
```

**Scalability**: O(n) where n is group size. Efficient for groups up to 100+ trajectories.

---

#### KL Divergence (K3 Approximation)

```python
def compute_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate K3 approximation of KL divergence.
    
    Args:
        old_log_probs: Log probabilities from old policy (at generation time)
        new_log_probs: Log probabilities from current policy (after update)
        
    Returns:
        Scalar tensor: mean KL divergence across tokens
        
    Formula:
        ratio = exp(old_log_probs - new_log_probs)  # π_old / π_new
        k3 = ratio - log(ratio) - 1
        return mean(k3)
        
    Note:
        This is the K3 approximation: KL(π_old || π_new) ≈ (π_old/π_new) - log(π_old/π_new) - 1
    """
```

**Scalability**: O(m) where m is number of tokens. Computationally efficient.

---

#### Policy Update

```python
def update_policy(self, trajectories: List[Trajectory]) -> Dict[str, float]:
    """
    Update policy using GRPO algorithm with PPO-style clipping.
    
    Args:
        trajectories: List of trajectories with pre-computed advantages
        
    Returns:
        Dictionary with keys: "loss", "kl_div", "avg_reward", "beta"
        
    Algorithm:
        1. Extract old_log_probs from trajectory.token_steps
        2. For each update iteration (config.update_times):
           a. Recompute new_log_probs using current model
           b. Calculate probability ratio: ratio = exp(new_log_probs - old_log_probs)
           c. Apply PPO clipping: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
           d. Policy loss: -mean(clipped_objective)
           e. KL penalty: beta * mean(K3_kl_divergence)
           f. Total loss: policy_loss + kl_penalty
           g. Backpropagate, clip gradients, update parameters
        3. Clear CUDA cache
        4. Return metrics
        
    Constraints:
        - Uses group-relative advantages (no separate value network)
        - Applies gradient clipping (config.max_grad_norm)
        - Calls torch.cuda.empty_cache() after updates
        - Handles mismatched sequence lengths gracefully
    """
```

**Scalability**:
- Memory: Stores all trajectories in memory during update
- Computation: O(update_times * num_trajectories * num_tokens)
- Consider gradient accumulation for very large batches

---

#### Training Step

```python
def train_step(self, queries: List[str], ground_truths: List[str], group_size: int = None) -> Dict[str, float]:
    """
    Execute one complete training step with group sampling.
    
    Args:
        queries: List of input queries (batch)
        ground_truths: List of ground truth answers (same length as queries)
        group_size: Number of trajectories per query (defaults to config.group_size)
        
    Returns:
        Dictionary with training metrics:
        - "loss": Policy loss value
        - "kl_div": Average KL divergence
        - "avg_reward": Average reward across all trajectories
        - "avg_tokens": Average tokens per trajectory
        - "search_trajectories": Fraction of trajectories that used search
        - "trajectories": List of all trajectories (for logging/debugging)
        
    Algorithm:
        1. For each (query, ground_truth) pair:
           a. Generate group_size trajectories
           b. Compute rewards for each trajectory
           c. Calculate group statistics (mean, std)
           d. Compute relative advantages: (r_i - μ) / σ
           e. Store advantage in trajectory.advantage
        2. Collect all trajectories
        3. Call update_policy(trajectories)
        4. Return metrics with additional statistics
    """
```

**Scalability**:
- Parallel generation: Can generate multiple trajectories concurrently
- Batch processing: Handles multiple queries per step
- Memory: All trajectories stored until update completes

---

#### Log Probability Recomputation

```python
def recompute_log_probs(self, trajectories: List[Trajectory]) -> List[torch.Tensor]:
    """
    Recompute log probabilities using current model parameters.
    
    Args:
        trajectories: List of trajectories with token_steps
        
    Returns:
        List of log probability tensors (one per trajectory)
        
    Algorithm:
        For each trajectory:
        1. Reconstruct prompt from trajectory.query
        2. Tokenize prompt
        3. For each token_step in trajectory.token_steps:
           a. Forward pass with current model
           b. Extract log_prob for token_step.token_id
           c. Append token to input for next step
        4. Return list of log_prob tensors
        
    Purpose:
        Used during policy updates to compute new policy probabilities
        after model parameters have changed.
    """
```

**Scalability**: O(num_trajectories * num_tokens). Can be parallelized across trajectories.

---

## Configuration Schema

### SearchR1Config

```python
@dataclass
class SearchR1Config:
    """Configuration for Search-R1 training - all hyperparameters externalized"""
    
    # Model Configuration
    use_openai: bool = False                    # Use OpenAI API vs local model
    openai_api_key: str = ""                    # OpenAI API key (from env var)
    openai_model: str = "gpt-4o-mini"           # OpenAI model name
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Local model path
    device: str = "cuda"                        # "cuda" or "cpu"
    
    # Generation Parameters
    max_tokens: int = 500                       # Maximum tokens per trajectory
    
    # GRPO Hyperparameters
    group_size: int = 2                         # Trajectories per query (must be >= 2)
    update_times: int = 4                       # Policy updates per batch
    clip_epsilon: float = 0.2                   # PPO clipping parameter
    beta: float = 0.1                           # KL divergence regularization coefficient
    
    # Optimization
    learning_rate: float = 1e-5                 # AdamW learning rate
    batch_size: int = 1                         # Queries per batch
    num_epochs: int = 10                        # Training epochs
    max_grad_norm: float = 0.5                  # Gradient clipping threshold
    
    # Reward Configuration
    format_reward: float = 0.5                  # Reward for correct format
    answer_reward: float = 1.0                  # Reward for correct answer
    exact_match_bonus: float = 2.0              # Bonus for exact match
    
    # Logging
    log_interval: int = 1                       # Log every N steps
```

**Constraints**:
- `group_size` must be >= 2 for meaningful advantage calculation
- `beta` controls KL penalty strength (higher = more conservative updates)
- `clip_epsilon` standard PPO value (0.1-0.3 range)
- `model_name_or_path` must be Qwen3-4B+ for production (0.6B only for tests)

**Scalability**:
- Config can be loaded from YAML/JSON files
- Environment variable support for sensitive values (API keys)
- Validation on initialization

---

## Training Data Schema

### Data Structure

```python
def create_training_data() -> Tuple[List[str], List[str]]:
    """
    Create training data pairs.
    
    Returns:
        Tuple of (queries: List[str], ground_truths: List[str])
        
    Format:
        queries: List of input questions/queries
        ground_truths: List of expected correct answers (same length)
    """
```

**Scalability**:
- Can be extended to load from files (JSON, CSV, etc.)
- Supports streaming data loaders for large datasets
- Consider data augmentation and validation splits

---

## Scalability Design Considerations

### Memory Management

1. **Trajectory Storage**: 
   - In-memory during training step
   - Consider checkpointing trajectories for large batches
   - Streaming generation for very long sequences

2. **Model Memory**:
   - Use gradient checkpointing for large models
   - Mixed precision training (float16)
   - `torch.cuda.empty_cache()` after updates

3. **Batch Processing**:
   - Configurable batch_size for memory constraints
   - Gradient accumulation for effective larger batches

### Computational Scalability

1. **Parallel Generation**:
   - Trajectories can be generated in parallel (different queries)
   - Consider async/threading for search engine calls
   - Batch inference for multiple trajectories

2. **Distributed Training**:
   - Multi-GPU support via device_map="auto"
   - Consider DDP for multi-node training
   - Trajectory serialization for distributed setups

3. **Search Engine**:
   - Caching for repeated queries
   - Rate limiting and batching for API calls
   - Async execution for non-blocking searches

### Data Scalability

1. **Data Loading**:
   - Streaming data loaders for large datasets
   - Shuffling and batching utilities
   - Validation/test splits

2. **Checkpointing**:
   - Model checkpointing (save/load)
   - Training state persistence
   - Resume from checkpoint capability

### API Scalability

1. **SearchEngine Interface**:
   - Abstract base class for easy swapping
   - Plugin architecture for different providers
   - Rate limiting and retry logic

2. **Configuration**:
   - External config files (YAML/JSON)
   - Environment variable support
   - Validation and type checking

---

## Error Handling & Edge Cases

### Generation Edge Cases

- Empty query → Return empty trajectory
- Max tokens reached before </answer> → Truncate and extract partial answer
- Invalid XML structure → Format reward = -1.0
- Search query extraction fails → Continue without search

### Training Edge Cases

- Single trajectory in group → Advantage = 0.0
- Zero reward variance → Use ε in std calculation
- Mismatched sequence lengths → Skip trajectory in update
- CUDA OOM → Fallback to CPU or reduce batch size

### Search Engine Edge Cases

- API timeout → Return empty string or cached result
- Rate limit exceeded → Queue and retry
- Invalid query → Return "No information found"

---

## Testing Requirements

### Unit Tests

1. **TokenStep/Trajectory**: Data structure validation
2. **SearchEngine**: Mock search functionality
3. **Reward Functions**: Format and answer correctness
4. **Advantage Calculation**: Mean/std normalization
5. **K3 KL Divergence**: Mathematical correctness
6. **Policy Update**: PPO clipping logic

### Integration Tests

1. **End-to-end training step**: Query → Trajectory → Reward → Update
2. **Search integration**: Search trigger and information insertion
3. **Multi-query batching**: Group sampling across queries

### Performance Tests

1. **Memory usage**: Large batch sizes
2. **Generation speed**: Tokens per second
3. **Training throughput**: Steps per second

---

## Future Extensions

1. **Dense Rewards**: Process supervision (reward per step, not just final)
2. **Multi-turn Dialogue**: Conversation history in trajectories
3. **Multi-modal Search**: Image/document search capabilities
4. **Active Learning**: Query selection strategies
5. **Transfer Learning**: Pre-training on search tasks
6. **Evaluation Metrics**: BLEU, ROUGE, semantic similarity

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Status**: Architecture Complete - Ready for Implementation
