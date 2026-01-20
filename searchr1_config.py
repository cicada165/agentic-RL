"""
Configuration file for Search-R1 training
Modify the model path here after obtaining the code
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchR1Config:
    """Configuration for Search-R1 training"""
    
    # Model configuration
    use_openai: bool = True  # Use OpenAI API instead of local model
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")  # OpenAI API key (from environment variable)
    openai_model: str = "gpt-4o-mini"  # OpenAI model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
    
    # Local model configuration (if use_openai=False)
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Modify this to your model path
    device: str = "cuda"  # or "cpu"
    
    # Training hyperparameters
    group_size: int = 2  # Number of trajectories to generate per query
    max_tokens: int = 500  # Maximum tokens to generate per trajectory
    update_times: int = 4  # Number of policy updates per batch
    
    # GRPO hyperparameters
    clip_epsilon: float = 0.2  # PPO clipping parameter
    beta: float = 0.1  # KL divergence regularization coefficient
    
    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 1  # Number of queries per batch
    num_epochs: int = 10
    
    # Reward configuration
    format_reward: float = 0.5  # Reward for correct format
    answer_reward: float = 1.0  # Reward for correct answer
    exact_match_bonus: float = 2.0  # Bonus for exact match
    
    # Gradient clipping
    max_grad_norm: float = 0.5
    
    # Logging
    log_interval: int = 1  # Log every N steps
