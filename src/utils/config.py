"""
Configuration system for Search-R1.
This module defines the configuration dataclass used throughout the project.
"""

import os
from dataclasses import dataclass

@dataclass
class SearchR1Config:
    """
    Configuration for Search-R1 training.
    All hyperparameters are externalized here.
    """
    
    # Model Configuration
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Local model path
    device: str = "cuda"  # "cuda" or "cpu"
    
    # OpenAI / API Configuration
    use_openai: bool = False
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_base: str = "https://models.inference.ai.azure.com"  # Default for GitHub Models
    openai_model: str = "gpt-4o"  # Default model name
    model_provider: str = "github"  # "openai", "github", or "azure"
    
    # Generation Parameters
    max_tokens: int = 500  # Maximum tokens per trajectory
    
    # GRPO Hyperparameters
    group_size: int = 2  # Trajectories per query (must be >= 2)
    update_times: int = 4  # Policy updates per batch
    clip_epsilon: float = 0.2  # PPO clipping parameter
    beta: float = 0.1  # KL divergence regularization coefficient
    
    # Optimization
    learning_rate: float = 1e-5  # AdamW learning rate
    batch_size: int = 1  # Queries per batch
    num_epochs: int = 10  # Training epochs
    max_grad_norm: float = 0.5  # Gradient clipping threshold
    
    # Reward Configuration
    format_reward: float = 0.5
    answer_reward: float = 1.0
    exact_match_bonus: float = 2.0
    
    # Logging
    log_interval: int = 1  # Log every N steps

    def __post_init__(self):
        """Validation logic."""
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {self.group_size}")
