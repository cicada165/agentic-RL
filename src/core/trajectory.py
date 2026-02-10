"""
Trajectory data structures for Search-R1.
This module defines the core data structures used to track generation steps and complete trajectories
during the reinforcement learning process.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class TokenStep:
    """
    Represents a single token generation step in a trajectory.
    
    Attributes:
        token_id: The ID of the token from the vocabulary.
        token_text: The decoded text representation of the token.
        log_prob: The log probability of this token under the current policy.
        position: The absolute position in the full sequence.
    """
    token_id: int
    token_text: str
    log_prob: float
    position: int


@dataclass
class Trajectory:
    """
    Represents a complete generation trajectory for one query used in GRPO training.
    
    Attributes:
        query: The input query string.
        token_steps: Ordered list of generated token steps.
        generated_text: Complete generated text (concatenated tokens).
        reward: Total reward (format + answer correctness).
        final_answer: Extracted answer from <answer> tags.
        full_input_ids: Full tokenized sequence (prompt + generated + information).
        generated_positions: Positions of generated tokens in full_input_ids.
        advantage: Group-relative advantage (computed post-generation).
    """
    query: str
    token_steps: List[TokenStep] = field(default_factory=list)
    generated_text: str = ""
    reward: float = 0.0
    final_answer: str = ""
    full_input_ids: List[int] = field(default_factory=list)
    generated_positions: List[int] = field(default_factory=list)
    advantage: float = 0.0

    def add_step(self, step: TokenStep) -> None:
        """
        Add a generation step to the trajectory.
        
        Args:
            step: The TokenStep to add.
        """
        self.token_steps.append(step)

    def compute_log_prob_sum(self) -> float:
        """
        Compute the sum of log probabilities for all steps in the trajectory.
        
        Returns:
            Sum of log probabilities.
        """
        return sum(step.log_prob for step in self.token_steps)
