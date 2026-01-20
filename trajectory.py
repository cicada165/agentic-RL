"""
Trajectory data structures for Search-R1
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TokenStep:
    """Represents a single token generation step"""
    token_id: int  # Token ID
    token_text: str  # Token text
    log_prob: float  # Log probability of the token
    position: int  # Position in the sequence


@dataclass
class Trajectory:
    """Represents a complete trajectory for one query"""
    query: str  # The input query
    token_steps: List[TokenStep] = field(default_factory=list)  # Token generation steps
    generated_text: str = ""  # Complete generated text
    reward: float = 0.0  # Reward value
    final_answer: str = ""  # Final answer extracted from generated text
    full_input_ids: List[int] = field(default_factory=list)  # Full input sequence (prompt + generated + information)
    generated_positions: List[int] = field(default_factory=list)  # Position of each generated token
    advantage: float = 0.0  # Calculated advantage for this trajectory
