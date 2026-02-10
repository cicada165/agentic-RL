"""
Composite reward function for Search-R1.
Combines format correctness and answer correctness rewards.
"""

from src.core.trajectory import Trajectory
from src.rewards.format_reward import check_format_correctness
from src.rewards.correctness import check_answer_correctness

def compute_reward(trajectory: Trajectory, ground_truth: str) -> float:
    """
    Computes total reward for a trajectory.
    
    Args:
        trajectory: Trajectory object with generated_text and final_answer.
        ground_truth: Expected correct answer.
        
    Returns:
        Total reward = format_reward + answer_reward.
        Range: [-1.0, 2.5]
    """
    format_reward = check_format_correctness(trajectory.generated_text)
    
    # Only compute answer reward if format is correct enough? 
    # Current specs say strict sum, but if format is broken, final_answer might be empty/invalid.
    # However, Trajectory.final_answer should be extracted before calling this if possible,
    # or we rely on what's generated.
    
    answer_reward = check_answer_correctness(trajectory.final_answer, ground_truth)
    
    return format_reward + answer_reward
