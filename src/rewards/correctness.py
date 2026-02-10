"""
Answer correctness reward function for Search-R1.
Checks if the generated answer matches the ground truth.
"""

from difflib import SequenceMatcher

def check_answer_correctness(final_answer: str, ground_truth: str) -> float:
    """
    Validates answer correctness using fuzzy matching.
    
    Args:
        final_answer: Extracted answer from <answer> tags.
        ground_truth: Expected correct answer.
        
    Returns:
        0.0 if incorrect (similarity < 0.5 or empty).
        1.0 if similar (0.5 <= similarity < 1.0).
        2.0 if exact match.
        0.5 if answer is "no content found" equivalent.
    """
    if not final_answer:
        return 0.0
        
    final_answer = final_answer.strip()
    ground_truth = ground_truth.strip()
    
    # Exact match
    if final_answer == ground_truth:
        return 2.0
        
    # Special case for "no info found"
    if "未找到相关内容" in final_answer:
        return 0.5
        
    # Fuzzy matching
    similarity = SequenceMatcher(None, final_answer, ground_truth).ratio()
    
    if similarity >= 0.5:
        return 1.0
        
    return 0.0
