"""
Format reward function for Search-R1.
Checks if the generated text follows the required XML structure.
"""

import re

def check_format_correctness(generated_text: str) -> float:
    """
    Validates XML tag structure and nesting.
    
    Rules:
    1. Must contain exactly one <answer> and one </answer> tag.
    2. <answer> must appear at the end of the text.
    3. If <search> appears, must have matching </search> and <information>...</information>.
    4. Search and information tags must appear in pairs.
    
    Args:
        generated_text: Complete generated text with XML tags.
        
    Returns:
        -1.0 if format is incorrect (invalid structure).
        0.5 if format is correct (proper tag nesting and closure).
    """
    # Check for exactly one <answer> block
    answer_blocks = re.findall(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
    if len(answer_blocks) != 1:
        return -1.0
        
    # Check if <answer> is at the very end (ignoring whitespace)
    if not generated_text.rstrip().endswith('</answer>'):
        return -1.0

    # Extract all tags to check order and nesting
    tags = re.findall(r'</?(?:search|information|answer|think)>', generated_text)
    
    # Basic tag balance check
    if tags.count('<search>') != tags.count('</search>'):
        return -1.0
    if tags.count('<information>') != tags.count('</information>'):
        return -1.0
        
    # Check search/information pairing
    # Search and information tags must appear in pairs
    if tags.count('<search>') != tags.count('<information>'):
        return -1.0
    
    return 0.5
