"""
Unit tests for reward functions.
"""
import unittest
from src.rewards.format_reward import check_format_correctness
from src.rewards.correctness import check_answer_correctness

class TestRewards(unittest.TestCase):
    
    def test_format_correctness(self):
        # Correct format
        text = "<think>...</think><answer>Result</answer>"
        self.assertEqual(check_format_correctness(text), 0.5)
        
        # Correct format with search
        text = "<think>...</think><search>Q</search><information>A</information><answer>Result</answer>"
        self.assertEqual(check_format_correctness(text), 0.5)
        
        # Missing answer
        text = "<think>...</think>"
        self.assertEqual(check_format_correctness(text), -1.0)
        
        # Answer not at end
        text = "<answer>Result</answer>Extra"
        self.assertEqual(check_format_correctness(text), -1.0)
        
        # Mismatched tags
        text = "<search>Q</search><answer>Result</answer>"
        self.assertEqual(check_format_correctness(text), -1.0)

    def test_answer_correctness(self):
        # Exact match
        self.assertEqual(check_answer_correctness("Foo", "Foo"), 2.0)
        
        # Fuzzy match
        self.assertEqual(check_answer_correctness("Foo Bar", "Foo Bar Baz"), 1.0)
        
        # No match
        self.assertEqual(check_answer_correctness("Abc", "Xyz"), 0.0)
        
        # No content found
        self.assertEqual(check_answer_correctness("未找到相关内容", "Xyz"), 0.5)

if __name__ == "__main__":
    unittest.main()
