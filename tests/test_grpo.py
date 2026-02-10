"""
Unit tests for GRPO logic.
"""
import unittest
import torch
from src.utils.config import SearchR1Config
from src.core.grpo import SearchR1Trainer

# Mock objects to avoid loading actual models
class MockDataset:
    pass

class TestGRPO(unittest.TestCase):
    
    def setUp(self):
        self.config = SearchR1Config(
            model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu" 
        )
        # We can't easily instantiate Trainer without loading model, 
        # so we'll test the static/detached methods if possible or use a mock.
        # For this test, we'll verify import and basic config.
        pass

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SearchR1Config(group_size=1)

    def test_advantage_calculation(self):
        # We can test the math logic by isolating it, 
        # but the method is an instance method of Trainer.
        # Let's mock the trainer if needed, or just subclass and override __init__
        
        class MockTrainer(SearchR1Trainer):
            def __init__(self, config):
                self.config = config
                self.device = torch.device("cpu")
                # Skip model loading
        
        trainer = MockTrainer(self.config)
        
        rewards = [1.0, 2.0, 3.0]
        advantages = trainer.compute_advantages(rewards)
        
        # Mean = 2.0, Std (pop) = 0.816...
        # Adv = [-1.22, 0.0, 1.22]
        self.assertTrue(torch.is_tensor(advantages))
        self.assertEqual(len(advantages), 3)
        self.assertAlmostEqual(advantages[1].item(), 0.0, places=5)
        
    def test_kl_divergence(self):
        class MockTrainer(SearchR1Trainer):
            def __init__(self, config):
                self.config = config
                self.device = torch.device("cpu")

        trainer = MockTrainer(self.config)
        
        old_probs = torch.tensor([0.0, 0.0]) # log(1)
        new_probs = torch.tensor([0.0, 0.0])
        
        kl = trainer.compute_kl_divergence(old_probs, new_probs)
        self.assertAlmostEqual(kl.item(), 0.0)
        
        # Test slightly different
        new_probs = torch.tensor([-0.1, -0.1])
        # ratio = exp(0.1) = 1.105
        # log_ratio = 0.1
        # k3 = 1.105 - 0.1 - 1 = 0.005
        kl = trainer.compute_kl_divergence(old_probs, new_probs)
        self.assertGreater(kl.item(), 0.0)

if __name__ == "__main__":
    unittest.main()
