"""
Evaluation script for Search-R1 with OpenAI/GitHub API.
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import SearchR1Config
from scripts.trainer_openai import SearchR1TrainerOpenAI
from src.data.dataset import create_training_data

def main():
    config = SearchR1Config()
    config.use_openai = True
    
    # Check if key is set
    if not config.openai_api_key:
        print("Please set OPENAI_API_KEY environment variable or config.")
        return

    print("=" * 50)
    print("Search-R1 Evaluation with API")
    print("=" * 50)
    print(f"Model: {config.openai_model}")
    print(f"Provider: {config.model_provider}")
    print(f"Base URL: {config.openai_api_base}")
    
    queries, ground_truths = create_training_data()
    print(f"\nLoaded {len(queries)} evaluation examples")
    
    print("\nInitializing trainer...")
    try:
        trainer = SearchR1TrainerOpenAI(config)
    except Exception as e:
        print(f"Failed to initialize trainer: {e}")
        return

    print("\nStarting evaluation...")
    metrics = trainer.train_step(queries, ground_truths)
    
    print(f"\nAverage Reward: {metrics['avg_reward']:.4f}")
    
    if metrics['trajectories']:
        print(f"\nSample Trajectory:")
        t = metrics['trajectories'][0]
        print(f"Query: {t.query}")
        print(f"Answer: {t.final_answer}")
        print(f"Text: {t.generated_text[:200]}...")

if __name__ == "__main__":
    main()
