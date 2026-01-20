"""
Main training/evaluation script for Search-R1 with OpenAI API
Note: OpenAI models cannot be updated, so this focuses on evaluation and reward computation.
For full RL training, you would need a local fine-tunable model.
"""
import os
from searchr1_config import SearchR1Config
from trainer_openai import SearchR1TrainerOpenAI
from data import create_training_data


def main():
    """Main evaluation function"""
    # Load configuration
    config = SearchR1Config()
    config.use_openai = True  # Ensure OpenAI is enabled
    
    # Print configuration
    print("=" * 50)
    print("Search-R1 Evaluation with OpenAI API")
    print("=" * 50)
    print(f"Model: {config.openai_model}")
    print(f"Group Size: {config.group_size}")
    print(f"Max Tokens: {config.max_tokens}")
    print("=" * 50)
    print("NOTE: OpenAI models cannot be updated.")
    print("This script evaluates trajectories and computes rewards/advantages.")
    print("For full RL training, use a local fine-tunable model.")
    print("=" * 50)
    
    # Create training data
    queries, ground_truths = create_training_data()
    print(f"\nLoaded {len(queries)} evaluation examples")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SearchR1TrainerOpenAI(config)
    
    # Evaluation loop
    print("\nStarting evaluation...")
    print("-" * 50)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Process in batches
        for i in range(0, len(queries), config.batch_size):
            batch_queries = queries[i:i + config.batch_size]
            batch_truths = ground_truths[i:i + config.batch_size]
            
            # Evaluation step
            metrics = trainer.train_step(batch_queries, batch_truths)
            
            # Log metrics
            if (i // config.batch_size) % config.log_interval == 0:
                print(f"  Step {i // config.batch_size + 1}:")
                print(f"    Avg Reward: {metrics['avg_reward']:.4f}")
                print(f"    Avg Tokens: {metrics['avg_tokens']:.2f}")
                print(f"    Search Trajectories: {metrics['search_trajectories']:.2%}")
                print(f"    Note: {metrics.get('note', 'N/A')}")
                
                # Show sample trajectories
                if metrics['trajectories']:
                    print(f"\n    Sample Trajectory:")
                    sample = metrics['trajectories'][0]
                    print(f"      Query: {sample.query}")
                    print(f"      Reward: {sample.reward:.4f}")
                    print(f"      Advantage: {sample.advantage:.4f}")
                    print(f"      Answer: {sample.final_answer[:100]}...")
        
        print("-" * 50)
    
    print("\nEvaluation completed!")
    print("\nTo perform actual RL training, you would need:")
    print("1. A local fine-tunable model (e.g., Qwen, Llama)")
    print("2. Use the original trainer.py with use_openai=False")
    print("3. The model can then be updated using GRPO algorithm")


if __name__ == "__main__":
    main()
