"""
Main training script for Search-R1
Run this file after setting up the environment and modifying the model path in searchr1_config.py
"""
import os
from searchr1_config import SearchR1Config
from trainer import SearchR1Trainer
from data import create_training_data


def main():
    """Main training function"""
    # Load configuration
    config = SearchR1Config()
    
    # Print configuration
    print("=" * 50)
    print("Search-R1 Training Configuration")
    print("=" * 50)
    print(f"Model: {config.model_name_or_path}")
    print(f"Device: {config.device}")
    print(f"Group Size: {config.group_size}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Update Times: {config.update_times}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Clip Epsilon: {config.clip_epsilon}")
    print(f"Beta (KL weight): {config.beta}")
    print("=" * 50)
    
    # Create training data
    queries, ground_truths = create_training_data()
    print(f"\nLoaded {len(queries)} training examples")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SearchR1Trainer(config)
    
    # Training loop
    print("\nStarting training...")
    print("-" * 50)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Process in batches
        for i in range(0, len(queries), config.batch_size):
            batch_queries = queries[i:i + config.batch_size]
            batch_truths = ground_truths[i:i + config.batch_size]
            
            # Training step
            metrics = trainer.train_step(batch_queries, batch_truths)
            
            # Log metrics
            if (i // config.batch_size) % config.log_interval == 0:
                print(f"  Step {i // config.batch_size + 1}:")
                print(f"    Loss: {metrics['loss']:.4f}")
                print(f"    KL Div: {metrics['kl_div']:.4f}")
                print(f"    Avg Reward: {metrics['avg_reward']:.4f}")
                print(f"    Avg Tokens: {metrics['avg_tokens']:.2f}")
                print(f"    Search Trajectories: {metrics['search_trajectories']:.2%}")
        
        print("-" * 50)
    
    print("\nTraining completed!")
    
    # Save model (optional)
    # trainer.model.save_pretrained("./saved_model")
    # trainer.tokenizer.save_pretrained("./saved_model")


if __name__ == "__main__":
    main()
