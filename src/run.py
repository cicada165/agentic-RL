"""
Main training script for Search-R1.
"""
import os
import argparse
import torch
from src.utils.config import SearchR1Config
from src.core.grpo import SearchR1Trainer
from src.data.dataset import create_training_data, load_from_json
from src.agent.search_engine import DuckDuckGoSearchEngine, GoogleSearchEngine, MockSearchEngine

def parse_args():
    parser = argparse.ArgumentParser(description="Train Search-R1")
    parser.add_argument("--data_path", type=str, default=None, help="Path to JSON dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--search_engine", type=str, default="mock", choices=["mock", "duckduckgo", "google"], help="Search engine to use")
    parser.add_argument("--google_api_key", type=str, default=None, help="Google API Key")
    parser.add_argument("--google_cse_id", type=str, default=None, help="Google CSE ID")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--group_size", type=int, default=None, help="Group size for GRPO")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens for generation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = SearchR1Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    
    if args.group_size:
        config.group_size = args.group_size
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    
    # Override device if needed
    if config.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        config.device = "cpu"
        
    print("=" * 50)
    print("Search-R1 Training Configuration")
    print("=" * 50)
    print(f"Model: {config.model_name_or_path}")
    print(f"Device: {config.device}")
    print(f"Search Engine: {args.search_engine}")
    
    # Load Data
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        queries, ground_truths = load_from_json(args.data_path)
    else:
        print("Using default dummy data...")
        queries, ground_truths = create_training_data()
    print(f"Loaded {len(queries)} training examples")
    
    # Initialize Trainer
    print("\nInitializing trainer...")
    trainer = SearchR1Trainer(config)
    
    # Setup Search Engine
    if args.search_engine == "duckduckgo":
        trainer.search_engine = DuckDuckGoSearchEngine()
    elif args.search_engine == "google":
        trainer.search_engine = GoogleSearchEngine(api_key=args.google_api_key, cse_id=args.google_cse_id)
    else:
        trainer.search_engine = MockSearchEngine()
        
    # Resume from checkpoint
    start_epoch = 0
    if args.resume_from:
        start_epoch, _ = trainer.load_checkpoint(args.resume_from)
        print(f"Resuming from epoch {start_epoch}")

    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train
    print("\nStarting training...")
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        for i in range(0, len(queries), config.batch_size):
            batch_q = queries[i:i+config.batch_size]
            batch_gt = ground_truths[i:i+config.batch_size]
            
            metrics = trainer.train_step(batch_q, batch_gt)
            
            step = i // config.batch_size + 1
            if (i // config.batch_size) % config.log_interval == 0:
                print(f"  Step {step}:")
                print(f"    Loss: {metrics['loss']:.4f}")
                print(f"    KL Div: {metrics['kl_div']:.4f}")
                print(f"    Avg Reward: {metrics['avg_reward']:.4f}")
                print(f"    Avg Tokens: {metrics['avg_tokens']:.2f}")
                print(f"    Search Trajectories: {metrics['search_trajectories']:.2%}")
        
        # Save checkpoint at end of epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.ckpt")
        trainer.save_checkpoint(checkpoint_path, epoch+1, 0)

    print("\nTraining completed!")

if __name__ == "__main__":
    main()
