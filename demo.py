"""
Quick demo script to test the entire pipeline.
This script downloads data, trains a small model, and generates text.
"""

import os
import sys
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import download_shakespeare_data, TextDataset, create_data_loaders
from models import create_model
from train import Trainer
from utils import get_device, generate_text, set_seed


def quick_demo():
    """
    Run a quick demo of the entire pipeline.
    Uses small model parameters for fast training.
    """
    print("="*60)
    print("CHARACTER-LEVEL LANGUAGE MODEL - QUICK DEMO")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Override config for quick demo
    config.embedding_dim = 64
    config.hidden_dim = 128
    config.num_layers = 2
    config.num_heads = 4
    config.sequence_length = 64
    config.batch_size = 32
    config.num_epochs = 5
    config.learning_rate = 1e-3
    config.log_interval = 50
    
    # Get device
    device = get_device()
    
    # Download data
    print("\n1. Downloading Shakespeare data...")
    download_shakespeare_data()
    
    if not os.path.exists(config.data_file):
        print(f"Error: Could not download data to {config.data_file}")
        return
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = TextDataset(config.data_file, config.sequence_length)
    config.vocab_size = dataset.vocab_size
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset, train_split=0.8)
    
    # Create model
    print("\n3. Creating Transformer model...")
    model = create_model(
        model_type="transformer",
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        sequence_length=config.sequence_length,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n4. Training model...")
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()
    
    # Generate text
    print("\n5. Generating text...")
    prompts = [
        "ROMEO:",
        "To be or not to be",
        "Once upon a time"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        generated = generate_text(
            model=model,
            tokenizer=dataset,
            prompt=prompt,
            max_length=100,
            temperature=0.8,
            device=device
        )
        print(generated)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("You can now:")
    print("1. Train a larger model with: python train.py --model_type transformer")
    print("2. Generate text with: python generate.py --model_path models/best_model.pt --prompt 'Your prompt'")
    print("3. Try the interactive mode: python generate.py --model_path models/best_model.pt --interactive")


if __name__ == "__main__":
    quick_demo()
