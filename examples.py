"""
Example scripts demonstrating different use cases of the LLM.
"""

import os
import sys
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import TextDataset, create_data_loaders
from models import create_model
from utils import generate_text, get_device


def example_1_custom_dataset():
    """
    Example 1: Training on a custom text dataset.
    """
    print("Example 1: Custom Dataset Training")
    print("="*50)
    
    # Create a custom text file
    custom_text = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    How vexingly quick daft zebras jump!
    Waltz, bad nymph, for quick jigs vex.
    """ * 100  # Repeat to have enough data
    
    # Save to file
    os.makedirs("examples", exist_ok=True)
    custom_file = "examples/custom_text.txt"
    
    with open(custom_file, 'w') as f:
        f.write(custom_text)
    
    # Create dataset
    dataset = TextDataset(custom_file, sequence_length=32)
    print(f"Custom dataset vocabulary size: {dataset.vocab_size}")
    print(f"Sample characters: {dataset.chars[:20]}")
    
    # You can now train on this dataset:
    # python train.py --data_file examples/custom_text.txt --model_type mlp


def example_2_model_comparison():
    """
    Example 2: Comparing MLP vs Transformer architectures.
    """
    print("\nExample 2: Model Architecture Comparison")
    print("="*50)
    
    vocab_size = 100
    embedding_dim = 64
    hidden_dim = 128
    num_layers = 2
    sequence_length = 32
    
    # Create sample input
    sample_input = torch.randint(0, vocab_size, (1, sequence_length))
    
    # MLP Model
    mlp_model = create_model(
        "mlp", vocab_size, embedding_dim, hidden_dim, 
        num_layers, sequence_length
    )
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    
    # Transformer Model
    transformer_model = create_model(
        "transformer", vocab_size, embedding_dim, hidden_dim,
        num_layers, sequence_length, num_heads=4
    )
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    
    print(f"MLP Model Parameters: {mlp_params:,}")
    print(f"Transformer Model Parameters: {transformer_params:,}")
    print(f"Transformer is {transformer_params/mlp_params:.1f}x larger")
    
    # Test forward pass
    with torch.no_grad():
        mlp_output = mlp_model(sample_input)
        transformer_output = transformer_model(sample_input)
        
        print(f"Both models output shape: {mlp_output.shape}")


def example_3_hyperparameter_tuning():
    """
    Example 3: Different hyperparameter configurations.
    """
    print("\nExample 3: Hyperparameter Configurations")
    print("="*50)
    
    configs = {
        "tiny": {
            "embedding_dim": 32,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "batch_size": 16
        },
        "small": {
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "batch_size": 32
        },
        "medium": {
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "batch_size": 64
        },
        "large": {
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "batch_size": 32
        }
    }
    
    print("Model Size Configurations:")
    print("-" * 30)
    
    for name, cfg in configs.items():
        # Estimate model size
        vocab_size = 100  # Approximate
        sequence_length = 128
        
        model = create_model(
            "transformer",
            vocab_size=vocab_size,
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            sequence_length=sequence_length,
            num_heads=cfg["num_heads"]
        )
        
        params = sum(p.numel() for p in model.parameters())
        memory_mb = params * 4 / (1024**2)  # Rough estimate (4 bytes per param)
        
        print(f"{name.upper():>8}: {params:>8,} params, ~{memory_mb:.1f}MB")
        print(f"         Train with: python train.py --embedding_dim {cfg['embedding_dim']} "
              f"--hidden_dim {cfg['hidden_dim']} --num_layers {cfg['num_layers']} "
              f"--batch_size {cfg['batch_size']}")
        print()


def example_4_generation_strategies():
    """
    Example 4: Different text generation strategies.
    """
    print("\nExample 4: Text Generation Strategies")
    print("="*50)
    
    print("Temperature effects on generation:")
    print("- Low temperature (0.1-0.5): More deterministic, repetitive")
    print("- Medium temperature (0.6-1.0): Balanced creativity and coherence")
    print("- High temperature (1.1-2.0): More creative but less coherent")
    print()
    
    print("Example commands:")
    print("# Conservative generation")
    print("python generate.py --model_path models/best_model.pt --prompt 'To be' --temperature 0.3")
    print()
    print("# Balanced generation")
    print("python generate.py --model_path models/best_model.pt --prompt 'To be' --temperature 0.8")
    print()
    print("# Creative generation")
    print("python generate.py --model_path models/best_model.pt --prompt 'To be' --temperature 1.2")
    print()
    
    print("Multiple prompts from file:")
    print("python generate.py --model_path models/best_model.pt --prompts_file prompts.txt")


def example_5_scaling_guide():
    """
    Example 5: Guide for scaling up the model.
    """
    print("\nExample 5: Scaling Up Guide")
    print("="*50)
    
    scaling_tips = [
        "1. Data Scaling:",
        "   - Use larger datasets (Project Gutenberg, Wikipedia)",
        "   - Preprocess data to remove noise",
        "   - Consider data augmentation techniques",
        "",
        "2. Model Scaling:",
        "   - Increase embedding_dim: 256 → 512 → 1024",
        "   - Increase hidden_dim: 512 → 1024 → 2048",
        "   - Add more layers: 4 → 8 → 12",
        "   - More attention heads: 8 → 16 → 32",
        "",
        "3. Training Scaling:",
        "   - Use gradient accumulation for larger effective batch sizes",
        "   - Implement mixed precision training (fp16)",
        "   - Use multiple GPUs with DataParallel",
        "   - Longer sequences: 128 → 512 → 1024",
        "",
        "4. Advanced Features:",
        "   - Implement learning rate warmup",
        "   - Add weight decay and dropout tuning",
        "   - Use gradient checkpointing to save memory",
        "   - Implement beam search for generation",
        "",
        "5. Infrastructure:",
        "   - Use cloud GPUs (AWS, Google Cloud, Azure)",
        "   - Monitor training with Weights & Biases",
        "   - Set up experiment tracking",
        "   - Implement model parallelism for very large models"
    ]
    
    for tip in scaling_tips:
        print(tip)


def create_sample_prompts_file():
    """
    Create a sample prompts file for batch generation.
    """
    prompts = [
        "To be or not to be",
        "Once upon a time",
        "It was the best of times",
        "Call me Ishmael",
        "In the beginning",
        "The quick brown fox",
        "Four score and seven years ago",
        "We hold these truths to be self-evident"
    ]
    
    os.makedirs("examples", exist_ok=True)
    with open("examples/sample_prompts.txt", 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    
    print(f"Created sample prompts file: examples/sample_prompts.txt")


def main():
    """
    Run all examples.
    """
    print("LLM FROM SCRATCH - EXAMPLES AND TUTORIALS")
    print("="*60)
    
    example_1_custom_dataset()
    example_2_model_comparison()
    example_3_hyperparameter_tuning()
    example_4_generation_strategies()
    example_5_scaling_guide()
    
    print("\nCreating sample files...")
    create_sample_prompts_file()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED!")
    print("="*60)
    print("Check the 'examples/' directory for sample files.")


if __name__ == "__main__":
    main()