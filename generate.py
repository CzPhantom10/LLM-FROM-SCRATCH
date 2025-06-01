"""
Text generation script for the trained character-level language model.
Loads a trained model and generates text based on a given prompt.
"""

import argparse
import os
import torch
from typing import Optional

# Import our modules
from config import config
from data_loader import TextDataset
from models import create_model
from utils import generate_text, get_device, load_checkpoint


def load_trained_model(model_path: str, data_file: str, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model
        data_file: Path to the data file (needed for vocabulary)
        device: Device to load model on
        
    Returns:
        model: Loaded model
        dataset: Dataset (for tokenization)
    """
    # Load dataset to get vocabulary
    dataset = TextDataset(data_file, config.sequence_length)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    
    # Get model parameters from checkpoint or use defaults
    vocab_size = model_config.get('vocab_size', dataset.vocab_size)
    embedding_dim = model_config.get('embedding_dim', config.embedding_dim)
    sequence_length = model_config.get('sequence_length', config.sequence_length)
    
    # Determine model type based on checkpoint (simple heuristic)
    # In a real implementation, you'd save this information
    state_dict = checkpoint['model_state_dict']
    
    # Check if it's a transformer (has attention layers)
    is_transformer = any('attention' in key for key in state_dict.keys())
    model_type = "transformer" if is_transformer else "mlp"
    
    print(f"Detected model type: {model_type}")
    
    # Create model
    model = create_model(
        model_type=model_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        sequence_length=sequence_length,
        num_heads=config.num_heads,
        dropout=0.0  # No dropout during inference
    )
    
    # Load model weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Vocabulary size: {vocab_size}")
    
    return model, dataset


def interactive_generation(model, dataset, device, temperature=0.8):
    """
    Interactive text generation mode.
    
    Args:
        model: Trained model
        dataset: Dataset for tokenization
        device: Device
        temperature: Sampling temperature
    """
    print("\n" + "="*60)
    print("INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("Commands:")
    print("  - 'temp X.X' to change temperature (e.g., 'temp 1.2')")
    print("  - 'length XXX' to change generation length (e.g., 'length 200')")
    print("  - 'quit' to exit")
    print("-"*60)
    
    generation_length = 100
    
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    temperature = new_temp
                    print(f"Temperature set to: {temperature}")
                    continue
                except (ValueError, IndexError):
                    print("Invalid temperature. Use format: temp 0.8")
                    continue
            
            if user_input.startswith('length '):
                try:
                    new_length = int(user_input.split()[1])
                    generation_length = new_length
                    print(f"Generation length set to: {generation_length}")
                    continue
                except (ValueError, IndexError):
                    print("Invalid length. Use format: length 100")
                    continue
            
            if not user_input:
                print("Please enter a prompt.")
                continue
            
            # Generate text
            print(f"\nGenerating (temp={temperature}, length={generation_length})...")
            print("-" * 40)
            
            generated = generate_text(
                model=model,
                tokenizer=dataset,
                prompt=user_input,
                max_length=generation_length,
                temperature=temperature,
                device=device
            )
            
            print(generated)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")


def batch_generation(model, dataset, device, prompts, temperature, generation_length):
    """
    Generate text for multiple prompts.
    
    Args:
        model: Trained model
        dataset: Dataset for tokenization
        device: Device
        prompts: List of prompts
        temperature: Sampling temperature
        generation_length: Length of generated text
    """
    print(f"\nGenerating text for {len(prompts)} prompts...")
    print(f"Temperature: {temperature}")
    print(f"Generation length: {generation_length}")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 40)
        
        generated = generate_text(
            model=model,
            tokenizer=dataset,
            prompt=prompt,
            max_length=generation_length,
            temperature=temperature,
            device=device
        )
        
        print(generated)
        print("="*60)


def main():
    """
    Main function for text generation.
    """
    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--data_file", type=str, default=None,
                       help="Path to the data file (for vocabulary)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="File containing multiple prompts (one per line)")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--length", type=int, default=100,
                       help="Length of generated text")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive generation mode")
    parser.add_argument("--output_file", type=str, default=None,
                       help="File to save generated text")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Set data file
    if not args.data_file:
        args.data_file = config.data_file
    
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    print("Character-Level Language Model - Text Generation")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, dataset = load_trained_model(args.model_path, args.data_file, device)
    
    # Prepare output file if specified
    output_file = None
    if args.output_file:
        output_file = open(args.output_file, 'w', encoding='utf-8')
        print(f"Output will be saved to: {args.output_file}")
    
    try:
        if args.interactive:
            # Interactive mode
            interactive_generation(model, dataset, device, args.temperature)
        
        elif args.prompts_file:
            # Multiple prompts from file
            if not os.path.exists(args.prompts_file):
                raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
            
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            if not prompts:
                raise ValueError("No prompts found in the file")
            
            batch_generation(model, dataset, device, prompts, args.temperature, args.length)
        
        elif args.prompt:
            # Single prompt
            print(f"Prompt: {args.prompt}")
            print(f"Temperature: {args.temperature}")
            print(f"Generation length: {args.length}")
            print("-" * 40)
            
            generated = generate_text(
                model=model,
                tokenizer=dataset,
                prompt=args.prompt,
                max_length=args.length,
                temperature=args.temperature,
                device=device
            )
            
            print(generated)
            
            if output_file:
                output_file.write(f"Prompt: {args.prompt}\n")
                output_file.write(f"Temperature: {args.temperature}\n")
                output_file.write(f"Length: {args.length}\n")
                output_file.write("-" * 40 + "\n")
                output_file.write(generated + "\n")
        
        else:
            # Default prompts
            default_prompts = [
                "To be or not to be",
                "Once upon a time",
                "The quick brown fox",
                "In the beginning",
                "It was the best of times"
            ]
            
            print("No prompt specified. Using default prompts:")
            batch_generation(model, dataset, device, default_prompts, args.temperature, args.length)
    
    finally:
        if output_file:
            output_file.close()
            print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
