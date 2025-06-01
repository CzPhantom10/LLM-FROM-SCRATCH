"""
Utility functions for training, evaluation, and text generation.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple
import time


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def count_parameters(model):
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_loss(model, batch, device):
    """
    Calculate loss for a batch.
    
    Args:
        model: Language model
        batch: Batch of input and target sequences
        device: Device to run on
        
    Returns:
        loss: Calculated loss
    """
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Forward pass
    logits = model(inputs)
    
    # Reshape for loss calculation
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    
    # Calculate cross-entropy loss
    loss = F.cross_entropy(logits, targets)
    
    return loss


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Language model
        data_loader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        average_loss: Average loss on the dataset
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            loss = calculate_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    
    average_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return average_loss


def generate_text(model, tokenizer, prompt: str, max_length: int = 100, 
                 temperature: float = 1.0, device=None):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        device: Device to run on
        
    Returns:
        str: Generated text
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Encode the prompt
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(prompt)
    else:
        # Assume it's our custom dataset with char_to_idx
        input_ids = [tokenizer.char_to_idx.get(ch, 0) for ch in prompt]
    
    # Convert to tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated_text = prompt
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(input_ids)
            
            # Get the logits for the last token
            last_token_logits = logits[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(last_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Decode the next character
            if hasattr(tokenizer, 'decode'):
                next_char = tokenizer.decode([next_token_id])
            else:
                # Assume it's our custom dataset with idx_to_char
                next_char = tokenizer.idx_to_char.get(next_token_id, '?')
            
            generated_text += next_char
            
            # Update input for next prediction
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # Keep only the recent context (to prevent memory issues)
            max_context = 512  # Adjust based on your model's sequence length
            if input_ids.size(1) > max_context:
                input_ids = input_ids[:, -max_context:]
    
    return generated_text


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        save_path: str = None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                   save_path: str, is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': {
            'vocab_size': getattr(model, 'vocab_size', None),
            'embedding_dim': getattr(model, 'embedding_dim', None),
            'sequence_length': getattr(model, 'sequence_length', None),
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to: {best_path}")


def load_checkpoint(model, optimizer, checkpoint_path: str, device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint
        device: Device to load on
        
    Returns:
        epoch: Loaded epoch
        train_loss: Training loss from checkpoint
        val_loss: Validation loss from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    
    return epoch, train_loss, val_loss


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            bool: Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class LearningRateScheduler:
    """
    Learning rate scheduler with warmup and decay.
    """
    
    def __init__(self, optimizer, warmup_steps: int = 1000, 
                 decay_factor: float = 0.9, decay_patience: int = 5):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            decay_factor: Factor to multiply LR by when decaying
            decay_patience: Patience for LR decay
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_patience = decay_patience
        
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Store initial learning rate
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, val_loss: float = None):
        """
        Update learning rate.
        
        Args:
            val_loss: Current validation loss (for decay scheduling)
        """
        self.step_count += 1
        
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        # Decay phase (if validation loss is provided)
        elif val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.decay_patience:
                    # Decay learning rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.decay_factor
                    
                    print(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
                    self.patience_counter = 0


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def print_model_summary(model, sample_input):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
    """
    print("Model Summary")
    print("=" * 50)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
    
    print("=" * 50)


def main():
    """
    Test utility functions.
    """
    print("Testing Utility Functions")
    print("=" * 40)
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test time formatting
    test_times = [30, 90, 3661, 7322]
    for t in test_times:
        print(f"{t} seconds = {format_time(t)}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3)
    losses = [1.0, 0.8, 0.7, 0.75, 0.8, 0.9]
    
    print("\nTesting Early Stopping:")
    for i, loss in enumerate(losses):
        stop = early_stopping(loss)
        print(f"Epoch {i+1}: Loss = {loss:.2f}, Stop = {stop}")
        if stop:
            break


if __name__ == "__main__":
    main()
