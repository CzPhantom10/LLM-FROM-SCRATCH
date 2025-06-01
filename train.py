"""
Training script for the character-level language model.
Supports both MLP and Transformer architectures with comprehensive logging and checkpointing.
"""

import argparse
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Import our modules
from config import config
from data_loader import TextDataset, create_data_loaders, download_shakespeare_data
from models import create_model, count_parameters
from utils import (
    set_seed, get_device, evaluate_model, calculate_loss,
    plot_training_curves, save_checkpoint, EarlyStopping,
    LearningRateScheduler, format_time, print_model_summary
)


class Trainer:
    """
    Trainer class for the language model.
    Handles training loop, validation, and checkpointing.
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Initialize the trainer.
        
        Args:
            model: Language model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            config: Configuration object
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        if config.lr_scheduler:
            self.lr_scheduler = LearningRateScheduler(
                self.optimizer,
                warmup_steps=min(1000, len(train_loader)),
                decay_factor=config.lr_decay_factor,
                decay_patience=config.lr_decay_patience
            )
        else:
            self.lr_scheduler = None
        
        # Initialize early stopping
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience
            )
        else:
            self.early_stopping = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create model save directory
        os.makedirs(config.model_save_path, exist_ok=True)
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            average_loss: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Calculate loss
            loss = calculate_loss(self.model, batch, self.device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}, "
                      f"Loss: {avg_loss:.4f}, "
                      f"LR: {current_lr:.6f}")
        
        average_loss = total_loss / num_batches
        return average_loss
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            average_loss: Average validation loss
        """
        return evaluate_model(self.model, self.val_loader, self.device)
    
    def train(self):
        """
        Main training loop.
        """
        print("Starting training...")
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate scheduler with validation loss
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Epoch Time: {format_time(epoch_time)}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")
            
            # Save model
            checkpoint_path = os.path.join(
                self.config.model_save_path,
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(
                self.model, self.optimizer, epoch + 1,
                train_loss, val_loss, checkpoint_path, is_best
            )
            
            # Early stopping check
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        plot_save_path = os.path.join(self.config.model_save_path, "training_curves.png")
        plot_training_curves(self.train_losses, self.val_losses, plot_save_path)
        
        # Save final model
        final_path = os.path.join(self.config.model_save_path, self.config.best_model_name)
        best_checkpoint_path = os.path.join(self.config.model_save_path, "checkpoint_epoch_1_best.pt")
        
        # Copy best model to final location
        if os.path.exists(best_checkpoint_path):
            import shutil
            shutil.copy2(best_checkpoint_path, final_path)
            print(f"Best model saved as: {final_path}")


def main():
    """
    Main function to set up and start training.
    """
    parser = argparse.ArgumentParser(description="Train Character-Level Language Model")
    parser.add_argument("--model_type", type=str, default="transformer",
                       choices=["mlp", "transformer"],
                       help="Model architecture to use")
    parser.add_argument("--data_file", type=str, default=None,
                       help="Custom data file path")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--embedding_dim", type=int, default=None,
                       help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of layers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Update config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.embedding_dim:
        config.embedding_dim = args.embedding_dim
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.num_layers:
        config.num_layers = args.num_layers
    
    print("Character-Level Language Model Training")
    print("=" * 50)
    print(config)
    
    # Get device
    device = get_device()
    config.device = str(device)
    
    # Download and prepare data
    if not args.data_file:
        download_shakespeare_data()
        data_file = config.data_file
    else:
        data_file = args.data_file
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Create dataset
    print(f"\nLoading dataset: {data_file}")
    dataset = TextDataset(data_file, config.sequence_length)
    config.vocab_size = dataset.vocab_size
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset)
    
    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    model = create_model(
        model_type=args.model_type,
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        sequence_length=config.sequence_length,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    
    # Print model summary
    sample_input = torch.randint(0, config.vocab_size, (1, config.sequence_length))
    print_model_summary(model, sample_input)
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {config.model_save_path}")


if __name__ == "__main__":
    main()
