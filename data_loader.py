"""
Data loading and preprocessing utilities.
Downloads Shakespeare text data and creates datasets for training.
"""

import os
import requests
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from config import config


class TextDataset(Dataset):
    """
    Character-level text dataset.
    
    This dataset takes a text file and creates training examples by:
    1. Converting text to character indices
    2. Creating sliding windows of fixed length
    3. Each window becomes an input-target pair
    """
    
    def __init__(self, text_file: str, sequence_length: int = None):
        """
        Initialize the dataset.
        
        Args:
            text_file: Path to the text file
            sequence_length: Length of each training sequence
        """
        self.sequence_length = sequence_length or config.sequence_length
        
        # Read the text file
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        print(f"Loaded text file: {len(self.text):,} characters")
        
        # Create character-to-index mapping
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {self.chars[:20]}")
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in self.text]
        
        # Calculate number of sequences
        self.num_sequences = len(self.data) - self.sequence_length
        
        print(f"Number of training sequences: {self.num_sequences:,}")
    
    def __len__(self):
        """Return the number of sequences in the dataset"""
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Returns:
            input_seq: Input sequence of character indices
            target_seq: Target sequence (shifted by one character)
        """
        # Get input sequence
        input_seq = self.data[idx:idx + self.sequence_length]
        
        # Target sequence is shifted by one position
        target_seq = self.data[idx + 1:idx + self.sequence_length + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
    
    def get_char_mapping(self):
        """Return character mapping dictionaries"""
        return self.char_to_idx, self.idx_to_char
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def encode(self, text: str) -> List[int]:
        """Convert text to indices"""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]


def download_shakespeare_data():
    """
    Download Shakespeare text data from the internet.
    This is a popular dataset for character-level language modeling.
    """
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(config.data_file):
        print(f"Data file already exists: {config.data_file}")
        return
    
    print("Downloading Shakespeare data...")
    
    try:
        response = requests.get(config.data_url)
        response.raise_for_status()
        
        with open(config.data_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded Shakespeare data to: {config.data_file}")
        print(f"File size: {len(response.text):,} characters")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("You can manually download the file and place it in the data/ directory")


def create_data_loaders(dataset: TextDataset, train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        dataset: The text dataset
        train_split: Fraction of data to use for training
    
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Calculate split indices
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    print(f"Created data loaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    """
    Main function to download and prepare data.
    Run this script to download the Shakespeare dataset.
    """
    print("=" * 50)
    print("Data Preparation for Character-Level LM")
    print("=" * 50)
    
    # Download data
    download_shakespeare_data()
    
    # Create dataset
    if os.path.exists(config.data_file):
        dataset = TextDataset(config.data_file)
        
        # Update config with vocabulary size
        config.vocab_size = dataset.vocab_size
        
        # Show some sample data
        print("\nSample data:")
        sample_input, sample_target = dataset[0]
        print(f"Input:  {dataset.decode(sample_input.tolist())}")
        print(f"Target: {dataset.decode(sample_target.tolist())}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(dataset)
        
        print(f"\nData preparation complete!")
        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Training sequences: {len(train_loader.dataset)}")
        print(f"Validation sequences: {len(val_loader.dataset)}")
    else:
        print("Error: Could not find data file. Please check the download.")


if __name__ == "__main__":
    main()
