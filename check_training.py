#!/usr/bin/env python3
import torch
import os

# Check the latest checkpoint
checkpoint_path = 'models/checkpoint_epoch_5_best.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Latest Model Info:")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'Unknown'):.4f}" if checkpoint.get('loss') else "Loss: Unknown")
    
    # Count parameters
    if 'model_state_dict' in checkpoint:
        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        print(f"Total parameters: {total_params:,}")
        
        # Show some key dimensions
        state_dict = checkpoint['model_state_dict']
        if 'embedding.weight' in state_dict:
            vocab_size, embed_dim = state_dict['embedding.weight'].shape
            print(f"Vocabulary size: {vocab_size}")
            print(f"Embedding dimension: {embed_dim}")
            
        # Count transformer layers
        layer_count = len([k for k in state_dict.keys() if k.startswith('transformer_blocks.')])
        if layer_count > 0:
            # Get unique layer indices
            layer_indices = set()
            for k in state_dict.keys():
                if k.startswith('transformer_blocks.'):
                    layer_idx = int(k.split('.')[1])
                    layer_indices.add(layer_idx)
            print(f"Number of transformer layers: {len(layer_indices)}")

# Check if the larger model exists
larger_model_paths = [
    'models/shakespeare_large_checkpoint.pt',
    'models/shakespeare_improved_checkpoint.pt'
]

for path in larger_model_paths:
    if os.path.exists(path):
        print(f"\nFound larger model: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            print(f"Parameters: {total_params:,}")
