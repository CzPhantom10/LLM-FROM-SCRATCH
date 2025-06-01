import torch
import os

# Check the best model info
if os.path.exists('models/best_model.pt'):
    checkpoint = torch.load('models/best_model.pt', map_location='cpu')
    print('Best model info:')
    print(f'  Epoch: {checkpoint.get("epoch", "Unknown")}')
    print(f'  Loss: {checkpoint.get("loss", "Unknown")}')
    print(f'  Model state dict keys: {list(checkpoint["model_state_dict"].keys())[:5]}...')
    
    # Try to infer model size from state dict
    if 'transformer.layers.0.self_attn.in_proj_weight' in checkpoint['model_state_dict']:
        embed_dim = checkpoint['model_state_dict']['embedding.weight'].shape[1]
        print(f'  Embedding dim: {embed_dim}')
        
        # Count transformer layers
        layer_count = 0
        for key in checkpoint['model_state_dict'].keys():
            if key.startswith('transformer.layers.') and key.endswith('.self_attn.in_proj_weight'):
                layer_num = int(key.split('.')[2])
                layer_count = max(layer_count, layer_num + 1)
        print(f'  Number of layers: {layer_count}')
else:
    print('No best model found')
