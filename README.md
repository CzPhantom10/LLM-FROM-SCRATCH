# Character-Level Language Model from Scratch

This project implements a complete character-level language model (mini-GPT) from scratch using PyTorch. It includes both simple MLP and Transformer architectures.

## Features

- 📝 Character-level tokenization
- 🧠 Two model architectures: MLP and Transformer
- 📊 Complete training pipeline with logging
- 🎯 Text generation capabilities
- 💾 Model saving and loading
- 📈 Training visualization
- 🔧 Easy to extend and modify

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and prepare data:**
   ```bash
   python data_loader.py
   ```

3. **Train the model:**
   ```bash
   # Train MLP model
   python train.py --model_type mlp

   # Train Transformer model
   python train.py --model_type transformer
   ```

4. **Generate text:**
   ```bash
   python generate.py --model_path models/best_model.pt --prompt "To be or not to be"
   ```

## Project Structure

```
├── data_loader.py      # Data downloading and preprocessing
├── tokenizer.py        # Character-level tokenizer
├── models.py          # MLP and Transformer architectures
├── train.py           # Training script
├── generate.py        # Text generation script
├── utils.py           # Utility functions
├── config.py          # Configuration settings
└── requirements.txt   # Dependencies
```

## Model Architectures

### 1. MLP Model
- Simple feedforward neural network
- Good for learning basic patterns
- Fast training and inference

### 2. Transformer Model
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Residual connections

## Training Features

- 📊 Loss tracking and visualization
- 🎛️ Learning rate scheduling
- ✂️ Gradient clipping
- 🛑 Early stopping
- 💾 Automatic checkpointing
- 📈 Real-time training progress

## Usage Examples

### Custom Dataset
```python
from data_loader import TextDataset

# Load your own text file
dataset = TextDataset("your_text_file.txt")
```

### Model Configuration
```python
from config import ModelConfig

# Customize model settings
config = ModelConfig(
    vocab_size=100,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=4
)
```

## Scaling Up

- Increase `embedding_dim` and `hidden_dim` for larger models
- Add more layers (`num_layers`)
- Use larger datasets
- Implement data parallel training for multiple GPUs

## License

MIT License - Feel free to use and modify!