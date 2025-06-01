# LLM from Scratch - Complete Tutorial

This is a beginner-friendly guide to understanding and using our character-level language model implementation.

## Table of Contents
1. [Understanding the Project](#understanding-the-project)
2. [Quick Start Guide](#quick-start-guide)
3. [Understanding Each Component](#understanding-each-component)
4. [Training Your First Model](#training-your-first-model)
5. [Text Generation](#text-generation)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Understanding the Project

### What is a Character-Level Language Model?

A character-level language model is a type of neural network that learns to predict the next character in a sequence of text. Unlike word-level models, it operates at the character level, making it:

- **Simple**: No need for complex tokenization
- **Flexible**: Can generate any text, including new words
- **Universal**: Works with any language or character set

### How Does It Work?

1. **Input**: A sequence of characters (e.g., "Hello worl")
2. **Processing**: The model analyzes patterns in the sequence
3. **Output**: Probability distribution over possible next characters
4. **Prediction**: Most likely next character (e.g., "d" to complete "world")

## Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU optional but recommended for larger models

### Installation

#### Option 1: Automated Setup (Windows)
```batch
# Run the setup script
.\setup_and_run.bat
```

#### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv myenv

# 2. Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data
python data_loader.py
```

### Your First Model

Train a simple MLP model:
```bash
python train.py --model_type mlp --num_epochs 5
```

Generate text:
```bash
python generate.py --model_path models/best_model.pt --prompt "To be or not to be"
```

## Understanding Each Component

### 1. Configuration (`config.py`)
Contains all hyperparameters and settings:
- Model architecture parameters
- Training settings
- Data paths
- Device configuration

**Key Parameters:**
- `embedding_dim`: Size of character embeddings (64-512)
- `hidden_dim`: Size of hidden layers (128-1024)
- `num_layers`: Number of model layers (2-12)
- `sequence_length`: Input sequence length (32-512)

### 2. Data Loading (`data_loader.py`)
Handles data downloading, preprocessing, and loading:
- Downloads Shakespeare text by default
- Converts text to character indices
- Creates training batches
- Supports custom datasets

**Usage:**
```python
from data_loader import TextDataset
dataset = TextDataset("your_text.txt")
```

### 3. Tokenization (`tokenizer.py`)
Converts between text and numbers:
- Character-level tokenization
- Byte-Pair Encoding (BPE) option
- Handles unknown characters

### 4. Models (`models.py`)
Two architectures available:

#### MLP Model
- Simple feedforward neural network
- Fast training and inference
- Good for learning basic patterns
- Limited context understanding

#### Transformer Model
- Uses self-attention mechanism
- Better context understanding
- More parameters but better results
- Standard in modern NLP

### 5. Training (`train.py`)
Complete training pipeline:
- Loss tracking
- Validation
- Early stopping
- Learning rate scheduling
- Model checkpointing

### 6. Generation (`generate.py`)
Text generation with options:
- Temperature control (creativity vs coherence)
- Interactive mode
- Batch generation
- Multiple sampling strategies

### 7. Utilities (`utils.py`)
Helper functions:
- Device detection (CPU/GPU)
- Model saving/loading
- Plotting training curves
- Performance monitoring

## Training Your First Model

### Step 1: Download Data
```bash
python data_loader.py
```
This downloads the Shakespeare dataset (~1MB text file).

### Step 2: Choose Model Type

#### For Quick Results (MLP):
```bash
python train.py --model_type mlp --num_epochs 10 --batch_size 32
```

#### For Better Quality (Transformer):
```bash
python train.py --model_type transformer --num_epochs 20 --batch_size 64
```

### Step 3: Monitor Training
Watch the training progress:
- Training loss should decrease over time
- Validation loss should also decrease (but may fluctuate)
- Early stopping will trigger if validation loss stops improving

### Step 4: Training Output
The training script will:
- Save checkpoints in `models/` directory
- Create training curve plots
- Save the best model as `models/best_model.pt`

## Text Generation

### Basic Generation
```bash
python generate.py --model_path models/best_model.pt --prompt "Romeo:"
```

### Temperature Control
Temperature controls randomness:

```bash
# Conservative (low creativity)
python generate.py --model_path models/best_model.pt --prompt "To be" --temperature 0.3

# Balanced
python generate.py --model_path models/best_model.pt --prompt "To be" --temperature 0.8

# Creative (high randomness)
python generate.py --model_path models/best_model.pt --prompt "To be" --temperature 1.5
```

### Interactive Mode
```bash
python generate.py --model_path models/best_model.pt --interactive
```

In interactive mode, you can:
- Enter any prompt
- Change temperature with `temp 0.8`
- Change length with `length 200`
- Type `quit` to exit

### Batch Generation
Create a file with prompts (one per line):
```
To be or not to be
Once upon a time
Call me Ishmael
```

Then generate:
```bash
python generate.py --model_path models/best_model.pt --prompts_file prompts.txt
```

## Advanced Usage

### Custom Datasets

#### Using Your Own Text
```python
# Create custom text file
with open("my_text.txt", "w") as f:
    f.write("Your custom text here...")

# Train on custom data
python train.py --data_file my_text.txt --model_type transformer
```

#### Preprocessing Tips
- Remove unwanted characters
- Ensure sufficient data (minimum 1MB recommended)
- Consider text quality and consistency

### Hyperparameter Tuning

#### Model Size
```bash
# Small model (fast, lower quality)
python train.py --embedding_dim 64 --hidden_dim 128 --num_layers 2

# Medium model (balanced)
python train.py --embedding_dim 128 --hidden_dim 256 --num_layers 4

# Large model (slow, higher quality)
python train.py --embedding_dim 256 --hidden_dim 512 --num_layers 6
```

#### Training Parameters
```bash
# Fast training
python train.py --batch_size 128 --learning_rate 1e-3 --num_epochs 10

# Careful training
python train.py --batch_size 32 --learning_rate 3e-4 --num_epochs 50
```

### GPU Usage
The code automatically detects and uses GPU if available:
```python
# Check device
python -c "from utils import get_device; print(get_device())"
```

For multiple GPUs, modify the training script to use `DataParallel`.

## Troubleshooting

### Common Issues

#### 1. Out of Memory
**Symptoms**: CUDA out of memory error
**Solutions**:
- Reduce `batch_size`: `--batch_size 16`
- Reduce `sequence_length`: `--sequence_length 64`
- Use smaller model: `--embedding_dim 64 --hidden_dim 128`

#### 2. Slow Training
**Symptoms**: Very slow progress
**Solutions**:
- Ensure GPU is being used
- Increase `batch_size` if memory allows
- Use MLP model for faster training

#### 3. Poor Generation Quality
**Symptoms**: Generated text is nonsensical
**Solutions**:
- Train for more epochs
- Use Transformer model instead of MLP
- Increase model size
- Check training loss convergence

#### 4. Model Not Learning
**Symptoms**: Loss not decreasing
**Solutions**:
- Reduce learning rate: `--learning_rate 1e-4`
- Check data quality
- Increase model capacity
- Train for more epochs

### Performance Tips

#### Training Speed
1. Use GPU if available
2. Increase batch size (within memory limits)
3. Use mixed precision training (advanced)
4. Enable DataLoader multiprocessing (set `num_workers > 0`)

#### Generation Quality
1. Train longer (more epochs)
2. Use larger models
3. Experiment with temperature
4. Use Transformer architecture
5. Ensure high-quality training data

### Debugging

#### Check Model Output
```python
from models import create_model
import torch

model = create_model("transformer", vocab_size=100, ...)
x = torch.randint(0, 100, (1, 32))
output = model(x)
print(f"Output shape: {output.shape}")  # Should be (1, 32, 100)
```

#### Verify Data Loading
```python
from data_loader import TextDataset
dataset = TextDataset("data/shakespeare.txt")
sample = dataset[0]
print(f"Input: {sample[0]}")
print(f"Target: {sample[1]}")
```

#### Monitor Training
Watch the loss curves in the generated plots to ensure proper training.

## Next Steps

### Scaling Up
1. **Larger Models**: Increase embedding_dim, hidden_dim, num_layers
2. **More Data**: Use larger datasets (Project Gutenberg, Wikipedia)
3. **Longer Sequences**: Increase sequence_length for better context
4. **Advanced Techniques**: Add techniques like gradient accumulation

### Advanced Features
1. **Beam Search**: Implement beam search for better generation
2. **Top-k/Top-p Sampling**: More sophisticated sampling methods
3. **Model Parallel**: Split large models across multiple GPUs
4. **Mixed Precision**: Use fp16 for faster training

### Production Deployment
1. **Model Serving**: Create REST API for text generation
2. **Optimization**: Quantization and pruning for faster inference
3. **Monitoring**: Add logging and metrics collection
4. **Caching**: Cache model outputs for repeated prompts

## Resources

### Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Karpathy's char-rnn](https://github.com/karpathy/char-rnn) - Inspiration for this project

### Datasets
- [Project Gutenberg](https://www.gutenberg.org/) - Free books
- [OpenWebText](https://github.com/jcpeterson/openwebtext2) - Web text
- [Wikipedia Dumps](https://dumps.wikimedia.org/) - Wikipedia articles

### Tools
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization
- [Hugging Face](https://huggingface.co/) - Pre-trained models and datasets

## Support

If you encounter issues:
1. Check this tutorial first
2. Look at the examples in `examples.py`
3. Run the demo with `python demo.py`
4. Check GitHub issues or create a new one

Happy training! 🚀
