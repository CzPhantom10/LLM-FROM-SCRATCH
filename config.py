"""
Configuration file for the character-level language model.
Contains all hyperparameters and settings.
"""

class ModelConfig:
    """Configuration class for model hyperparameters"""
    
    def __init__(self):
        # Data settings
        self.data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.data_file = "data/shakespeare.txt"
        self.sequence_length = 64   # Length of input sequences (reduced for CPU)
        self.batch_size = 32        # Batch size for training (reduced for CPU)
        
        # Model architecture
        self.vocab_size = None      # Will be set based on data
        self.embedding_dim = 128    # Embedding dimension (reasonable for CPU)
        self.hidden_dim = 256       # Hidden dimension for MLP/Transformer
        self.num_layers = 3         # Number of layers (reasonable for CPU)
        self.num_heads = 4          # Number of attention heads (for Transformer)
        self.dropout = 0.1          # Dropout rate
        
        # Training settings
        self.learning_rate = 1e-3   # Learning rate (slightly higher for faster convergence)
        self.num_epochs = 20        # Number of training epochs (reasonable for CPU)
        self.weight_decay = 1e-5    # Weight decay for regularization
        self.gradient_clip = 1.0    # Gradient clipping value
        
        # Learning rate scheduling
        self.lr_scheduler = True    # Whether to use learning rate scheduling
        self.lr_decay_factor = 0.9  # Learning rate decay factor
        self.lr_decay_patience = 5  # Patience for LR decay
        
        # Early stopping
        self.early_stopping = True  # Whether to use early stopping
        self.early_stopping_patience = 10  # Patience for early stopping
        
        # Logging and saving
        self.log_interval = 100     # How often to log training progress
        self.save_interval = 1000   # How often to save checkpoints
        self.model_save_path = "models/"  # Directory to save models
        self.best_model_name = "best_model.pt"  # Best model filename
        
        # Generation settings
        self.generation_length = 500  # Length of generated text
        self.temperature = 0.8       # Temperature for text generation
        
        # Device settings
        self.device = "cuda" if self._cuda_available() else "cpu"
        
    def _cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def __str__(self):
        """String representation of the configuration"""
        config_str = "Model Configuration:\n"
        config_str += "-" * 30 + "\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"{key}: {value}\n"
        return config_str

# Default configuration instance
config = ModelConfig()
