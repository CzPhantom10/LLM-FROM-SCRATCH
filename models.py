"""
Model architectures: MLP and Transformer for character-level language modeling.
Both models predict the next character given a sequence of previous characters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLPLanguageModel(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) language model.
    
    This model:
    1. Embeds input characters
    2. Flattens the sequence into a vector
    3. Passes through MLP layers
    4. Outputs probabilities for next character
    
    Good for learning basic patterns but limited context understanding.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, sequence_length, dropout=0.1):
        """
        Initialize the MLP model.
        
        Args:
            vocab_size: Size of the character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden dimension of MLP layers
            num_layers: Number of hidden layers
            sequence_length: Length of input sequences
            dropout: Dropout rate for regularization
        """
        super(MLPLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = sequence_length * embedding_dim  # Flattened embedding dimension
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, vocab_size))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            logits: Output logits of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Embed characters
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Flatten embeddings
        flattened = embeddings.view(batch_size, -1)  # (batch_size, seq_len * embedding_dim)
        
        # Pass through MLP
        output = self.mlp(flattened)  # (batch_size, vocab_size)
        
        # Reshape to match target format (for compatibility with loss calculation)
        output = output.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, vocab_size)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    This is the core component of the Transformer architecture.
    """
    
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
            mask: Attention mask (optional)
            
        Returns:
            Output tensor (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        # Linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, embedding_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embedding_dim
        )
        
        # Final linear projection
        output = self.out_proj(attention_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    A single Transformer block with self-attention and feedforward layers.
    """
    
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        """
        Initialize Transformer block.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        # Self-attention
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass of Transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
            mask: Attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, embedding_dim)
        """
        # Self-attention with residual connection
        attn_output = self.attention(self.ln1(x), mask)
        x = x + attn_output
        
        # Feedforward with residual connection
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding to give the model information about token positions.
    """
    
    def __init__(self, embedding_dim, max_seq_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Create sinusoidal encodings
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           -(math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (batch_size, seq_len, embedding_dim)
            
        Returns:
            Input embeddings with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerLanguageModel(nn.Module):
    """
    Transformer-based character-level language model.
    
    This model uses self-attention to understand context and relationships
    between characters in the sequence.
    """
    
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, 
                 num_layers, sequence_length, dropout=0.1):
        """
        Initialize the Transformer model.
        
        Args:
            vocab_size: Size of the character vocabulary
            embedding_dim: Dimension of character embeddings
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension of feedforward layers
            num_layers: Number of Transformer blocks
            sequence_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(TransformerLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, sequence_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def _create_causal_mask(self, seq_len, device):
        """
        Create causal mask to prevent attention to future tokens.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask tensor
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, x):
        """
        Forward pass of the Transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            logits: Output logits of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len, device)
        
        # Embed characters and add positional encoding
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pass through Transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, mask)
        
        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Output projection
        logits = self.head(hidden_states)  # (batch_size, seq_len, vocab_size)
        
        return logits


def create_model(model_type, vocab_size, embedding_dim, hidden_dim, 
                num_layers, sequence_length, num_heads=8, dropout=0.1):
    """
    Factory function to create models.
    
    Args:
        model_type: "mlp" or "transformer"
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        sequence_length: Sequence length
        num_heads: Number of attention heads (for Transformer)
        dropout: Dropout rate
        
    Returns:
        Initialized model
    """
    if model_type.lower() == "mlp":
        return MLPLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            sequence_length=sequence_length,
            dropout=dropout
        )
    elif model_type.lower() == "transformer":
        return TransformerLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            sequence_length=sequence_length,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Test model creation and forward pass.
    """
    # Test parameters
    vocab_size = 100
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    sequence_length = 32
    batch_size = 4
    
    print("Testing Model Architectures")
    print("=" * 40)
    
    # Create test input
    x = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    # Test MLP model
    print("\nTesting MLP Model:")
    mlp_model = create_model("mlp", vocab_size, embedding_dim, hidden_dim, 
                            num_layers, sequence_length)
    mlp_output = mlp_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {mlp_output.shape}")
    print(f"Parameters: {count_parameters(mlp_model):,}")
    
    # Test Transformer model
    print("\nTesting Transformer Model:")
    transformer_model = create_model("transformer", vocab_size, embedding_dim, 
                                   hidden_dim, num_layers, sequence_length)
    transformer_output = transformer_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {transformer_output.shape}")
    print(f"Parameters: {count_parameters(transformer_model):,}")


if __name__ == "__main__":
    main()
