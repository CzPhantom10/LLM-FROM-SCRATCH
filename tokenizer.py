"""
Character-level tokenizer for the language model.
Provides simple character-level tokenization and encoding/decoding utilities.
"""

from typing import List, Dict, Tuple


class CharacterTokenizer:
    """
    Simple character-level tokenizer.
    
    This tokenizer:
    1. Maps each unique character to an integer ID
    2. Provides encoding (text -> IDs) and decoding (IDs -> text)
    3. Handles unknown characters gracefully
    """
    
    def __init__(self, text: str = None):
        """
        Initialize the tokenizer.
        
        Args:
            text: Training text to build vocabulary from
        """
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
        if text is not None:
            self.build_vocab(text)
    
    def build_vocab(self, text: str):
        """
        Build vocabulary from text.
        
        Args:
            text: Training text to analyze
        """
        # Get unique characters and sort them for consistency
        unique_chars = sorted(list(set(text)))
        
        # Create mappings
        self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        self.id_to_char = {i: char for i, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        
        print(f"Built vocabulary with {self.vocab_size} characters")
        print(f"Sample characters: {unique_chars[:20]}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to a list of character IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of character IDs
        """
        encoded = []
        for char in text:
            if char in self.char_to_id:
                encoded.append(self.char_to_id[char])
            else:
                # Handle unknown character (you might want to add a special UNK token)
                print(f"Warning: Unknown character '{char}' encountered")
        
        return encoded
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of character IDs to text.
        
        Args:
            ids: List of character IDs to decode
            
        Returns:
            Decoded text string
        """
        decoded_chars = []
        for id in ids:
            if id in self.id_to_char:
                decoded_chars.append(self.id_to_char[id])
            else:
                print(f"Warning: Unknown ID '{id}' encountered")
                decoded_chars.append('?')  # Replace unknown IDs with '?'
        
        return ''.join(decoded_chars)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size
    
    def get_char_to_id(self) -> Dict[str, int]:
        """Return character to ID mapping"""
        return self.char_to_id.copy()
    
    def get_id_to_char(self) -> Dict[int, str]:
        """Return ID to character mapping"""
        return self.id_to_char.copy()
    
    def save_vocab(self, filepath: str):
        """
        Save vocabulary to a file.
        
        Args:
            filepath: Path to save vocabulary
        """
        import json
        
        vocab_data = {
            'char_to_id': self.char_to_id,
            'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str):
        """
        Load vocabulary from a file.
        
        Args:
            filepath: Path to load vocabulary from
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_id = vocab_data['char_to_id']
        self.id_to_char = {int(k): v for k, v in vocab_data['id_to_char'].items()}
        self.vocab_size = vocab_data['vocab_size']
        
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {self.vocab_size}")


class BytePairTokenizer:
    """
    Simple Byte-Pair Encoding (BPE) tokenizer.
    
    This is a more advanced tokenizer that can handle subword units.
    It learns to merge frequently occurring character pairs.
    """
    
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.merges = []
        self.trained = False
    
    def train(self, text: str):
        """
        Train the BPE tokenizer on text.
        
        Args:
            text: Training text
        """
        print("Training BPE tokenizer...")
        
        # Start with character-level vocabulary
        chars = sorted(list(set(text)))
        vocab = {char: i for i, char in enumerate(chars)}
        
        # Convert text to list of characters
        words = [list(word) for word in text.split()]
        
        # Iteratively merge most frequent pairs
        while len(vocab) < self.vocab_size:
            # Count pairs
            pairs = {}
            for word in words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + 1
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair in all words
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        # Merge the pair
                        merged = word[i] + word[i + 1]
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            
            words = new_words
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            vocab[merged_token] = len(vocab)
            self.merges.append(best_pair)
        
        # Create mappings
        self.char_to_id = vocab
        self.id_to_char = {i: char for char, i in vocab.items()}
        self.trained = True
        
        print(f"BPE training complete. Vocabulary size: {len(vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if not self.trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        # This is a simplified version - a full implementation would be more complex
        encoded = []
        for char in text:
            if char in self.char_to_id:
                encoded.append(self.char_to_id[char])
        
        return encoded
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        tokens = [self.id_to_char.get(id, '?') for id in ids]
        return ''.join(tokens)


def create_tokenizer(text: str, tokenizer_type: str = "character") -> CharacterTokenizer:
    """
    Factory function to create a tokenizer.
    
    Args:
        text: Training text
        tokenizer_type: Type of tokenizer ("character" or "bpe")
        
    Returns:
        Trained tokenizer
    """
    if tokenizer_type == "character":
        tokenizer = CharacterTokenizer(text)
    elif tokenizer_type == "bpe":
        tokenizer = BytePairTokenizer()
        tokenizer.train(text)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    return tokenizer


def main():
    """
    Test the tokenizer functionality.
    """
    # Sample text for testing
    sample_text = "Hello, world! This is a simple test of our character-level tokenizer."
    
    print("Testing Character-Level Tokenizer")
    print("=" * 40)
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(sample_text)
    
    # Test encoding and decoding
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {sample_text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Match:    {sample_text == decoded}")
    
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Character mapping: {tokenizer.get_char_to_id()}")


if __name__ == "__main__":
    main()
