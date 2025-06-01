"""
Setup script for the Character-Level Language Model project.
This script sets up the environment and verifies all components work correctly.
"""

import os
import sys
import subprocess
import importlib.util


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True


def check_and_install_requirements():
    """Check and install required packages."""
    print("\n📦 Checking requirements...")
    
    requirements = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("requests", "requests")
    ]
    
    missing_packages = []
    
    for package_name, import_name in requirements:
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"] + missing_packages
            )
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["data", "models", "examples", "logs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/")


def test_imports():
    """Test if all project modules can be imported."""
    print("\n🧪 Testing module imports...")
    
    modules = [
        "config",
        "data_loader", 
        "tokenizer",
        "models",
        "utils",
        "train",
        "generate"
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}.py")
        except Exception as e:
            print(f"❌ {module}.py - Error: {e}")
            return False
    
    return True


def test_model_creation():
    """Test model creation and basic functionality."""
    print("\n🤖 Testing model creation...")
    
    try:
        from models import create_model
        import torch
        
        # Test parameters
        vocab_size = 50
        embedding_dim = 32
        hidden_dim = 64
        num_layers = 2
        sequence_length = 16
        
        # Test MLP model
        mlp_model = create_model(
            "mlp", vocab_size, embedding_dim, hidden_dim, 
            num_layers, sequence_length
        )
        
        # Test Transformer model
        transformer_model = create_model(
            "transformer", vocab_size, embedding_dim, hidden_dim,
            num_layers, sequence_length, num_heads=4
        )
        
        # Test forward pass
        sample_input = torch.randint(0, vocab_size, (2, sequence_length))
        
        with torch.no_grad():
            mlp_output = mlp_model(sample_input)
            transformer_output = transformer_model(sample_input)
        
        print(f"✅ MLP model: {mlp_output.shape}")
        print(f"✅ Transformer model: {transformer_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing data loading...")
    
    try:
        from data_loader import TextDataset
        
        # Create a small test file
        test_text = "Hello world! This is a test file for our language model. " * 20
        test_file = "test_data.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_text)
        
        # Test dataset creation
        dataset = TextDataset(test_file, sequence_length=10)
        
        # Test data loading
        sample_input, sample_target = dataset[0]
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        print(f"✅ Vocabulary size: {dataset.vocab_size}")
        print(f"✅ Sample shapes: {sample_input.shape}, {sample_target.shape}")
        
        # Clean up
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def run_quick_test():
    """Run a very quick training test."""
    print("\n⚡ Running quick training test...")
    
    try:
        # Import required modules
        from config import config
        from data_loader import TextDataset, create_data_loaders
        from models import create_model
        from train import Trainer
        from utils import get_device, set_seed
        
        # Set up for quick test
        set_seed(42)
        device = get_device()
        
        # Create test data
        test_text = "abcdefghijklmnopqrstuvwxyz " * 100
        test_file = "quick_test_data.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_text)
        
        # Override config for quick test
        config.sequence_length = 8
        config.batch_size = 4
        config.num_epochs = 1
        config.log_interval = 10
        
        # Create dataset
        dataset = TextDataset(test_file, config.sequence_length)
        config.vocab_size = dataset.vocab_size
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(dataset, train_split=0.8)
        
        # Create small model
        model = create_model(
            model_type="mlp",
            vocab_size=config.vocab_size,
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1,
            sequence_length=config.sequence_length
        )
        
        # Quick training
        trainer = Trainer(model, train_loader, val_loader, device, config)
        trainer.train()
        
        # Clean up
        os.remove(test_file)
        
        print("✅ Quick training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False


def print_usage_guide():
    """Print usage guide."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📖 Quick Start Guide:")
    print("\n1. Run the demo:")
    print("   python demo.py")
    print("\n2. Download data and train a model:")
    print("   python data_loader.py")
    print("   python train.py --model_type transformer")
    print("\n3. Generate text:")
    print("   python generate.py --model_path models/best_model.pt --prompt \"To be or not to be\"")
    print("\n4. Interactive generation:")
    print("   python generate.py --model_path models/best_model.pt --interactive")
    print("\n5. See examples:")
    print("   python examples.py")
    print("\n📚 Model Types:")
    print("   - mlp: Simple feedforward network (fast, basic)")
    print("   - transformer: Attention-based model (slower, better quality)")
    print("\n🎛️ Key Parameters:")
    print("   --embedding_dim: Size of character embeddings (64, 128, 256)")
    print("   --hidden_dim: Hidden layer size (128, 256, 512)")
    print("   --num_layers: Number of layers (2, 4, 6)")
    print("   --batch_size: Batch size (16, 32, 64)")
    print("   --learning_rate: Learning rate (1e-4, 3e-4, 1e-3)")
    print("\n💡 Tips:")
    print("   - Start with small models for testing")
    print("   - Use GPU if available for faster training")
    print("   - Experiment with different temperatures for generation")
    print("   - Check the README.md for detailed documentation")


def main():
    """Main setup function."""
    print("🚀 CHARACTER-LEVEL LANGUAGE MODEL SETUP")
    print("="*60)
    
    success = True
    
    # Run all checks
    success &= check_python_version()
    success &= check_and_install_requirements()
    
    if success:
        create_directories()
        success &= test_imports()
        success &= test_model_creation()
        success &= test_data_loading()
        
        # Optional quick test (can be slow)
        print("\n❓ Would you like to run a quick training test? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                success &= run_quick_test()
        except (EOFError, KeyboardInterrupt):
            print("Skipped quick test.")
    
    if success:
        print_usage_guide()
    else:
        print("\n❌ Setup encountered errors. Please check the messages above.")
        print("You may need to install dependencies manually or fix import issues.")
    
    return success


if __name__ == "__main__":
    main()
