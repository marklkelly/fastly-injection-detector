#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly for train_cls.py
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported."""
    required_modules = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('sklearn', 'Scikit-learn'),
        ('numpy', 'NumPy'),
        ('accelerate', 'Accelerate'),
        ('safetensors', 'Safetensors'),
    ]
    
    all_good = True
    for module_name, display_name in required_modules:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {display_name:20} {version}")
        except ImportError as e:
            print(f"❌ {display_name:20} NOT INSTALLED - {e}")
            all_good = False
    
    return all_good

def test_models():
    """Test if the models can be loaded."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    models_to_test = [
        "nreimers/MiniLM-L6-H384-uncased",
        "microsoft/MiniLM-L6-v2",
    ]
    
    print("\nTesting model loading...")
    for model_name in models_to_test:
        try:
            print(f"  Loading {model_name}...")
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            print(f"  ✅ {model_name} loaded successfully")
            del model, tok  # Free memory
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            return False
    
    return True

def test_dataset():
    """Test if the dataset can be loaded."""
    from datasets import load_dataset
    import os
    
    print("\nTesting dataset loading...")
    
    # Check if dataset files exist
    dataset_path = "../datasets/data/pi_mix_v1"
    train_file = os.path.join(dataset_path, "train.jsonl")
    val_file = os.path.join(dataset_path, "val.jsonl")
    
    if not os.path.exists(train_file):
        print(f"  ❌ Training file not found: {train_file}")
        return False
    
    if not os.path.exists(val_file):
        print(f"  ❌ Validation file not found: {val_file}")
        return False
    
    # Try loading a sample
    try:
        ds = load_dataset("json", data_files={"train": train_file, "validation": val_file})
        print(f"  ✅ Dataset loaded: {len(ds['train'])} train, {len(ds['validation'])} val examples")
        
        # Check structure
        sample = ds['train'][0]
        if 'text' not in sample:
            print(f"  ❌ Missing 'text' field in dataset")
            return False
        if 'label' not in sample:
            print(f"  ❌ Missing 'label' field in dataset")
            return False
            
        print(f"  ✅ Dataset structure valid")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        return False

def test_mps():
    """Test Apple Silicon MPS availability."""
    import torch
    
    print("\nHardware acceleration:")
    if torch.backends.mps.is_available():
        print("  ✅ Apple Silicon MPS available")
    elif torch.cuda.is_available():
        print("  ✅ CUDA available")
    else:
        print("  ⚠️  No GPU acceleration (CPU only)")
    
    return True

def main():
    print("=" * 50)
    print("Testing train_cls.py dependencies and setup")
    print("=" * 50)
    print()
    
    # Test imports
    print("Testing imports...")
    if not test_imports():
        print("\n❌ Missing dependencies. Install with:")
        print("  pip install -r requirements.txt")
        print("  OR")
        print("  uv pip install torch transformers datasets scikit-learn numpy accelerate safetensors")
        sys.exit(1)
    
    # Test models
    if not test_models():
        print("\n❌ Model loading failed")
        sys.exit(1)
    
    # Test dataset
    if not test_dataset():
        print("\n❌ Dataset loading failed")
        sys.exit(1)
    
    # Test hardware
    test_mps()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Ready to train.")
    print("=" * 50)
    print("\nRun training with:")
    print("  python train_cls.py --train_path ../datasets/data/pi_mix_v1/train.jsonl \\")
    print("                      --val_path ../datasets/data/pi_mix_v1/val.jsonl \\")
    print("                      --output_dir ../models/your-model \\")
    print("                      --model nreimers/MiniLM-L6-H384-uncased")

if __name__ == "__main__":
    main()