#!/usr/bin/env python3
"""Debug script to identify dataset structure issues"""

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

# Dataset configurations
dataset_configs = [
    {
        'name': 'deepset/prompt-injections',
        'text_column': 'text',
        'label_column': 'label',
        'label_mapping': {0: 'SAFE', 1: 'INJECTION'},
        'split': 'train'
    },
    {
        'name': 'hackaprompt/hackaprompt-dataset',
        'text_column': 'user_input',
        'label_column': None,
        'label_mapping': None,
        'split': 'train'
    },
    {
        'name': 'markush1/LLM-Jailbreak-Classifier',
        'text_column': 'prompt',
        'label_column': 'label',
        'label_mapping': {0: 'SAFE', 1: 'INJECTION'},
        'split': 'train'
    },
    {
        'name': 'rubend18/ChatGPT-Jailbreak-Prompts',
        'text_column': 'prompt',
        'label_column': None,
        'label_mapping': None,
        'split': 'train'
    },
    {
        'name': 'imoxto/prompt_injection_cleaned_dataset_v2',
        'text_column': 'prompt',
        'label_column': 'label',
        'label_mapping': {0: 'SAFE', 1: 'INJECTION'},
        'split': 'train'
    }
]

def debug_dataset(config):
    """Debug a single dataset"""
    print(f"\n{'='*60}")
    print(f"Dataset: {config['name']}")
    print(f"{'='*60}")

    try:
        # Load dataset
        print(f"Loading dataset...")
        ds = load_dataset(config['name'], split=config['split'])
        print(f"✓ Loaded {len(ds)} samples")

        # Show column names
        print(f"\n📋 Column names: {ds.column_names}")

        # Inspect first few samples
        print(f"\n📝 First 3 samples:")
        for i in range(min(3, len(ds))):
            sample = ds[i]
            print(f"\n  Sample {i}:")
            print(f"    Keys: {list(sample.keys())}")

            # Check configured text column
            text_col = config['text_column']
            if text_col in sample:
                text = sample[text_col]
                print(f"    {text_col} type: {type(text)}")
                if text is not None:
                    text_preview = str(text)[:100].replace('\n', ' ')
                    print(f"    {text_col} preview: {text_preview}...")
                else:
                    print(f"    {text_col}: None")
            else:
                print(f"    ⚠️ Column '{text_col}' NOT FOUND!")
                # Try to find text-like columns
                for key in sample.keys():
                    if any(word in key.lower() for word in ['text', 'prompt', 'input', 'content', 'message']):
                        value = sample[key]
                        if value and isinstance(value, str):
                            preview = str(value)[:100].replace('\n', ' ')
                            print(f"    💡 Possible text column '{key}': {preview}...")

            # Check label column if configured
            if config['label_column']:
                label_col = config['label_column']
                if label_col in sample:
                    label = sample[label_col]
                    print(f"    {label_col}: {label} (type: {type(label)})")
                else:
                    print(f"    ⚠️ Label column '{label_col}' NOT FOUND!")

        # Convert to pandas to check for any issues
        print(f"\n📊 Converting to pandas...")
        df = ds.to_pandas()
        print(f"✓ DataFrame shape: {df.shape}")
        print(f"✓ DataFrame columns: {df.columns.tolist()}")

        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"\n⚠️ Null values found:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"    {col}: {count} nulls ({count/len(df)*100:.1f}%)")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("Starting dataset debugging...")
    print(f"Testing {len(dataset_configs)} datasets\n")

    results = []
    for config in dataset_configs:
        success = debug_dataset(config)
        results.append((config['name'], success))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")

if __name__ == "__main__":
    main()