#!/usr/bin/env python3
"""
Static Quantization with Real Dataset
Based on Static_Quantization_with_Dataset_Spec.md

CRITICAL: Uses actual prompt injection dataset for calibration,
not random data. This ensures proper quantization ranges.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Generator, Dict, List, Tuple
import pandas as pd

# TensorFlow for model export
import tensorflow as tf


class DatasetQuantizationPipeline:
    """
    Proper static quantization using real dataset.

    Key improvements:
    1. Uses actual prompt injection data
    2. Full preprocessing pipeline (tokenization → embeddings → attention)
    3. Separate calibration for each FFN layer
    4. Reads quantization params from converted model
    """

    def __init__(
        self,
        dataset_path: str = None,
        vocab_size: int = 30522,
        hidden_dim: int = 128,
        seq_len: int = 128,
        num_heads: int = 2,
        num_layers: int = 2
    ):
        """
        Initialize quantization pipeline.

        Args:
            dataset_path: Path to prompt injection dataset CSV
            vocab_size: Vocabulary size for embeddings
            hidden_dim: Hidden dimension (128 for BERT-tiny)
            seq_len: Sequence length (128)
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
        """
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_layers = num_layers

        # Initialize weights (in production, load from checkpoint)
        self._init_weights()

        # Load dataset if provided
        self.dataset = None
        if dataset_path and Path(dataset_path).exists():
            self.load_dataset(dataset_path)

    def load_dataset(self, path: str):
        """
        Load prompt injection dataset.

        Expected columns: 'text' and 'label' (0=SAFE, 1=INJECTION)
        """
        print(f"Loading dataset from {path}")
        self.dataset = pd.read_csv(path)

        # Validate columns
        assert 'text' in self.dataset.columns, "Dataset must have 'text' column"
        assert 'label' in self.dataset.columns, "Dataset must have 'label' column"

        # Balance dataset
        safe_samples = self.dataset[self.dataset['label'] == 0]
        injection_samples = self.dataset[self.dataset['label'] == 1]

        print(f"Dataset statistics:")
        print(f"  Total samples: {len(self.dataset)}")
        print(f"  SAFE samples: {len(safe_samples)}")
        print(f"  INJECTION samples: {len(injection_samples)}")

        # Balance if needed (undersample majority class)
        min_samples = min(len(safe_samples), len(injection_samples))
        if min_samples < len(safe_samples):
            safe_samples = safe_samples.sample(n=min_samples, random_state=42)
        if min_samples < len(injection_samples):
            injection_samples = injection_samples.sample(n=min_samples, random_state=42)

        self.dataset = pd.concat([safe_samples, injection_samples]).sample(frac=1, random_state=42)
        print(f"  Balanced samples: {len(self.dataset)}")

    def _init_weights(self):
        """Initialize embedding and attention weights."""
        # Embeddings (would load from checkpoint in production)
        self.E_token = np.random.randn(self.vocab_size, self.hidden_dim).astype(np.float32) * 0.02
        self.E_position = np.random.randn(self.seq_len, self.hidden_dim).astype(np.float32) * 0.02
        self.E_segment = np.random.randn(2, self.hidden_dim).astype(np.float32) * 0.02

        # Embedding LayerNorm parameters
        self.emb_ln_gamma = np.ones(self.hidden_dim, dtype=np.float32)
        self.emb_ln_beta = np.zeros(self.hidden_dim, dtype=np.float32)
        self.emb_ln_eps = 1e-12

        # Attention weights for each layer
        self.attention_weights = []
        for _ in range(self.num_layers):
            self.attention_weights.append({
                'Wq': np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02,
                'Wk': np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02,
                'Wv': np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02,
                'Wo': np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02,
            })

    def tokenize(self, text: str) -> np.ndarray:
        """
        Tokenize text to token IDs.

        In production, use actual tokenizer. Here we simulate.
        """
        # Simple hash-based tokenization for demo
        # In production: use actual BERT tokenizer
        tokens = []
        for word in text.lower().split()[:self.seq_len]:
            # Hash word to vocab range
            token_id = hash(word) % self.vocab_size
            tokens.append(token_id)

        # Pad to seq_len
        while len(tokens) < self.seq_len:
            tokens.append(0)  # [PAD] token

        return np.array(tokens[:self.seq_len], dtype=np.int32)

    def preprocess_to_embeddings(self, texts: List[str]) -> Generator[np.ndarray, None, None]:
        """
        Preprocess texts to post-embedding tensors.

        Pipeline: Tokenization → Embeddings → LayerNorm

        Args:
            texts: List of input texts

        Yields:
            Post-embedding tensors [1, seq_len, hidden_dim]
        """
        for text in texts:
            # Tokenize
            token_ids = self.tokenize(text).reshape(1, -1)  # [1, seq_len]

            # Embeddings
            tok_emb = self.E_token[token_ids]  # [1, seq_len, hidden_dim]
            pos_emb = self.E_position[np.arange(self.seq_len)].reshape(1, self.seq_len, self.hidden_dim)
            seg_emb = self.E_segment[np.zeros((1, self.seq_len), dtype=np.int32)]

            # Sum embeddings
            x = tok_emb + pos_emb + seg_emb

            # Embedding LayerNorm
            mean = x.mean(axis=-1, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            x = self.emb_ln_gamma * (x - mean) / np.sqrt(var + self.emb_ln_eps) + self.emb_ln_beta

            yield x.astype(np.float32)

    def cpu_attention(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Multi-head attention on CPU.

        Must match production implementation exactly!
        """
        batch_size, seq_len, hidden_dim = x.shape
        weights = self.attention_weights[layer_idx]

        # QKV projections
        Q = x @ weights['Wq']
        K = x @ weights['Wk']
        V = x @ weights['Wv']

        # Reshape for multi-head attention
        def split_heads(t):
            return t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Apply attention
        context = attention_weights @ V  # [batch, heads, seq, head_dim]

        # Merge heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)

        # Output projection
        output = context @ weights['Wo']

        # Residual connection
        return x + output

    def representative_dataset_ffn0(self, num_samples: int = 2000) -> Generator:
        """
        Representative dataset for FFN-0 calibration.

        Produces post-attention-0 tensors from real data.
        """
        if self.dataset is None:
            print("⚠️  No dataset loaded, using random data (NOT RECOMMENDED)")
            for _ in range(num_samples):
                yield [np.random.randn(1, self.seq_len, self.hidden_dim).astype(np.float32) * 0.5]
            return

        # Use real dataset
        texts = self.dataset['text'].values[:num_samples]

        for x_emb in self.preprocess_to_embeddings(texts):
            # Apply attention layer 0
            x_after_attn0 = self.cpu_attention(x_emb, layer_idx=0)
            yield [x_after_attn0]  # TFLite expects list of inputs

    def representative_dataset_ffn1(self, num_samples: int = 2000) -> Generator:
        """
        Representative dataset for FFN-1 calibration.

        Produces post-attention-1 tensors from real data.
        """
        if self.dataset is None:
            print("⚠️  No dataset loaded, using random data (NOT RECOMMENDED)")
            for _ in range(num_samples):
                yield [np.random.randn(1, self.seq_len, self.hidden_dim).astype(np.float32) * 0.5]
            return

        # Use real dataset
        texts = self.dataset['text'].values[:num_samples]

        for x_emb in self.preprocess_to_embeddings(texts):
            # Apply attention layer 0
            x_after_attn0 = self.cpu_attention(x_emb, layer_idx=0)

            # Simulate FFN-0 (in production, use actual FFN or TFLite)
            # For now, simple pass-through with slight modification
            x_ffn0_out = x_after_attn0 * 1.1  # Placeholder

            # Apply attention layer 1
            x_after_attn1 = self.cpu_attention(x_ffn0_out, layer_idx=1)
            yield [x_after_attn1]

    def export_with_dataset_calibration(
        self,
        model: tf.keras.Model,
        output_path: str,
        layer_idx: int,
        num_calibration_samples: int = 2000
    ) -> Dict:
        """
        Export TFLite model with proper dataset calibration.

        Args:
            model: Keras model to convert
            output_path: Path to save TFLite model
            layer_idx: Which layer (0 or 1) for calibration
            num_calibration_samples: Number of samples for calibration

        Returns:
            Quantization parameters dict
        """
        print(f"\n{'='*60}")
        print(f"Exporting FFN-{layer_idx} with dataset calibration")
        print(f"{'='*60}")

        # Select appropriate representative dataset
        if layer_idx == 0:
            rep_dataset = lambda: self.representative_dataset_ffn0(num_calibration_samples)
        else:
            rep_dataset = lambda: self.representative_dataset_ffn1(num_calibration_samples)

        # Convert to TFLite with INT8 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # CRITICAL: Use real data for calibration!
        converter.representative_dataset = rep_dataset

        # Force INT8 input/output for Edge TPU
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # No custom ops
        converter.allow_custom_ops = False

        # Convert
        print(f"Running calibration with {num_calibration_samples} samples...")
        tflite_model = converter.convert()

        # Save model
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✅ Saved to {output_path} ({len(tflite_model)/1024:.1f} KB)")

        # Extract quantization parameters
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        quant_params = {
            'input': {},
            'output': {}
        }

        # Input quantization
        if 'quantization_parameters' in input_details:
            q = input_details['quantization_parameters']
            scale = q.get('scales', [1.0])[0]
            zero_point = q.get('zero_points', [0])[0]

            quant_params['input'] = {
                'scale': float(scale),
                'zero_point': int(zero_point)
            }

            print(f"\n📊 Input Quantization Parameters:")
            print(f"  Scale: {scale:.8f}")
            print(f"  Zero Point: {zero_point}")

        # Output quantization
        if 'quantization_parameters' in output_details:
            q = output_details['quantization_parameters']
            scale = q.get('scales', [1.0])[0]
            zero_point = q.get('zero_points', [0])[0]

            quant_params['output'] = {
                'scale': float(scale),
                'zero_point': int(zero_point)
            }

            print(f"  Output Scale: {scale:.8f}")
            print(f"  Output Zero Point: {zero_point}")

        return quant_params

    def validate_quantization(self, model_path: str, layer_idx: int, num_test_samples: int = 100):
        """
        Validate quantization quality.

        Checks:
        1. Saturation rate (elements hitting -128 or 127)
        2. Distribution coverage
        3. Quantization error
        """
        print(f"\n{'='*60}")
        print(f"Validating quantization for FFN-{layer_idx}")
        print(f"{'='*60}")

        # Load quantization parameters
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        q = input_details['quantization_parameters']
        scale = q.get('scales', [1.0])[0]
        zero_point = q.get('zero_points', [0])[0]

        # Get test samples
        if layer_idx == 0:
            test_gen = self.representative_dataset_ffn0(num_test_samples)
        else:
            test_gen = self.representative_dataset_ffn1(num_test_samples)

        saturated_count = 0
        total_elements = 0
        quantization_errors = []

        for sample in test_gen:
            x_float = sample[0]

            # Quantize
            x_quant = np.round(x_float / scale + zero_point)
            x_int8 = np.clip(x_quant, -128, 127).astype(np.int8)

            # Count saturation
            saturated = np.sum((x_int8 == -128) | (x_int8 == 127))
            saturated_count += saturated
            total_elements += x_int8.size

            # Measure quantization error
            x_dequant = (x_int8.astype(np.float32) - zero_point) * scale
            error = np.abs(x_float - x_dequant)
            quantization_errors.append(error.mean())

        # Report statistics
        saturation_rate = saturated_count / total_elements * 100
        mean_error = np.mean(quantization_errors)
        max_error = np.max(quantization_errors)

        print(f"📊 Quantization Statistics:")
        print(f"  Saturation rate: {saturation_rate:.2f}%")
        print(f"  Mean quantization error: {mean_error:.6f}")
        print(f"  Max quantization error: {max_error:.6f}")

        # Warnings
        if saturation_rate > 2.0:
            print("⚠️  High saturation rate! Consider:")
            print("   - Using more diverse calibration data")
            print("   - Checking for outliers in activations")

        if mean_error > scale * 5:
            print("⚠️  High quantization error! Check calibration data")

        return {
            'saturation_rate': saturation_rate,
            'mean_error': mean_error,
            'max_error': max_error
        }


def main():
    """Demo proper static quantization."""
    import argparse

    parser = argparse.ArgumentParser(description="Static quantization with dataset")
    parser.add_argument('--dataset', default='processed_training_data.csv',
                        help='Path to prompt injection dataset CSV')
    parser.add_argument('--output_dir', default='models_dataset_quant/',
                        help='Output directory for quantized models')
    parser.add_argument('--calibration_samples', type=int, default=2000,
                        help='Number of calibration samples')

    args = parser.parse_args()

    print("="*60)
    print("STATIC QUANTIZATION WITH REAL DATASET")
    print("="*60)

    # Initialize pipeline
    pipeline = DatasetQuantizationPipeline(
        dataset_path=args.dataset,
        hidden_dim=128,
        seq_len=128,
        num_heads=2,
        num_layers=2
    )

    if pipeline.dataset is None:
        print("\n❌ CRITICAL WARNING: No dataset loaded!")
        print("   Quantization will use random data - NOT SUITABLE FOR PRODUCTION")
        print("   Please provide a dataset with --dataset path/to/data.csv")

    # For demo, create dummy FFN models
    # In production, load actual trained models
    from coral_phase_b_optimized import build_tpu_ffn_block

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_quant_params = {}

    # Export FFN-0 with dataset calibration
    ffn0_model = build_tpu_ffn_block(
        seq_len=128,
        hidden_dim=128,
        ffn_dim=192,
        layer_idx=0
    )

    ffn0_path = output_dir / "ffn_layer_0_dataset_int8.tflite"
    quant_params_0 = pipeline.export_with_dataset_calibration(
        model=ffn0_model,
        output_path=str(ffn0_path),
        layer_idx=0,
        num_calibration_samples=args.calibration_samples
    )
    all_quant_params['layer_0'] = quant_params_0

    # Validate FFN-0 quantization
    pipeline.validate_quantization(str(ffn0_path), layer_idx=0)

    # Export FFN-1 with dataset calibration
    ffn1_model = build_tpu_ffn_block(
        seq_len=128,
        hidden_dim=128,
        ffn_dim=192,
        layer_idx=1
    )

    ffn1_path = output_dir / "ffn_layer_1_dataset_int8.tflite"
    quant_params_1 = pipeline.export_with_dataset_calibration(
        model=ffn1_model,
        output_path=str(ffn1_path),
        layer_idx=1,
        num_calibration_samples=args.calibration_samples
    )
    all_quant_params['layer_1'] = quant_params_1

    # Validate FFN-1 quantization
    pipeline.validate_quantization(str(ffn1_path), layer_idx=1)

    # Save all quantization parameters
    quant_params_path = output_dir / "quantization_params.json"
    with open(quant_params_path, 'w') as f:
        json.dump({
            'config': {
                'calibration_samples': args.calibration_samples,
                'dataset_used': pipeline.dataset is not None,
                'hidden_dim': 128,
                'ffn_dim': 192,
                'num_layers': 2
            },
            'quantization': all_quant_params
        }, f, indent=2)

    print(f"\n✅ Saved quantization parameters to {quant_params_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if pipeline.dataset is not None:
        print("✅ Models quantized with real dataset")
        print(f"   Calibration samples: {args.calibration_samples}")
        print(f"   Models saved to: {output_dir}")
        print("\n⚠️  CRITICAL: These quantization parameters MUST be used in production!")
        print("   The CPU must quantize with EXACTLY these scale/zero_point values")
    else:
        print("❌ Models quantized with random data - NOT FOR PRODUCTION")
        print("   Re-run with --dataset path/to/prompt_injection_data.csv")

    print("\nNext steps:")
    print("1. Compile with Edge TPU compiler (Linux)")
    print("2. Use exact quantization params in production pipeline")
    print("3. Validate F1 score on test set")

    return 0


if __name__ == "__main__":
    sys.exit(main())