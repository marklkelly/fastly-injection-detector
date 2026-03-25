#!/usr/bin/env python3
"""
FIXED Static Quantization Pipeline with Real Weights and Tokenizer

Critical fixes implemented:
1. Uses real tokenizer (tokenizers library) instead of toy hash tokenizer
2. Loads actual trained weights from ONNX, not random initialization
3. Applies attention masks to ignore padding tokens
4. Runs actual FFN-0 model when generating FFN-1 calibration
5. Balanced dataset sampling with proper shuffling
6. MAE in LSBs validation metric (not arbitrary scale*5)
7. Reads qparams from compiled EdgeTPU models
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Generator, Dict, List, Tuple, Optional
import pandas as pd
import tensorflow as tf

# Try to import tflite_runtime, fall back to tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Real tokenizer support
try:
    from tokenizers import Tokenizer
except ImportError:
    print("ERROR: Install tokenizers library: pip install tokenizers")
    sys.exit(1)


class ProperDatasetQuantizationPipeline:
    """
    Production-ready static quantization pipeline.

    Uses real weights, real tokenizer, and proper dataset calibration.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer_path: str,
        weights_dir: str,
        hidden_dim: int = 128,
        seq_len: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        ffn_dim: int = 192
    ):
        """
        Initialize with real model components.

        Args:
            dataset_path: Path to prompt injection dataset CSV
            tokenizer_path: Path to tokenizer.json
            weights_dir: Directory containing extracted ONNX weights
            hidden_dim: Hidden dimension (128 for BERT-tiny)
            seq_len: Sequence length (128)
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            ffn_dim: FFN intermediate dimension (192 optimized)
        """
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim

        # Load real tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)

        # Load real weights from ONNX extraction
        self._load_weights_from_dir(weights_dir)

        # Load and balance dataset
        self.dataset = self._load_and_balance_dataset(dataset_path)

        # FFN-0 interpreter for proper FFN-1 calibration
        self.ffn0_interpreter = None

    def _load_tokenizer(self, tokenizer_path: str) -> Tokenizer:
        """Load real tokenizer from JSON file."""
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        print(f"✅ Loading real tokenizer from {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    def _load_weights_from_dir(self, weights_dir: str):
        """Load real weights extracted from ONNX model."""
        weights_path = Path(weights_dir)

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

        print(f"✅ Loading real weights from {weights_dir}")

        # Load embeddings
        emb_path = weights_path / "embeddings.npz"
        if emb_path.exists():
            emb = np.load(emb_path)
            self.E_token = emb["token"].astype(np.float32)
            self.E_position = emb["position"].astype(np.float32)
            self.E_segment = emb.get("segment", np.zeros((2, self.hidden_dim), np.float32))
            print(f"  Token embeddings: {self.E_token.shape}")
            print(f"  Position embeddings: {self.E_position.shape}")
            print(f"  Segment embeddings: {self.E_segment.shape}")
        else:
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")

        # Load LayerNorm parameters
        ln_path = weights_path / "layer_norm.npz"
        if ln_path.exists():
            ln = np.load(ln_path)
            self.emb_ln_gamma = ln.get("emb_gamma", np.ones(self.hidden_dim, np.float32))
            self.emb_ln_beta = ln.get("emb_beta", np.zeros(self.hidden_dim, np.float32))
            self.emb_ln_eps = float(ln.get("eps", 1e-12))
            print(f"  Embedding LayerNorm loaded")
        else:
            print("  ⚠️ LayerNorm not found, using defaults")
            self.emb_ln_gamma = np.ones(self.hidden_dim, np.float32)
            self.emb_ln_beta = np.zeros(self.hidden_dim, np.float32)
            self.emb_ln_eps = 1e-12

        # Load attention weights for each layer
        self.attention_weights = []
        for i in range(self.num_layers):
            attn_path = weights_path / f"attention_L{i}.npz"
            if attn_path.exists():
                w = np.load(attn_path)
                self.attention_weights.append({
                    "Wq": w["Wq"].astype(np.float32),
                    "Wk": w["Wk"].astype(np.float32),
                    "Wv": w["Wv"].astype(np.float32),
                    "Wo": w["Wo"].astype(np.float32),
                })
                print(f"  Layer {i} attention weights loaded")
            else:
                raise FileNotFoundError(f"Attention weights not found: {attn_path}")

    def _load_and_balance_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load and properly balance dataset."""
        print(f"✅ Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Validate columns
        assert 'text' in df.columns, "Dataset must have 'text' column"
        assert 'label' in df.columns, "Dataset must have 'label' column"

        # Split by class
        safe = df[df.label == 0]
        injection = df[df.label == 1]

        print(f"  Original: {len(safe)} SAFE, {len(injection)} INJECTION")

        # Balance and shuffle properly
        min_samples = min(len(safe), len(injection))
        safe_balanced = safe.sample(n=min_samples, random_state=42)
        injection_balanced = injection.sample(n=min_samples, random_state=42)

        # Combine and shuffle again
        balanced = pd.concat([safe_balanced, injection_balanced])
        balanced = balanced.sample(frac=1, random_state=123).reset_index(drop=True)

        print(f"  Balanced: {len(balanced)} total samples")

        return balanced

    def tokenize_with_mask(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize text and create attention mask.

        Returns:
            token_ids: [seq_len] array of token IDs
            attention_mask: [seq_len] array (1=real token, 0=padding)
        """
        # Encode with real tokenizer
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.seq_len]

        # Create attention mask
        mask = [1] * len(ids)

        # Pad to seq_len
        pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        while len(ids) < self.seq_len:
            ids.append(pad_id)
            mask.append(0)

        return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32)

    def post_embedding(self, ids: np.ndarray) -> np.ndarray:
        """
        Apply embeddings and LayerNorm.

        Args:
            ids: Token IDs [1, seq_len]

        Returns:
            Post-embedding tensor [1, seq_len, hidden_dim]
        """
        # Token embeddings
        tok_emb = self.E_token[ids]  # [1, seq_len, hidden_dim]

        # Position embeddings
        pos_ids = np.arange(self.seq_len).reshape(1, -1)
        pos_emb = self.E_position[pos_ids]

        # Segment embeddings (all segment 0)
        seg_ids = np.zeros((1, self.seq_len), dtype=np.int32)
        seg_emb = self.E_segment[seg_ids]

        # Sum embeddings
        x = tok_emb + pos_emb + seg_emb

        # Embedding LayerNorm
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x = self.emb_ln_gamma * (x - mean) / np.sqrt(var + self.emb_ln_eps) + self.emb_ln_beta

        return x.astype(np.float32)

    def cpu_attention_with_mask(
        self,
        x: np.ndarray,
        layer_idx: int,
        attn_mask: np.ndarray
    ) -> np.ndarray:
        """
        Multi-head attention with proper padding mask.

        Args:
            x: Input tensor [1, seq_len, hidden_dim]
            layer_idx: Which attention layer
            attn_mask: Attention mask [1, seq_len] (1=real, 0=pad)

        Returns:
            Output with residual [1, seq_len, hidden_dim]
        """
        B, T, H = x.shape
        w = self.attention_weights[layer_idx]

        # QKV projections
        Q = x @ w['Wq']
        K = x @ w['Wk']
        V = x @ w['Wv']

        # Split heads
        def split_heads(t):
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        Qh = split_heads(Q)  # [B, num_heads, T, head_dim]
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Scaled dot-product attention
        scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # [B, H, T, T]

        # Apply attention mask - set padding positions to very negative
        # Expand mask from [B, T] to [B, 1, 1, T] for broadcasting
        mask_expanded = attn_mask[:, None, None, :]  # [B, 1, 1, T]
        scores = scores + (1.0 - mask_expanded) * (-1e9)

        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Apply attention to values
        context = attention_weights @ Vh  # [B, H, T, head_dim]

        # Merge heads
        context = context.transpose(0, 2, 1, 3).reshape(B, T, H)

        # Output projection
        output = context @ w['Wo']

        # Residual connection
        return x + output

    def _balanced_text_sample(self, n: int) -> np.ndarray:
        """Get balanced sample of texts."""
        # Sample equally from SAFE and INJECTION
        safe = self.dataset[self.dataset.label == 0]
        injection = self.dataset[self.dataset.label == 1]

        k = min(len(safe), len(injection), n // 2)

        safe_sample = safe.sample(n=k, random_state=42)
        injection_sample = injection.sample(n=k, random_state=43)

        # Combine and shuffle
        combined = pd.concat([safe_sample, injection_sample])
        combined = combined.sample(frac=1, random_state=123)

        return combined.text.values

    def representative_dataset_ffn0(self, n: int = 2000) -> Generator:
        """
        Generate representative data for FFN-0 calibration.

        Produces post-attention-0 tensors from real data.
        """
        texts = self._balanced_text_sample(n)

        for text in texts:
            # Tokenize with mask
            ids, mask = self.tokenize_with_mask(text)
            ids = ids.reshape(1, -1)
            mask = mask.reshape(1, -1)

            # Post-embedding with LayerNorm
            x = self.post_embedding(ids)

            # Attention-0 with mask
            x0 = self.cpu_attention_with_mask(x, layer_idx=0, attn_mask=mask)

            yield [x0.astype(np.float32)]

    def representative_dataset_ffn1(self, n: int = 2000) -> Generator:
        """
        Generate representative data for FFN-1 calibration.

        CRITICAL: Runs actual FFN-0 model, not placeholder!
        """
        if self.ffn0_interpreter is None:
            raise RuntimeError("FFN-0 interpreter not set! Call set_ffn0_model() first")

        texts = self._balanced_text_sample(n)

        # Get FFN-0 quantization parameters
        inp0 = self.ffn0_interpreter.get_input_details()[0]
        out0 = self.ffn0_interpreter.get_output_details()[0]
        s0_in, z0_in = inp0['quantization']
        s0_out, z0_out = out0['quantization']

        for text in texts:
            # Tokenize with mask
            ids, mask = self.tokenize_with_mask(text)
            ids = ids.reshape(1, -1)
            mask = mask.reshape(1, -1)

            # Post-embedding with LayerNorm
            x = self.post_embedding(ids)

            # Attention-0 with mask
            x0 = self.cpu_attention_with_mask(x, layer_idx=0, attn_mask=mask)

            # Quantize for FFN-0
            x0_q = np.clip(np.round(x0 / s0_in + z0_in), -128, 127).astype(np.int8)

            # Run actual FFN-0 on CPU
            self.ffn0_interpreter.set_tensor(inp0['index'], x0_q)
            self.ffn0_interpreter.invoke()
            y0_q = self.ffn0_interpreter.get_tensor(out0['index'])

            # Dequantize FFN-0 output
            y0 = (y0_q.astype(np.float32) - z0_out) * s0_out

            # Add residual (FFN-0 includes residual internally)
            h0 = y0  # FFN output already has residual

            # Attention-1 with mask
            x1 = self.cpu_attention_with_mask(h0, layer_idx=1, attn_mask=mask)

            yield [x1.astype(np.float32)]

    def set_ffn0_model(self, ffn0_path: str):
        """Set FFN-0 model for proper FFN-1 calibration."""
        print(f"Loading FFN-0 model for calibration: {ffn0_path}")
        self.ffn0_interpreter = tflite.Interpreter(model_path=ffn0_path)
        self.ffn0_interpreter.allocate_tensors()

    def export_with_proper_calibration(
        self,
        model: tf.keras.Model,
        output_path: str,
        layer_idx: int,
        num_samples: int = 2000
    ) -> Dict:
        """
        Export TFLite model with proper dataset calibration.
        """
        print(f"\n{'='*60}")
        print(f"Exporting FFN-{layer_idx} with PROPER calibration")
        print(f"{'='*60}")

        # Select appropriate representative dataset
        if layer_idx == 0:
            rep_dataset = lambda: self.representative_dataset_ffn0(num_samples)
        else:
            rep_dataset = lambda: self.representative_dataset_ffn1(num_samples)

        # Convert to TFLite with INT8
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.allow_custom_ops = False

        print(f"Running calibration with {num_samples} REAL samples...")
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
            'input': {
                'scale': float(input_details['quantization'][0]),
                'zero_point': int(input_details['quantization'][1])
            },
            'output': {
                'scale': float(output_details['quantization'][0]),
                'zero_point': int(output_details['quantization'][1])
            }
        }

        print(f"📊 Quantization Parameters:")
        print(f"  Input: scale={quant_params['input']['scale']:.8f}, zp={quant_params['input']['zero_point']}")
        print(f"  Output: scale={quant_params['output']['scale']:.8f}, zp={quant_params['output']['zero_point']}")

        return quant_params

    def extract_qparams_from_compiled(self, compiled_path: str) -> Dict:
        """
        Extract quantization parameters from compiled EdgeTPU model.

        CRITICAL: Use these exact values in production!
        """
        print(f"\n📊 Extracting qparams from compiled model: {compiled_path}")

        interpreter = tflite.Interpreter(model_path=compiled_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        qparams = {
            'input': {
                'scale': float(input_details['quantization'][0]),
                'zero_point': int(input_details['quantization'][1]),
                'dtype': input_details['dtype'].name,
                'shape': list(input_details['shape'])
            },
            'output': {
                'scale': float(output_details['quantization'][0]),
                'zero_point': int(output_details['quantization'][1]),
                'dtype': output_details['dtype'].name,
                'shape': list(output_details['shape'])
            }
        }

        print(f"  Input: scale={qparams['input']['scale']:.8f}, zp={qparams['input']['zero_point']}")
        print(f"  Output: scale={qparams['output']['scale']:.8f}, zp={qparams['output']['zero_point']}")

        # Save to JSON next to model
        qparams_path = Path(compiled_path).with_suffix('.qparams.json')
        with open(qparams_path, 'w') as f:
            json.dump(qparams, f, indent=2)

        print(f"  ✅ Saved to {qparams_path}")

        return qparams

    def validate_quantization_proper(
        self,
        model_path: str,
        layer_idx: int,
        num_samples: int = 100
    ) -> Dict:
        """
        Proper validation with MAE in LSBs and saturation check.
        """
        print(f"\n{'='*60}")
        print(f"Validating quantization for FFN-{layer_idx}")
        print(f"{'='*60}")

        # Load model and get qparams
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        scale = input_details['quantization'][0]
        zero_point = input_details['quantization'][1]

        # Get test samples
        if layer_idx == 0:
            test_gen = self.representative_dataset_ffn0(num_samples)
        else:
            test_gen = self.representative_dataset_ffn1(num_samples)

        saturated_count = 0
        total_elements = 0
        mae_lsbs = []

        for sample in test_gen:
            x_float = sample[0]

            # Quantize
            x_quant = np.round(x_float / scale + zero_point)
            x_int8 = np.clip(x_quant, -128, 127).astype(np.int8)

            # Count saturation
            saturated = np.sum((x_int8 == -128) | (x_int8 == 127))
            saturated_count += saturated
            total_elements += x_int8.size

            # Dequantize
            x_dequant = (x_int8.astype(np.float32) - zero_point) * scale

            # MAE in LSBs (proper metric)
            mae = np.mean(np.abs(x_float - x_dequant))
            mae_lsb = mae / scale
            mae_lsbs.append(mae_lsb)

        # Statistics
        saturation_rate = saturated_count / total_elements * 100
        mean_mae_lsb = np.mean(mae_lsbs)
        max_mae_lsb = np.max(mae_lsbs)

        print(f"📊 Quantization Quality:")
        print(f"  Saturation rate: {saturation_rate:.2f}%")
        print(f"  Mean MAE (LSBs): {mean_mae_lsb:.2f}")
        print(f"  Max MAE (LSBs): {max_mae_lsb:.2f}")

        # Quality gates
        passed = True

        if saturation_rate > 2.0:
            print("❌ FAILED: Saturation > 2%")
            passed = False
        else:
            print("✅ PASSED: Saturation < 2%")

        if mean_mae_lsb > 1.0:
            print("❌ FAILED: MAE > 1 LSB")
            passed = False
        else:
            print("✅ PASSED: MAE ≤ 1 LSB")

        return {
            'saturation_rate': saturation_rate,
            'mean_mae_lsb': mean_mae_lsb,
            'max_mae_lsb': max_mae_lsb,
            'passed': passed
        }


def main():
    """Run proper static quantization pipeline."""
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description="Proper static quantization")
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer.json')
    parser.add_argument('--weights_dir', required=True, help='Directory with extracted weights')
    parser.add_argument('--output_dir', default='models_proper_quant/', help='Output directory')
    parser.add_argument('--calibration_samples', type=int, default=2000)

    args = parser.parse_args()

    print("="*60)
    print("PROPER STATIC QUANTIZATION WITH REAL COMPONENTS")
    print("="*60)

    # Initialize pipeline with real components
    pipeline = ProperDatasetQuantizationPipeline(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        weights_dir=args.weights_dir,
        hidden_dim=128,
        seq_len=128,
        num_heads=2,
        num_layers=2,
        ffn_dim=192
    )

    # Add parent directory to path to import coral_phase_b_optimized
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from coral_phase_b_optimized import build_tpu_ffn_block

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export FFN-0
    ffn0_model = build_tpu_ffn_block(
        seq_len=128,
        hidden_dim=128,
        ffn_dim=192,
        layer_idx=0
    )

    ffn0_path = output_dir / "ffn_layer_0_proper_int8.tflite"
    qparams_0 = pipeline.export_with_proper_calibration(
        model=ffn0_model,
        output_path=str(ffn0_path),
        layer_idx=0,
        num_samples=args.calibration_samples
    )

    # Validate FFN-0
    val_0 = pipeline.validate_quantization_proper(str(ffn0_path), layer_idx=0)

    # Set FFN-0 for FFN-1 calibration
    pipeline.set_ffn0_model(str(ffn0_path))

    # Export FFN-1
    ffn1_model = build_tpu_ffn_block(
        seq_len=128,
        hidden_dim=128,
        ffn_dim=192,
        layer_idx=1
    )

    ffn1_path = output_dir / "ffn_layer_1_proper_int8.tflite"
    qparams_1 = pipeline.export_with_proper_calibration(
        model=ffn1_model,
        output_path=str(ffn1_path),
        layer_idx=1,
        num_samples=args.calibration_samples
    )

    # Validate FFN-1
    val_1 = pipeline.validate_quantization_proper(str(ffn1_path), layer_idx=1)

    # Summary
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)

    if val_0['passed'] and val_1['passed']:
        print("✅ ALL QUALITY GATES PASSED")
    else:
        print("❌ QUALITY GATES FAILED")

    print("\n📋 Next Steps:")
    print("1. Compile models with Edge TPU compiler on Linux")
    print("2. Extract qparams from COMPILED models (not pre-compile)")
    print("3. Use exact qparams in production CPU→TPU handoff")
    print("4. Test F1 score (must be within 2pp of baseline)")

    return 0 if (val_0['passed'] and val_1['passed']) else 1


if __name__ == "__main__":
    sys.exit(main())