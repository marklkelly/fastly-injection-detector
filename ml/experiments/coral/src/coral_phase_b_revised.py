#!/usr/bin/env python3
"""
Phase B REVISED: Post-Embedding Hybrid with CPU Attention + TPU FFN
Based on PHASE_B_FEEDBACK.md recommendations.

Architecture:
- CPU: Tokenization, Embeddings, LayerNorm, ATTENTION
- TPU: FFN blocks (Conv1×1 → ReLU6 → Conv1×1)
- Head: CLS token selection (not GlobalAveragePooling)
"""

import numpy as np
from pathlib import Path
import argparse
import sys

# Note: Run in clean TF 2.13 environment
# python3.9 -m venv .venv_tf && source .venv_tf/bin/activate
# pip install "tensorflow==2.13.0" "numpy<2" "flatbuffers<24"

import tensorflow as tf


def build_tpu_ffn_block(seq_len=128, hidden_dim=128, ffn_dim=256, layer_idx=0):
    """
    Build a single FFN block for TPU execution.

    This accepts the output from CPU attention and runs:
    FFN: x + Conv1×1(H→F) → ReLU6 → Conv1×1(F→H)

    Args:
        seq_len: Sequence length (128)
        hidden_dim: Hidden dimension (128)
        ffn_dim: FFN intermediate dimension (256 or 192)
        layer_idx: Layer index for naming

    Returns:
        Keras model for one FFN block
    """
    # Input: post-attention tensor from CPU
    inputs = tf.keras.Input(
        shape=(seq_len, hidden_dim),
        dtype=tf.float32,
        name=f'layer_{layer_idx}_ffn_input'
    )

    x = inputs
    residual = x

    # FFN projection 1: expand (H → F)
    x = tf.keras.layers.Conv1D(
        filters=ffn_dim,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name=f'layer_{layer_idx}_ffn_conv1'
    )(x)

    # ReLU6 activation (Edge TPU compatible)
    x = tf.keras.layers.ReLU(
        max_value=6.0,
        name=f'layer_{layer_idx}_relu6'
    )(x)

    # FFN projection 2: contract (F → H)
    x = tf.keras.layers.Conv1D(
        filters=hidden_dim,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name=f'layer_{layer_idx}_ffn_conv2'
    )(x)

    # Residual connection
    outputs = tf.keras.layers.Add(
        name=f'layer_{layer_idx}_residual'
    )([x, residual])

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f'tpu_ffn_layer_{layer_idx}'
    )

    return model


def build_tpu_classifier_head(seq_len=128, hidden_dim=128, num_classes=2):
    """
    Build classifier head using CLS token (position 0).

    Args:
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_classes: Number of output classes (2 for SAFE/INJECTION)

    Returns:
        Keras model for classification head
    """
    inputs = tf.keras.Input(
        shape=(seq_len, hidden_dim),
        dtype=tf.float32,
        name='classifier_input'
    )

    # Extract CLS token (position 0)
    # Using Lambda to select first position
    cls_token = tf.keras.layers.Lambda(
        lambda x: x[:, 0, :],
        name='extract_cls'
    )(inputs)

    # Classification layer
    # Could also use Conv1D with kernel_size=1 on CLS position only
    logits = tf.keras.layers.Dense(
        num_classes,
        name='classifier'
    )(cls_token)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=logits,
        name='tpu_classifier_head'
    )

    return model


def export_tflite_int8(model, output_path, representative_data=None):
    """
    Export model to INT8 TFLite format for Edge TPU.

    CRITICAL: Extracts and prints quantization parameters.
    These MUST be used exactly when quantizing CPU tensors.

    Args:
        model: Keras model
        output_path: Path to save TFLite model
        representative_data: Representative dataset for quantization

    Returns:
        Dict with quantization parameters
    """
    if representative_data is None:
        def representative_dataset():
            # Generate samples matching expected distribution
            for _ in range(100):
                # Post-attention tensor (after LayerNorm if used)
                data = np.random.randn(1, 128, 128).astype(np.float32)
                # Simulate post-LayerNorm distribution
                data = (data - data.mean(axis=-1, keepdims=True)) / (data.std(axis=-1, keepdims=True) + 1e-5)
                yield [data]
        representative_data = representative_dataset

    # Convert to TFLite with INT8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data

    # Force INT8 input/output for Edge TPU
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # No custom ops for our Conv1D model
    converter.allow_custom_ops = False

    # Convert
    tflite_model = converter.convert()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ Exported INT8 TFLite model to {output_path}")
    print(f"   Size: {len(tflite_model) / 1024:.1f} KB")

    # Extract quantization parameters - CRITICAL!
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("\n" + "="*60)
    print("⚠️  CRITICAL QUANTIZATION PARAMETERS")
    print("="*60)

    quant_params = {}

    if 'quantization_parameters' in input_details:
        q = input_details['quantization_parameters']
        scale = q.get('scales', [1.0])[0]
        zero_point = q.get('zero_points', [0])[0]

        quant_params['input'] = {
            'scale': float(scale),
            'zero_point': int(zero_point)
        }

        print(f"Input Quantization:")
        print(f"  Scale: {scale}")
        print(f"  Zero Point: {zero_point}")
        print(f"\nUse EXACTLY these values in embed_cpu.py:")
        print(f"  x_int8 = np.round(x_float / {scale} + {zero_point}).clip(-128, 127).astype(np.int8)")

    if 'quantization_parameters' in output_details:
        q = output_details['quantization_parameters']
        scale = q.get('scales', [1.0])[0]
        zero_point = q.get('zero_points', [0])[0]

        quant_params['output'] = {
            'scale': float(scale),
            'zero_point': int(zero_point)
        }

        print(f"\nOutput Dequantization:")
        print(f"  logits_float = (logits_int8 - {zero_point}) * {scale}")

    print("="*60)

    return quant_params


def main():
    parser = argparse.ArgumentParser(description="Build Phase B REVISED models")
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--ffn_dim', type=int, default=256, help='FFN dimension (try 192 if segments > 3)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--output_dir', default='models/', help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("PHASE B REVISED: CPU Attention + TPU FFN")
    print("="*60)
    print("\nArchitecture:")
    print("  CPU: Embeddings → LayerNorm → Attention (per layer)")
    print("  TPU: FFN blocks (Conv1×1 → ReLU6 → Conv1×1)")
    print("  Head: CLS token selection → Dense(2)")
    print(f"\nConfiguration:")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  FFN dim: {args.ffn_dim}")
    print(f"  Encoder layers: {args.num_layers}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build and export FFN blocks for each layer
    print("\n" + "="*60)
    print("Building FFN Blocks")
    print("="*60)

    quant_params_all = {}

    for layer_idx in range(args.num_layers):
        print(f"\nLayer {layer_idx}:")

        # Build FFN block for this layer
        ffn_model = build_tpu_ffn_block(
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            ffn_dim=args.ffn_dim,
            layer_idx=layer_idx
        )

        print(f"  Parameters: {ffn_model.count_params():,}")

        # Export to INT8 TFLite
        output_path = output_dir / f"ffn_layer_{layer_idx}_int8.tflite"
        quant_params = export_tflite_int8(ffn_model, output_path)
        quant_params_all[f'layer_{layer_idx}'] = quant_params

    # Build and export classifier head
    print("\n" + "="*60)
    print("Building Classifier Head")
    print("="*60)

    classifier_model = build_tpu_classifier_head(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_classes=2
    )

    print(f"  Parameters: {classifier_model.count_params():,}")

    # Export classifier
    output_path = output_dir / "classifier_head_int8.tflite"
    quant_params = export_tflite_int8(classifier_model, output_path)
    quant_params_all['classifier'] = quant_params

    # Save quantization parameters
    import json
    quant_path = output_dir / "quantization_params.json"
    with open(quant_path, 'w') as f:
        json.dump(quant_params_all, f, indent=2)
    print(f"\n✅ Saved quantization parameters to {quant_path}")

    # Next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Compile each model with Edge TPU compiler:")
    for layer_idx in range(args.num_layers):
        print(f"   edgetpu_compiler -s models/ffn_layer_{layer_idx}_int8.tflite")
    print(f"   edgetpu_compiler -s models/classifier_head_int8.tflite")

    print("\n2. Check compilation gates for EACH model:")
    print("   - Expect ≥90% mapping (Conv1D + ReLU6 fully supported)")
    print("   - Expect 1 segment per model (contiguous FFN)")
    print("   - If segments > 3, reduce ffn_dim to 192")

    print("\n3. Implement CPU pipeline (embed_cpu.py):")
    print("   - Load embeddings + attention weights from original model")
    print("   - For each layer:")
    print("     * Run attention on CPU")
    print("     * Quantize with EXACT params from quantization_params.json")
    print("     * Send to TPU FFN")
    print("     * Dequantize output")
    print("   - Send final hidden states to classifier")

    print("\n4. Benchmark:")
    print("   - Boundary budget: 2 layers = 2 CPU↔TPU transfers")
    print("   - Compare E2E latency vs CPU INT8 baseline")
    print("   - Verify F1 within 2pp of WASM model")

    return 0


if __name__ == "__main__":
    sys.exit(main())