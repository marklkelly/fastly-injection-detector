#!/usr/bin/env python3
"""
Phase A: Quick falsification - Convert ONNX model to TFLite INT8.
Expected to fail Edge TPU compilation gates due to unsupported ops.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import json
import argparse
from pathlib import Path


def load_tokenizer_vocab(tokenizer_path):
    """Load tokenizer to get vocab size."""
    with open(tokenizer_path, 'r') as f:
        tokenizer_config = json.load(f)
    vocab = tokenizer_config.get('model', {}).get('vocab', {})
    return len(vocab) if vocab else 30522  # BERT default


def create_representative_dataset(batch_size=1, seq_len=128, num_samples=100):
    """Create representative dataset for quantization calibration."""
    def dataset_gen():
        for _ in range(num_samples):
            # Generate random token IDs (typical for BERT)
            input_ids = np.random.randint(0, 30522, size=(batch_size, seq_len), dtype=np.int32)
            yield [input_ids.astype(np.float32)]  # TFLite expects float32 for calibration
    return dataset_gen


def convert_onnx_to_tflite(onnx_path, output_path, tokenizer_path=None):
    """
    Convert ONNX model to fully quantized INT8 TFLite.

    Phase A attempt - expected to fail Edge TPU compilation due to:
    - Softmax ops in attention
    - LayerNorm ops
    - GELU activations
    - Gather ops from embeddings
    """
    print(f"Loading ONNX model from {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    # Check ONNX model ops to predict failure
    print("\nAnalyzing ONNX ops (predicting Edge TPU incompatibility):")
    op_types = set()
    for node in onnx_model.graph.node:
        op_types.add(node.op_type)

    unsupported_ops = {'Softmax', 'LayerNormalization', 'Erf', 'Gather', 'Unsqueeze', 'Squeeze'}
    found_unsupported = op_types & unsupported_ops
    if found_unsupported:
        print(f"⚠️  Found Edge TPU-unsupported ops: {found_unsupported}")
        print("   (Phase A is expected to fail compilation gates)")

    # Convert ONNX to TensorFlow
    print("\nConverting ONNX to TensorFlow...")
    tf_rep = prepare(onnx_model)

    # Export to SavedModel format
    saved_model_dir = 'models/tf_saved_model'
    print(f"Exporting TensorFlow SavedModel to {saved_model_dir}")
    tf_rep.export_graph(saved_model_dir)

    # Convert to TFLite with full INT8 quantization
    print("\nConverting to TFLite with full INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Enable full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = create_representative_dataset()

    # Force INT8 for Edge TPU compatibility
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Allow custom ops as fallback (will fail Edge TPU compilation)
    converter.allow_custom_ops = True

    try:
        tflite_model = converter.convert()

        # Save TFLite model
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"\n✅ TFLite model saved to {output_path}")
        print(f"   Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

        # Analyze TFLite ops
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        print("\nTFLite model analysis:")
        print(f"  Input shape: {interpreter.get_input_details()[0]['shape']}")
        print(f"  Input dtype: {interpreter.get_input_details()[0]['dtype']}")
        print(f"  Output shape: {interpreter.get_output_details()[0]['shape']}")
        print(f"  Output dtype: {interpreter.get_output_details()[0]['dtype']}")

        return True

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nThis is expected for Phase A - BERT ops are not Edge TPU compatible")
        print("Proceeding to Phase B (post-embedding hybrid) is recommended")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase A: Convert ONNX to TFLite INT8")
    parser.add_argument('--onnx', default='models/student_1x128_int8.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--output', default='models/phase_a_int8.tflite',
                        help='Output TFLite path')
    parser.add_argument('--tokenizer', default='/Users/mkelly/Documents/projects/fastly-injection-detector/fastly/assets/tokenizer.json',
                        help='Path to tokenizer.json')

    args = parser.parse_args()

    print("="*60)
    print("PHASE A: Quick Falsification (2-4 hour time box)")
    print("="*60)
    print("\nExpected outcome: Compilation will fail Edge TPU gates")
    print("Reason: BERT contains Softmax, LayerNorm, GELU, Gather ops")
    print("Purpose: Establish baseline before Phase B")

    success = convert_onnx_to_tflite(args.onnx, args.output, args.tokenizer)

    if success:
        print("\n" + "="*60)
        print("Next step: Run edgetpu_compiler")
        print("="*60)
        print(f"edgetpu_compiler -s {args.output}")
        print("\nThen check gates with:")
        print("python scripts/compile_and_gate.py --report compile.log")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())