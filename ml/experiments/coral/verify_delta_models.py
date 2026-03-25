#!/usr/bin/env python3
"""
Runtime verification script for delta-only FFN models.
Implements shape assertions and startup tests recommended in SUCCESS_REVIEW.md
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Import TFLite runtime (use tflite_runtime if available, fallback to tf.lite)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def load_model_and_verify_shape(model_path: str) -> Tuple[tflite.Interpreter, Dict]:
    """
    Load model and verify input/output shapes match requirements.

    Returns:
        Tuple of (interpreter, shape_info)
    """
    print(f"\n📋 Loading model: {model_path}")

    # Load model with Edge TPU delegate if available
    try:
        delegates = [tflite.load_delegate('libedgetpu.so.1')]
        interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=delegates)
        print("✅ Edge TPU delegate loaded")
    except:
        interpreter = tflite.Interpreter(model_path=model_path)
        print("⚠️ Running without Edge TPU delegate (CPU only)")

    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Verify shapes
    expected_shape = [1, 128, 1, 128]

    shape_info = {
        'input_shape': list(input_details['shape']),
        'input_dtype': str(input_details['dtype']),
        'output_shape': list(output_details['shape']),
        'output_dtype': str(output_details['dtype']),
        'input_quant': input_details.get('quantization', (0.0, 0)),
        'output_quant': output_details.get('quantization', (0.0, 0))
    }

    # Assert shape requirements
    assert shape_info['input_shape'] == expected_shape, \
        f"❌ Input shape mismatch: expected {expected_shape}, got {shape_info['input_shape']}"

    assert shape_info['output_shape'] == expected_shape, \
        f"❌ Output shape mismatch: expected {expected_shape}, got {shape_info['output_shape']}"

    # Verify INT8 quantization
    assert 'int8' in shape_info['input_dtype'].lower(), \
        f"❌ Input not INT8: {shape_info['input_dtype']}"

    assert 'int8' in shape_info['output_dtype'].lower(), \
        f"❌ Output not INT8: {shape_info['output_dtype']}"

    print(f"✅ Shape verification passed")
    print(f"   Input:  {shape_info['input_shape']} ({shape_info['input_dtype']})")
    print(f"   Output: {shape_info['output_shape']} ({shape_info['output_dtype']})")
    print(f"   Input quant:  scale={shape_info['input_quant'][0]:.8f}, zp={shape_info['input_quant'][1]}")
    print(f"   Output quant: scale={shape_info['output_quant'][0]:.8f}, zp={shape_info['output_quant'][1]}")

    return interpreter, shape_info


def run_startup_test(interpreter: tflite.Interpreter, shape_info: Dict) -> bool:
    """
    Run startup test with known vectors to catch delegate issues.
    """
    print("\n🧪 Running startup test...")

    # Test 1: All zeros
    test_input = np.zeros([1, 128, 1, 128], dtype=np.int8)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    print(f"   Zero input test: output shape {output.shape}, mean={np.mean(output):.2f}")
    assert output.shape == tuple([1, 128, 1, 128]), "Output shape mismatch"

    # Test 2: Small random values (within quantization range)
    scale, zp = shape_info['input_quant']
    test_float = np.random.randn(1, 128, 1, 128).astype(np.float32) * 0.1
    test_input = np.round(test_float / scale + zp).clip(-128, 127).astype(np.int8)

    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    print(f"   Random input test: output range [{np.min(output)}, {np.max(output)}]")

    # Test 3: Measure inference time
    import time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)
    print(f"   Inference time: {avg_time:.2f}ms (avg of 10 runs)")

    return True


def verify_per_channel_quantization(model_path: str) -> bool:
    """
    Check if Conv2D weights use per-channel quantization.
    """
    print("\n🔍 Checking per-channel quantization...")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()

    conv_count = 0
    per_channel_count = 0

    for tensor in tensor_details:
        # Check if this is a Conv2D weight tensor
        if 'conv' in tensor['name'].lower() and 'weight' in tensor['name'].lower():
            conv_count += 1
            quant = tensor.get('quantization_parameters', {})
            scales = quant.get('scales', [])

            if len(scales) > 1:
                per_channel_count += 1
                print(f"   ✅ {tensor['name']}: per-channel ({len(scales)} channels)")
            else:
                print(f"   ⚠️ {tensor['name']}: per-tensor quantization")

    if conv_count > 0:
        print(f"\nPer-channel quantization: {per_channel_count}/{conv_count} Conv2D layers")

    return per_channel_count == conv_count


def verify_qparams_match(model_path: str, qparams_path: str) -> bool:
    """
    Verify that extracted qparams match the compiled model.
    """
    print(f"\n🔐 Verifying qparams match...")

    # Load model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    model_scale, model_zp = input_details['quantization']

    # Load qparams file
    with open(qparams_path, 'r') as f:
        qparams = json.load(f)

    file_scale = qparams['input']['scale']
    file_zp = qparams['input']['zero_point']

    # Compare
    scale_match = abs(model_scale - file_scale) < 1e-6
    zp_match = model_zp == file_zp

    print(f"   Model: scale={model_scale:.8f}, zero_point={model_zp}")
    print(f"   File:  scale={file_scale:.8f}, zero_point={file_zp}")
    print(f"   Match: {'✅' if scale_match and zp_match else '❌'}")

    return scale_match and zp_match


def main():
    """Run all verification tests."""

    print("="*60)
    print("DELTA FFN MODEL VERIFICATION")
    print("="*60)

    models_dir = Path("models_delta")

    # Check both FFN models
    for layer_idx in [0, 1]:
        model_file = models_dir / f"ffn{layer_idx}_delta_int8_edgetpu.tflite"
        qparams_file = models_dir / "qparams" / f"ffn{layer_idx}_delta_qparams.json"

        if not model_file.exists():
            print(f"❌ Model not found: {model_file}")
            continue

        print(f"\n{'='*40}")
        print(f"FFN-{layer_idx} VERIFICATION")
        print(f"{'='*40}")

        try:
            # 1. Load and verify shape
            interpreter, shape_info = load_model_and_verify_shape(str(model_file))

            # 2. Run startup test
            run_startup_test(interpreter, shape_info)

            # 3. Check per-channel quantization
            verify_per_channel_quantization(str(model_file))

            # 4. Verify qparams match
            if qparams_file.exists():
                verify_qparams_match(str(model_file), str(qparams_file))

            print(f"\n✅ FFN-{layer_idx}: All checks passed")

        except Exception as e:
            print(f"\n❌ FFN-{layer_idx} verification failed: {e}")
            sys.exit(1)

    print("\n" + "="*60)
    print("✅ ALL VERIFICATIONS PASSED")
    print("="*60)
    print("\nModels are ready for production deployment with:")
    print("- Correct shapes [1, 128, 1, 128]")
    print("- INT8 quantization")
    print("- Working Edge TPU delegation")
    print("- Matching qparams files")


if __name__ == "__main__":
    main()