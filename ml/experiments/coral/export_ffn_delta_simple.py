#!/usr/bin/env python3
"""
⚠️ DO NOT SHIP - TEST ONLY ⚠️
Simplified export of delta-only FFN models for Edge TPU with ≥90% mapping.

WARNING: This script uses RANDOM calibration data and is for CI smoke-testing only.
For production models, use export_ffn_delta.py with real representative datasets.
"""

import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path


def build_ffn_delta(layer_idx, input_channels=128, mid_channels=192):
    """
    Build delta-only FFN model with Conv2D layers.
    No residual add in the model - done on CPU at runtime.
    """
    # Static shapes to keep the converter happy
    inp = tf.keras.Input(
        batch_size=1,
        shape=(128, 1, input_channels),
        dtype=tf.float32,
        name="x_in"
    )

    # 1x1 Conv → ReLU6 (TPU-friendly activation)
    x = tf.keras.layers.Conv2D(
        filters=mid_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu6,
        use_bias=True,
        name=f"ffn{layer_idx}_conv1"
    )(inp)

    # 1x1 Conv (linear) - outputs delta only
    delta = tf.keras.layers.Conv2D(
        filters=input_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation=None,
        use_bias=True,
        name=f"ffn{layer_idx}_conv2"
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=delta, name=f"ffn{layer_idx}_delta")
    return model


def load_weights_for_layer(weights_dir, layer_idx):
    """Load pre-trained weights for FFN layer."""
    weights_dir = Path(weights_dir)

    # Load FFN weights
    ffn_file = weights_dir / f"ffn_L{layer_idx}.npz"
    if not ffn_file.exists():
        raise FileNotFoundError(f"FFN weights not found: {ffn_file}")

    ffn_weights = np.load(ffn_file)
    w_in = ffn_weights['W_in']  # Original: [512, 128]
    w_out = ffn_weights['W_out']  # Original: [128, 512]

    # Truncate to 192-wide (using first 192 channels)
    print(f"  Original W_in shape: {w_in.shape}")
    w_in_192 = w_in[:192, :]  # [192, 128]
    w_out_192 = w_out[:, :192]  # [128, 192]
    print(f"  Truncated to W_in: {w_in_192.shape}, W_out: {w_out_192.shape}")

    # Use zero biases for 192 channels
    b_in = np.zeros(192, dtype=np.float32)
    b_out = np.zeros(128, dtype=np.float32)

    # Reshape for Conv2D: [kernel_h, kernel_w, in_channels, out_channels]
    w_in_conv = w_in_192.T.reshape(1, 1, 128, 192)
    w_out_conv = w_out_192.T.reshape(1, 1, 192, 128)

    return {
        'conv1': {'kernel': w_in_conv, 'bias': b_in},
        'conv2': {'kernel': w_out_conv, 'bias': b_out}
    }


def set_model_weights(model, weights):
    """Set pre-trained weights on the model."""
    # Set conv1 weights
    conv1_layer = model.get_layer(name=[l.name for l in model.layers if 'conv1' in l.name][0])
    conv1_layer.set_weights([weights['conv1']['kernel'], weights['conv1']['bias']])

    # Set conv2 weights
    conv2_layer = model.get_layer(name=[l.name for l in model.layers if 'conv2' in l.name][0])
    conv2_layer.set_weights([weights['conv2']['kernel'], weights['conv2']['bias']])


def create_simple_representative_dataset(n_samples=2000):
    """
    Create simple representative dataset using random data.
    In production, this would use real post-attention tensors.
    """
    def representative_dataset():
        for _ in range(n_samples):
            # Generate random post-attention-like data
            # Shape: [1, 128, 1, 128] for Conv2D
            x = np.random.randn(1, 128, 1, 128).astype(np.float32)
            # Scale to typical activation range
            x = x * 0.5
            yield [x]

    return representative_dataset


def export_int8_tflite_delta(model, representative_dataset, output_path):
    """Export model as INT8 TFLite with delta-only architecture."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Use INT8 only operations
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Enable new quantizer for more stable ranges (if available)
    try:
        converter.experimental_new_quantizer = True
    except:
        pass  # Not available in this TF version

    # Convert
    tflite_model = converter.convert()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    print(f"✅ Exported: {output_path} ({len(tflite_model)/1024:.1f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export delta-only FFN models for Edge TPU")
    parser.add_argument("--layer", type=int, required=True, choices=[0, 1],
                        help="FFN layer index (0 or 1)")
    parser.add_argument("--weights_dir", type=str, default="weights_extracted",
                        help="Directory containing extracted weights")
    parser.add_argument("--output_dir", type=str, default="models_delta_192",
                        help="Output directory for models")
    parser.add_argument("--mid_channels", type=int, default=192,
                        help="FFN intermediate dimension")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"EXPORTING DELTA-ONLY FFN-{args.layer} MODEL")
    print(f"{'='*60}")

    # Build model
    print(f"\n📦 Building delta-only FFN-{args.layer} model...")
    model = build_ffn_delta(
        layer_idx=args.layer,
        input_channels=128,
        mid_channels=args.mid_channels
    )
    model.summary()

    # Load pre-trained weights
    print(f"\n📂 Loading weights from {args.weights_dir}...")
    weights = load_weights_for_layer(args.weights_dir, args.layer)
    set_model_weights(model, weights)
    print("✅ Weights loaded successfully")

    # Create representative dataset (simplified for now)
    print(f"\n📊 Creating representative dataset...")
    rep_dataset = create_simple_representative_dataset(n_samples=2000)

    # Export as INT8 TFLite
    print(f"\n🔧 Exporting INT8 TFLite model...")
    output_path = Path(args.output_dir) / f"ffn{args.layer}_delta_int8.tflite"
    export_int8_tflite_delta(model, rep_dataset, output_path)

    print(f"\n{'='*60}")
    print(f"✅ EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Transfer to Linux x86_64")
    print(f"2. Run: edgetpu_compiler -s {output_path}")
    print(f"3. Verify mapping ≥90%")


if __name__ == "__main__":
    main()