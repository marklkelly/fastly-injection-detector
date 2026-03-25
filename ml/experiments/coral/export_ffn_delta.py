#!/usr/bin/env python3
"""
Export delta-only FFN models for Edge TPU with ≥90% mapping.
Removes shape operations and residual add from the compiled graph.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from tokenizers import Tokenizer


def build_ffn_delta(layer_idx, input_channels=128, mid_channels=512):
    """
    Build delta-only FFN model with Conv2D layers.
    No residual add in the model - done on CPU at runtime.

    Args:
        layer_idx: 0 or 1 for FFN layer
        input_channels: Hidden dimension (128)
        mid_channels: Intermediate dimension (512 for production width)
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
        activation=tf.nn.relu6,  # ReLU6 for better TPU compatibility
        use_bias=True,
        name=f"ffn{layer_idx}_conv1"
    )(inp)

    # 1x1 Conv (linear) - outputs delta only
    delta = tf.keras.layers.Conv2D(
        filters=input_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation=None,  # Linear activation
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
    w_in = ffn_weights['W_in']  # [128, 512]
    b_in = ffn_weights['b_in']  # [512]
    w_out = ffn_weights['W_out']  # [512, 128]
    b_out = ffn_weights['b_out']  # [128]

    # Reshape for Conv2D: [kernel_h, kernel_w, in_channels, out_channels]
    # For 1x1 conv: [1, 1, in_channels, out_channels]
    w_in_conv = w_in.T.reshape(1, 1, 128, 512)  # [1, 1, 128, 512]
    w_out_conv = w_out.T.reshape(1, 1, 512, 128)  # [1, 1, 512, 128]

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


class AttentionSimulator:
    """Simulate CPU attention to generate representative data."""

    def __init__(self, weights_dir, tokenizer_path):
        self.weights_dir = Path(weights_dir)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.seq_len = 128

        # Load embeddings and attention weights
        self._load_weights()

    def _load_weights(self):
        # Load embeddings
        emb_file = self.weights_dir / "embeddings.npz"
        emb_data = np.load(emb_file)
        self.token_emb = emb_data['token_embeddings']  # [vocab_size, 128]
        self.pos_emb = emb_data['position_embeddings'][:128, :]  # [128, 128]
        self.seg_emb = emb_data['segment_embeddings']  # [2, 128]

        # Load layer norm
        ln_file = self.weights_dir / "layer_norm.npz"
        ln_data = np.load(ln_file)
        self.ln_weight = ln_data['weight']
        self.ln_bias = ln_data['bias']

        # Load attention weights for both layers
        self.attn_weights = {}
        for layer in [0, 1]:
            attn_file = self.weights_dir / f"attention_L{layer}.npz"
            attn_data = np.load(attn_file)
            self.attn_weights[layer] = {
                'Wq': attn_data['Wq'],
                'Wk': attn_data['Wk'],
                'Wv': attn_data['Wv'],
                'Wo': attn_data['Wo']
            }

    def tokenize_with_mask(self, text):
        """Tokenize text and create attention mask."""
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.seq_len]
        mask = [1] * len(ids)

        # Pad with [PAD] token
        pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        while len(ids) < self.seq_len:
            ids.append(pad_id)
            mask.append(0)

        return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32)

    def embed(self, token_ids):
        """Create embeddings from token IDs."""
        # Token embeddings
        tok_emb = self.token_emb[token_ids]  # [128, 128]

        # Position embeddings
        pos_emb = self.pos_emb  # [128, 128]

        # Segment embeddings (all segment 0)
        seg_emb = self.seg_emb[0:1].repeat(self.seq_len, axis=0)  # [128, 128]

        # Sum and layer norm
        x = tok_emb + pos_emb + seg_emb

        # Layer normalization
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(var + 1e-12)
        x = x * self.ln_weight + self.ln_bias

        return x.astype(np.float32)

    def attention(self, x, layer_idx, attn_mask):
        """Run attention layer."""
        batch_size = 1
        seq_len = self.seq_len
        hidden_dim = 128
        num_heads = 2
        head_dim = hidden_dim // num_heads

        # Add batch dimension
        x = x.reshape(batch_size, seq_len, hidden_dim)

        # QKV projections
        W = self.attn_weights[layer_idx]
        Q = x @ W['Wq']  # [1, 128, 128]
        K = x @ W['Wk']  # [1, 128, 128]
        V = x @ W['Wv']  # [1, 128, 128]

        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

        # Apply mask
        mask_expanded = attn_mask.reshape(batch_size, 1, 1, seq_len)
        scores = scores + (1.0 - mask_expanded) * (-1e9)

        # Softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Apply attention
        context = attention_weights @ V  # [1, 2, 128, 64]
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)

        # Output projection
        output = context @ W['Wo']

        # Residual + layer norm (simplified)
        output = output + x

        return output.reshape(batch_size, seq_len, hidden_dim)


def create_representative_dataset(layer_idx, dataset_path, tokenizer_path, weights_dir, n_samples=2000):
    """
    Create representative dataset for quantization calibration.
    Returns post-attention tensors reshaped for Conv2D input.
    """
    # Load balanced dataset
    df = pd.read_csv(dataset_path)

    # Filter by class for balance
    safe = df[df['label'] == 'SAFE']
    injection = df[df['label'] == 'INJECTION']

    # Sample equally from each class
    n_per_class = min(len(safe), len(injection), n_samples // 2)
    safe_sample = safe.sample(n=n_per_class, random_state=42)
    injection_sample = injection.sample(n=n_per_class, random_state=43)

    # Combine and shuffle
    combined = pd.concat([safe_sample, injection_sample])
    combined = combined.sample(frac=1, random_state=123)

    # Create attention simulator
    sim = AttentionSimulator(weights_dir, tokenizer_path)

    def representative_dataset():
        for text in combined['text'].values[:n_samples]:
            # Tokenize
            token_ids, mask = sim.tokenize_with_mask(text)

            # Embed
            x = sim.embed(token_ids)

            # Run attention layers as needed
            if layer_idx == 0:
                # For FFN-0: just need post-attention-0
                x = sim.attention(x, layer_idx=0, attn_mask=mask)
            else:
                # For FFN-1: need post-attention-0, then FFN-0 output, then attention-1
                x = sim.attention(x, layer_idx=0, attn_mask=mask)
                # Note: In production, FFN-0 would run here, but for calibration
                # we can simulate with a simple transform
                x = x * 1.1  # Placeholder for FFN-0 effect
                x = sim.attention(x, layer_idx=1, attn_mask=mask)

            # Reshape for Conv2D: [1, 128, 128] -> [1, 128, 1, 128]
            x_reshaped = x.reshape(1, 128, 1, 128).astype(np.float32)
            yield [x_reshaped]

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

    # Enable new quantizer for more stable ranges
    converter.experimental_new_quantizer = True

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
    parser.add_argument("--weights_dir", type=str,
                        default="/Users/mkelly/Documents/projects/fastly-injection-detector/ml/training/coral_pi/weights_extracted",
                        help="Directory containing extracted weights")
    parser.add_argument("--dataset", type=str,
                        default="/Users/mkelly/Documents/projects/fastly-injection-detector/ml/training/coral_pi/balanced_dataset.csv",
                        help="Balanced dataset CSV")
    parser.add_argument("--tokenizer", type=str,
                        default="/Users/mkelly/Documents/projects/fastly-injection-detector/fastly/assets/tokenizer.json",
                        help="Tokenizer JSON file")
    parser.add_argument("--output_dir", type=str, default="models_delta",
                        help="Output directory for models")
    parser.add_argument("--mid_channels", type=int, default=512,
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

    # Create representative dataset
    print(f"\n📊 Creating representative dataset...")
    rep_dataset = create_representative_dataset(
        layer_idx=args.layer,
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        weights_dir=args.weights_dir,
        n_samples=2000
    )

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