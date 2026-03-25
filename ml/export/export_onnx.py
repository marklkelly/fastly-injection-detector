#!/usr/bin/env python3
"""
Export clean ONNX model with 2 inputs (int64) for production deployment.
This creates a cleaner model without token_type_ids and with proper int64 inputs.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime


def export_clean_model(model_path: str, output_path: str):
    """Export model with 2 inputs (input_ids, attention_mask) as int64."""
    
    print(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torchscript=True,
        attn_implementation="eager",  # Avoid SDPA which requires opset 14
    )
    model.eval()
    
    # Create dummy inputs (int64, as BERT expects)
    batch_size = 1
    seq_length = 128
    dummy_input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    
    print("Exporting to ONNX with 2 inputs (int64)...")
    
    # Export with only 2 inputs - no token_type_ids
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            verbose=False,
            dynamo=False,  # Force TorchScript exporter; opset 11 uses static Unsqueeze axes
        )
    
    print(f"Model exported to {output_path}")
    
    # Verify the export
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Check inputs
    inputs = onnx_model.graph.input
    print(f"Number of inputs: {len(inputs)}")
    for i, input_tensor in enumerate(inputs):
        print(f"  Input {i}: {input_tensor.name}, type: {input_tensor.type.tensor_type.elem_type}")
    
    # Verify with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    session = ort.InferenceSession(output_path)
    
    # Test inference
    test_ids = np.zeros((1, 128), dtype=np.int64)
    test_mask = np.ones((1, 128), dtype=np.int64)
    
    outputs = session.run(None, {
        'input_ids': test_ids,
        'attention_mask': test_mask
    })
    
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output sample: {outputs[0][0][:5]}...")  # First 5 logits
    
    return True


def quantize_model(input_path: str, output_path: str):
    """Quantize the clean ONNX model to INT8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print(f"\nQuantizing model to INT8...")
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True
        )
        
        # Check size reduction
        original_size = Path(input_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"Original size: {original_size:.2f} MB")
        print(f"Quantized size: {quantized_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        
        return True
        
    except ImportError:
        print("WARNING: onnxruntime-tools not installed. Skipping quantization.")
        print("Install with: pip install onnxruntime-tools")
        return False


def backup_existing_model(model_path: str):
    """Backup existing model before replacing."""
    if Path(model_path).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{model_path}.backup_{timestamp}"
        shutil.copy2(model_path, backup_path)
        print(f"Backed up existing model to: {backup_path}")
        return backup_path
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Export clean ONNX model with 2 inputs")
    parser.add_argument(
        "--model-path",
        default="ml/models/bert-tiny-pi-v1",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--output-dir",
        default="fastly/assets",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip INT8 quantization"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true", 
        help="Don't backup existing model"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_model_path = output_dir / "student_2inputs_f32.onnx"
    quantized_model_path = output_dir / "student_2inputs_int8.onnx"
    final_model_path = output_dir / "student_1x128_int8.onnx"
    
    # Backup existing model
    if not args.no_backup:
        backup_existing_model(final_model_path)
    
    # Export clean model
    success = export_clean_model(args.model_path, str(clean_model_path))
    if not success:
        print("ERROR: Model export failed")
        return 1
    
    # Quantize if requested
    if not args.skip_quantization:
        success = quantize_model(str(clean_model_path), str(quantized_model_path))
        if success:
            print(f"\nModel successfully quantized to: {quantized_model_path}")
            print("\nTo deploy this model:")
            print(f"  1. cp {quantized_model_path} {final_model_path}")
            print(f"  2. cargo build --release --target wasm32-wasip1 --features 'inference two_inputs_i64'")
            print(f"  3. fastly compute serve  # Test locally")
            print(f"  4. fastly compute deploy  # Deploy to production")
        else:
            print(f"\nUsing unquantized model: {clean_model_path}")
    
    print("\n✅ Export complete!")
    print("\nNote: The new model is NOT yet deployed. Test it first, then copy to production path.")
    
    return 0


if __name__ == "__main__":
    exit(main())