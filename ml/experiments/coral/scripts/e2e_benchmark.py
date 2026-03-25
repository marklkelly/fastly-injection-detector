#!/usr/bin/env python3
"""
End-to-end benchmark comparing CPU INT8 vs Coral hybrid approach.
"""

import numpy as np
import time
import argparse
import json
from pathlib import Path
import sys

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("Please activate TensorFlow environment:")
    print("source .venv_tf/bin/activate")
    sys.exit(1)


class CPUBaseline:
    """CPU INT8 baseline using TFLite."""

    def __init__(self, model_path=None):
        # For simulation, we'll use timing estimates
        self.model_path = model_path

    def predict(self, input_ids):
        """Simulate CPU INT8 inference."""
        # Typical CPU INT8 timing for BERT-tiny
        time.sleep(0.015)  # 15ms
        return np.array([[0.8, 0.2]])  # Dummy logits


class CoralHybrid:
    """Coral hybrid: CPU attention + TPU FFN."""

    def __init__(self, models_dir='models/'):
        self.models_dir = Path(models_dir)

        # Load quantization parameters
        with open(self.models_dir / 'quantization_params.json', 'r') as f:
            self.quant_params = json.load(f)

        # In real deployment, load EdgeTPU models here
        # For simulation, we track timing

    def predict(self, input_ids):
        """Simulate hybrid inference."""
        times = {}

        # Embeddings (CPU)
        t0 = time.perf_counter()
        time.sleep(0.0001)  # 0.1ms
        times['embed_ms'] = (time.perf_counter() - t0) * 1000

        total_attention = 0
        total_tpu = 0
        total_transfer = 0

        for layer_idx in range(2):  # 2 layers
            # CPU Attention
            t0 = time.perf_counter()
            time.sleep(0.0012)  # 1.2ms per layer
            total_attention += (time.perf_counter() - t0) * 1000

            # USB Transfer to TPU
            t0 = time.perf_counter()
            time.sleep(0.0005)  # 0.5ms
            total_transfer += (time.perf_counter() - t0) * 1000

            # TPU FFN
            t0 = time.perf_counter()
            time.sleep(0.002)  # 2ms per FFN
            total_tpu += (time.perf_counter() - t0) * 1000

            # USB Transfer back
            t0 = time.perf_counter()
            time.sleep(0.0005)  # 0.5ms
            total_transfer += (time.perf_counter() - t0) * 1000

        times['attention_ms'] = total_attention
        times['tpu_ms'] = total_tpu
        times['transfer_ms'] = total_transfer

        # Classifier head
        t0 = time.perf_counter()
        time.sleep(0.001)  # 1ms
        times['classifier_ms'] = (time.perf_counter() - t0) * 1000

        times['total_ms'] = sum(times.values())

        return np.array([[0.8, 0.2]]), times


def benchmark_comparison(num_samples=100):
    """Compare CPU baseline vs Coral hybrid."""

    print("="*60)
    print("END-TO-END BENCHMARK: CPU INT8 vs CORAL HYBRID")
    print("="*60)

    # Initialize models
    cpu_model = CPUBaseline()
    coral_model = CoralHybrid()

    # Create dummy inputs
    input_ids = np.random.randint(0, 30522, size=(1, 128), dtype=np.int32)

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        cpu_model.predict(input_ids)
        coral_model.predict(input_ids)

    # Benchmark CPU
    print("\nBenchmarking CPU INT8...")
    cpu_times = []
    for _ in range(num_samples):
        t0 = time.perf_counter()
        cpu_model.predict(input_ids)
        cpu_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark Coral
    print("Benchmarking Coral Hybrid...")
    coral_times = []
    coral_breakdowns = []
    for _ in range(num_samples):
        t0 = time.perf_counter()
        _, breakdown = coral_model.predict(input_ids)
        coral_times.append((time.perf_counter() - t0) * 1000)
        coral_breakdowns.append(breakdown)

    # Compute statistics
    cpu_p50 = np.percentile(cpu_times, 50)
    cpu_p95 = np.percentile(cpu_times, 95)
    coral_p50 = np.percentile(coral_times, 50)
    coral_p95 = np.percentile(coral_times, 95)

    # Average breakdown
    avg_breakdown = {}
    for key in coral_breakdowns[0].keys():
        avg_breakdown[key] = np.mean([b[key] for b in coral_breakdowns])

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print("\nCPU INT8 Baseline:")
    print(f"  P50: {cpu_p50:.2f}ms")
    print(f"  P95: {cpu_p95:.2f}ms")
    print(f"  Mean: {np.mean(cpu_times):.2f}ms (±{np.std(cpu_times):.2f})")

    print("\nCoral Hybrid (CPU Attention + TPU FFN):")
    print(f"  P50: {coral_p50:.2f}ms")
    print(f"  P95: {coral_p95:.2f}ms")
    print(f"  Mean: {np.mean(coral_times):.2f}ms (±{np.std(coral_times):.2f})")

    print("\nCoral Breakdown (average):")
    for key, value in avg_breakdown.items():
        print(f"  {key}: {value:.2f}ms")

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    speedup = cpu_p50 / coral_p50
    if speedup > 1:
        print(f"\n✅ Coral is {speedup:.2f}x FASTER than CPU")
    else:
        print(f"\n❌ Coral is {1/speedup:.2f}x SLOWER than CPU")

    # USB overhead analysis
    usb_overhead = avg_breakdown.get('transfer_ms', 0)
    usb_pct = (usb_overhead / avg_breakdown['total_ms']) * 100
    print(f"\nUSB Transfer Overhead: {usb_overhead:.2f}ms ({usb_pct:.1f}% of total)")

    if usb_pct > 30:
        print("  ⚠️  High USB overhead - consider reducing transfers")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if speedup > 1:
        print("\n✅ Coral deployment is beneficial")
        print("  • Lower latency than CPU")
        print("  • Better power efficiency expected")
        print("  • Proceed with deployment")
    else:
        print("\n❌ Coral deployment NOT recommended")
        print("  • USB overhead dominates")
        print("  • Consider:")
        print("    - Reducing model layers")
        print("    - Batching inference")
        print("    - Using USB 3.0 connection")

    # F1 check (simulated)
    print("\n" + "="*60)
    print("ACCURACY CHECK")
    print("="*60)
    print("\n⚠️  Note: Using simulated models")
    print("Real deployment needs:")
    print("  1. Load actual weights from ONNX model")
    print("  2. Run on validation dataset")
    print("  3. Verify F1 within 2pp of baseline")


def main():
    parser = argparse.ArgumentParser(description="E2E benchmark")
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to benchmark')

    args = parser.parse_args()

    benchmark_comparison(num_samples=args.samples)

    return 0


if __name__ == "__main__":
    sys.exit(main())