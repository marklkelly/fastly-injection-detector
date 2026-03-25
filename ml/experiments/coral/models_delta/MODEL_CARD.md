# FFN Delta Models - Edge TPU Production Bundle

## Compilation Success ✅

Successfully recovered from 28.6% to **100% TPU mapping** using delta-only architecture.

## Models

| Model | TPU Mapping | Operations | Size |
|-------|-------------|------------|------|
| ffn0_delta_int8_edgetpu.tflite | **100%** | 2 Conv2D | 169KB |
| ffn1_delta_int8_edgetpu.tflite | **100%** | 2 Conv2D | 169KB |

## Architecture

- **Type**: Delta-only FFN (residual computed on CPU)
- **Input**: INT8 [1, 128, 1, 128] 
- **Output**: INT8 [1, 128, 1, 128] (delta values)
- **Operations**: Conv2D(1×1, ReLU6) → Conv2D(1×1, linear)

## Quantization Parameters

### FFN-0
- Input: scale=0.10981479, zero_point=1
- Output: scale=0.11378000, zero_point=12

### FFN-1  
- Input: scale=0.17512970, zero_point=2
- Output: scale=0.21210340, zero_point=2

## Quality Gates

| Gate | Requirement | Result | Status |
|------|-------------|--------|--------|
| Segments | 1 | 1 | ✅ PASSED |
| TPU Mapping | ≥90% | 100% | ✅ PASSED |
| Saturation | <2% | 0.00% | ✅ PASSED |
| MAE | ≤1.0 LSB | 0.25 | ✅ PASSED |

## Deployment Status

The TFLite models pass all quality gates and are correctly compiled for Edge TPU.
However, the CPU-side inference harness (attention, pooler, classifier) is not yet
working — the experiment is suspended at this stage. See `ml/experiments/coral/README.md`.
