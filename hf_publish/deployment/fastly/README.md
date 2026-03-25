# Fastly Compute Deployment

This directory contains artifacts for deploying `bert-tiny-injection-detector` on
[Fastly Compute](https://www.fastly.com/products/edge-compute) using
[`tract-onnx`](https://github.com/sonos/tract) in a Rust WASM service.

## Files

| File | Description |
|---|---|
| `calibrated_thresholds.json` | Calibrated block and review thresholds for the injection model |

## calibrated_thresholds.json

```json
{
  "injection": {
    "T_block_at_1pct_FPR": 0.9403,
    "T_review_lower_at_2pct_FPR": 0.8692
  }
}
```

| Threshold | Score range | Decision |
|---|---|---|
| Below `T_review` | score < 0.8692 | Allow |
| Review band | 0.8692 ≤ score < 0.9403 | Review |
| At or above `T_block` | score ≥ 0.9403 | Block |

## ONNX requirements for tract-onnx

- Use `onnx/opset11/model.int8.onnx` (or `model.fp32.onnx` for debugging)
- **Opset 11 is required.** Opset ≥ 13 uses dynamic `Unsqueeze` axes that `tract` cannot
  resolve statically. The opset-11 graph has only 2 static `Unsqueeze` nodes.
- Input tensors must be `int64` of shape `[1, 128]`
- Apply `head_tail` truncation before inference for inputs longer than 128 tokens

## Memory and latency

Measured on Fastly Compute (production, service v11: opt-level=3, Wizer pre-init, simd128):

| Metric | Value |
|---|---|
| Median inference | ~69 ms |
| Median total service elapsed | ~70 ms |
| p95 total service elapsed | ~85 ms |
| Memory footprint | < 128 MB budget |

The inference time exceeds the nominal 50 ms Fastly CPU budget by ~1.4×. This is WASM
overhead — INT8 SIMD paths are not accelerated in the sandbox. The service is functional
at this latency. Wizer pre-initialization eliminates the lazy-static init cost (~163 ms
in earlier versions); the remaining time is pure BERT inference.
