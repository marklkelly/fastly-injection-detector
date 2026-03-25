# Coral Edge TPU Experiment

**Status: Active experiment — suspended pending inference harness fix**

This experiment explores deploying the bert-tiny injection classifier on Google
Coral Edge TPU hardware. 

---

## The fundamental challenge

Coral's Edge TPU is optimised for convolutional networks. It runs INT8
operations fused into single-segment TFLite graphs. Transformer architectures
(like BERT) are not a natural fit for three reasons:

1. **Dynamic shapes in attention.** The `Gather`/`Unsqueeze` chains in
   standard multi-head attention reference dynamic shapes at runtime. The Edge
   TPU compiler requires all tensor shapes to be statically known at compile
   time, so the attention block cannot be mapped to the TPU.

2. **Complex op mix.** Softmax, layer norm, and the various residual additions
   in the encoder use ops the TPU either cannot execute or cannot fuse into a
   single TPU segment. Any break in the segment forces a CPU round-trip, which
   kills the latency benefit.

3. **Single-segment requirement.** For the TPU's throughput advantage to
   matter, the entire model (or a large continuous subgraph) needs to compile
   as one segment. Splitting into many small TPU segments with CPU fallbacks
   typically produces worse latency than running the whole model on CPU.

These constraints rule out a direct BERT deployment. The work here is about
finding a decomposition that gives the TPU enough of the model to be useful.

---

## What has been done

### Phase A: Quantization pipeline

Static INT8 quantization using real, balanced calibration data (50/50
injection/safe split). The critical lessons from this phase:

- **Random calibration data is catastrophic.** The first attempt used randomly
  generated inputs, producing a quantized model with valid structure but
  garbage parameters — all predictions converged to SAFE.
- **Dataset balance matters.** Even with real inputs, a 99.94%/0.06%
  SAFE/INJECTION imbalance in the calibration set collapsed the injection
  signal entirely.

### Phase B: Delta-only FFN architecture

Since the attention block cannot map to the TPU, the approach shifted to
extracting only the FFN (feed-forward network) sub-blocks and running them as
standalone TPU models. The transformer residual structure means this is
well-defined: each FFN computes a delta that is added back to the layer input.

The FFN sub-block (`Linear -> GELU -> Linear`, 128 -> 512 -> 128) maps cleanly
to two 1x1 Conv2D operations, which the Edge TPU compiler handles in a single
segment.

**Result: 100% TPU mapping for FFN layers** (up from 28.6% when attempting to
compile the full model).

The compiled models are in `models_delta/`:

| Model | TPU ops | Size | Quality |
|-------|---------|------|---------|
| `ffn0_delta_int8_edgetpu.tflite` | 2x Conv2D, 100% | 169KB | MAE 0.25 LSB, 0% sat |
| `ffn1_delta_int8_edgetpu.tflite` | 2x Conv2D, 100% | 169KB | MAE 0.25 LSB, 0% sat |

FFN-only TPU latency: ~10ms per forward pass (USB Coral). Full model latency
with CPU-side attention: ~95ms total (TPU 10ms + CPU 85ms). This is the
measured operating point — not yet competitive with the Fastly Wasm service
at 69ms median.

### Where the work stopped

End-to-end inference with trained weights does not produce correct predictions.
The inference harness runs the FFN layers on the TPU correctly, but the
CPU-side components (attention, layer norm, pooler, classifier) are implemented
in custom NumPy code using weights extracted from the ONNX model. That
extraction and reimplementation is producing incorrect output — all inputs
classify as SAFE at ~0.57–0.65 confidence regardless of content.

Multiple weight extraction and harness debug attempts were made without
resolving the root cause. The leading hypotheses (see `EDGE_TPU_DEPLOYMENT_ISSUES.md`
— now removed; detail preserved in Obsidian at
`projects/fastly-injection-detector/experiments/coral-edge-tpu-experiment`)
were: incorrect dequantization of attention weights from INT8 ONNX, subtle
transposition errors in the MatMul weight mapping, and possible layer norm
placement mismatches.

---

## Where to look next

The custom NumPy inference harness is the wrong approach. Reimplementing
the BERT forward pass by hand is error-prone and fragile. Two better paths:

**Option A (most likely to succeed): ONNX Runtime for CPU layers.**
Run the full ONNX model through ORT on the host CPU for everything except the
FFN layers. Intercept the FFN inputs and outputs, route those through the TPU
models, and inject the TPU outputs back into the computation graph. This gives
a correct CPU baseline with a well-defined, narrow TPU insertion point — no
custom BERT reimplementation required. The weight extraction and harness
correctness problems go away entirely.

**Option B: Validate FFN-only pass first.**
Before any end-to-end work, verify that the TPU FFN outputs match the
corresponding ONNX subgraph outputs for the same inputs. This rules out
quantization and scale/zero-point issues as a source of error. If the FFNs
match, the problem is definitively in the CPU harness. If they don't, the
TFLite compilation or quantization parameters need revisiting.

**Option C: Latency-first question.**
The ~85ms CPU attention cost dominates. Even if accuracy is fixed, the total
~95ms doesn't beat 69ms Fastly Wasm. Before further debugging, it's worth
profiling where the CPU time actually goes — if attention on a real host CPU
(not Raspberry Pi) gets to ~20ms, the total (~30ms) becomes compelling. The
`PRODUCTION_READINESS_PLAN.md` (now in Obsidian) noted that OpenBLAS + INT8
attention quantization could reduce the CPU component significantly.

---

## Repository layout

```
ml/experiments/coral/
  README.md                        # This file
  requirements_coral.txt           # Python 3.9 dependencies
  models_delta/                    # Compiled Edge TPU artifacts
    ffn0_delta_int8_edgetpu.tflite
    ffn1_delta_int8_edgetpu.tflite
    ffn0_edgetpu.tflite
    ffn1_edgetpu.tflite
    MODEL_CARD.md                  # Compilation quality report
    qparams/                       # Quantization parameters per layer
    compile_logs/                  # edgetpu_compiler output
    SHA256SUMS
  export_ffn_delta.py              # Export FFN deltas from ONNX to TFLite
  export_ffn_delta_simple.py       # Simplified export variant
  src/
    coral_phase_b_revised.py       # Delta-only FFN architecture (final)
    production_pipeline.py         # E2E pipeline skeleton
    static_quantization.py         # INT8 quantization pipeline
  scripts/
    compile_and_gate.py            # Compile TFLite + quality gate check
    e2e_benchmark.py               # End-to-end latency benchmark
  inference_harness.py             # Current (broken) end-to-end harness
  venv_coral/                      # Python 3.9 venv — gitignored, root-owned
```

## Setup

Requires Python 3.9 (pycoral is not compatible with later versions) and a
Coral USB Accelerator or M.2 module.

```bash
python3.9 -m venv venv_coral
source venv_coral/bin/activate
pip install -r requirements_coral.txt
```

The venv is gitignored. If files are owned by root (from a prior Docker/sudo
run), recreate with: `sudo rm -rf venv_coral` then the steps above.
