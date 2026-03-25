# Edge Inference for Prompt Injection Detection: Deploying a Distilled Classifier on Fastly Compute

**Mark Kelly** — March 2026

## Introduction

Prompt injection defences are typically implemented deep in the application stack, after requests have already reached a hosted model or agent runtime. That design has three costs. It increases latency, because malicious traffic still traverses the full application path before being rejected. It increases spend, because blocked prompts may still consume inference calls before downstream safeguards fire. And it widens the blast radius: once a request reaches a powerful model, even a failed attack can trigger logging, tool selection, or intermediate reasoning that the platform would rather avoid.

Running a detector at the network edge eliminates those costs for the requests it catches. A Fastly Compute service can reject suspicious inputs before they reach an origin, and it can do so in a language-agnostic way because the detector only sees raw text. The question is whether you can make it work within the constraints.

This paper describes the end-to-end process of training a distilled BERT-tiny prompt injection classifier, quantising it to 4.3 MB, and deploying it inside a Fastly Compute WebAssembly sandbox. The production service runs at 69 ms median on Fastly edge hardware, with PR-AUC 0.993 and recall 0.956 at 1% FPR on a held-out validation set of 20,027 examples.

The two most instructive parts of the project were not the model itself. They were a data-curation mistake that actively degraded performance, and a latency problem that required build-toolchain engineering rather than model changes to resolve. Both are covered in detail because they are the kind of problems that determine whether an edge ML deployment actually ships.

## Platform Constraints

Fastly Compute runs your code in a WebAssembly sandbox (`wasm32-wasip1`) that spins up per request. The constraints that matter for ML inference are:

- **128 MB heap memory.** This is the total memory budget for the WASM module, including the model weights, tokenizer vocabulary, inference graph, and all intermediate tensors. There is no swap, no memory-mapped I/O, and no way to stream weights from storage.
- **Tight vCPU budget.** Fastly sets a soft per-request CPU time target of 50 ms. The production deployment described here runs at 69 ms median — approximately 1.4× over the soft target. Zero `compute_service_vcpu_exceeded_error` events were recorded across 200 benchmark requests, confirming the service operates within the hard enforcement boundary at this latency.
- **No filesystem.** All assets — model weights, tokenizer vocabulary, threshold configuration — must be embedded directly into the compiled WASM binary via `include_bytes!` or equivalent. There is no runtime file I/O.
- **No persistent processes.** Each request gets a fresh execution context. There is no connection pooling, no warm model server, no background process that keeps weights in memory between requests. Every optimisation that relies on amortising startup cost across requests is unavailable — unless you can move initialisation to build time.
- **No threading.** Fastly does not expose Wasmtime's shared-memory threading surface to guest code. All inference is single-threaded within a single request.

These constraints rule out Python-based inference runtimes, ONNX Runtime's native builds, and any model large enough to exceed the memory budget once you account for the inference graph and intermediate activations. They also rule out the standard approach of running a model server and sending requests to it — the model has to live inside the request handler.

## Model Design

The detector uses `prajjwal1/bert-tiny`, a 2-layer BERT variant with approximately 4.4 million parameters and a hidden size of 128. It was chosen for three reasons: it fits comfortably in the memory budget after quantisation, it is architecturally simple enough that every operation maps to operators supported by the WASM inference runtime, and it is large enough to learn a useful decision boundary for a focused binary classification task.

### Knowledge Distillation

The model was trained using knowledge distillation from `protectai/deberta-v3-small-prompt-injection-v2`, a DeBERTa-v3-small model fine-tuned specifically for prompt injection detection. The distillation loss combines hard labels and teacher soft labels with equal weight (`alpha=0.5`, `temperature=2.0`). The teacher sees up to 256 tokens while the student is capped at 128, so the teacher can encode richer soft targets for longer prompts even when the deployed model cannot see the full input.

Training used 3 epochs, learning rate `5e-5`, weight decay `0.01`, batch size 32, cosine learning-rate schedule with 5% warmup, and early stopping with patience 3. The best checkpoint was selected by `recall_at_1pct_fpr` rather than PR-AUC, because deployment decisions are made at a fixed low-FPR operating point.

### Quantisation

The trained model is exported to ONNX and then quantised using `onnxruntime.quantization.quantize_dynamic` with INT8 weights and per-channel quantisation. This reduces the model from 16.8 MB (FP32 ONNX) to 4.3 MB (INT8 ONNX) — a 74% size reduction. Dynamic quantisation compresses the weight-heavy linear layers while quantising activations at inference time. It requires no calibration dataset, which keeps the export pipeline simple.

The 4.3 MB quantised model, combined with the tokenizer vocabulary, fits easily within the 128 MB heap budget with room to spare for the inference graph and intermediate tensors.

## The Data Story

This section describes the single most impactful decision in the project — and the mistake that preceded it.

### The Mixed-Dataset Hypothesis

The initial training set (`pi_mix_v1`) combined prompt injection examples with jailbreak examples from `allenai/wildjailbreak`. The hypothesis was that broader coverage of adversarial input types would produce a more robust classifier. The resulting dataset contained 199,822 balanced training examples: 158,289 from `jayavibhav/prompt-injection`, 39,583 from `allenai/wildjailbreak`, and smaller contributions from two other injection-focused sources.

This was a plausible design choice — prompt injection and jailbreaking are often discussed together in the security literature, and a classifier that could catch both would be more useful than one that only caught direct injection. But it turned out to be a category error.

### Source-Sliced Evaluation

The problem surfaced through source-sliced evaluation — breaking validation metrics down by the original data source rather than looking only at aggregate numbers. On the mixed dataset, the injection student achieved recall at 1% FPR of 0.943 on `jayavibhav/prompt-injection` examples, but only 0.070 on `allenai/wildjailbreak` examples.

Qualitative review explained the gap. Wildjailbreak positive examples are typically indirect harmful requests or jailbreak role-play prompts rather than explicit instruction overrides. Their benign counterparts use almost identical structural patterns — harm intent, not syntax, provides the discriminative signal. That is a fundamentally different classification problem from prompt injection.

### Teacher-Task Mismatch

The teacher model confirmed the mismatch quantitatively. `protectai/deberta-v3-small-prompt-injection-v2` scored 51.5% accuracy on 4,949 wildjailbreak validation examples — effectively random. Recall at a 0.5 threshold was 36.2%, and 1,474 true harmful examples received scores between 0.0 and 0.1. The teacher was not just weak on this subset; it was actively providing anti-signal. The student was being trained to match a teacher that could not distinguish the classes it was being asked to separate.

### The Fix and Its Effect

Removing all 39,583 wildjailbreak rows from the training set produced the cleaned `injection_only` dataset: 160,239 training examples and 20,027 validation examples. The effect on model performance was unusually large:

| Dataset | PR-AUC | Recall @ 1% FPR | Recall @ 1% FPR (long sequences) |
|---|---:|---:|---:|
| Mixed (with wildjailbreak) | 0.932 | 0.759 | 0.533 |
| Injection only | 0.993 | 0.956 | 0.956 |

Global recall at 1% FPR improved from 0.759 to 0.956. On long inputs (sequences exceeding 128 tokens), the improvement was even more dramatic: from 0.533 to 0.956. The effect size is too large to attribute to training variance. It is direct evidence that the wrong data can actively train a small student away from the intended decision boundary.

The biggest single gain in the entire project came from getting the task definition right, not from a better model, a better optimizer, or a larger architecture.

Building the mixed dataset was my own design decision, and in retrospect it was overly optimistic about the overlap between injection and jailbreak as classification tasks. But the diagnostic process — source-sliced evaluation revealing per-source performance collapse, followed by direct teacher evaluation confirming the mismatch — is exactly the kind of systematic debugging that catches these problems before they reach production.

## Deployment Engineering

### The ONNX Compatibility Problem

The hardest deployment problem was not model accuracy. It was ONNX compatibility with `tract-onnx`, the Rust crate used for WASM-compatible inference.

Early attempts to load the model failed with `Unsqueeze13` analysis errors during graph initialisation. The root cause was an interaction between ONNX opset versioning and tract's static analysis requirements. Starting at opset 13, the `Unsqueeze` operator represents its axes as a second input tensor rather than a static attribute. In BERT's self-attention layers, those axes are produced through a `Shape → Gather → Unsqueeze` chain that tract cannot resolve statically at graph analysis time. The opset-14 export of bert-tiny contained 19 such dynamically-parameterised `Unsqueeze` nodes and would not load.

The fix was to export with ONNX opset 11, which represents axes as static attributes. This reduced the problematic nodes from 19 to 2, both statically analysable, and tract loaded the model successfully.

That change exposed a second issue: newer Transformer implementations in the Hugging Face library default to Scaled Dot-Product Attention (SDPA), which maps to operators requiring higher opsets. Setting `attn_implementation="eager"` on model load forces the classic `BertSelfAttention` path, which exports cleanly to opset 11.

### The Deployment Recipe

The final deployment configuration requires three properties that are non-obvious and interdependent:

1. **ONNX opset 11** — for tract compatibility with static Unsqueeze axes
2. **Eager attention** — to avoid SDPA operators that require higher opsets
3. **Static input shapes** — `int64` tensors of shape `[1, 128]` for both `input_ids` and `attention_mask`

Without that exact combination, the model may export successfully but fail during WASM-side initialisation. The `token_type_ids` input is deliberately omitted from the export to simplify the runtime signature.

At runtime, all assets are embedded with `include_bytes!`. The tokenizer, ONNX model, and threshold configuration are loaded once into `once_cell::sync::Lazy` statics. The service exposes `GET /health` and `POST /classify`, with content-type validation and a 64 KB body cap at the HTTP layer. Input text is tokenised, truncated to 128 tokens using a head-tail strategy (first 63 + last 63 content tokens, preserving `[CLS]` and `[SEP]`), and padded into `Array2<i64>` tensors.

The inference runtime is `tract-onnx 0.21.15`, compiled to `wasm32-wasip1`. The entire service — model weights, tokenizer, inference engine, HTTP handler — compiles into a single WASM binary.

## Latency Optimisation

The initial production deployment ran at 414 ms median elapsed time on Fastly edge hardware — roughly 6x over the target budget. The model accuracy was fine; the runtime performance was not shippable. This section describes the three-experiment optimisation campaign that brought it to 69 ms.

### Experiment 1: Compiler Optimisation Level

The initial build used `opt-level="z"` (size-optimised), which is a common default for WASM targets where binary size matters. Switching to `opt-level=3` (speed-optimised) enabled loop vectorisation and aggressive inlining, reducing BERT forward-pass time from approximately 192 ms to 64 ms — a 3x speedup. The binary grew larger, but the Fastly Compute binary size limit accommodated it comfortably.

### Experiment 2: Wizer Pre-Initialisation

Even with the faster forward pass, every request was paying approximately 163 ms of initialisation cost. The `once_cell::sync::Lazy` statics — tokenizer loading, ONNX graph optimisation and compilation, threshold parsing — were being evaluated on the first access within each request's execution context. Since Fastly Compute has no persistent processes, every request was "first access."

[Wizer](https://github.com/aspect-build/wizer) solves this by running initialisation at build time and snapshotting the resulting WASM heap. The build pipeline calls `wizer.initialize` to force all lazy statics to evaluate, then captures the memory state. Every production request starts with the tokenizer, compiled inference graph, and thresholds already resident in memory. The initialisation gap dropped from 163 ms to zero.

This was the single largest latency improvement in the project — and it had nothing to do with the model.

### Experiment 3: SIMD

Enabling `+simd128` in the WASM target features had no meaningful effect on latency. Investigation revealed that `tract-linalg 0.21.15` provides SIMD-optimised kernels only for `f32` matrix multiplication, not the INT8 quantised path used by the deployed model. The SIMD flag was retained (it does no harm) but it is not contributing to the current performance.

### Results

| Configuration | Elapsed median | Elapsed p95 | Init gap | Service version |
|---|---:|---:|---:|---|
| Baseline (`opt-level="z"`) | 414 ms | 494 ms | ~222 ms | v6 |
| `opt-level=3` | 227 ms | 263 ms | 163 ms | v9 |
| + Wizer pre-init | 70 ms | 84 ms | 0 ms | v10 |
| + `+simd128` | **69 ms** | **85 ms** | **0 ms** | v11 |

The production service (v11) runs at 69 ms median wall-clock elapsed time. Zero `compute_service_vcpu_exceeded_error` events were recorded across 200 benchmark requests, consistent with the service operating below the hard enforcement boundary despite exceeding the 50 ms soft target. Individual requests on fast Fastly PoPs reach below 50 ms (observed minimum: 41 ms). The remaining cost is entirely BERT forward-pass time; initialisation overhead is now zero.

## Evaluation

All metrics were measured on the held-out validation set of 20,027 examples. The operating threshold (`T_block = 0.9403`) was calibrated at 1% false positive rate.

| Metric | Value |
|---|---:|
| PR-AUC | 0.993 |
| Recall @ 1% FPR | 0.956 |
| Precision @ 1% FPR | 0.989 |
| F1 @ 1% FPR | 0.973 |
| Operating threshold (T_block) | 0.9403 |
| Review threshold (T_review @ 2% FPR) | 0.8692 |

The model is effectively length-robust after dataset cleanup: recall at 1% FPR on sequences longer than 128 tokens is 0.956, matching the global figure. This is notable because the model only sees 128 tokens — the head-tail truncation strategy preserves enough signal from long inputs to maintain detection quality.

## Limitations and Future Work

**Validation set used for both checkpoint selection and evaluation.** The 20,027-example evaluation set is a proper stratified split — no examples overlap with the 160,239 training examples. However, the same split was used for early stopping (best checkpoint selected by `recall_at_1pct_fpr`), which means the training loop had indirect access to it through checkpoint selection. A fully independent test set — ideally including out-of-distribution examples from sources not seen during training — would provide stronger evidence of generalisation.

**128-token sequence cap.** The model truncates inputs to 128 tokens. While the head-tail strategy mitigates this for most prompt injection patterns (which tend to concentrate attack tokens at the beginning or end of the input), adversarial inputs specifically designed to hide injection in the middle of long sequences could evade detection.

**Single-task classifier.** The deployed model detects prompt injection only — it does not detect jailbreak attempts, which require a different decision boundary. A jailbreak model was trained and validated but not deployed, because sequential two-model inference on Fastly Compute would land at approximately 136 ms median (estimated as 2× the single-model elapsed time; parallel inference is architecturally unavailable in the current sandbox). The correct path to combined detection is a shared-encoder architecture with multiple classification heads rather than two independent model passes.

**Threshold calibration on validation data.** The operating threshold was calibrated on the same validation set used for evaluation. In production, the FPR may differ depending on the distribution of real traffic, which will not match the class balance or stylistic distribution of the validation set.

**No adversarial robustness evaluation.** The model has not been tested against adaptive attacks — inputs specifically crafted to evade this particular classifier. Token-level perturbations, encoding tricks, or prompt structures designed to exploit the 128-token window could degrade performance below the reported figures.

## Relevance: Why Edge Classifiers Now

The timing for edge-deployed prompt injection detection is driven by three concurrent developments. MCP (Model Context Protocol) now has [over 10,000 active public servers](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) and a 2026 roadmap focused on agent-to-agent communication. Google's A2A (Agent-to-Agent) protocol is [under Linux Foundation governance](https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/), and its specification treats agent cards and inter-agent messages as untrusted input by design. NIST's CAISI division is treating agent security and indirect prompt injection as formal standards work.

As agents increasingly communicate through open protocols, every message in the communication path becomes a potential injection vector. Lightweight classifiers that can screen these messages at wire speed — before they reach the agent runtime — move from being a nice-to-have to being infrastructure. The same way a firewall is not optional once you open a port to the internet, input screening is not optional once you open an agent to messages from other agents.

A 4.3 MB INT8 model running at 69 ms is small enough and fast enough to sit in that path without meaningful impact on end-to-end latency. The edge deployment model — no origin round-trip, no model server, no Python runtime — makes it feasible to add this layer without architectural changes to the agent system it protects.

## Acknowledgements

This work was conducted on Fastly's Compute platform using production edge infrastructure. The base model is `prajjwal1/bert-tiny`; the distillation teacher is `protectai/deberta-v3-small-prompt-injection-v2`. Training data was drawn from public datasets including `jayavibhav/prompt-injection`, `xTRam1/safe-guard-prompt-injection`, and `darkknight25/Prompt_Injection_Benign_Prompt_Dataset`. The Wizer pre-initialisation tool is maintained by the aspect-build team.

---

*Model artefact and inference code: [marklkelly/bert-tiny-injection-detector](https://huggingface.co/marklkelly/bert-tiny-injection-detector) · [github.com/marklkelly/fastly-injection-detector](https://github.com/marklkelly/fastly-injection-detector)*
