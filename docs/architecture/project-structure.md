# Project Structure

This project is organized around five distinct concerns. Each has its own canonical directory.

## The Five Concerns

### 1. Data (`ml/data/`)

Dataset assembly, deduplication, and versioning. The single entry point is `ml/data/build.py`. Dataset recipes live in `ml/data/recipes/` and assembled datasets in `ml/data/versions/`.

### 2. Local Training (`ml/training/`)

Train the BERT-tiny distillation model locally. Shared training configuration lives in `ml/training/config/model.yaml`; this is the single source of truth for all hyperparameters. Entry point: `python ml/training/train_cls.py --config ml/training/config/model.yaml`.

### 3. Google Cloud Training (`ml/cloud/`)

Submit the same training job to Vertex AI. `ml/cloud/submit.py` is the entry point. It reads `ml/cloud/config.yaml` for infrastructure settings and references `ml/training/config/model.yaml` for model hyperparameters so the cloud path does not duplicate the training defaults. `ml/cloud/entrypoint.py` is the container entrypoint that calls the shared trainer.

### 4. Coral TPU Experiment (`ml/experiments/coral/`)

Experimental work to deploy the model to Google Coral Edge TPU hardware. This is not production-ready. See `ml/experiments/coral/README.md`.

### 5. Fastly Service (`service/`)

The Rust/Wasm Compute@Edge service. This is the only production deployment target. Nothing under `ml/` is required at runtime on Fastly; the model is embedded at compile time via `include_bytes!`.

## Data Flow

```text
Raw datasets (HuggingFace)
    -> ml/data/build.py
ml/data/versions/pi_mix_v1/
    -> ml/training/train_cls.py (or ml/cloud/submit.py)
ml/models/bert-tiny-injection-only-20260317/  <- trained PyTorch model
    -> ml/export/export_onnx.py
service/assets/injection_1x128_int8.onnx     <- INT8 quantization included in export
    -> cargo build --features inference (in service/)
Fastly Compute (WASM binary)
```

## Key Design Decisions

**BERT-tiny over larger models:** The Fastly platform enforces a 50 ms CPU budget and 128 MB memory limit. BERT-tiny (2 layers, hidden size 128) fits the edge target at about 18.3 MB after INT8 quantization while keeping inference in the right range for Compute@Edge.

**Knowledge distillation:** The student model (`prajjwal1/bert-tiny`) is trained with soft labels from the teacher `protectai/deberta-v3-small-prompt-injection-v2`. This preserves strong prompt-injection recall without moving the runtime outside the edge deployment envelope.

**Calibrated thresholds:** The model uses a two-threshold policy calibrated from the validation set at specific false-positive-rate operating points. Those thresholds are stored in `service/assets/calibrated_thresholds.json` and loaded by the service at startup.

**`tract-onnx`:** The ONNX runtime for embedded and WASM systems. Inference does not depend on a Python runtime.

**INT8 quantization:** Quantization reduces the model bundle to about 18.3 MB with minimal accuracy loss, which keeps the deployment comfortably within the Fastly memory budget.

## Shared Configuration

`ml/training/config/model.yaml` is the canonical model and training config. Both local and cloud training load it. The cloud config (`ml/cloud/config.yaml`) should only provide infrastructure settings and targeted overrides; defaults belong in `model.yaml`.
