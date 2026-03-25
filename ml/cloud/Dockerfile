# Vertex AI Training Container for Prompt Injection Detector
# Supports GPU acceleration with PyTorch 2.0+

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
# Exclude torch — it is provided by the base image with CUDA support.
# Reinstalling from PyPI would replace it with a CPU-only build.
COPY ml/training/requirements.txt ./requirements.txt
RUN grep -v '^torch' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Install Google Cloud Storage support
# protobuf is needed by the transformers tokenizer error-handling codepath for
# SentencePiece-based models (e.g. DeBERTa); sentencepiece for SPM tokenizers.
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-aiplatform \
    gcsfs \
    tensorboard \
    sentencepiece \
    protobuf

# Copy training code into ml/training/ so config_runtime.py's parents[2] resolves to /app/
COPY ml/training/train_cls.py ./ml/training/train_cls.py
COPY ml/training/trainer_ext.py ./ml/training/trainer_ext.py
COPY ml/training/teacher_cache.py ./ml/training/teacher_cache.py
COPY ml/training/config_runtime.py ./ml/training/config_runtime.py
COPY ml/training/slice_metrics.py ./ml/training/slice_metrics.py
COPY ml/training/eval_ood.py ./ml/training/eval_ood.py
COPY ml/training/eval_utils.py ./ml/training/eval_utils.py
COPY ml/training/config ./ml/training/config
COPY ml/cloud/entrypoint.py ./entrypoint.py
COPY ml/cloud/precache_models.py ./precache_models.py

# Copy datasets if including them in container (optional)
# COPY datasets/ ./datasets/

# Pre-cache the backbone and teacher models at build time.
# prajjwal1/bert-tiny is public; protectai/deberta-v3-small-prompt-injection-v2
# is gated and requires a HuggingFace token.  Pass your token as a BuildKit
# secret so it is never embedded in any image layer:
#
#   docker build --secret id=hf_token,src=/path/to/.cache/huggingface/token ...
#
# Without a token only bert-tiny is cached; the teacher model downloads at
# Vertex AI training runtime (containers have internet access).
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) python precache_models.py

# Create directories for outputs
RUN mkdir -p /tmp/outputs /tmp/checkpoints

# Set environment variables for GPU support
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Vertex AI expects the training script to be the entrypoint
ENTRYPOINT ["python", "entrypoint.py"]
