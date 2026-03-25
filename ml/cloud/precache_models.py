"""Pre-cache HuggingFace models at Docker build time.

Run at build time via:
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null) python precache_models.py

prajjwal1/bert-tiny is public and always cached.
protectai/deberta-v3-small-prompt-injection-v2 is gated; requires HF_TOKEN.
If no token is provided it is skipped and will download at training runtime.
"""
import os

from huggingface_hub import snapshot_download

snapshot_download("prajjwal1/bert-tiny")
print("bert-tiny cached.")

token = (os.environ.get("HF_TOKEN") or "").strip() or None
if token:
    snapshot_download(
        "protectai/deberta-v3-small-prompt-injection-v2", token=token
    )
    print("Teacher model cached.")
else:
    print("No HF_TOKEN supplied — teacher model will download at training runtime.")

print("Model cache warmed.")
