# train_cls.py
# Fine-tune (and optionally distil) a binary SAFE/INJECTION classifier.
# - Optional knowledge distillation from a teacher model (--teacher, --alpha, --temperature)
# - Metrics: PR-AUC, F1 at 1% / 2% FPR (plus classic precision/recall/F1)
# - Calibrates operating thresholds (block/review) on the validation set
# - Emits a tiny model card + thresholds JSON alongside weights
# - Optional FP16 safetensors export for Candle/WASM under output_dir/edge_export
#
# Apple Silicon notes:
# - Uses MPS automatically if available (PyTorch on macOS, e.g., M4).
# - On MPS, keep default FP32 unless you intentionally force bf16/fp16;
#   mixed precision on MPS is evolving and not always faster/safer.

import os
import json
import argparse
import inspect
import platform
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(
    0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
)
from eval_utils import evaluate_at_prior

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from config_runtime import (
    SUPPORTED_BEST_MODEL_METRICS,
    SUPPORTED_DEVICES,
    SUPPORTED_MIXED_PRECISION,
    SUPPORTED_SCHEDULERS,
    SUPPORTED_TRUNCATION_STRATEGIES,
    resolve_config,
    resolve_project_path,
    summarize_resolution,
)
from slice_metrics import (
    compute_slice_report,
    length_bucket_for_token_count,
    summarize_slice_report,
)
from teacher_cache import (
    DEFAULT_CACHE_ROOT,
    build_teacher_cache_manifest,
    cache_paths_for_manifest,
    load_cached_teacher_logits,
    write_cached_teacher_logits,
)
from trainer_ext import DistillationTrainer


# ---------------------------
# CLI
# ---------------------------
def parse_cli_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def add_optional_bool_argument(parser, positive_flags, negative_flags, dest, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        *positive_flags,
        dest=dest,
        nargs="?",
        const=True,
        default=None,
        type=parse_cli_bool,
        help=help_text,
    )
    group.add_argument(
        *negative_flags,
        dest=dest,
        action="store_const",
        const=False,
        help="Disable {0}".format(help_text[0].lower() + help_text[1:]),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train (and optionally distil) a binary prompt-injection classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/training/config/model.yaml",
        help="Path to the training config YAML",
    )

    parser.add_argument(
        "--train-path", "--train_path", dest="train_path", type=str, default=None
    )
    parser.add_argument(
        "--val-path", "--val_path", dest="val_path", type=str, default=None
    )
    parser.add_argument(
        "--test-path", "--test_path", dest="test_path", type=str, default=None
    )
    parser.add_argument(
        "--labels-path", "--labels_path", dest="labels_path", type=str, default=None
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=str, default=None
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Backbone model to fine-tune"
    )
    parser.add_argument(
        "--artifact-name",
        "--artifact_name",
        dest="artifact_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Teacher model name/path for distillation",
    )
    add_optional_bool_argument(
        parser,
        ("--distillation-enabled", "--distillation_enabled"),
        ("--no-distillation-enabled", "--no-distillation_enabled"),
        "distillation_enabled",
        "Enable knowledge distillation when a teacher model is configured",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Distillation weight (0..1)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Distillation temperature T"
    )
    add_optional_bool_argument(
        parser,
        ("--cache-teacher-logits", "--cache_teacher_logits"),
        ("--no-cache-teacher-logits", "--no-cache_teacher_logits"),
        "cache_teacher_logits",
        "Cache raw teacher logits on disk",
    )
    parser.add_argument(
        "--max-length", "--max_length", dest="max_length", type=int, default=None
    )
    parser.add_argument(
        "--truncation-strategy",
        "--truncation_strategy",
        dest="truncation_strategy",
        choices=sorted(SUPPORTED_TRUNCATION_STRATEGIES),
        default=None,
    )
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None
    )
    parser.add_argument(
        "--eval-batch-size",
        "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=None,
    )
    add_optional_bool_argument(
        parser,
        ("--grad-checkpointing", "--grad_checkpointing"),
        ("--no-grad-checkpointing", "--no-grad_checkpointing"),
        "grad_checkpointing",
        "Enable gradient checkpointing",
    )
    parser.add_argument(
        "--label-smoothing-factor",
        "--label_smoothing_factor",
        dest="label_smoothing_factor",
        type=float,
        default=None,
    )
    add_optional_bool_argument(
        parser,
        ("--bf16",),
        ("--no-bf16",),
        "bf16",
        "Use bf16 mixed precision",
    )
    add_optional_bool_argument(
        parser,
        ("--fp16",),
        ("--no-fp16",),
        "fp16",
        "Use fp16 mixed precision",
    )
    parser.add_argument(
        "--mixed-precision",
        "--mixed_precision",
        dest="mixed_precision",
        choices=sorted(SUPPORTED_MIXED_PRECISION),
        default=None,
    )
    add_optional_bool_argument(
        parser,
        (
            "--save-edge-export",
            "--save_edge_export",
            "--save-fp16-safetensors",
            "--save_fp16_safetensors",
        ),
        (
            "--no-save-edge-export",
            "--no-save_edge_export",
            "--no-save-fp16-safetensors",
            "--no-save_fp16_safetensors",
        ),
        "save_edge_export",
        "Export FP16 safetensors under output_dir/edge_export",
    )
    add_optional_bool_argument(
        parser,
        ("--save-hf-checkpoint", "--save_hf_checkpoint"),
        ("--no-save-hf-checkpoint", "--no-save_hf_checkpoint"),
        "save_hf_checkpoint",
        "Save the Hugging Face checkpoint",
    )
    add_optional_bool_argument(
        parser,
        ("--write-model-card", "--write_model_card"),
        ("--no-write-model-card", "--no-write_model_card"),
        "write_model_card",
        "Write model_card.json",
    )
    add_optional_bool_argument(
        parser,
        ("--write-eval-metrics", "--write_eval_metrics"),
        ("--no-write-eval-metrics", "--no-write_eval_metrics"),
        "write_eval_metrics",
        "Write eval_metrics.json",
    )
    add_optional_bool_argument(
        parser,
        ("--quick-test", "--quick_test"),
        ("--no-quick-test", "--no-quick_test"),
        "quick_test",
        "Run quick test mode with a 2-step smoke train",
    )
    parser.add_argument(
        "--max-steps", "--max_steps", dest="max_steps", type=int, default=None
    )
    parser.add_argument(
        "--logging-steps",
        "--logging_steps",
        dest="logging_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save-steps", "--save_steps", dest="save_steps", type=int, default=None
    )
    parser.add_argument(
        "--eval-steps", "--eval_steps", dest="eval_steps", type=int, default=None
    )
    parser.add_argument(
        "--early-stopping-patience",
        "--early_stopping_patience",
        dest="early_stopping_patience",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--best-model-metric",
        "--best_model_metric",
        dest="best_model_metric",
        choices=sorted(SUPPORTED_BEST_MODEL_METRICS),
        default=None,
    )
    parser.add_argument(
        "--lr-scheduler-type",
        "--lr_scheduler_type",
        dest="lr_scheduler_type",
        choices=sorted(SUPPORTED_SCHEDULERS),
        default=None,
    )
    parser.add_argument(
        "--warmup-ratio",
        "--warmup_ratio",
        dest="warmup_ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=sorted(SUPPORTED_DEVICES), default=None)
    parser.add_argument(
        "--dataloader-workers",
        "--dataloader_workers",
        dest="dataloader_workers",
        type=int,
        default=None,
    )
    parser.add_argument("--eval-ood-path", dest="eval_ood_path", type=str, default=None)
    parser.add_argument("--eval-prior", dest="eval_prior", type=float, default=None)
    return parser.parse_args()


def load_label_config(config: Dict[str, Any]) -> Dict[str, str]:
    labels = {str(key): str(value) for key, value in config["model"]["labels"].items()}
    labels_path = resolve_project_path(config["dataset"].get("labels_path"))
    if labels_path is None:
        return labels

    with labels_path.open("r", encoding="utf-8") as handle:
        raw_labels = json.load(handle)

    if isinstance(raw_labels, list):
        loaded = {str(index): str(label) for index, label in enumerate(raw_labels)}
    elif isinstance(raw_labels, dict):
        loaded = {str(key): str(value) for key, value in raw_labels.items()}
    else:
        raise ValueError("Unsupported labels file format: expected list or mapping")

    if set(loaded.keys()) != {"0", "1"}:
        raise ValueError("Labels config must define exactly keys '0' and '1'")
    return loaded


def select_student_token_window(
    token_ids: Sequence[int],
    max_tokens: int,
    truncation_strategy: str,
) -> list[int]:
    if max_tokens <= 0:
        raise ValueError("Student token budget must be positive")

    raw_ids = list(token_ids)
    if len(raw_ids) <= max_tokens:
        return raw_ids

    if truncation_strategy == "head":
        return raw_ids[:max_tokens]
    if truncation_strategy == "tail":
        return raw_ids[-max_tokens:]
    if truncation_strategy == "head_tail":
        head_tokens = max_tokens // 2
        tail_tokens = max_tokens - head_tokens
        return raw_ids[:head_tokens] + raw_ids[-tail_tokens:]

    raise ValueError("Unsupported truncation strategy: {0}".format(truncation_strategy))


def build_student_encoding_from_token_ids(
    tokenizer,
    raw_token_ids: Sequence[int],
    max_length: int,
    truncation_strategy: str,
    padding: str | bool = "max_length",
) -> Dict[str, Any]:
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_budget = max_length - special_tokens
    if content_budget <= 0:
        raise ValueError(
            "model.max_length={0} leaves no room for content tokens".format(max_length)
        )

    raw_token_ids = list(raw_token_ids)
    original_token_length = len(raw_token_ids)
    selected_token_ids = select_student_token_window(
        raw_token_ids,
        max_tokens=content_budget,
        truncation_strategy=truncation_strategy,
    )
    prepared = tokenizer.prepare_for_model(
        selected_token_ids,
        truncation=False,
        max_length=max_length,
        padding=padding,
        return_attention_mask=True,
    )
    prepared["original_token_length"] = original_token_length
    prepared["length_bucket"] = length_bucket_for_token_count(original_token_length)
    return prepared


def build_student_batch(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    truncation_strategy: str,
    padding: str | bool = "max_length",
) -> Dict[str, list[Any]]:
    raw_encodings = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=False,
    )
    encoded_batch: Dict[str, list[Any]] = {}
    for raw_token_ids in raw_encodings["input_ids"]:
        prepared = build_student_encoding_from_token_ids(
            tokenizer=tokenizer,
            raw_token_ids=raw_token_ids,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
            padding=padding,
        )
        for key, value in prepared.items():
            encoded_batch.setdefault(key, []).append(value)
    return encoded_batch


def build_teacher_batch(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    padding: str | bool,
) -> Dict[str, list[Any]]:
    return tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=padding,
    )


def normalize_binary_labels(
    raw_labels: Sequence[Any],
    positive_label: str,
    negative_label: str,
) -> list[int]:
    labels = []
    for raw_label in raw_labels:
        if isinstance(raw_label, str):
            label_upper = raw_label.strip().upper()
            if label_upper in (positive_label, "1"):
                labels.append(1)
            elif label_upper in (negative_label, "0"):
                labels.append(0)
            else:
                raise ValueError("Unknown label value: {0}".format(raw_label))
            continue

        label_int = int(raw_label)
        if label_int not in {0, 1}:
            raise ValueError("Unknown numeric label value: {0}".format(raw_label))
        labels.append(label_int)

    return labels


def extract_logits(predictions: Any) -> np.ndarray:
    if isinstance(predictions, tuple):
        return predictions[0]
    if isinstance(predictions, Mapping):
        return predictions.get("logits", predictions)
    return predictions


def collect_split_metadata(dataset_split, split_name: str) -> list[Dict[str, Any]]:
    sources = dataset_split["source"] if "source" in dataset_split.column_names else []
    labels = dataset_split["labels"]
    original_lengths = dataset_split["original_token_length"]
    length_buckets = dataset_split["length_bucket"]

    if not sources:
        sources = ["unknown"] * len(labels)

    return [
        {
            "source": str(source),
            "labels": int(label),
            "original_token_length": int(original_length),
            "length_bucket": str(length_bucket),
            "split": split_name,
        }
        for source, label, original_length, length_bucket in zip(
            sources,
            labels,
            original_lengths,
            length_buckets,
        )
    ]


def predict_labels_and_probs(
    trainer: Trainer, dataset_split
) -> tuple[np.ndarray, np.ndarray]:
    prediction_output = trainer.predict(dataset_split)
    logits = np.asarray(extract_logits(prediction_output.predictions), dtype=np.float32)
    labels = np.asarray(prediction_output.label_ids).reshape(-1)
    return labels, probs_from_logits(logits)


def resolve_runtime_device(requested_device: str, use_mps: bool) -> torch.device:
    if use_mps:
        return torch.device("mps")
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_pretrained_revision(pretrained_obj) -> str | None:
    init_kwargs = getattr(pretrained_obj, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        for key in ("_commit_hash", "revision"):
            value = init_kwargs.get(key)
            if value:
                return str(value)

    for value in (
        getattr(pretrained_obj, "_commit_hash", None),
        getattr(getattr(pretrained_obj, "config", None), "_commit_hash", None),
        getattr(getattr(pretrained_obj, "config", None), "revision", None),
    ):
        if value:
            return str(value)
    return None


def compute_teacher_logits_for_split(
    *,
    teacher_model,
    teacher_tokenizer,
    dataset_split,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    if len(dataset_split) == 0:
        return torch.empty(
            (0, teacher_model.config.num_labels),
            dtype=torch.float32,
        )

    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    logits_batches = []
    for start in range(0, len(dataset_split), batch_size):
        stop = min(start + batch_size, len(dataset_split))
        texts = dataset_split[start:stop]["text"]
        batch = teacher_tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = teacher_model(**batch)
        logits_batches.append(outputs.logits.detach().cpu().to(dtype=torch.float32))

    return torch.cat(logits_batches, dim=0)


def prepare_teacher_logit_cache(
    *,
    datasets_by_split,
    dataset_paths_by_split: Mapping[str, Path],
    teacher_name: str,
    teacher_model,
    teacher_tokenizer,
    teacher_max_length: int,
    student_truncation_strategy: str,
    batch_size: int,
    requested_device: str,
    use_mps: bool,
) -> dict[str, torch.Tensor]:
    teacher_identifier = getattr(teacher_model, "name_or_path", None) or teacher_name
    tokenizer_identifier = (
        getattr(teacher_tokenizer, "name_or_path", None) or teacher_name
    )
    teacher_model_revision = resolve_pretrained_revision(teacher_model)
    teacher_tokenizer_revision = resolve_pretrained_revision(teacher_tokenizer)
    cache_root = DEFAULT_CACHE_ROOT
    inference_device = resolve_runtime_device(requested_device, use_mps)

    cached_logits: dict[str, torch.Tensor] = {}
    for split_name, dataset_split in datasets_by_split.items():
        dataset_path = dataset_paths_by_split[split_name]
        manifest = build_teacher_cache_manifest(
            split=split_name,
            dataset_path=dataset_path,
            teacher_model=teacher_identifier,
            teacher_model_revision=teacher_model_revision,
            teacher_tokenizer=tokenizer_identifier,
            teacher_tokenizer_revision=teacher_tokenizer_revision,
            teacher_max_length=teacher_max_length,
            student_truncation_strategy=student_truncation_strategy,
            example_count=len(dataset_split),
        )
        cache_paths = cache_paths_for_manifest(cache_root, manifest)
        logits = load_cached_teacher_logits(cache_root, manifest)
        if logits is None:
            logits = compute_teacher_logits_for_split(
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                dataset_split=dataset_split,
                batch_size=batch_size,
                max_length=teacher_max_length,
                device=inference_device,
            )
            write_cached_teacher_logits(cache_root, manifest, logits)
            print(
                "Teacher logits cache write [{0}] -> {1}".format(
                    split_name,
                    cache_paths.cache_dir,
                )
            )
        else:
            print(
                "Teacher logits cache hit [{0}] -> {1}".format(
                    split_name,
                    cache_paths.cache_dir,
                )
            )
        if logits.shape[0] != len(dataset_split):
            raise ValueError(
                "Teacher logits cache row count mismatch for split '{0}': expected {1}, got {2}".format(
                    split_name,
                    len(dataset_split),
                    logits.shape[0],
                )
            )
        cached_logits[split_name] = logits

    return cached_logits


# ---------------------------
# TensorBoard → GCS sync callback (Vertex AI)
# ---------------------------
class _TBGCSSyncCallback(TrainerCallback):
    """Syncs local TensorBoard event files to GCS during Vertex AI training.

    The Vertex AI managed TensorBoard sidecar streams events from
    AIP_TENSORBOARD_LOG_DIR while the job is running. Uploading only at the
    end of training risks missing the sidecar's final sync window. This
    callback uploads incrementally so data appears in the viewer during training.
    """

    def __init__(self, local_dir: str, gcs_dir: str, sync_every_n_logs: int = 5):
        self._local_dir = local_dir
        self._gcs_dir = gcs_dir
        self._sync_every = sync_every_n_logs
        self._log_count = 0
        self._client = None
        parts = gcs_dir.replace("gs://", "").split("/", 1)
        self._bucket_name = parts[0]
        self._prefix = parts[1] if len(parts) > 1 else ""

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client()
        return self._client

    def _sync(self):
        try:
            bucket = self._get_client().bucket(self._bucket_name)
            for root_dir, _, files in os.walk(self._local_dir):
                for fname in files:
                    local_file = os.path.join(root_dir, fname)
                    rel = os.path.relpath(local_file, self._local_dir)
                    blob_name = os.path.join(self._prefix, rel).replace("\\", "/")
                    bucket.blob(blob_name).upload_from_filename(local_file)
        except Exception as exc:
            print(f"[TBGCSSyncCallback] Warning: GCS sync failed: {exc}")

    def on_log(self, args, state, control, **kwargs):
        self._log_count += 1
        if self._log_count % self._sync_every == 0:
            self._sync()

    def on_train_end(self, args, state, control, **kwargs):
        self._sync()


# ---------------------------
# Distillation wrapper
# ---------------------------
class DistilledStudent(nn.Module):
    """
    Wraps a student classification model and (optionally) a frozen teacher.
    Expects batch keys:
      - input_ids, attention_mask (student)
      - labels
      - teacher_input_ids, teacher_attention_mask (if teacher provided)
    """

    def __init__(
        self,
        student_model,
        teacher_model=None,
        alpha=0.5,
        temperature=2.0,
        label_smoothing_factor=0.0,
    ):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = alpha
        self.T = temperature
        self.label_smoothing_factor = label_smoothing_factor
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def _supervised_loss(self, logits, labels):
        if labels is None:
            return None
        logits = logits.to(torch.float32)
        if self.label_smoothing_factor <= 0:
            return F.cross_entropy(logits, labels)

        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)
        return (
            (1 - self.label_smoothing_factor) * nll
            + self.label_smoothing_factor * smooth
        ).mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,  # present for BERT/DistilBERT tokenizers
        labels=None,
        teacher_input_ids=None,  # added by our prep() when --teacher is set
        teacher_attention_mask=None,
        teacher_logits=None,
        **kwargs,  # keep a catch-all for HF quirks (e.g., position_ids)
    ):
        # Keep the supervised loss explicit so label smoothing cannot replace distillation.
        student_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            student_inputs["token_type_ids"] = token_type_ids

        stu_out = self.student(**student_inputs)
        logits = stu_out.logits
        loss = self._supervised_loss(logits, labels)

        # ---- Optional distillation term (teacher is frozen) ----
        if labels is not None and (
            teacher_logits is not None
            or (self.teacher is not None and teacher_input_ids is not None)
        ):
            if teacher_logits is None and teacher_input_ids is not None:
                with torch.no_grad():
                    tea_out = self.teacher(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                    )
                    tea_logits = tea_out.logits
            else:
                tea_logits = teacher_logits

            student_logits = logits.to(torch.float32)
            teacher_logits_fp32 = tea_logits.to(torch.float32)
            s = student_logits / self.T
            t = teacher_logits_fp32 / self.T
            kl = F.kl_div(
                F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean"
            ) * (self.T**2)

            loss = (1 - self.alpha) * loss + self.alpha * kl

        # Return in format expected by HF Trainer
        # We need to return an object that has .loss and .logits attributes
        from transformers.modeling_outputs import SequenceClassifierOutput

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------
# Metrics helpers
# ---------------------------
def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    Return p(class=1) as a 1-D vector of length N.
    Accepts:
      - [N, C] logits (normal case; we softmax over C then take column 1)
      - [N, 1] scores (we squeeze)
      - [N] probabilities/scores (we return as-is)
      - [N, seq_len, C] (rare): we take the first token then proceed
    """
    logits = np.asarray(logits)

    # Token-level logits -> take first token
    if logits.ndim == 3:
        logits = logits[:, 0, :]

    # Already 1-D
    if logits.ndim == 1:
        return logits

    # [N,1] -> squeeze
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]

    # General case: softmax over class dim, then take col 1
    z = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    sm = e / e.sum(axis=1, keepdims=True)  # softmax matrix [N, C]
    if sm.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 columns for binary classification, got {sm.shape}"
        )
    return sm[:, 1]  # shape (N,)


def threshold_at_fpr(
    y_true: np.ndarray, p: np.ndarray, target_fpr: float = 0.01
) -> float:
    """Find a probability threshold that yields FPR <= target_fpr (closest from above)."""
    fpr, tpr, thr = roc_curve(y_true, p)
    # thr aligns with fpr/tpr; pick the most conservative threshold within the FPR target
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 1.0  # nothing hits the FPR target; effectively block nothing
    t = float(thr[ok[-1]])
    if not np.isfinite(t):
        # roc_curve can emit inf as a sentinel; fall back to high threshold
        return 1.0
    return t


def f1_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1, precision, recall, tp, fp, fn


def compute_metrics_builder(store_last_eval: Dict):
    """
    Returns a compute_metrics function capturing a dict to store last eval curves & probs.
    Logs:
      - classic F1/precision/recall (argmax)
      - PR-AUC
      - F1@1% FPR and F1@2% FPR with their thresholds
    """

    def compute_metrics(eval_pred):
        # Handle different output formats from the model
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Coerce to arrays and flatten labels
        logits = np.asarray(extract_logits(predictions), dtype=np.float32)
        labels = np.asarray(labels).reshape(-1)

        # If token-level logits slipped through, use first token
        if logits.ndim == 3:
            logits = logits[:, 0, :]

        # Expect at least 2 classes
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError(f"Expected logits shape [N, >=2], got {logits.shape}")

        preds = np.argmax(logits, axis=-1)

        # Classic metrics (argmax)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Probabilities for PR-AUC and threshold-based metrics
        p = probs_from_logits(logits)  # now guaranteed (N,)
        p = np.asarray(p).reshape(-1)  # final belt-and-braces

        # Handle degenerate validation slices (only one class present)
        if labels.min() == labels.max():
            # Only one class present; PR-AUC is undefined. Use NaN and conservative thresholds.
            pr_auc = float("nan")
            thr_1 = thr_2 = 1.0
            f1_1 = f1_2 = prec_1 = prec_2 = rec_1 = rec_2 = 0.0
        else:
            pr_auc = float(average_precision_score(labels, p))
            thr_1 = threshold_at_fpr(labels, p, 0.01)
            f1_1, prec_1, rec_1, *_ = f1_at_threshold(labels, p, thr_1)
            thr_2 = threshold_at_fpr(labels, p, 0.02)
            f1_2, prec_2, rec_2, *_ = f1_at_threshold(labels, p, thr_2)

        # Stash for later (model card / thresholds)
        store_last_eval.clear()
        store_last_eval.update(
            dict(
                labels=labels,
                probs=p,
                pr_auc=pr_auc,
                thr_1_fpr=thr_1,
                f1_at_1_fpr=f1_1,
                precision_at_1_fpr=prec_1,
                recall_at_1_fpr=rec_1,
                thr_2_fpr=thr_2,
                f1_at_2_fpr=f1_2,
                precision_at_2_fpr=prec_2,
                recall_at_2_fpr=rec_2,
            )
        )
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "pr_auc": pr_auc,
            "f1_at_1pct_fpr": f1_1,
            "recall_at_1pct_fpr": rec_1,
            "f1_at_2pct_fpr": f1_2,
            "threshold_at_1pct_fpr": thr_1,
            "threshold_at_2pct_fpr": thr_2,
        }

    return compute_metrics


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    resolution = resolve_config(args.config, args)
    config = resolution.resolved_config

    # Disable tokenizers parallelism to avoid forking issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(summarize_resolution(resolution))

    dataset_config = config["dataset"]
    evaluation_config = config["evaluation"]
    model_config = config["model"]
    training_config = config["training"]
    runtime_config = config["runtime"]
    output_config = config["outputs"]

    train_path = resolve_project_path(dataset_config["train_path"])
    val_path = resolve_project_path(dataset_config["val_path"])
    test_path = resolve_project_path(dataset_config["test_path"])
    output_dir = resolution.resolved_output_dir
    eval_ood_path = resolve_project_path(evaluation_config["ood_path"])

    requested_device = runtime_config["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda requested, but CUDA is not available")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("runtime.device=mps requested, but MPS is not available")

    # Apple Silicon (MPS) hint – Trainer/Accelerate will pick MPS if available.
    use_mps = torch.backends.mps.is_available() and requested_device in {"auto", "mps"}
    if use_mps:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print("🔋 Using Apple Silicon (MPS) acceleration.")
    elif requested_device == "cpu":
        print("🧠 Using CPU runtime.")
    elif requested_device == "cuda" and torch.cuda.is_available():
        print("🚀 Using CUDA acceleration.")

    labels_config = load_label_config(config)
    negative_label = labels_config["0"].strip().upper()
    positive_label = labels_config["1"].strip().upper()

    # Tokenisers
    tok = AutoTokenizer.from_pretrained(model_config["backbone"], use_fast=True)
    # Add padding token if missing (some models like MiniLM might not have it)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"

    teacher_name = model_config["teacher_model"]
    distillation_enabled = (
        bool(teacher_name) and model_config["distillation"]["enabled"]
    )
    teacher_max_length = model_config["distillation"]["teacher_max_length"]
    cache_teacher_logits = (
        distillation_enabled and model_config["distillation"]["cache_teacher_logits"]
    )
    teacher_tok = None
    if distillation_enabled:
        teacher_tok = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)
        if teacher_tok.pad_token is None:
            teacher_tok.pad_token = (
                teacher_tok.eos_token if teacher_tok.eos_token else "[PAD]"
            )

    # Load datasets
    data_files = {"train": str(train_path), "validation": str(val_path)}
    if test_path is not None:
        data_files["test"] = str(test_path)
    dataset_paths_by_split = {
        "train": train_path,
        "validation": val_path,
    }
    if test_path is not None:
        dataset_paths_by_split["test"] = test_path
    ds = load_dataset(
        "json",
        data_files=data_files,
    )

    # Quick test mode: use small subset
    if runtime_config["quick_test"]:
        print("🚀 QUICK TEST MODE: Using 100 train, 50 val samples, 2 training steps")
        ds["train"] = ds["train"].select(range(min(100, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(
            range(min(50, len(ds["validation"])))
        )
        if "test" in ds:
            ds["test"] = ds["test"].select(range(min(50, len(ds["test"]))))

    teacher_logits_by_split: dict[str, torch.Tensor] = {}
    if cache_teacher_logits:
        teacher_for_cache = AutoModelForSequenceClassification.from_pretrained(
            teacher_name,
            num_labels=model_config["num_labels"],
        )
        teacher_for_cache.eval()
        teacher_logits_by_split = prepare_teacher_logit_cache(
            datasets_by_split=ds,
            dataset_paths_by_split=dataset_paths_by_split,
            teacher_name=teacher_name,
            teacher_model=teacher_for_cache,
            teacher_tokenizer=teacher_tok,
            teacher_max_length=teacher_max_length,
            student_truncation_strategy=model_config["truncation_strategy"],
            batch_size=training_config["eval_batch_size"],
            requested_device=requested_device,
            use_mps=use_mps,
        )
        del teacher_for_cache

    # Map -> tokenise; also tokenise for teacher if provided
    def prep(batch, indices, split_name):
        texts = batch["text"]
        out = build_student_batch(
            tokenizer=tok,
            texts=texts,
            max_length=model_config["max_length"],
            truncation_strategy=model_config["truncation_strategy"],
            padding=False,
        )
        raw = batch.get("label", None)
        if raw is None:
            raise ValueError("Missing 'label' field in dataset.")
        labels = normalize_binary_labels(raw, positive_label, negative_label)
        out["labels"] = labels
        out["source"] = [
            "unknown" if source is None else str(source)
            for source in batch.get("source", ["unknown"] * len(texts))
        ]
        out["split"] = [split_name] * len(texts)

        if distillation_enabled:
            if cache_teacher_logits:
                out["teacher_logits"] = teacher_logits_by_split[split_name][
                    indices
                ].tolist()
            else:
                t = build_teacher_batch(
                    teacher_tok,
                    texts,
                    max_length=teacher_max_length,
                    padding=False,
                )
                out["teacher_input_ids"] = t["input_ids"]
                out["teacher_attention_mask"] = t["attention_mask"]
                if "token_type_ids" in t:
                    out["teacher_token_type_ids"] = t["token_type_ids"]
        return out

    for split_name in list(ds.keys()):
        ds[split_name] = ds[split_name].map(
            lambda batch, indices, split_name=split_name: prep(
                batch,
                indices,
                split_name,
            ),
            batched=True,
            with_indices=True,
            remove_columns=ds[split_name].column_names,
        )

    slice_metadata_by_split = {}
    for split_name in ("validation", "test"):
        if split_name in ds:
            slice_metadata_by_split[split_name] = collect_split_metadata(
                ds[split_name], split_name
            )

    # Models
    student = AutoModelForSequenceClassification.from_pretrained(
        model_config["backbone"],
        num_labels=model_config["num_labels"],
    )
    if training_config["grad_checkpointing"] and hasattr(
        student, "gradient_checkpointing_enable"
    ):
        student.gradient_checkpointing_enable()

    teacher = None
    if distillation_enabled and not cache_teacher_logits:
        teacher = AutoModelForSequenceClassification.from_pretrained(
            teacher_name,
            num_labels=model_config["num_labels"],
        )
        teacher.eval()

    use_wrapper = distillation_enabled or training_config["label_smoothing_factor"] > 0
    model = (
        DistilledStudent(
            student,
            teacher,
            alpha=model_config["distillation"]["alpha"],
            temperature=model_config["distillation"]["temperature"],
            label_smoothing_factor=training_config["label_smoothing_factor"],
        )
        if use_wrapper
        else student
    )

    tensorboard_gcs_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
    local_tensorboard_dir = str(output_dir / "runs")

    # TrainingArguments (robust to version differences)
    base_kwargs = dict(
        output_dir=str(output_dir),
        logging_dir=local_tensorboard_dir,
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["eval_batch_size"],
        num_train_epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        logging_steps=training_config["logging_steps"],
        save_total_limit=training_config["save_total_limit"],
        report_to=["none"] if runtime_config["quick_test"] else ["tensorboard"],
        dataloader_num_workers=0
        if runtime_config["quick_test"]
        else runtime_config["dataloader_workers"],
        optim="adamw_torch",
        gradient_checkpointing=training_config["grad_checkpointing"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        seed=training_config["seed"],
        label_smoothing_factor=training_config["label_smoothing_factor"],
        dataloader_pin_memory=torch.cuda.is_available() and requested_device != "cpu",
        # DistillationTrainer's custom collator reads raw `text` columns directly,
        # so we must keep all columns. The plain Trainer uses the default HF
        # DataCollator which tries to tensorize every column — string columns like
        # `source` must be removed or it raises ValueError.
        remove_unused_columns=not use_wrapper,
    )

    # Add max_steps if specified
    if training_config["max_steps"] is not None:
        base_kwargs["max_steps"] = training_config["max_steps"]

    if training_config["save_strategy"] == "steps":
        base_kwargs["save_steps"] = training_config["save_steps"]
        base_kwargs["eval_steps"] = training_config["eval_steps"]

    params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    # Filter base_kwargs to only include supported parameters
    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in params}

    # Mixed precision flags: default to auto unless explicitly resolved to a concrete dtype.
    if runtime_config["mixed_precision"] == "bf16":
        if "bf16" in params:
            filtered_kwargs["bf16"] = True
        if "fp16" in params:
            filtered_kwargs["fp16"] = False
    elif runtime_config["mixed_precision"] == "fp16":
        if "fp16" in params:
            filtered_kwargs["fp16"] = True
        if "bf16" in params:
            filtered_kwargs["bf16"] = False
    elif runtime_config["mixed_precision"] == "fp32":
        if "bf16" in params:
            filtered_kwargs["bf16"] = False
        if "fp16" in params:
            filtered_kwargs["fp16"] = False

    if requested_device == "cpu":
        if "use_cpu" in params:
            filtered_kwargs["use_cpu"] = True
        elif "no_cuda" in params:
            filtered_kwargs["no_cuda"] = True
    elif requested_device == "mps" and "use_mps_device" in params:
        filtered_kwargs["use_mps_device"] = True

    if "evaluation_strategy" in params:
        filtered_kwargs["evaluation_strategy"] = training_config["evaluation_strategy"]
    elif "eval_strategy" in params:
        filtered_kwargs["eval_strategy"] = training_config["evaluation_strategy"]
    if "save_strategy" in params:
        filtered_kwargs["save_strategy"] = training_config["save_strategy"]
    if "load_best_model_at_end" in params:
        filtered_kwargs["load_best_model_at_end"] = training_config[
            "load_best_model_at_end"
        ]
    if "metric_for_best_model" in params:
        filtered_kwargs["metric_for_best_model"] = training_config[
            "metric_for_best_model"
        ]
    if "greater_is_better" in params:
        filtered_kwargs["greater_is_better"] = training_config["greater_is_better"]
    if "lr_scheduler_type" in params:
        filtered_kwargs["lr_scheduler_type"] = training_config["lr_scheduler_type"]
    if training_config["warmup_steps"] is not None and "warmup_steps" in params:
        filtered_kwargs["warmup_steps"] = training_config["warmup_steps"]
    elif training_config["warmup_ratio"] is not None and "warmup_ratio" in params:
        filtered_kwargs["warmup_ratio"] = training_config["warmup_ratio"]

    training_args = TrainingArguments(**filtered_kwargs)

    # Metrics
    last_eval_store: Dict = {}
    compute_metrics = compute_metrics_builder(last_eval_store)

    # Callbacks
    callbacks = []
    if training_config["enable_early_stopping"]:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"],
                early_stopping_threshold=0.001,
            )
        )
    if tensorboard_gcs_dir and tensorboard_gcs_dir.startswith("gs://"):
        callbacks.append(_TBGCSSyncCallback(local_tensorboard_dir, tensorboard_gcs_dir))

    if use_wrapper:
        trainer = DistillationTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            processing_class=tok,
            student_tokenizer=tok,
            teacher_tokenizer=None if cache_teacher_logits else teacher_tok,
            student_max_length=model_config["max_length"],
            teacher_max_length=teacher_max_length,
            cache_teacher_logits=cache_teacher_logits,
            compute_metrics=compute_metrics,
            callbacks=callbacks if callbacks else None,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            processing_class=tok,
            compute_metrics=compute_metrics,
            callbacks=callbacks if callbacks else None,
        )

    # Train
    os.makedirs(output_dir, exist_ok=True)
    trainer.train()

    # Final evaluation (with probs preserved for calibration)
    trainer.evaluate()

    # Calibrate thresholds on validation set
    labels = last_eval_store["labels"]
    probs = last_eval_store["probs"]
    thr_1 = float(last_eval_store["thr_1_fpr"])
    thr_2 = float(last_eval_store["thr_2_fpr"])

    calibrated = {
        "T_block_at_1pct_FPR": thr_1,
        "T_review_lower_at_2pct_FPR": thr_2,
        "policy": "p < T_review_lower -> Allow; T_review_lower <= p < T_block -> Review; p >= T_block -> Block.",
    }
    with open(os.path.join(output_dir, "calibrated_thresholds.json"), "w") as f:
        json.dump(calibrated, f, indent=2)

    # Helper: balanced metrics at a threshold
    def _balanced_at_thr(y, p_scores, thr):
        y = np.asarray(y)
        p_scores = np.asarray(p_scores)
        pred = (p_scores >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        n_pos = tp + fn
        n_neg = fp + tn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "threshold": float(thr),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "fpr": float(fp / n_neg) if n_neg > 0 else 0.0,
            "tpr": float(tp / n_pos) if n_pos > 0 else 0.0,
        }

    validation_slice_report = compute_slice_report(
        labels=labels,
        probs=probs,
        metadata=slice_metadata_by_split["validation"],
        threshold=thr_1,
        threshold_source="validation_overall",
    )
    test_slice_report = None
    if "test" in ds:
        test_labels, test_probs = predict_labels_and_probs(trainer, ds["test"])
        test_slice_report = compute_slice_report(
            labels=test_labels,
            probs=test_probs,
            metadata=slice_metadata_by_split["test"],
            threshold=thr_1,
            threshold_source="validation_overall",
        )

    # Restructured eval_metrics.json
    pr_auc_val = (
        float(last_eval_store["pr_auc"])
        if not (
            isinstance(last_eval_store["pr_auc"], float)
            and np.isnan(last_eval_store["pr_auc"])
        )
        else None
    )
    in_domain_balanced = {
        "threshold_source": "validation",
        "auc_pr": pr_auc_val,
        "auc_roc": float(
            roc_auc_score(last_eval_store["labels"], last_eval_store["probs"])
        ),
        "at_block_threshold": _balanced_at_thr(labels, probs, thr_1),
        "at_review_threshold": _balanced_at_thr(labels, probs, thr_2),
    }
    in_domain_estimated = {
        "prior": evaluation_config["prior"],
        "threshold_source": "validation",
        "at_block_threshold": evaluate_at_prior(
            labels.tolist(),
            probs.tolist(),
            thr_1,
            prior=evaluation_config["prior"],
        ),
        "at_review_threshold": evaluate_at_prior(
            labels.tolist(),
            probs.tolist(),
            thr_2,
            prior=evaluation_config["prior"],
        ),
    }
    structured_eval_metrics = {
        "in_domain_balanced": in_domain_balanced,
        "in_domain_estimated_at_prior": in_domain_estimated,
        "validation_slices": validation_slice_report,
    }
    if test_slice_report is not None:
        structured_eval_metrics["test_slices"] = test_slice_report

    # OOD evaluation (if requested)
    ood_metrics_for_card = None
    if eval_ood_path is not None:
        print(f"\n🔍 Running OOD evaluation on {eval_ood_path} ...")
        ood_examples = []
        with open(eval_ood_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                raw = str(obj.get("label", "")).strip()
                if raw in ("1", positive_label):
                    label_int = 1
                elif raw in ("0", negative_label):
                    label_int = 0
                else:
                    raise ValueError(
                        f"Unknown label {raw!r} in OOD file at line {line_number}. Expected 0/1/SAFE/INJECTION."
                    )
                ood_examples.append((obj["text"], label_int))

        ood_texts = [e[0] for e in ood_examples]
        ood_labels = [e[1] for e in ood_examples]

        # Run inference with the trained student model
        student.eval()
        inf_device = resolve_runtime_device(requested_device, use_mps)
        student_inf = student.to(inf_device)

        ood_probs = []
        eval_batch_size = training_config["eval_batch_size"]
        max_len = model_config["max_length"]
        for i in range(0, len(ood_texts), eval_batch_size):
            batch = ood_texts[i : i + eval_batch_size]
            enc = tok(
                batch,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            enc = {k: v.to(inf_device) for k, v in enc.items()}
            with torch.no_grad():
                out = student_inf(**enc)
            logits = out.logits.float().cpu().numpy()
            z = logits - logits.max(axis=1, keepdims=True)
            e_exp = np.exp(z)
            sm = e_exp / e_exp.sum(axis=1, keepdims=True)
            ood_probs.extend(sm[:, 1].tolist())

        ood_labels_arr = np.asarray(ood_labels)
        ood_probs_arr = np.asarray(ood_probs)
        ood_auc_pr = float(average_precision_score(ood_labels_arr, ood_probs_arr))
        ood_auc_roc = float(roc_auc_score(ood_labels_arr, ood_probs_arr))

        ood_block_balanced = _balanced_at_thr(ood_labels, ood_probs, thr_1)
        ood_review_balanced = _balanced_at_thr(ood_labels, ood_probs, thr_2)
        ood_block_adjusted = evaluate_at_prior(
            ood_labels,
            ood_probs,
            thr_1,
            prior=evaluation_config["prior"],
        )
        ood_review_adjusted = evaluate_at_prior(
            ood_labels,
            ood_probs,
            thr_2,
            prior=evaluation_config["prior"],
        )

        structured_eval_metrics["ood_balanced"] = {
            "eval_path": str(eval_ood_path),
            "threshold_source": "calibrated_thresholds.json",
            "auc_pr": ood_auc_pr,
            "auc_roc": ood_auc_roc,
            "at_block_threshold": ood_block_balanced,
            "at_review_threshold": ood_review_balanced,
        }
        structured_eval_metrics["ood_estimated_at_prior"] = {
            "prior": evaluation_config["prior"],
            "threshold_source": "calibrated_thresholds.json",
            "at_block_threshold": ood_block_adjusted,
            "at_review_threshold": ood_review_adjusted,
        }

        ood_metrics_for_card = {
            "eval_path": str(eval_ood_path),
            "n_examples": len(ood_examples),
            "auc_pr": ood_auc_pr,
            "auc_roc": ood_auc_roc,
            "f1_at_block_thr": ood_block_balanced["f1"],
            "f1_at_review_thr": ood_review_balanced["f1"],
            "estimated_ppv_at_block_thr": ood_block_adjusted["estimated_ppv"],
            "estimated_f1_at_block_thr": ood_block_adjusted["estimated_f1"],
        }
        print(f"   OOD AUC-PR: {ood_auc_pr:.4f}  AUC-ROC: {ood_auc_roc:.4f}")
        print(
            f"   F1@block: {ood_block_balanced['f1']:.4f}   F1@review: {ood_review_balanced['f1']:.4f}"
        )

    if output_config["write_eval_metrics"]:
        with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
            json.dump(structured_eval_metrics, f, indent=2)

    # (A) Always save a standard HF checkpoint of the *student* (not the wrapper)
    if output_config["save_hf_checkpoint"]:
        student.save_pretrained(output_dir, safe_serialization=True)
        tok.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "labels.json"), "w") as f:
            json.dump([labels_config["0"], labels_config["1"]], f)

    # (B) Optional: export FP16 safetensors for Candle/WASM under edge_export/
    if output_config["save_edge_export"]:
        export_dir = os.path.join(output_dir, "edge_export")
        os.makedirs(export_dir, exist_ok=True)

        # Cast on CPU to float16 for portability
        student_cpu_fp16 = student.to("cpu").to(dtype=torch.float16)
        student_cpu_fp16.save_pretrained(export_dir, safe_serialization=True)
        tok.save_pretrained(export_dir)

        # Remove any accidental .bin files to keep export lean
        for name in os.listdir(export_dir):
            if name.endswith(".bin"):
                os.remove(os.path.join(export_dir, name))

        # Sanity check
        if not any(n.endswith(".safetensors") for n in os.listdir(export_dir)):
            raise RuntimeError(
                "Expected a .safetensors file in edge_export, but none was written."
            )

    # Compute size of saved core artefacts (rough estimate)
    size_bytes = 0
    for name in os.listdir(output_dir):
        if name.endswith((".safetensors", ".bin", ".pt", ".json")) and name.startswith(
            ("pytorch_model", "model", "config", "tokenizer")
        ):
            size_bytes += os.path.getsize(os.path.join(output_dir, name))

    resolved_pr_auc = (
        None
        if isinstance(last_eval_store["pr_auc"], float)
        and np.isnan(last_eval_store["pr_auc"])
        else float(last_eval_store["pr_auc"])
    )
    resolved_dtype = (
        "bf16"
        if getattr(training_args, "bf16", False)
        else ("fp16" if getattr(training_args, "fp16", False) else "fp32")
    )

    # Tiny model card
    card = {
        "purpose": "Edge prompt-injection binary classifier (SAFE vs INJECTION).",
        "base_model": model_config["backbone"],
        "teacher": teacher_name or None,
        "distillation": distillation_enabled,
        "alpha": model_config["distillation"]["alpha"]
        if distillation_enabled
        else None,
        "temperature": model_config["distillation"]["temperature"]
        if distillation_enabled
        else None,
        "task": "sequence_classification_binary",
        "labels": labels_config,
        "training": {
            "epochs": training_config["epochs"],
            "max_steps": training_config["max_steps"],
            "lr": training_config["learning_rate"],
            "weight_decay": training_config["weight_decay"],
            "batch_size": training_config["batch_size"],
            "eval_batch_size": training_config["eval_batch_size"],
            "gradient_accumulation_steps": training_config[
                "gradient_accumulation_steps"
            ],
            "max_length": model_config["max_length"],
            "truncation_strategy": model_config["truncation_strategy"],
            "teacher_max_length": teacher_max_length if distillation_enabled else None,
            "cache_teacher_logits": cache_teacher_logits,
            "grad_checkpointing": training_config["grad_checkpointing"],
            "label_smoothing_factor": training_config["label_smoothing_factor"],
            "evaluation_strategy": training_config["evaluation_strategy"],
            "save_strategy": training_config["save_strategy"],
            "eval_steps": training_config["eval_steps"],
            "save_steps": training_config["save_steps"],
            "lr_scheduler_type": training_config["lr_scheduler_type"],
            "warmup_ratio": training_config["warmup_ratio"],
            "warmup_steps": training_config["warmup_steps"],
            "early_stopping_patience": training_config["early_stopping_patience"],
            "metric_for_best_model": training_config["metric_for_best_model"],
            "greater_is_better": training_config["greater_is_better"],
            "device": str(trainer.args.device),
            "dtype": resolved_dtype,
            "hardware": platform.platform(),
        },
        "dataset": {
            "version": dataset_config["version"],
            "train_path": dataset_config["train_path"],
            "val_path": dataset_config["val_path"],
            "test_path": dataset_config["test_path"],
            "train_examples": len(ds["train"]),
            "validation_examples": len(ds["validation"]),
            "test_examples": len(ds["test"]) if "test" in ds else None,
            "pos_rate_validation": float(np.mean(labels)),
        },
        "metrics_validation": {
            "pr_auc": resolved_pr_auc,
            "f1_at_1pct_fpr": float(last_eval_store["f1_at_1_fpr"]),
            "precision_at_1pct_fpr": float(last_eval_store["precision_at_1_fpr"]),
            "recall_at_1pct_fpr": float(last_eval_store["recall_at_1_fpr"]),
            "threshold_at_1pct_fpr": thr_1,
            "f1_at_2pct_fpr": float(last_eval_store["f1_at_2_fpr"]),
            "precision_at_2pct_fpr": float(last_eval_store["precision_at_2_fpr"]),
            "recall_at_2pct_fpr": float(last_eval_store["recall_at_2_fpr"]),
            "threshold_at_2pct_fpr": thr_2,
        },
        "slice_metrics_summary": {
            "validation": summarize_slice_report(validation_slice_report),
        },
        "calibrated_thresholds": calibrated,
        "artifacts": {
            "output_dir": str(output_dir),
            "approx_total_model_files_size_bytes": int(size_bytes),
            "edge_export": os.path.join(output_dir, "edge_export")
            if output_config["save_edge_export"]
            else None,
        },
        "resolved_config": config,
        "decision_policy": "Two-threshold: allow / review / block using calibrated thresholds.",
        "limitations": [
            "Not a general content moderation system.",
            "Performance depends on dataset coverage (e.g., obfuscations, multilingual).",
        ],
    }
    if test_slice_report is not None:
        card["slice_metrics_summary"]["test"] = summarize_slice_report(
            test_slice_report
        )
    if ood_metrics_for_card is not None:
        card["ood_metrics"] = ood_metrics_for_card
    if output_config["write_model_card"]:
        with open(os.path.join(output_dir, "model_card.json"), "w") as f:
            json.dump(card, f, indent=2)

    print("✅ Training complete.")
    pr_auc_display = (
        "nan"
        if card["metrics_validation"]["pr_auc"] is None
        else "{0:.4f}".format(card["metrics_validation"]["pr_auc"])
    )
    print(f"   PR-AUC: {pr_auc_display}")
    print(
        f"   F1@1% FPR: {card['metrics_validation']['f1_at_1pct_fpr']:.4f} @ thr={thr_1:.4f}"
    )
    print(
        f"   F1@2% FPR: {card['metrics_validation']['f1_at_2pct_fpr']:.4f} @ thr={thr_2:.4f}"
    )
    if output_config["save_edge_export"]:
        print(f"   Edge export ready: {os.path.join(output_dir, 'edge_export')}")

    # Log final hyperparameters and metrics to Vertex AI Experiments when running on Vertex AI.
    # RUN_NAME is injected by submit.py; if absent we are running locally and skip this.
    _run_name = os.environ.get("RUN_NAME")
    if _run_name and not runtime_config.get("quick_test"):
        try:
            from google.cloud import aiplatform
            _experiment = os.environ.get("EXPERIMENT_NAME", "injection-detector")
            aiplatform.init(experiment=_experiment)
            with aiplatform.start_run(_run_name, resume=True):
                aiplatform.log_params({
                    "backbone": model_config["backbone"],
                    "teacher_model": model_config.get("teacher_model") or "none",
                    "distillation_enabled": bool(distillation_enabled),
                    "epochs": training_config["epochs"],
                    "batch_size": training_config["batch_size"],
                    "learning_rate": training_config["learning_rate"],
                    "alpha": model_config.get("distillation", {}).get("alpha"),
                    "temperature": model_config.get("distillation", {}).get("temperature"),
                    "max_length": model_config["max_length"],
                    "truncation_strategy": model_config.get("truncation_strategy", "head"),
                    "cache_teacher_logits": model_config.get("distillation", {}).get("cache_teacher_logits"),
                    "dataset_version": dataset_config.get("version"),
                    "train_examples": card.get("dataset", {}).get("train_examples"),
                })
                _mv = card["metrics_validation"]
                aiplatform.log_metrics({
                    "pr_auc": _mv.get("pr_auc") or float("nan"),
                    "f1_at_1pct_fpr": _mv.get("f1_at_1pct_fpr") or float("nan"),
                    "precision_at_1pct_fpr": _mv.get("precision_at_1pct_fpr") or float("nan"),
                    "recall_at_1pct_fpr": _mv.get("recall_at_1pct_fpr") or float("nan"),
                    "f1_at_2pct_fpr": _mv.get("f1_at_2pct_fpr") or float("nan"),
                    "threshold_at_1pct_fpr": _mv.get("threshold_at_1pct_fpr") or float("nan"),
                })
            print("   Vertex AI Experiments: metrics logged.")
        except Exception as _e:
            print(f"   Warning: Vertex AI Experiments logging failed: {_e}")


if __name__ == "__main__":
    main()
