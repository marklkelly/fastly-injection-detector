from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

LENGTH_BUCKET_SHORT = "<=128"
LENGTH_BUCKET_LONG = ">128"
DEFAULT_LENGTH_BUCKETS = (LENGTH_BUCKET_SHORT, LENGTH_BUCKET_LONG)


def length_bucket_for_token_count(
    original_token_length: int,
    student_contract_tokens: int = 128,
) -> str:
    if original_token_length <= student_contract_tokens:
        return LENGTH_BUCKET_SHORT
    return LENGTH_BUCKET_LONG


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _slice_metrics_at_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    threshold_source: str,
) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)

    if labels.shape[0] != probs.shape[0]:
        raise ValueError("labels and probs must have the same length")

    example_count = int(labels.shape[0])
    if example_count == 0:
        return {
            "example_count": 0,
            "pr_auc": None,
            "precision_at_1pct_fpr": None,
            "recall_at_1pct_fpr": None,
            "f1_at_1pct_fpr": None,
            "threshold_source": threshold_source,
        }

    unique_labels = np.unique(labels)
    pr_auc = None
    if unique_labels.shape[0] > 1:
        pr_auc = _safe_float(average_precision_score(labels, probs))

    pred = (probs >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    precision = None
    if (tp + fp) > 0:
        precision = tp / (tp + fp)

    recall = None
    if (tp + fn) > 0:
        recall = tp / (tp + fn)

    f1 = None
    if precision is not None and recall is not None:
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

    return {
        "example_count": example_count,
        "pr_auc": pr_auc,
        "precision_at_1pct_fpr": _safe_float(precision),
        "recall_at_1pct_fpr": _safe_float(recall),
        "f1_at_1pct_fpr": _safe_float(f1),
        "threshold_source": threshold_source,
    }


def _ordered_slice_names(
    observed_names: Iterable[str],
    expected_names: Sequence[str] | None = None,
) -> list[str]:
    ordered = []
    if expected_names is not None:
        ordered.extend(expected_names)
    for name in sorted(set(observed_names)):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _group_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    group_names: Sequence[str],
    threshold: float,
    threshold_source: str,
    expected_names: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    grouped = {}
    all_names = _ordered_slice_names(group_names, expected_names=expected_names)
    for name in all_names:
        idx = [index for index, value in enumerate(group_names) if value == name]
        if not idx:
            grouped[name] = _slice_metrics_at_threshold(
                labels=[],
                probs=[],
                threshold=threshold,
                threshold_source=threshold_source,
            )
            continue
        grouped[name] = _slice_metrics_at_threshold(
            labels=labels[idx],
            probs=probs[idx],
            threshold=threshold,
            threshold_source=threshold_source,
        )
    return grouped


def _metadata_value(metadata: Mapping[str, Any], key: str, default: str) -> str:
    value = metadata.get(key)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def compute_slice_report(
    labels: Sequence[int] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    threshold: float,
    threshold_source: str = "validation_overall",
) -> Dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs_array = np.asarray(probs, dtype=np.float32).reshape(-1)

    if labels_array.shape[0] != probs_array.shape[0]:
        raise ValueError("labels and probs must have the same length")
    if labels_array.shape[0] != len(metadata):
        raise ValueError("metadata length must match labels/probs length")

    sources = [_metadata_value(item, "source", "unknown") for item in metadata]
    length_buckets = [
        _metadata_value(
            item,
            "length_bucket",
            length_bucket_for_token_count(int(item.get("original_token_length", 0))),
        )
        for item in metadata
    ]

    return {
        "threshold_source": threshold_source,
        "threshold_at_1pct_fpr": _safe_float(threshold),
        "by_source": _group_metrics(
            labels=labels_array,
            probs=probs_array,
            group_names=sources,
            threshold=threshold,
            threshold_source=threshold_source,
        ),
        "by_length_bucket": _group_metrics(
            labels=labels_array,
            probs=probs_array,
            group_names=length_buckets,
            threshold=threshold,
            threshold_source=threshold_source,
            expected_names=DEFAULT_LENGTH_BUCKETS,
        ),
    }


def summarize_slice_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    by_length_bucket = report.get("by_length_bucket", {})
    return {
        "source_count": len(report.get("by_source", {})),
        "length_buckets": {
            name: {
                "example_count": bucket["example_count"],
                "pr_auc": bucket["pr_auc"],
                "f1_at_1pct_fpr": bucket["f1_at_1pct_fpr"],
            }
            for name, bucket in by_length_bucket.items()
        },
    }
