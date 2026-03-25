from __future__ import annotations

from ml.training.benchmark_granite_guardian_jailbreak_teacher import (
    build_distribution_table,
    compute_binary_metrics,
    select_stratified_sample,
)


def test_select_stratified_sample_is_reproducible() -> None:
    examples = [
        {"label": 1, "text": "inj-a"},
        {"label": 1, "text": "inj-b"},
        {"label": 1, "text": "inj-c"},
        {"label": 0, "text": "safe-a"},
        {"label": 0, "text": "safe-b"},
        {"label": 0, "text": "safe-c"},
    ]

    first = select_stratified_sample(
        examples,
        positive_count=2,
        negative_count=2,
        seed=42,
    )
    second = select_stratified_sample(
        examples,
        positive_count=2,
        negative_count=2,
        seed=42,
    )

    assert first == second
    assert sum(item["label"] for item in first) == 2
    assert len(first) == 4


def test_compute_binary_metrics_matches_expected_rates() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.3, 0.8, 0.1]

    metrics = compute_binary_metrics(labels, scores, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["fpr"] == 0.5


def test_build_distribution_table_uses_expected_bins() -> None:
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.05, 0.25, 0.45, 0.65, 0.85, 0.95]

    rows = build_distribution_table(labels, scores)

    assert [row["bin"] for row in rows] == [
        "0.0-0.1",
        "0.1-0.3",
        "0.3-0.5",
        "0.5-0.7",
        "0.7-0.9",
        "0.9-1.0",
    ]
    assert rows[0]["injection"] == 1
    assert rows[1]["safe"] == 1
    assert rows[5]["safe"] == 1
