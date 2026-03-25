"""Tests for split invariants."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def make_rows(n=100, n_clusters=10):
    """Create synthetic rows with cluster_ids."""
    import random
    random.seed(42)
    rows = []
    for i in range(n):
        rows.append({
            "text": f"example text number {i} with unique content here",
            "label": i % 2,
            "source": "test/source" if i < 80 else "test/source2",
            "cluster_id": i % n_clusters,
        })
    return rows


def test_no_overlap_between_splits():
    """train/val/test must have zero text overlap."""
    from ml.data.build import cluster_stratified_split
    rows = make_rows(200)
    train, val, test = cluster_stratified_split(rows, seed=42)
    train_texts = {r["text"] for r in train}
    val_texts = {r["text"] for r in val}
    test_texts = {r["text"] for r in test}
    assert not (train_texts & val_texts), "Overlap between train and val"
    assert not (train_texts & test_texts), "Overlap between train and test"
    assert not (val_texts & test_texts), "Overlap between val and test"


def test_minimum_bucket_rule():
    """Buckets with <10 rows go to train only."""
    from ml.data.build import cluster_stratified_split
    # Create a bucket with only 5 rows
    rows = [{"text": f"tiny bucket {i}", "label": 1, "source": "tiny/source", "cluster_id": i} for i in range(5)]
    # Add enough rows to make the overall dataset valid
    rows += make_rows(100)
    train, val, test = cluster_stratified_split(rows, seed=42)
    tiny_texts = {f"tiny bucket {i}" for i in range(5)}
    train_texts = {r["text"] for r in train}
    val_texts = {r["text"] for r in val}
    test_texts = {r["text"] for r in test}
    # All tiny bucket rows must be in train
    for t in tiny_texts:
        assert t in train_texts, f"'{t}' not in train"
        assert t not in val_texts, f"'{t}' in val (violates min-bucket rule)"
        assert t not in test_texts, f"'{t}' in test (violates min-bucket rule)"


def test_cluster_integrity():
    """Two rows with the same cluster_id and same (source, label) must be in the same split."""
    from ml.data.build import cluster_stratified_split
    rows = make_rows(200, n_clusters=5)
    train, val, test = cluster_stratified_split(rows, seed=42)

    split_map = {}
    for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
        for r in split_rows:
            key = (r["source"], r["label"], r["cluster_id"])
            if key in split_map:
                assert split_map[key] == split_name, (
                    f"Cluster integrity violated: cluster {key} appears in both "
                    f"{split_map[key]} and {split_name}"
                )
            split_map[key] = split_name


def test_ood_sources_absent_from_train():
    """OOD rows must not appear in train/val/test (they go to ood_rows separately)."""
    ood_texts = {"ood example 1", "ood example 2"}
    train_texts = {"train example 1", "train example 2"}
    assert not (ood_texts & train_texts), "OOD texts leaked into train"


def test_ood_no_leakage_after_global_dedup():
    """No text should appear in both test_ood and any of train/val/test after dedup+split."""
    from ml.data.build import build_minhash_clusters, cluster_stratified_split

    ood_source = "ood/source"
    in_domain_source = "in_domain/source"

    # 20 in-domain rows
    in_domain = [
        {"text": f"in-domain unique example number {i}", "label": i % 2, "source": in_domain_source}
        for i in range(20)
    ]
    # 5 OOD rows (3 exact duplicates shared with in-domain, 2 near-duplicates)
    shared_texts = [in_domain[0]["text"], in_domain[1]["text"], in_domain[2]["text"]]
    near_dup_texts = [
        "in domain unique example number 5 extra word",   # near-dup of in_domain[5]
        "in domain unique example number 6 extra word",   # near-dup of in_domain[6]
    ]
    ood = [
        {"text": shared_texts[0], "label": 1, "source": ood_source},
        {"text": shared_texts[1], "label": 1, "source": ood_source},
        {"text": shared_texts[2], "label": 1, "source": ood_source},
        {"text": near_dup_texts[0], "label": 1, "source": ood_source},
        {"text": near_dup_texts[1], "label": 1, "source": ood_source},
    ]

    all_rows = in_domain + ood

    # Run global MinHash dedup
    deduped, _num_clusters, _removed = build_minhash_clusters(all_rows, num_perm=64, threshold=0.7)

    # Route OOD after dedup
    ood_source_repos = {ood_source}
    test_ood_rows = [r for r in deduped if r["source"] in ood_source_repos]
    final_in_domain = [r for r in deduped if r["source"] not in ood_source_repos]

    # Ensure in-domain has enough rows to split
    assert len(final_in_domain) >= 3, "Need at least 3 in-domain rows for split"

    train, val, test = cluster_stratified_split(final_in_domain, seed=42)

    # Assert no text overlap between test_ood and any train/val/test split
    ood_texts = {r["text"] for r in test_ood_rows}
    train_texts = {r["text"] for r in train}
    val_texts = {r["text"] for r in val}
    test_texts = {r["text"] for r in test}

    assert not (ood_texts & train_texts), f"OOD leaked into train: {ood_texts & train_texts}"
    assert not (ood_texts & val_texts), f"OOD leaked into val: {ood_texts & val_texts}"
    assert not (ood_texts & test_texts), f"OOD leaked into test: {ood_texts & test_texts}"
