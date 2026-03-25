"""Tests for manifest schema contract (Phase 2)."""
import pytest
import json
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def make_valid_manifest():
    return {
        "version": "pi_mix_v2",
        "build_date": "2026-03-12T15:00:00Z",
        "build_command": "python ml/data/build.py --recipe ml/data/recipes/pi_mix_v2.yaml --output ml/data/versions/pi_mix_v2",
        "recipe_path": "/path/to/ml/data/recipes/pi_mix_v2.yaml",
        "recipe_sha256": "abc123def456",
        "python_dependencies": {
            "datasets": "2.18.0",
            "datasketch": "1.6.4",
        },
        "sources": [
            {
                "repo": "jayavibhav/prompt-injection",
                "revision": "a5bd9869c16d22ffebda9fa91a13a759b8669bb6",
                "status": "used",
                "rows_loaded": 1000,
            },
            {
                "repo": "deepset/prompt-injections",
                "revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
                "status": "used",
                "ood_eval": True,
                "rows_loaded": 200,
            },
            {
                "repo": "markush1/LLM-Injection-Dataset",
                "revision": None,
                "status": "skipped",
                "rows_loaded": 0,
            },
        ],
        "dedup": {
            "exact_removed": 150,
            "minhash_removed": 40,
            "threshold": 0.7,
            "num_perm": 128,
        },
        "final": {
            "train": 1440,
            "val": 180,
            "test": 180,
            "test_ood": 200,
        },
    }


def test_required_top_level_keys():
    """manifest.json must have all required Phase 2 top-level keys."""
    m = make_valid_manifest()
    required = [
        "version", "build_date", "build_command", "recipe_path",
        "recipe_sha256", "python_dependencies", "sources", "dedup", "final",
    ]
    for field in required:
        assert field in m, f"Missing required field: {field}"


def test_sources_have_provenance():
    """Each source entry must have repo, revision, status, rows_loaded."""
    m = make_valid_manifest()
    for src in m["sources"]:
        assert "repo" in src, "source missing 'repo'"
        assert "revision" in src, "source missing 'revision'"
        assert "status" in src, "source missing 'status'"
        assert "rows_loaded" in src, "source missing 'rows_loaded'"


def test_source_status_values():
    """status must be one of 'used', 'replaced', 'skipped'."""
    m = make_valid_manifest()
    valid_statuses = {"used", "replaced", "skipped"}
    for src in m["sources"]:
        assert src["status"] in valid_statuses, (
            f"Invalid status '{src['status']}' for repo {src['repo']}"
        )


def test_dedup_has_required_fields():
    """dedup block must have exact_removed, minhash_removed, threshold, num_perm."""
    m = make_valid_manifest()
    dedup = m["dedup"]
    assert "exact_removed" in dedup, "dedup missing 'exact_removed'"
    assert "minhash_removed" in dedup, "dedup missing 'minhash_removed'"
    assert "threshold" in dedup, "dedup missing 'threshold'"
    assert "num_perm" in dedup, "dedup missing 'num_perm'"


def test_final_has_required_splits():
    """final block must have train, val, test, test_ood — all non-negative integers."""
    m = make_valid_manifest()
    final = m["final"]
    for split in ["train", "val", "test", "test_ood"]:
        assert split in final, f"final missing split '{split}'"
        assert isinstance(final[split], int), f"final.{split} must be an integer"
        assert final[split] >= 0, f"final.{split} must be non-negative"


def test_manifest_roundtrip():
    """Manifest should serialize and deserialize cleanly."""
    m = make_valid_manifest()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(m, f, indent=2)
        path = f.name
    try:
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == m
    finally:
        os.unlink(path)


def test_python_dependencies_has_datasets_and_datasketch():
    """python_dependencies must contain 'datasets' and 'datasketch' keys."""
    m = make_valid_manifest()
    deps = m["python_dependencies"]
    assert "datasets" in deps, "python_dependencies missing 'datasets'"
    assert "datasketch" in deps, "python_dependencies missing 'datasketch'"

