from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.training.teacher_cache import (
    ManifestMismatchError,
    build_teacher_cache_manifest,
    cache_key_for_manifest,
    load_cached_teacher_logits,
    write_cached_teacher_logits,
)


def _write_dataset(tmp_path: Path, name: str = "train.jsonl") -> Path:
    dataset_path = tmp_path / name
    dataset_path.write_text(
        '{"text":"hello","label":"SAFE"}\n{"text":"world","label":"INJECTION"}\n',
        encoding="utf-8",
    )
    return dataset_path


def _build_manifest(
    tmp_path: Path,
    *,
    dataset_name: str = "train.jsonl",
    **overrides,
):
    dataset_path = _write_dataset(tmp_path, dataset_name)
    return build_teacher_cache_manifest(
        split="train",
        dataset_path=dataset_path,
        teacher_model="teacher/model",
        teacher_model_revision="rev-a",
        teacher_tokenizer="teacher/tokenizer",
        teacher_tokenizer_revision="tok-rev-a",
        teacher_max_length=256,
        student_truncation_strategy="head",
        **overrides,
    )


def test_cache_key_is_stable_for_equivalent_manifest_content(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    reordered_manifest = dict(reversed(list(manifest.items())))

    assert cache_key_for_manifest(manifest) == cache_key_for_manifest(
        reordered_manifest
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("split", "validation"),
        ("dataset_file_hash", "different-hash"),
        ("teacher_model_revision", "rev-b"),
        ("student_truncation_strategy", "head_tail"),
        ("teacher_max_length", 128),
    ],
)
def test_manifest_mismatch_blocks_cache_reuse(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    manifest = _build_manifest(tmp_path)
    logits = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32)
    write_cached_teacher_logits(tmp_path, manifest, logits)

    mismatched_manifest = dict(manifest)
    mismatched_manifest[field_name] = value

    with pytest.raises(ManifestMismatchError, match=field_name):
        load_cached_teacher_logits(tmp_path, mismatched_manifest)


def test_write_and_load_cached_teacher_logits_round_trip(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    expected = torch.tensor(
        [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
        dtype=torch.float32,
    )

    write_cached_teacher_logits(tmp_path, manifest, expected)
    loaded = load_cached_teacher_logits(tmp_path, manifest)

    assert loaded is not None
    assert loaded.shape == (3, 2)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, expected)
