from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "ml/training/.cache/teacher_logits"


class ManifestMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherCachePaths:
    cache_key: str
    cache_dir: Path
    manifest_path: Path
    logits_path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_teacher_cache_manifest(
    *,
    split: str,
    dataset_path: Path,
    teacher_model: str,
    teacher_model_revision: str | None,
    teacher_tokenizer: str,
    teacher_tokenizer_revision: str | None,
    teacher_max_length: int,
    student_truncation_strategy: str,
    example_count: int | None = None,
) -> dict[str, Any]:
    resolved_dataset_path = Path(dataset_path).resolve()
    manifest = {
        "split": split,
        "dataset_path": str(resolved_dataset_path),
        "dataset_file_hash": file_sha256(resolved_dataset_path),
        "teacher_model": teacher_model,
        "teacher_model_revision": teacher_model_revision,
        "teacher_tokenizer": teacher_tokenizer,
        "teacher_tokenizer_revision": teacher_tokenizer_revision,
        "teacher_max_length": teacher_max_length,
        "teacher_truncation_mode": "head",
        "student_truncation_strategy": student_truncation_strategy,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    if example_count is not None:
        manifest["example_count"] = example_count
    return manifest


def cache_key_for_manifest(manifest: Mapping[str, Any]) -> str:
    key_payload = {
        "dataset_path": manifest["dataset_path"],
        "teacher_model": manifest["teacher_model"],
        "teacher_tokenizer": manifest["teacher_tokenizer"],
    }
    payload = json.dumps(
        key_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_paths_for_manifest(
    cache_root: Path | str,
    manifest: Mapping[str, Any],
) -> TeacherCachePaths:
    root = Path(cache_root)
    cache_key = cache_key_for_manifest(manifest)
    cache_dir = root / cache_key
    return TeacherCachePaths(
        cache_key=cache_key,
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.json",
        logits_path=cache_dir / "teacher_logits.pt",
    )


def _manifest_mismatch_message(
    actual_manifest: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> str:
    mismatches = []
    for key in sorted(set(actual_manifest) | set(expected_manifest)):
        actual_value = actual_manifest.get(key)
        expected_value = expected_manifest.get(key)
        if actual_value == expected_value:
            continue
        mismatches.append(
            "{0}: expected {1!r}, found {2!r}".format(
                key,
                expected_value,
                actual_value,
            )
        )
    return "; ".join(mismatches)


def _validate_manifest(
    manifest_path: Path,
    expected_manifest: Mapping[str, Any],
) -> None:
    if not manifest_path.exists():
        raise ManifestMismatchError(
            "Teacher logits cache is missing manifest: {0}".format(manifest_path)
        )
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dict(actual_manifest) == dict(expected_manifest):
        return
    raise ManifestMismatchError(
        "Teacher logits cache manifest mismatch at {0}: {1}".format(
            manifest_path,
            _manifest_mismatch_message(actual_manifest, expected_manifest),
        )
    )


def _ensure_2d_float_tensor(logits: Any) -> torch.Tensor:
    tensor = torch.as_tensor(logits).detach().cpu().to(dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(
            "Teacher logits cache only supports 2D raw logits, got shape {0}".format(
                tuple(tensor.shape)
            )
        )
    return tensor


def load_cached_teacher_logits(
    cache_root: Path | str,
    expected_manifest: Mapping[str, Any],
) -> torch.Tensor | None:
    paths = cache_paths_for_manifest(cache_root, expected_manifest)
    if not paths.logits_path.exists():
        return None
    _validate_manifest(paths.manifest_path, expected_manifest)
    return _ensure_2d_float_tensor(torch.load(paths.logits_path, map_location="cpu"))


def write_cached_teacher_logits(
    cache_root: Path | str,
    manifest: Mapping[str, Any],
    logits: Any,
) -> TeacherCachePaths:
    paths = cache_paths_for_manifest(cache_root, manifest)
    if paths.cache_dir.exists() and paths.manifest_path.exists():
        _validate_manifest(paths.manifest_path, manifest)

    tensor = _ensure_2d_float_tensor(logits)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(tensor, paths.logits_path)
    return paths
