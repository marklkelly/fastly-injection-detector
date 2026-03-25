#!/usr/bin/env python3
"""Cloud training entrypoint for the refactored training pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from google.cloud import storage

APP_DIR = Path("/app")
DATA_DIR = Path("/tmp/data")
TRAIN_DATA_PATH = DATA_DIR / "train.jsonl"
VAL_DATA_PATH = DATA_DIR / "val.jsonl"
LABELS_PATH = DATA_DIR / "labels.json"
OUTPUT_DIR = Path("/tmp/outputs")
RESOLVED_CONFIG_PATH = Path("/tmp/resolved_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training from a patched config")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-uri", required=True)
    parser.add_argument("--val-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    return parser.parse_args()


def split_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {uri}")

    bucket_name, _, blob_name = uri[5:].partition("/")
    if not bucket_name:
        raise ValueError(f"Missing bucket name in URI: {uri}")

    return bucket_name, blob_name


def download_blob(client: storage.Client, uri: str, destination: Path) -> None:
    bucket_name, blob_name = split_gcs_uri(uri)
    if not blob_name:
        raise ValueError(f"Missing object path in URI: {uri}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(destination))
    print(f"Downloaded {uri} -> {destination}")


def upload_directory(client: storage.Client, source_dir: Path, output_uri: str) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {source_dir}")

    bucket_name, prefix = split_gcs_uri(output_uri)
    normalized_prefix = prefix.rstrip("/")
    bucket = client.bucket(bucket_name)

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(source_dir).as_posix()
        blob_name = (
            relative_path
            if not normalized_prefix
            else f"{normalized_prefix}/{relative_path}"
        )
        bucket.blob(blob_name).upload_from_filename(str(path))
        print(f"Uploaded {path} -> gs://{bucket_name}/{blob_name}")


def resolve_model_config_path(config_arg: str) -> Path:
    requested = Path(config_arg)
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    add_candidate(requested)
    if not requested.is_absolute():
        add_candidate(APP_DIR / requested)

    parts = requested.parts
    if "config" in parts:
        config_relative = Path(*parts[parts.index("config") :])
        add_candidate(config_relative)
        add_candidate(APP_DIR / config_relative)

    add_candidate(Path("config") / requested.name)
    add_candidate(APP_DIR / "config" / requested.name)
    add_candidate(APP_DIR / "ml" / "training" / "config" / requested.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve model config '{config_arg}'. Tried: {tried}"
    )


def patch_config(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Config file must contain a mapping: {config_path}")

    dataset = config.get("dataset")
    if dataset is None:
        dataset = {}
        config["dataset"] = dataset
    if not isinstance(dataset, dict):
        raise TypeError("Config key 'dataset' must be a mapping")

    outputs = config.get("outputs")
    if outputs is None:
        outputs = {}
        config["outputs"] = outputs
    if not isinstance(outputs, dict):
        raise TypeError("Config key 'outputs' must be a mapping")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset["train_path"] = str(TRAIN_DATA_PATH)
    dataset["val_path"] = str(VAL_DATA_PATH)
    dataset["labels_path"] = str(LABELS_PATH)
    outputs["model_dir"] = str(OUTPUT_DIR)

    with RESOLVED_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(f"Resolved config written to {RESOLVED_CONFIG_PATH}")
    return RESOLVED_CONFIG_PATH


def run_training(config_path: Path) -> int:
    command = ["python", "ml/training/train_cls.py", "--config", str(config_path)]
    print(f"Running: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(
            result.stderr,
            file=sys.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
        )

    return result.returncode


def main() -> int:
    args = parse_args()
    client = storage.Client()

    try:
        download_blob(client, args.train_uri, TRAIN_DATA_PATH)
        download_blob(client, args.val_uri, VAL_DATA_PATH)
        labels_uri = args.train_uri.rsplit("/", 1)[0] + "/labels.json"
        download_blob(client, labels_uri, LABELS_PATH)
        config_path = resolve_model_config_path(args.model_config)
        resolved_config = patch_config(config_path)
        training_exit_code = run_training(resolved_config)
        if training_exit_code != 0:
            return training_exit_code
        upload_directory(client, OUTPUT_DIR, args.output_uri)
    except Exception as exc:
        print(f"Entrypoint failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
