from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ml.training.config_runtime import ConfigError, resolve_config


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def _write_config(tmp_path: Path, override=None) -> Path:
    config = {
        "dataset": {
            "train_path": "ml/data/versions/pi_mix_v2/train.jsonl",
            "val_path": "ml/data/versions/pi_mix_v2/val.jsonl",
        },
        "outputs": {
            "model_dir": "ml/models/test-config-runtime",
        },
    }
    if override:
        config = _deep_merge(config, override)

    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_yaml_load_validation_and_defaults(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "backbone": "prajjwal1/bert-tiny",
                "distillation": {"alpha": 0.25},
            },
            "training": {
                "epochs": 6,
            },
        },
    )

    result = resolve_config(str(config_path))

    assert result.resolved_config["model"]["backbone"] == "prajjwal1/bert-tiny"
    assert result.resolved_config["model"]["distillation"]["alpha"] == 0.25
    assert result.resolved_config["model"]["truncation_strategy"] == "head"
    assert result.resolved_config["model"]["distillation"]["teacher_max_length"] == 128
    assert (
        result.resolved_config["model"]["distillation"]["cache_teacher_logits"] is False
    )
    assert result.resolved_config["training"]["epochs"] == 6
    assert result.resolved_config["training"]["lr_scheduler_type"] == "cosine"
    assert result.resolved_config["training"]["warmup_ratio"] == 0.05


def test_teacher_max_length_defaults_to_256_when_teacher_is_enabled(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "teacher_model": "teacher/model",
            },
        },
    )

    result = resolve_config(str(config_path))

    assert result.resolved_config["model"]["teacher_model"] == "teacher/model"
    assert result.resolved_config["model"]["distillation"]["teacher_max_length"] == 256


def test_cache_teacher_logits_cli_override_updates_distillation_config(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "teacher_model": "teacher/model",
                "distillation": {
                    "cache_teacher_logits": False,
                },
            },
        },
    )

    result = resolve_config(str(config_path), {"cache_teacher_logits": True})

    assert (
        result.resolved_config["model"]["distillation"]["cache_teacher_logits"] is True
    )
    assert result.cli_overrides["model"]["distillation"]["cache_teacher_logits"] is True


def test_invalid_truncation_strategy_fails_fast(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "truncation_strategy": "middle",
            },
        },
    )

    with pytest.raises(ConfigError, match="model.truncation_strategy"):
        resolve_config(str(config_path))


def test_unknown_yaml_keys_fail_fast(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"training": {"unknown_knob": 123}})

    with pytest.raises(ConfigError, match="training.unknown_knob"):
        resolve_config(str(config_path))


def test_cli_overrides_take_precedence_over_yaml(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "training": {"epochs": 6},
            "outputs": {"model_dir": "ml/models/from-yaml"},
        },
    )

    result = resolve_config(
        str(config_path),
        {
            "epochs": 8,
            "output_dir": "/tmp/train-phase1-cli",
        },
    )

    assert result.resolved_config["training"]["epochs"] == 8
    assert result.resolved_config["outputs"]["model_dir"] == "/tmp/train-phase1-cli"
    assert result.cli_overrides["training"]["epochs"] == 8
    assert result.cli_overrides["outputs"]["model_dir"] == "/tmp/train-phase1-cli"


def test_boolean_disable_overrides_yaml_true(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "training": {"grad_checkpointing": True},
            "outputs": {"save_edge_export": True},
        },
    )

    result = resolve_config(
        str(config_path),
        {
            "grad_checkpointing": False,
            "save_edge_export": False,
        },
    )

    assert result.resolved_config["training"]["grad_checkpointing"] is False
    assert result.resolved_config["outputs"]["save_edge_export"] is False


def test_warmup_steps_take_precedence_over_ratio(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = resolve_config(
        str(config_path),
        {
            "warmup_steps": 200,
        },
    )

    assert result.resolved_config["training"]["lr_scheduler_type"] == "cosine"
    assert result.resolved_config["training"]["warmup_steps"] == 200
    assert result.resolved_config["training"]["warmup_ratio"] is None


def test_cadence_normalization_for_step_strategies(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "training": {
                "eval_steps": 123,
                "save_steps": None,
            }
        },
    )

    result = resolve_config(str(config_path))

    assert result.resolved_config["training"]["evaluation_strategy"] == "steps"
    assert result.resolved_config["training"]["save_strategy"] == "steps"
    assert result.resolved_config["training"]["eval_steps"] == 123
    assert result.resolved_config["training"]["save_steps"] == 123


def test_cadence_normalization_uses_epoch_when_steps_omitted(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "training": {
                "eval_steps": None,
                "save_steps": None,
            }
        },
    )

    result = resolve_config(str(config_path))

    assert result.resolved_config["training"]["evaluation_strategy"] == "epoch"
    assert result.resolved_config["training"]["save_strategy"] == "epoch"
    assert result.resolved_config["training"]["eval_steps"] is None
    assert result.resolved_config["training"]["save_steps"] is None


def test_best_model_metric_selection_updates_resolved_fields(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = resolve_config(
        str(config_path),
        {
            "best_model_metric": "f1_at_1pct_fpr",
            "early_stopping_patience": 4,
        },
    )

    assert result.resolved_config["training"]["best_model_metric"] == "f1_at_1pct_fpr"
    assert (
        result.resolved_config["training"]["metric_for_best_model"] == "f1_at_1pct_fpr"
    )
    assert result.resolved_config["training"]["greater_is_better"] is True
    assert result.resolved_config["training"]["enable_early_stopping"] is True
