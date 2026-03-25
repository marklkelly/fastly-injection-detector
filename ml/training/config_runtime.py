from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml/training/config/model.yaml"

SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}
SUPPORTED_MIXED_PRECISION = {"auto", "fp32", "bf16", "fp16"}
SUPPORTED_TRUNCATION_STRATEGIES = {"head", "tail", "head_tail"}
SUPPORTED_SCHEDULERS = {
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
}
SUPPORTED_BEST_MODEL_METRICS = {"pr_auc", "f1_at_1pct_fpr", "recall_at_1pct_fpr"}

SCHEMA_DEFAULTS = {
    "model": {
        "artifact_name": "prompt-injection-classifier",
        "backbone": "microsoft/MiniLM-L6-v2",
        "num_labels": 2,
        "max_length": 128,
        "truncation_strategy": "head",
        "labels": {"0": "SAFE", "1": "INJECTION"},
        "teacher_model": None,
        "distillation": {
            "enabled": True,
            "alpha": 0.5,
            "temperature": 2.0,
            "teacher_max_length": None,
            "cache_teacher_logits": False,
        },
    },
    "dataset": {
        "version": None,
        "train_path": None,
        "val_path": None,
        "test_path": None,
        "labels_path": None,
    },
    "training": {
        "epochs": 3,
        "max_steps": None,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "batch_size": 32,
        "eval_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "grad_checkpointing": False,
        "label_smoothing_factor": 0.0,
        "logging_steps": 100,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 2,
        "seed": 42,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "warmup_steps": None,
        "early_stopping_patience": None,
        "best_model_metric": "pr_auc",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "pr_auc",
        "greater_is_better": True,
        "enable_early_stopping": False,
    },
    "runtime": {
        "device": "auto",
        "mixed_precision": "auto",
        "dataloader_workers": 4,
        "quick_test": False,
    },
    "outputs": {
        "model_dir": "./model_out",
        "save_hf_checkpoint": True,
        "save_edge_export": False,
        "write_model_card": True,
        "write_eval_metrics": True,
    },
    "evaluation": {
        "ood_path": None,
        "prior": 0.02,
    },
}

CLI_TO_CONFIG_PATH = {
    "train_path": "dataset.train_path",
    "val_path": "dataset.val_path",
    "test_path": "dataset.test_path",
    "labels_path": "dataset.labels_path",
    "output_dir": "outputs.model_dir",
    "save_hf_checkpoint": "outputs.save_hf_checkpoint",
    "save_edge_export": "outputs.save_edge_export",
    "write_model_card": "outputs.write_model_card",
    "write_eval_metrics": "outputs.write_eval_metrics",
    "model": "model.backbone",
    "teacher": "model.teacher_model",
    "distillation_enabled": "model.distillation.enabled",
    "alpha": "model.distillation.alpha",
    "temperature": "model.distillation.temperature",
    "cache_teacher_logits": "model.distillation.cache_teacher_logits",
    "artifact_name": "model.artifact_name",
    "max_length": "model.max_length",
    "truncation_strategy": "model.truncation_strategy",
    "epochs": "training.epochs",
    "max_steps": "training.max_steps",
    "lr": "training.learning_rate",
    "weight_decay": "training.weight_decay",
    "batch_size": "training.batch_size",
    "eval_batch_size": "training.eval_batch_size",
    "gradient_accumulation_steps": "training.gradient_accumulation_steps",
    "grad_checkpointing": "training.grad_checkpointing",
    "label_smoothing_factor": "training.label_smoothing_factor",
    "logging_steps": "training.logging_steps",
    "save_steps": "training.save_steps",
    "eval_steps": "training.eval_steps",
    "early_stopping_patience": "training.early_stopping_patience",
    "best_model_metric": "training.best_model_metric",
    "lr_scheduler_type": "training.lr_scheduler_type",
    "warmup_ratio": "training.warmup_ratio",
    "warmup_steps": "training.warmup_steps",
    "seed": "training.seed",
    "device": "runtime.device",
    "mixed_precision": "runtime.mixed_precision",
    "dataloader_workers": "runtime.dataloader_workers",
    "quick_test": "runtime.quick_test",
    "eval_ood_path": "evaluation.ood_path",
    "eval_prior": "evaluation.prior",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ResolutionResult:
    config_path: Path
    config_from_yaml: Dict[str, Any]
    raw_yaml: Dict[str, Any]
    cli_overrides: Dict[str, Any]
    resolved_config: Dict[str, Any]
    resolved_output_dir: Path


Validator = Callable[[str, Any], None]


def _deepcopy_defaults() -> Dict[str, Any]:
    return copy.deepcopy(SCHEMA_DEFAULTS)


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _path_join(prefix: str, key: str) -> str:
    return key if not prefix else "{0}.{1}".format(prefix, key)


def _collect_unknown_keys(
    config: Dict[str, Any], schema: Dict[str, Any], prefix: str = ""
) -> Iterable[str]:
    unknown = []
    for key, value in config.items():
        path = _path_join(prefix, key)
        if key not in schema:
            unknown.append(path)
            continue
        schema_value = schema[key]
        if isinstance(schema_value, dict):
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError(
                    "Expected mapping for '{0}', got {1}".format(
                        path, type(value).__name__
                    )
                )
            unknown.extend(_collect_unknown_keys(value, schema_value, path))
    return unknown


def _get_by_path(config: Mapping[str, Any], dotted_path: str) -> Any:
    current = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_by_path(config: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _format_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _validate_enum(allowed: Iterable[str]) -> Validator:
    allowed_values = set(allowed)

    def _inner(path: str, value: Any) -> None:
        if value not in allowed_values:
            raise ConfigError(
                "Invalid value for '{0}': {1}. Allowed: {2}".format(
                    path,
                    _format_value(value),
                    ", ".join(sorted(allowed_values)),
                )
            )

    return _inner


def _validate_bool(path: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ConfigError(
            "Expected boolean for '{0}', got {1}".format(path, type(value).__name__)
        )


def _validate_non_empty_string(path: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Expected non-empty string for '{0}'".format(path))


def _validate_optional_string(path: str, value: Any) -> None:
    if value is None:
        return
    _validate_non_empty_string(path, value)


def _validate_positive_int(path: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(
            "Expected positive integer for '{0}', got {1}".format(
                path, _format_value(value)
            )
        )


def _validate_optional_positive_int(path: str, value: Any) -> None:
    if value is None:
        return
    _validate_positive_int(path, value)


def _validate_non_negative_int(path: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(
            "Expected non-negative integer for '{0}', got {1}".format(
                path, _format_value(value)
            )
        )


def _validate_optional_non_negative_int(path: str, value: Any) -> None:
    if value is None:
        return
    _validate_non_negative_int(path, value)


def _validate_non_negative_float(path: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ConfigError(
            "Expected non-negative number for '{0}', got {1}".format(
                path, _format_value(value)
            )
        )


def _validate_ratio(path: str, value: Any) -> None:
    if value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        or value > 1
    ):
        raise ConfigError(
            "Expected number in [0, 1] for '{0}', got {1}".format(
                path, _format_value(value)
            )
        )


def _validate_labels(path: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise ConfigError(
            "Expected mapping for '{0}', got {1}".format(path, type(value).__name__)
        )
    if set(value.keys()) != {"0", "1"}:
        raise ConfigError("Expected label keys {{'0', '1'}} for '{0}'".format(path))
    for key, label in value.items():
        if not isinstance(label, str) or not label.strip():
            raise ConfigError(
                "Expected non-empty label string for '{0}.{1}'".format(path, key)
            )


VALIDATORS = {
    "model.artifact_name": _validate_non_empty_string,
    "model.backbone": _validate_non_empty_string,
    "model.num_labels": _validate_positive_int,
    "model.max_length": _validate_positive_int,
    "model.truncation_strategy": _validate_enum(SUPPORTED_TRUNCATION_STRATEGIES),
    "model.labels": _validate_labels,
    "model.teacher_model": _validate_optional_string,
    "model.distillation.enabled": _validate_bool,
    "model.distillation.alpha": _validate_ratio,
    "model.distillation.temperature": _validate_non_negative_float,
    "model.distillation.teacher_max_length": _validate_optional_positive_int,
    "model.distillation.cache_teacher_logits": _validate_bool,
    "dataset.version": _validate_optional_string,
    "dataset.train_path": _validate_non_empty_string,
    "dataset.val_path": _validate_non_empty_string,
    "dataset.test_path": _validate_optional_string,
    "dataset.labels_path": _validate_optional_string,
    "training.epochs": _validate_positive_int,
    "training.max_steps": _validate_optional_positive_int,
    "training.learning_rate": _validate_non_negative_float,
    "training.weight_decay": _validate_non_negative_float,
    "training.batch_size": _validate_positive_int,
    "training.eval_batch_size": _validate_positive_int,
    "training.gradient_accumulation_steps": _validate_positive_int,
    "training.grad_checkpointing": _validate_bool,
    "training.label_smoothing_factor": _validate_ratio,
    "training.logging_steps": _validate_positive_int,
    "training.save_steps": _validate_optional_positive_int,
    "training.eval_steps": _validate_optional_positive_int,
    "training.save_total_limit": _validate_positive_int,
    "training.seed": _validate_non_negative_int,
    "training.lr_scheduler_type": _validate_enum(SUPPORTED_SCHEDULERS),
    "training.warmup_ratio": _validate_ratio,
    "training.warmup_steps": _validate_optional_non_negative_int,
    "training.early_stopping_patience": _validate_optional_non_negative_int,
    "training.best_model_metric": _validate_enum(SUPPORTED_BEST_MODEL_METRICS),
    "runtime.device": _validate_enum(SUPPORTED_DEVICES),
    "runtime.mixed_precision": _validate_enum(SUPPORTED_MIXED_PRECISION),
    "runtime.dataloader_workers": _validate_non_negative_int,
    "runtime.quick_test": _validate_bool,
    "outputs.model_dir": _validate_non_empty_string,
    "outputs.save_hf_checkpoint": _validate_bool,
    "outputs.save_edge_export": _validate_bool,
    "outputs.write_model_card": _validate_bool,
    "outputs.write_eval_metrics": _validate_bool,
    "evaluation.ood_path": _validate_optional_string,
    "evaluation.prior": _validate_ratio,
}


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ConfigError(
            "Expected top-level mapping in config file '{0}', got {1}".format(
                config_path, type(loaded).__name__
            )
        )

    unknown_keys = list(_collect_unknown_keys(loaded, SCHEMA_DEFAULTS))
    if unknown_keys:
        raise ConfigError(
            "Unknown config key(s) in '{0}': {1}".format(
                config_path,
                ", ".join(sorted(unknown_keys)),
            )
        )

    return loaded


def _validate_config(config: Dict[str, Any]) -> None:
    for path, validator in VALIDATORS.items():
        validator(path, _get_by_path(config, path))

    if _get_by_path(config, "model.num_labels") != 2:
        raise ConfigError("This trainer expects 'model.num_labels' to be 2")


def _normalize_namespace(cli_args: Optional[Any]) -> Dict[str, Any]:
    if cli_args is None:
        return {}
    if isinstance(cli_args, argparse.Namespace):
        return dict(vars(cli_args))
    if isinstance(cli_args, Mapping):
        return dict(cli_args)
    raise TypeError("cli_args must be an argparse.Namespace or mapping")


def _normalize_mixed_precision_override(cli_values: Dict[str, Any]) -> Optional[str]:
    explicit_mixed_precision = cli_values.get("mixed_precision")
    bf16 = cli_values.get("bf16")
    fp16 = cli_values.get("fp16")

    alias_mixed_precision = None
    if bf16 is True and fp16 is True:
        raise ConfigError("Cannot enable both --bf16 and --fp16 in the same command")
    if bf16 is True:
        alias_mixed_precision = "bf16"
    elif fp16 is True:
        alias_mixed_precision = "fp16"
    elif bf16 is False or fp16 is False:
        alias_mixed_precision = "fp32"

    if explicit_mixed_precision is None:
        return alias_mixed_precision
    if (
        alias_mixed_precision is None
        or alias_mixed_precision == explicit_mixed_precision
    ):
        return explicit_mixed_precision
    raise ConfigError(
        "Conflicting mixed precision CLI overrides: explicit={0}, alias={1}".format(
            explicit_mixed_precision,
            alias_mixed_precision,
        )
    )


def normalize_cli_overrides(cli_args: Optional[Any]) -> Dict[str, Any]:
    cli_values = _normalize_namespace(cli_args)
    normalized = {}

    for cli_key, config_path in CLI_TO_CONFIG_PATH.items():
        if cli_key not in cli_values:
            continue
        value = cli_values[cli_key]
        if value is None:
            continue
        _set_by_path(normalized, config_path, value)

    mixed_precision = _normalize_mixed_precision_override(cli_values)
    if mixed_precision is not None:
        _set_by_path(normalized, "runtime.mixed_precision", mixed_precision)

    return normalized


def _normalize_steps(value: Any) -> Optional[int]:
    if value in (None, 0):
        return None
    return value


def _apply_warmup_rules(config: Dict[str, Any]) -> None:
    warmup_steps = config["training"]["warmup_steps"]
    if warmup_steps in (None, 0):
        config["training"]["warmup_steps"] = None
        return
    if warmup_steps < 0:
        raise ConfigError(
            "'training.warmup_steps' must be null or a non-negative integer"
        )
    config["training"]["warmup_ratio"] = None


def _apply_cadence_rules(config: Dict[str, Any]) -> None:
    training = config["training"]
    runtime = config["runtime"]

    if runtime["quick_test"]:
        training["max_steps"] = 2
        training["epochs"] = 1
        training["logging_steps"] = 1
        training["eval_steps"] = 2
        training["save_steps"] = 2
        training["evaluation_strategy"] = "steps"
        training["save_strategy"] = "steps"
        return

    eval_steps = _normalize_steps(training["eval_steps"])
    save_steps = _normalize_steps(training["save_steps"])
    training["eval_steps"] = eval_steps
    training["save_steps"] = save_steps

    if eval_steps is None and save_steps is None:
        training["evaluation_strategy"] = "epoch"
        training["save_strategy"] = "epoch"
        return

    if eval_steps is None:
        eval_steps = save_steps
    if save_steps is None:
        save_steps = eval_steps

    if eval_steps is None or save_steps is None:
        raise ConfigError(
            "Step cadence normalization failed to produce eval/save steps"
        )

    if save_steps % eval_steps != 0:
        raise ConfigError(
            "'training.save_steps' must be a round multiple of 'training.eval_steps' when using step cadence"
        )

    training["eval_steps"] = eval_steps
    training["save_steps"] = save_steps
    training["evaluation_strategy"] = "steps"
    training["save_strategy"] = "steps"


def _apply_best_model_rules(config: Dict[str, Any]) -> None:
    metric = config["training"]["best_model_metric"]
    config["training"]["metric_for_best_model"] = metric
    config["training"]["greater_is_better"] = True
    config["training"]["load_best_model_at_end"] = True

    patience = config["training"]["early_stopping_patience"]
    if patience in (None, 0):
        config["training"]["early_stopping_patience"] = None
        config["training"]["enable_early_stopping"] = False
        return

    config["training"]["enable_early_stopping"] = True


def _apply_distillation_rules(config: Dict[str, Any]) -> None:
    model = config["model"]
    distillation = model["distillation"]
    teacher_enabled = bool(model["teacher_model"]) and distillation["enabled"]

    if teacher_enabled:
        if distillation["teacher_max_length"] is None:
            distillation["teacher_max_length"] = 256
    else:
        distillation["teacher_max_length"] = model["max_length"]

    if distillation["cache_teacher_logits"] and not teacher_enabled:
        raise ConfigError(
            "'model.distillation.cache_teacher_logits' requires an enabled teacher model"
        )


def resolve_config(
    config_path: Optional[str] = None, cli_args: Optional[Any] = None
) -> ResolutionResult:
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT / config_file).resolve()

    if not config_file.exists():
        raise ConfigError("Config file not found: {0}".format(config_file))

    raw_yaml = load_yaml_config(config_file)
    config_from_yaml = _deep_merge(_deepcopy_defaults(), raw_yaml)
    _validate_config(config_from_yaml)

    cli_overrides = normalize_cli_overrides(cli_args)
    resolved = _deep_merge(config_from_yaml, cli_overrides)
    _apply_warmup_rules(resolved)
    _apply_cadence_rules(resolved)
    _apply_best_model_rules(resolved)
    _apply_distillation_rules(resolved)
    _validate_config(resolved)

    resolved_output_dir = resolve_project_path(resolved["outputs"]["model_dir"])
    if resolved_output_dir is None:
        raise ConfigError("Resolved output directory cannot be null")
    return ResolutionResult(
        config_path=config_file,
        config_from_yaml=config_from_yaml,
        raw_yaml=raw_yaml,
        cli_overrides=cli_overrides,
        resolved_config=resolved,
        resolved_output_dir=resolved_output_dir,
    )


def summarize_resolution(result: ResolutionResult) -> str:
    lines = [
        "Configuration summary:",
        "  config: {0}".format(result.config_path),
        "  output_dir: {0}".format(result.resolved_output_dir),
    ]

    if not result.cli_overrides:
        lines.append("  cli_overrides: none")
        return "\n".join(lines)

    lines.append("  cli_overrides:")
    for path in sorted(_flatten_leaf_paths(result.cli_overrides)):
        before = _get_by_path(result.config_from_yaml, path)
        after = _get_by_path(result.resolved_config, path)
        if before == after:
            continue
        lines.append(
            "    {0}: {1} -> {2}".format(
                path, _format_value(before), _format_value(after)
            )
        )

    if lines[-1] == "  cli_overrides:":
        lines.append("    none")

    return "\n".join(lines)


def _flatten_leaf_paths(config: Mapping[str, Any], prefix: str = "") -> Iterable[str]:
    paths = []
    for key, value in config.items():
        path = _path_join(prefix, key)
        if isinstance(value, Mapping):
            paths.extend(_flatten_leaf_paths(value, path))
            continue
        paths.append(path)
    return paths
