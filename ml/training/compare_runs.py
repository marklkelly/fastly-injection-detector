#!/usr/bin/env python3
"""Compare training runs by scanning model_card.json artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODELS_DIR = Path("ml/models")

FIELD_ORDER = [
    "run_name",
    "output_dir",
    "dataset_version",
    "epochs",
    "batch_size",
    "lr",
    "alpha",
    "temperature",
    "truncation_strategy",
    "pr_auc",
    "f1_at_1pct_fpr",
    "f1_at_2pct_fpr",
    "threshold_at_1pct_fpr",
    "train_examples",
    "hardware",
    "device",
    "dtype",
    "early_stopped",
]

FIELD_LABELS = {
    "run_name": "run_name",
    "output_dir": "output_dir",
    "dataset_version": "dataset_version",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "lr": "lr",
    "alpha": "alpha",
    "temperature": "temperature",
    "truncation_strategy": "truncation_strategy",
    "pr_auc": "pr_auc",
    "f1_at_1pct_fpr": "f1_at_1pct_fpr",
    "f1_at_2pct_fpr": "f1_at_2pct_fpr",
    "threshold_at_1pct_fpr": "threshold_at_1pct_fpr",
    "train_examples": "train_examples",
    "hardware": "hardware",
    "device": "device",
    "dtype": "dtype",
    "early_stopped": "early_stopped",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare training runs by scanning model_card.json files."
    )
    parser.add_argument(
        "models_dir",
        nargs="?",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory to scan for model run subdirectories (default: ml/models).",
    )
    parser.add_argument(
        "--sort-by",
        default="pr_auc",
        choices=FIELD_ORDER,
        help="Field to sort by (default: pr_auc).",
    )
    parser.add_argument(
        "--output",
        default="table",
        choices=("table", "csv"),
        help="Output format (default: table).",
    )
    return parser.parse_args()


def dig(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_present(*values: Any) -> Any:
    for value in values:
        if not is_missing(value):
            return value
    return None


def load_model_card(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def infer_early_stopped(card: dict[str, Any]) -> bool | None:
    training = dig(card, "training")
    if not isinstance(training, dict):
        return None

    direct = as_bool(training.get("early_stopped"))
    if direct is not None:
        return direct

    epochs = training.get("epochs")
    best_epoch = training.get("best_epoch")
    if isinstance(best_epoch, (int, float)) and isinstance(epochs, (int, float)):
        return best_epoch < epochs

    return None


def extract_row(model_card_path: Path, card: dict[str, Any]) -> dict[str, Any]:
    output_dir = dig(card, "artifacts", "output_dir")
    run_dir = model_card_path.parent

    return {
        "run_name": run_dir.name,
        "output_dir": output_dir or str(run_dir),
        "dataset_version": first_present(
            dig(card, "dataset", "version"),
            dig(card, "resolved_config", "dataset", "version"),
        ),
        "epochs": first_present(
            dig(card, "training", "epochs"),
            dig(card, "resolved_config", "training", "epochs"),
        ),
        "batch_size": first_present(
            dig(card, "training", "batch_size"),
            dig(card, "resolved_config", "training", "batch_size"),
        ),
        "lr": first_present(
            dig(card, "training", "lr"),
            dig(card, "resolved_config", "training", "learning_rate"),
        ),
        "alpha": first_present(
            card.get("alpha"),
            dig(card, "resolved_config", "model", "distillation", "alpha"),
        ),
        "temperature": first_present(
            card.get("temperature"),
            dig(card, "resolved_config", "model", "distillation", "temperature"),
        ),
        "truncation_strategy": first_present(
            dig(card, "training", "truncation_strategy"),
            dig(card, "resolved_config", "model", "truncation_strategy"),
        ),
        "pr_auc": first_present(
            dig(card, "metrics_validation", "pr_auc"),
            dig(card, "metrics", "pr_auc"),
        ),
        "f1_at_1pct_fpr": first_present(
            dig(card, "metrics_validation", "f1_at_1pct_fpr"),
            dig(card, "metrics", "f1_at_1pct_fpr"),
        ),
        "f1_at_2pct_fpr": first_present(
            dig(card, "metrics_validation", "f1_at_2pct_fpr"),
            dig(card, "metrics", "f1_at_2pct_fpr"),
        ),
        "threshold_at_1pct_fpr": first_present(
            dig(card, "metrics_validation", "threshold_at_1pct_fpr"),
            dig(card, "metrics", "threshold_at_1pct_fpr"),
        ),
        "train_examples": first_present(
            dig(card, "dataset", "train_examples"),
            dig(card, "resolved_config", "dataset", "train_examples"),
        ),
        "hardware": dig(card, "training", "hardware"),
        "device": first_present(
            dig(card, "training", "device"),
            dig(card, "resolved_config", "runtime", "device"),
        ),
        "dtype": first_present(
            dig(card, "training", "dtype"),
            dig(card, "resolved_config", "runtime", "mixed_precision"),
        ),
        "early_stopped": infer_early_stopped(card),
    }


def sort_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    present = [row for row in rows if not is_missing(row.get(field))]
    missing = [row for row in rows if is_missing(row.get(field))]
    if not present:
        return missing

    sample = present[0].get(field)
    if isinstance(sample, str):
        present.sort(key=lambda row: str(row.get(field)).lower())
    else:
        present.sort(key=lambda row: row.get(field), reverse=True)
    return present + missing


def format_cell(value: Any) -> str:
    if is_missing(value):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def collect_rows(models_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_card_path in sorted(models_dir.rglob("model_card.json")):
        card = load_model_card(model_card_path)
        if card is None:
            continue
        rows.append(extract_row(model_card_path, card))
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [FIELD_LABELS[field] for field in FIELD_ORDER]
    rendered_rows = [
        [format_cell(row.get(field)) for field in FIELD_ORDER] for row in rows
    ]
    widths = [
        max(len(header), *(len(rendered_row[index]) for rendered_row in rendered_rows))
        for index, header in enumerate(headers)
    ]

    def render_line(values: list[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    print(render_line(headers))
    print(render_line(["-" * width for width in widths]))
    for rendered_row in rendered_rows:
        print(render_line(rendered_row))


def print_csv(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELD_ORDER, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: "" if is_missing(row.get(field)) else row.get(field)
                for field in FIELD_ORDER
            }
        )


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir)
    rows = sort_rows(collect_rows(models_dir), args.sort_by)

    if args.output == "csv":
        print_csv(rows)
    elif rows:
        print_table(rows)
    else:
        print(f"No model_card.json files found under {models_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
