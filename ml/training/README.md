# Training

`train_cls.py` is the canonical supported training entrypoint. Runtime behavior is resolved from `ml/training/config/model.yaml`, then optionally overridden by explicit CLI flags.

## Install

```bash
pip install -r ml/training/requirements.txt
```

Or with `uv`:

```bash
uv pip install -r ml/training/requirements.txt
```

## Canonical Command

Run from the repository root:

```bash
python ml/training/train_cls.py --config ml/training/config/model.yaml
```

## Common Overrides

CLI flags only override the fields you set explicitly. Everything else comes from the YAML file.

```bash
python ml/training/train_cls.py \
  --config ml/training/config/model.yaml \
  --output_dir /tmp/train-run \
  --epochs 8 \
  --warmup_steps 200 \
  --truncation-strategy head_tail \
  --best-model-metric f1_at_1pct_fpr \
  --no-grad-checkpointing
```

Legacy aliases still work:

- `--teacher` -> `model.teacher_model`
- `--alpha` -> `model.distillation.alpha`
- `--temperature` -> `model.distillation.temperature`
- `--max_length` -> `model.max_length`
- `--bf16` / `--fp16` -> `runtime.mixed_precision`

## Quick Test

```bash
python ml/training/train_cls.py \
  --config ml/training/config/model.yaml \
  --quick_test \
  --output_dir /tmp/train-quick
```

This uses the config-backed dataset paths, truncates the run to 2 steps, and still emits the full artifact set.

## Config Surface

`ml/training/config/model.yaml` includes:

- `model`: backbone, max length, truncation strategy, label map, teacher, distillation settings
- `dataset`: train/validation/test paths plus labels metadata
- `training`: epochs, scheduler, warmup, eval/save cadence, early stopping, batching, gradient accumulation, checkpointing
- `runtime`: device, mixed precision, dataloader workers
- `outputs`: output directory and artifact toggles
- `evaluation`: optional OOD path and prior-adjusted metrics settings

Unknown YAML keys and invalid enum values fail fast at startup.

## Defaults

- Scheduler: `cosine`
- Warmup: `warmup_ratio: 0.05` unless `warmup_steps` is set
- Student truncation: `head` by default, with `tail` and `head_tail` available for long examples
- Teacher truncation: standard head truncation, with `model.distillation.teacher_max_length: 256` by default when a teacher is enabled
- Teacher cache: `model.distillation.cache_teacher_logits: false`; enable it with `--cache-teacher-logits` to store raw teacher logits under `ml/training/.cache/teacher_logits/`
- Eval/save cadence: step-based when either `eval_steps` or `save_steps` is set; epoch-based when both are omitted
- Epoch budget: `6`
- Early stopping: enabled with `early_stopping_patience: 3`
- Best model metric: `pr_auc`

## Long Inputs And Slice Metrics

- `model.truncation_strategy` controls deterministic student truncation for examples whose tokenized length exceeds the 128-token student contract:
  - `head`: keep the first tokens (matches the pre-Phase-2 behavior)
  - `tail`: keep the last tokens
  - `head_tail`: keep the first half and last half
- Training uses dynamic padding to the longest student sequence in each batch. Validation and test stay fixed at the student max length for deterministic metrics.
- When distillation is enabled without caching, teacher inputs are padded separately using the teacher tokenizer. When caching is enabled, batches carry fixed `teacher_logits` tensors instead of teacher token IDs.
- `original_token_length` is recorded before student truncation so slice metrics can distinguish short and long examples consistently.
- `dataset.test_path` optionally loads a test split for post-training slice reporting.
- `eval_metrics.json` now includes `validation_slices` and, when a test split is configured, `test_slices`.
- Slice reports are broken out by `source` and by length bucket (`<=128`, `>128`) using the overall validation 1% FPR threshold.

## Output Files

Training writes:

```text
output_dir/
├── model.safetensors
├── config.json
├── tokenizer_config.json
├── tokenizer.json
├── labels.json
├── eval_metrics.json
├── calibrated_thresholds.json
├── model_card.json
└── edge_export/
    ├── model.safetensors
    └── tokenizer.json
```

`model_card.json` includes the full `resolved_config` that was used for the run.
