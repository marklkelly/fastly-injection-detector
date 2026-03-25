# Experiment Tracking

This project uses each run's `model_card.json` as the source of truth. Every training run already writes the resolved config, validation metrics, dataset metadata, and runtime details into the output directory, so the lightest reliable tracking approach is to keep those run directories and scan them when you want a comparison view. That avoids a second log file drifting out of sync with the actual artifact.

Recommended naming is:

```text
ml/models/<artifact_name>-<YYYYMMDD>-<short-description>/
```

Examples: `ml/models/bert-tiny-pi-v1-20260313-higher-alpha/` or `ml/models/bert-tiny-pi-v1-20260313-new-dataset-v3/`. The directory name becomes the human-facing run label, while `model_card.json` preserves the exact config and final metrics needed for reproduction.

Use `ml/training/compare_runs.py` to compare historical runs side by side:

```bash
python ml/training/compare_runs.py
python ml/training/compare_runs.py --sort-by f1_at_1pct_fpr
python ml/training/compare_runs.py --output csv > runs.csv
```

The script scans `ml/models/` for `model_card.json` files and prints the key fields that matter for model selection and reproducibility: dataset version, epochs, batch size, learning rate, alpha, temperature, truncation strategy, PR-AUC, F1 at fixed FPR points, threshold, train example count, and runtime information such as hardware, device, and dtype.

For run intent, use a lightweight annotation pattern alongside the run itself. In Vertex AI submissions, add a human-only `notes` entry in `ml/cloud/config.yaml` next to the override block you are changing, for example `notes: testing higher alpha on pi_mix_v3`. That note is for operators and code review, not for `train_cls.py`. For ad hoc local runs, add a shell comment directly above the command, such as `# run: compare head_tail truncation against baseline`. The run description should also be reflected in the output directory name so it remains visible after artifacts are downloaded into `ml/models/`.

