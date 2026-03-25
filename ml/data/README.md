# ml/data — Dataset Management

Entry point: `python ml/data/build.py --recipe ml/data/recipes/pi_mix_v3.yaml --output ml/data/versions/pi_mix_v3`

## Recipes
- `recipes/pi_mix_v1.yaml` — v1 policy: matches original hardcoded defaults; sources pinned to 2026-03-12 revisions
- `recipes/pi_mix_v2.yaml` — v2 policy: reduced jayavibhav cap, adds deepset OOD source, frozen source policy
- `recipes/pi_mix_v3.yaml` — v3 policy: adds neuralchemy/Prompt-injection-dataset and wambosec/prompt-injections-subtle; removes Harelix and markush1
- `recipes/legacy-v0-sources.md` — provenance record for the original 2025-09-18 `data_out/` build (pre-recipe)

## Dataset Versioning

Versions live in `ml/data/versions/pi_mix_vN/`. Each version directory contains:

| File | Description |
|------|-------------|
| `train.jsonl` | Training split |
| `val.jsonl` | Validation split |
| `test.jsonl` | In-domain test split |
| `test_ood.jsonl` | Out-of-domain test split |
| `labels.json` | Label mapping |
| `manifest.json` | Build metadata and row counts |

To build a version:

```bash
python ml/data/build.py --recipe ml/data/recipes/pi_mix_v2.yaml --output ml/data/versions/pi_mix_v2
```

**Current active version:** `pi_mix_v3` (see `ml/training/config/model.yaml`)

## Versions
- `versions/pi_mix_v1/` — Assembled dataset (train/val/test splits)
- `versions/pi_mix_v2/` — Reduced jayavibhav cap, deepset OOD
- `versions/pi_mix_v3/` — Adds neuralchemy and wambosec sources

## Adding a New Source

Before adding any source to a recipe:
- [ ] Run `python ml/data/audit_source.py --repo <name>` and review output
- [ ] Schema documented (columns, dtypes)
- [ ] Label semantics reviewed manually
- [ ] License/terms approved for training use
- [ ] Revision pinned in recipe YAML
- [ ] Loader test added to `ml/data/tests/test_loaders.py`
- [ ] Split invariants still pass: `pytest ml/data/tests/`
- [ ] Manifest contract still passes: `pytest ml/data/tests/`

## Source scripts
- `build.py` — Unified dataset builder

## CLI reference

```
python ml/data/build.py [OPTIONS]

Options:
  --recipe PATH           YAML recipe (seed, min_chars, max_chars, balance, per-source caps).
                          Explicit CLI flags override recipe values.
  --output PATH           Output directory (alias for --out-dir).
  --out-dir PATH          Output directory (default: data/pi_mix_v1).
  --seed INT              Random seed (default: 42, or recipe value).
  --min-chars INT         Minimum text length in characters (default: 8).
  --max-chars INT         Maximum text length in characters (default: 4000).
  --balance {off,downsample,upsample}
                          Class balancing strategy (default: downsample).
  --limit-per-source INT  Override cap for every source (0 = use recipe/default caps).
  --include-wildjailbreak Include allenai/wildjailbreak (requires HF terms acceptance).
                          Superseded by recipe sources list when --recipe is given.
  --save-hf               Also save the combined Dataset to disk.
```

## Recipe YAML keys

```yaml
version: pi_mix_v1
seed: 42
min_chars: 8
max_chars: 4000
balance: downsample

dedup:
  method: sha256_exact   # informational; exact dedup is always applied

sources:
  - repo: <hf-owner/dataset>
    revision: <commit-sha>   # pinned HF revision
    cap: <int>               # max rows from this source (before --limit-per-source)
    optional: true           # if true, load errors do not abort the build
    fallback_repo: <repo>    # fallback when optional source is unavailable
    gated: true              # skipped unless HF access is granted
    ood_eval: true           # informational: used only in OOD eval splits (Phase 2+)
    pending_license_review: true  # skipped until license cleared
```

