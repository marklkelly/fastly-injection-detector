# ml/cloud — Vertex AI training

Scripts for submitting and monitoring training jobs on Google Cloud Vertex AI.

## Files

| File | Purpose |
|------|---------|
| `submit.py` | Submit a training job from a YAML config |
| `job_status.py` | List and monitor jobs, tail logs |
| `entrypoint.py` | Container entrypoint (runs inside Vertex AI) |
| `precache_models.py` | Pre-download HuggingFace models to GCS before training |
| `Dockerfile` | Trainer container definition |
| `config.yaml` | Reference config (deprecated: pi_mix_v1 with wildjailbreak) |
| `config_injection_only.yaml` | Config for injection model (pi_mix_v1_injection_only) |
| `config_jailbreak.yaml` | Config for jailbreak model (pi_mix_jailbreak_v1) |

## Submitting a job

Fill in project-specific values in the config file, then:

```bash
python3 ml/cloud/submit.py --config ml/cloud/config_injection_only.yaml
python3 ml/cloud/submit.py --config ml/cloud/config_injection_only.yaml --dry-run
```

The config requires:
- `project_id`, `region`, `staging_bucket`, `service_account`
- `container.image_uri` — pre-built trainer image in Artifact Registry
- `data.train_uri`, `data.val_uri`, `data.output_uri` — GCS paths

## Building the trainer image

Use Cloud Build (write the YAML to a temp file — stdin does not work):

```bash
gcloud builds submit --project <project> --config /tmp/trainer-cloudbuild.yaml .
```

The base image is `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`. Do **not** reinstall `torch` in the Dockerfile — the base image provides a CUDA-enabled build and reinstalling from PyPI overwrites it with a CPU-only version.

## Monitoring

```bash
# List recent jobs
python3 ml/cloud/job_status.py list --project <project>

# Watch a running job
python3 ml/cloud/job_status.py wait <job-id> --project <project>

# Tail logs
python3 ml/cloud/job_status.py logs <job-id> --project <project>
```

## Config notes

- `config.yaml` is kept for reference only. It uses `pi_mix_v1` which includes
  `allenai/wildjailbreak` examples that degraded injection recall by ~20pp.
- Use `config_injection_only.yaml` for the injection model (cleaned dataset).
- Use `config_jailbreak.yaml` for the jailbreak model (no teacher distillation).
