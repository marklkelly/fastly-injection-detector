#!/usr/bin/env python3
"""Submit a training job to Vertex AI.

Usage:
    python ml/cloud/submit.py --config ml/cloud/config.yaml [--dry-run]
"""
import argparse
import os
import yaml
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def _noop():
    """No-op context manager used when experiment tracking is not configured."""
    yield


def _read_hf_token() -> str | None:
    """Read HuggingFace token from standard local cache location."""
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        token = open(token_path).read().strip()
        return token or None
    return None


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI training job")
    parser.add_argument("--config", default="ml/cloud/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    job_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    display_name = f"{cfg['job']['display_name_prefix']}-{job_ts}"

    print(f"Submitting job: {display_name}")
    if args.dry_run:
        print("[dry-run] would submit to Vertex AI with config:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    from google.cloud import aiplatform

    monitoring = cfg.get("monitoring", {})
    experiment_name = monitoring.get("experiment_name")
    tb_resource = monitoring.get("tensorboard_resource_name")

    aiplatform.init(
        project=cfg["project_id"],
        location=cfg["region"],
        staging_bucket=cfg["staging_bucket"],
        experiment=experiment_name,
        experiment_tensorboard=tb_resource,
    )

    output_uri = cfg["data"]["output_uri"].replace("${JOB_TS}", job_ts)
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=cfg["container"]["image_uri"],
    )

    # Pass HuggingFace token so the training container can access gated repos
    # (e.g. protectai/deberta-v3-small-prompt-injection-v2 teacher model).
    env_vars = {}
    hf_token = _read_hf_token()
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
        env_vars["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HuggingFace token found — will be passed to training container.")
    else:
        print("Warning: no HuggingFace token found; gated models may fail to download.")

    # Pass experiment context into the container so it can log final metrics.
    env_vars["RUN_NAME"] = job_ts
    if experiment_name:
        env_vars["EXPERIMENT_NAME"] = experiment_name

    if experiment_name:
        print(f"Vertex AI Experiment: {experiment_name} / run: {job_ts}")

    with aiplatform.start_run(job_ts) if experiment_name else _noop():
        if experiment_name:
            aiplatform.log_params({
                "machine_type": cfg["job"]["machine_type"],
                "accelerator_type": cfg["job"].get("accelerator_type", "none"),
                "accelerator_count": cfg["job"].get("accelerator_count", 0),
                "model_config": cfg["training"]["model_config"],
                "train_uri": cfg["data"]["train_uri"],
            })

        job.run(
            args=[
                "--model-config", cfg["training"]["model_config"],
                "--train-uri", cfg["data"]["train_uri"],
                "--val-uri", cfg["data"]["val_uri"],
                "--output-uri", output_uri,
            ],
            replica_count=cfg["job"]["replica_count"],
            machine_type=cfg["job"]["machine_type"],
            accelerator_type=cfg["job"].get("accelerator_type"),
            accelerator_count=cfg["job"].get("accelerator_count", 0),
            base_output_dir=output_uri,
            service_account=cfg.get("service_account"),
            environment_variables=env_vars or None,
            tensorboard=tb_resource,
        )


if __name__ == "__main__":
    main()
