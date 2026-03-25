from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from ml.cloud import job_status


def test_load_cli_defaults_reads_project_and_region(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_id": "test-project",
                "region": "europe-west4",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    defaults = job_status.load_cli_defaults(config_path)

    assert defaults["project"] == "test-project"
    assert defaults["region"] == "europe-west4"


def test_load_cli_defaults_falls_back_when_config_is_missing(tmp_path: Path) -> None:
    defaults = job_status.load_cli_defaults(tmp_path / "missing.yaml")

    assert defaults["project"] == job_status.DEFAULT_PROJECT
    assert defaults["region"] == job_status.DEFAULT_REGION


def test_normalize_custom_job_name_expands_numeric_id() -> None:
    name = job_status.normalize_custom_job_name(
        "4416686651988770816",
        project="test-project",
        region="us-central1",
    )

    assert (
        name
        == "projects/test-project/locations/us-central1/customJobs/4416686651988770816"
    )


def test_build_logs_command_filters_for_custom_job_resource() -> None:
    command = job_status.build_logs_command(
        project="test-project",
        region="us-central1",
        job_id="4416686651988770816",
        limit=25,
    )

    assert command[:3] == ["gcloud", "logging", "read"]
    assert "--project=test-project" in command
    assert "--limit=25" in command
    assert "--order=desc" in command
    assert "--format=json" in command
    assert 'resource.type="aiplatform.googleapis.com/CustomJob"' in command[3]
    assert 'resource.labels.job_id="4416686651988770816"' in command[3]


def test_format_duration_handles_running_jobs() -> None:
    start_time = datetime(2026, 3, 13, 10, 0, tzinfo=UTC)
    now = datetime(2026, 3, 13, 12, 5, 42, tzinfo=UTC)

    assert job_status.format_duration(start_time, None, now=now) == "2h05m42s"
