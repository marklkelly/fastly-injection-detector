#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import JobState
from google.protobuf import json_format

DEFAULT_PROJECT = ""
DEFAULT_REGION = "us-central1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
DEFAULT_LIST_LIMIT = 10
DEFAULT_LOG_LIMIT = 50
WAIT_POLL_SECONDS = 30

SUCCESS_STATES = {"JOB_STATE_SUCCEEDED"}
FAILURE_STATES = {
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_FAILED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}
TERMINAL_STATES = SUCCESS_STATES | FAILURE_STATES


def load_cli_defaults(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    defaults = {
        "project": DEFAULT_PROJECT,
        "region": DEFAULT_REGION,
    }
    if not config_path.exists():
        return defaults

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    project = config.get("project_id")
    region = config.get("region")
    if isinstance(project, str) and project.strip():
        defaults["project"] = project.strip()
    if isinstance(region, str) and region.strip():
        defaults["region"] = region.strip()
    return defaults


def normalize_custom_job_name(job_id: str, project: str, region: str) -> str:
    normalized = job_id.strip()
    if not normalized:
        raise ValueError("job_id must not be empty")
    if normalized.startswith("projects/"):
        return normalized
    return f"projects/{project}/locations/{region}/customJobs/{normalized}"


def build_logs_command(
    project: str,
    region: str,
    job_id: str,
    limit: int,
) -> list[str]:
    job_name = normalize_custom_job_name(job_id, project=project, region=region)
    filter_parts = [
        'resource.type="aiplatform.googleapis.com/CustomJob"',
        f'resource.labels.job_id="{job_name.rsplit("/", maxsplit=1)[-1]}"',
        "("
        f'textPayload:"{job_name}" OR '
        f'textPayload:"customJobs/{job_name.rsplit("/", maxsplit=1)[-1]}" OR '
        f'jsonPayload.job:"{job_name}" OR '
        f'protoPayload.resourceName:"{job_name}"'
        ")",
    ]
    log_filter = " AND ".join(filter_parts)
    return [
        "gcloud",
        "logging",
        "read",
        log_filter,
        f"--project={project}",
        f"--limit={limit}",
        "--order=desc",
        "--format=json",
        "--freshness=30d",
    ]


def format_duration(
    start_time: Any,
    end_time: Any,
    *,
    now: datetime | None = None,
) -> str:
    start = _coerce_datetime(start_time)
    if start is None:
        return "-"

    resolved_now = now or datetime.now(UTC)
    end = _coerce_datetime(end_time) or resolved_now
    total_seconds = max(int((end - start).total_seconds()), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if hasattr(value, "seconds") and hasattr(value, "nanos"):
        if value.seconds == 0 and value.nanos == 0:
            return None
    if hasattr(value, "ToDatetime"):
        return value.ToDatetime(tzinfo=UTC)
    return None


def _format_timestamp(value: Any) -> str:
    resolved = _coerce_datetime(value)
    if resolved is None:
        return "-"
    return resolved.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_id_from_name(job_name: str) -> str:
    return job_name.rsplit("/", maxsplit=1)[-1]


def _job_state_name(state: Any) -> str:
    if hasattr(state, "name") and state.name:
        return str(state.name)
    if state is None:
        return "UNKNOWN"
    return JobState(state).name


def _job_duration_from_proto(job_proto: Any, *, now: datetime | None = None) -> str:
    start_time = getattr(job_proto, "start_time", None) or getattr(
        job_proto,
        "create_time",
        None,
    )
    return format_duration(start_time, getattr(job_proto, "end_time", None), now=now)


def _job_output_uri(job_proto: Any) -> str:
    base_output = getattr(job_proto.job_spec, "base_output_directory", None)
    output_uri = getattr(base_output, "output_uri_prefix", "")
    return output_uri or "-"


def _job_machine_type(job_proto: Any) -> str:
    worker_pool_specs = getattr(job_proto.job_spec, "worker_pool_specs", None) or []
    if not worker_pool_specs:
        return "-"
    machine_spec = getattr(worker_pool_specs[0], "machine_spec", None)
    if machine_spec is None:
        return "-"
    return getattr(machine_spec, "machine_type", "") or "-"


def _job_accelerator(job_proto: Any) -> str:
    worker_pool_specs = getattr(job_proto.job_spec, "worker_pool_specs", None) or []
    if not worker_pool_specs:
        return "-"
    machine_spec = getattr(worker_pool_specs[0], "machine_spec", None)
    if machine_spec is None:
        return "-"

    accelerator_type = getattr(machine_spec, "accelerator_type", None)
    accelerator_count = getattr(machine_spec, "accelerator_count", 0) or 0
    accelerator_name = getattr(accelerator_type, "name", str(accelerator_type))
    if (
        accelerator_name in {"0", "ACCELERATOR_TYPE_UNSPECIFIED"}
        or accelerator_count < 1
    ):
        return "-"
    return f"{accelerator_name} x{accelerator_count}"


def _job_error_message(job_proto: Any) -> str:
    error = getattr(job_proto, "error", None)
    message = getattr(error, "message", "")
    return message or "-"


def _job_proto_to_dict(job_proto: Any) -> dict[str, Any]:
    return json_format.MessageToDict(
        job_proto._pb,
        preserving_proto_field_name=True,
    )


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return f"{value[: width - 3]}..."


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    for row in rows:
        lines.append(
            "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        )
    return "\n".join(lines)


def _extract_log_message(entry: dict[str, Any]) -> str:
    for key in ("textPayload", "protoPayload", "jsonPayload"):
        payload = entry.get(key)
        if payload is None:
            continue
        message = _payload_to_message(payload)
        if message:
            return message
    return json.dumps(entry, sort_keys=True)


def _payload_to_message(payload: Any) -> str:
    if isinstance(payload, str):
        return " ".join(payload.splitlines()).strip()
    if isinstance(payload, dict):
        for key in ("message", "msg", "text", "@message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.splitlines()).strip()
        return json.dumps(payload, sort_keys=True)
    return str(payload)


def _init_vertex(project: str, region: str) -> None:
    aiplatform.init(project=project, location=region)


def _get_job(job_id: str, project: str, region: str) -> aiplatform.CustomJob:
    _init_vertex(project, region)
    job_name = normalize_custom_job_name(job_id, project=project, region=region)
    return aiplatform.CustomJob.get(job_name, project=project, location=region)


def _command_list(args: argparse.Namespace) -> int:
    _init_vertex(args.project, args.region)
    jobs = aiplatform.CustomJob.list(
        order_by="create_time desc",
        project=args.project,
        location=args.region,
    )
    selected_jobs = jobs[: args.limit]
    if not selected_jobs:
        print("No custom jobs found.")
        return 0

    rows = []
    for job in selected_jobs:
        job_proto = job._gca_resource
        rows.append(
            [
                _job_id_from_name(job.resource_name),
                _truncate(job.display_name, 40),
                _job_state_name(job_proto.state),
                _format_timestamp(job_proto.create_time),
                _job_duration_from_proto(job_proto),
            ]
        )
    print(
        _render_table(
            ["job_id", "display_name", "state", "create_time", "duration"],
            rows,
        )
    )
    return 0


def _command_status(args: argparse.Namespace) -> int:
    job = _get_job(args.job_id, project=args.project, region=args.region)
    job_proto = job._gca_resource
    details = {
        "job_id": _job_id_from_name(job.resource_name),
        "name": job.resource_name,
        "display_name": job.display_name,
        "state": _job_state_name(job_proto.state),
        "create_time": _format_timestamp(job_proto.create_time),
        "start_time": _format_timestamp(job_proto.start_time),
        "end_time": _format_timestamp(job_proto.end_time),
        "duration": _job_duration_from_proto(job_proto),
        "machine_type": _job_machine_type(job_proto),
        "accelerator": _job_accelerator(job_proto),
        "output_uri": _job_output_uri(job_proto),
        "error_message": _job_error_message(job_proto),
    }
    for key, value in details.items():
        print(f"{key}: {value}")
    print("job_proto:")
    print(json.dumps(_job_proto_to_dict(job_proto), indent=2, sort_keys=True))
    return 0


def _command_logs(args: argparse.Namespace) -> int:
    command = build_logs_command(
        project=args.project,
        region=args.region,
        job_id=args.job_id,
        limit=args.limit,
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        print("gcloud CLI was not found on PATH.", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(result.stderr.strip() or "Failed to read logs.", file=sys.stderr)
        return result.returncode

    entries = json.loads(result.stdout or "[]")
    if not entries:
        print("No log entries found.")
        return 0

    for entry in entries:
        timestamp = entry.get("timestamp", "-")
        severity = entry.get("severity", "DEFAULT")
        message = _extract_log_message(entry)
        print(f"{timestamp} {severity:<8} {message}")
    return 0


def _command_cancel(args: argparse.Namespace) -> int:
    job = _get_job(args.job_id, project=args.project, region=args.region)
    job_proto = job._gca_resource
    state = _job_state_name(job_proto.state)

    print(f"name: {job.resource_name}")
    print(f"display_name: {job.display_name}")
    print(f"state: {state}")
    if state in TERMINAL_STATES:
        print("Job is already in a terminal state; nothing to cancel.", file=sys.stderr)
        return 1

    response = input("Cancel this job? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancellation aborted.")
        return 0

    job.cancel()
    print(f"Cancellation requested for {job.resource_name}.")
    return 0


def _command_wait(args: argparse.Namespace) -> int:
    try:
        while True:
            job = _get_job(args.job_id, project=args.project, region=args.region)
            job_proto = job._gca_resource
            state = _job_state_name(job_proto.state)
            now = datetime.now(UTC)
            print(
                f"{_format_timestamp(now)} "
                f"job_id={_job_id_from_name(job.resource_name)} "
                f"state={state} "
                f"duration={_job_duration_from_proto(job_proto, now=now)}"
            )

            if state in SUCCESS_STATES:
                return 0
            if state in FAILURE_STATES:
                error_message = _job_error_message(job_proto)
                if error_message != "-":
                    print(f"error_message: {error_message}", file=sys.stderr)
                return 1

            time.sleep(WAIT_POLL_SECONDS)
    except KeyboardInterrupt:
        print("Interrupted while waiting for job completion.", file=sys.stderr)
        return 130


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = load_cli_defaults()

    parser = argparse.ArgumentParser(
        description="Inspect and manage Vertex AI custom training jobs.",
    )
    parser.add_argument("--project", default=defaults["project"])
    parser.add_argument("--region", default=defaults["region"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List recent custom jobs sorted newest first.",
    )
    list_parser.add_argument("--limit", type=int, default=DEFAULT_LIST_LIMIT)
    list_parser.set_defaults(func=_command_list)

    status_parser = subparsers.add_parser(
        "status",
        help="Show detailed status for a custom job.",
    )
    status_parser.add_argument("job_id")
    status_parser.set_defaults(func=_command_status)

    logs_parser = subparsers.add_parser(
        "logs",
        help="Read recent logs for a custom job.",
    )
    logs_parser.add_argument("job_id")
    logs_parser.add_argument("--limit", type=int, default=DEFAULT_LOG_LIMIT)
    logs_parser.set_defaults(func=_command_logs)

    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel a running custom job.",
    )
    cancel_parser.add_argument("job_id")
    cancel_parser.set_defaults(func=_command_cancel)

    wait_parser = subparsers.add_parser(
        "wait",
        help="Poll a custom job until it reaches a terminal state.",
    )
    wait_parser.add_argument("job_id")
    wait_parser.set_defaults(func=_command_wait)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
