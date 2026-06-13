from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_METADATA_PATH = Path("datasets/finetune/openai_finetune_job.json")


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")


def read_job_id(metadata_path: Path) -> str:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    job_id = metadata.get("fine_tuning_job_id") or metadata.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"No fine-tuning job ID found in {metadata_path}")
    return job_id


def get_error_message(job: Any) -> str | None:
    error = getattr(job, "error", None)
    if error is None:
        return None
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    message = getattr(error, "message", None)
    if message:
        return str(message)
    return str(error)


def retrieve_job_status(client: Any, job_id: str) -> dict[str, Any]:
    job = client.fine_tuning.jobs.retrieve(job_id)
    return {
        "id": getattr(job, "id", job_id),
        "status": getattr(job, "status", None),
        "fine_tuned_model": getattr(job, "fine_tuned_model", None),
        "error": get_error_message(job),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check an OpenAI fine-tuning job status.")
    parser.add_argument("--job-id", default=None, help="Fine-tuning job ID. Overrides metadata file.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="Saved job metadata JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_api_key()

    from openai import OpenAI

    job_id = args.job_id or read_job_id(args.metadata)
    status = retrieve_job_status(OpenAI(), job_id)

    print(f"Fine-tuning job ID: {status['id']}")
    print(f"Status: {status['status']}")
    print(f"Fine-tuned model: {status['fine_tuned_model'] or 'not available yet'}")
    print(f"Error: {status['error'] or 'none'}")


if __name__ == "__main__":
    main()
