from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from finetune_utils import read_jsonl, validate_finetune_example


DEFAULT_TRAIN_PATH = Path("datasets/finetune/train.jsonl")
DEFAULT_VAL_PATH = Path("datasets/finetune/val.jsonl")
DEFAULT_METADATA_PATH = Path("datasets/finetune/openai_finetune_job.json")


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")
    return api_key


def to_plain_object(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: to_plain_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_object(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: to_plain_object(item) for key, item in vars(value).items()}
    return value


def validate_finetune_jsonl(path: Path) -> int:
    records = read_jsonl(path)
    for index, record in enumerate(records, start=1):
        try:
            validate_finetune_example(record)
        except ValueError as exc:
            raise ValueError(f"{path}:{index} failed validation: {exc}") from exc
    return len(records)


def upload_file(client: Any, path: Path) -> Any:
    with path.open("rb") as file_obj:
        return client.files.create(file=file_obj, purpose="fine-tune")


def start_finetune(
    client: Any,
    train_path: Path,
    val_path: Path,
    model: str,
    output_path: Path,
    suffix: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    train_count = validate_finetune_jsonl(train_path)
    val_count = validate_finetune_jsonl(val_path)

    train_file = upload_file(client, train_path)
    val_file = upload_file(client, val_path)

    job_kwargs: dict[str, Any] = {
        "model": model,
        "training_file": train_file.id,
        "validation_file": val_file.id,
    }
    if suffix:
        job_kwargs["suffix"] = suffix
    if seed is not None:
        job_kwargs["seed"] = seed

    job = client.fine_tuning.jobs.create(**job_kwargs)

    metadata = {
        "model": model,
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "train_examples": train_count,
        "validation_examples": val_count,
        "training_file_id": train_file.id,
        "validation_file_id": val_file.id,
        "fine_tuning_job_id": job.id,
        "status": getattr(job, "status", None),
        "fine_tuned_model": getattr(job, "fine_tuned_model", None),
        "training_file": to_plain_object(train_file),
        "validation_file": to_plain_object(val_file),
        "job": to_plain_object(job),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload JSONL files and start an OpenAI fine-tuning job.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH, help="Training JSONL file.")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH, help="Validation JSONL file.")
    parser.add_argument("--model", required=True, help="Base model to fine-tune.")
    parser.add_argument("--output", type=Path, default=DEFAULT_METADATA_PATH, help="Job metadata JSON output path.")
    parser.add_argument("--suffix", default=None, help="Optional fine-tuned model suffix.")
    parser.add_argument("--seed", type=int, default=None, help="Optional fine-tuning seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_api_key()

    from openai import OpenAI

    metadata = start_finetune(
        client=OpenAI(),
        train_path=args.train,
        val_path=args.val,
        model=args.model,
        output_path=args.output,
        suffix=args.suffix,
        seed=args.seed,
    )

    print("OpenAI fine-tuning job started.")
    print(f"Training file ID: {metadata['training_file_id']}")
    print(f"Validation file ID: {metadata['validation_file_id']}")
    print(f"Fine-tuning job ID: {metadata['fine_tuning_job_id']}")
    print(f"Status: {metadata['status']}")
    print(f"Metadata saved to: {args.output}")


if __name__ == "__main__":
    main()
