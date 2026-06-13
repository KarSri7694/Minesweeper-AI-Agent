from __future__ import annotations

import argparse
import random
from pathlib import Path

from finetune_utils import read_jsonl, validate_finetune_example, write_jsonl


def split_records(
    records: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("need at least two records to create train and validation splits")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    val_count = round(len(shuffled) * val_ratio)
    val_count = max(1, min(len(shuffled) - 1, val_count))

    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def split_file(
    input_path: Path,
    train_output: Path,
    val_output: Path,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    records = read_jsonl(input_path)
    for index, record in enumerate(records, start=1):
        try:
            validate_finetune_example(record)
        except ValueError as exc:
            raise ValueError(f"{input_path}:{index} failed validation: {exc}") from exc

    train_records, val_records = split_records(records, val_ratio=val_ratio, seed=seed)
    write_jsonl(train_output, train_records)
    write_jsonl(val_output, val_records)
    return len(train_records), len(val_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split chat fine-tuning JSONL into train and validation files.")
    parser.add_argument("--input", required=True, type=Path, help="Input fine-tuning JSONL path.")
    parser.add_argument("--train-output", required=True, type=Path, help="Output path for training JSONL.")
    parser.add_argument("--val-output", required=True, type=Path, help="Output path for validation JSONL.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction between 0 and 1.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count, val_count = split_file(
        input_path=args.input,
        train_output=args.train_output,
        val_output=args.val_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("Fine-tuning split complete.")
    print(f"Train count: {train_count}")
    print(f"Validation count: {val_count}")
    print(f"Train output: {args.train_output}")
    print(f"Validation output: {args.val_output}")


if __name__ == "__main__":
    main()
