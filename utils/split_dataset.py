import argparse
import os
import random
from typing import List


def split_dataset(
    input_path: str,
    output_dir: str | None = None,
    train_ratio: float = 0.7,
    seed: int | None = None,
) -> tuple[str, str, int, int]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."

    with open(input_path, "r", encoding="utf-8") as input_file:
        rows: List[str] = [line for line in input_file if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(rows)

    split_index = int(len(rows) * train_ratio)
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]

    input_name = os.path.basename(input_path)
    input_stem, input_ext = os.path.splitext(input_name)
    if not input_ext:
        input_ext = ".jsonl"

    train_path = os.path.join(output_dir, f"{input_stem}_train{input_ext}")
    test_path = os.path.join(output_dir, f"{input_stem}_test{input_ext}")

    with open(train_path, "w", encoding="utf-8") as train_file:
        train_file.writelines(train_rows)

    with open(test_path, "w", encoding="utf-8") as test_file:
        test_file.writelines(test_rows)

    return train_path, test_path, len(train_rows), len(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a JSONL dataset into train and test files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the source JSONL dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the split output files. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of rows to place in the train split. Default is 0.7.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic shuffling before splitting.",
    )
    args = parser.parse_args()

    train_path, test_path, train_count, test_count = split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(f"Train dataset written to: {train_path}")
    print(f"Test dataset written to: {test_path}")
    print(f"Train rows: {train_count}")
    print(f"Test rows: {test_count}")


if __name__ == "__main__":
    main()
