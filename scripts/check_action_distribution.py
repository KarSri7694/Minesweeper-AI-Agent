from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from finetune_utils import get_message, parse_assistant_action, read_jsonl


BOARD_SIZE_RE = re.compile(r"Board size:\s*(\d+)\s*rows\s*x\s*(\d+)\s*columns")


def extract_board_size(user_message: str) -> tuple[int, int] | None:
    match = BOARD_SIZE_RE.search(user_message)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def analyze_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    actions: Counter[str] = Counter()
    board_sizes: Counter[tuple[int, int] | str] = Counter()
    x_counts: Counter[int] = Counter()
    y_counts: Counter[int] = Counter()
    coordinates: Counter[tuple[int, int]] = Counter()
    invalid_assistant_json = 0

    for example in examples:
        try:
            user_message = get_message(example, "user")
            board_size = extract_board_size(user_message)
            board_sizes[board_size if board_size is not None else "unknown"] += 1
        except ValueError:
            board_sizes["unknown"] += 1

        try:
            action = parse_assistant_action(example)
        except ValueError:
            invalid_assistant_json += 1
            continue

        action_name = str(action["action"])
        x = int(action["x"])
        y = int(action["y"])

        actions[action_name] += 1
        x_counts[x] += 1
        y_counts[y] += 1
        coordinates[(x, y)] += 1

    return {
        "total_examples": len(examples),
        "actions": actions,
        "board_sizes": board_sizes,
        "unique_coordinates": len(coordinates),
        "x_counts": x_counts,
        "y_counts": y_counts,
        "invalid_assistant_json": invalid_assistant_json,
    }


def format_counter(counter: Counter) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items(), key=lambda item: item[0]))


def print_report(stats: dict[str, Any]) -> None:
    actions = stats["actions"]
    print("Fine-tuning action distribution")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Reveal actions: {actions.get('reveal', 0)}")
    print(f"Flag actions: {actions.get('flag', 0)}")
    print(f"Board sizes: {format_counter(stats['board_sizes'])}")
    print(f"Unique coordinates: {stats['unique_coordinates']}")
    print(f"X distribution: {format_counter(stats['x_counts'])}")
    print(f"Y distribution: {format_counter(stats['y_counts'])}")
    print(f"Invalid assistant JSON responses: {stats['invalid_assistant_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report action and coordinate distribution for fine-tuning JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Fine-tuning JSONL file to analyze.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = analyze_examples(read_jsonl(args.input))
    print_report(stats)


if __name__ == "__main__":
    main()
