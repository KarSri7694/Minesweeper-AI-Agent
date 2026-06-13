from __future__ import annotations

import argparse
from pathlib import Path

from finetune_utils import get_message, parse_assistant_action, read_jsonl, validate_finetune_example


def preview_text(text: str, limit: int = 240) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def extract_board(user_message: str) -> str:
    marker = "Board:\n"
    if marker not in user_message:
        return user_message
    board_and_tail = user_message.split(marker, 1)[1]
    return board_and_tail.split("\n\n", 1)[0]


def inspect_examples(path: Path, count: int = 5) -> list[str]:
    examples = read_jsonl(path)
    rendered: list[str] = []

    for index, example in enumerate(examples[:count], start=1):
        validate_finetune_example(example)
        system_preview = preview_text(get_message(example, "system"))
        user_message = get_message(example, "user")
        board = extract_board(user_message)
        action = parse_assistant_action(example)

        rendered.append(
            "\n".join(
                [
                    f"Example {index}",
                    f"System preview: {system_preview}",
                    "User board state:",
                    board,
                    f"Assistant action: {action}",
                ]
            )
        )

    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print readable previews from a chat fine-tuning JSONL file.")
    parser.add_argument("--input", required=True, type=Path, help="Fine-tuning JSONL file to inspect.")
    parser.add_argument("--count", type=int, default=5, help="Number of examples to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for example in inspect_examples(args.input, count=args.count):
        print(example)
        print()


if __name__ == "__main__":
    main()
