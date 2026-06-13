from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "system_prompt.md"
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI agent playing Minesweeper. Given the visible board state, "
    "return exactly one JSON move with action, x, and y."
)
VALID_ACTIONS = {"reveal", "flag"}


@dataclass(frozen=True)
class ConversionSummary:
    input_records: int
    output_examples: int
    skipped_records: int
    output_path: Path


def load_system_prompt(path: Path | None = None) -> str:
    prompt_path = path or DEFAULT_SYSTEM_PROMPT_PATH
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT


def format_compact_board(board: Any) -> str:
    if not isinstance(board, list) or not board:
        raise ValueError("compact_board_before must be a non-empty 2D list")

    rows: list[str] = []
    width: int | None = None
    for row in board:
        if not isinstance(row, list) or not row:
            raise ValueError("compact_board_before must contain non-empty rows")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("compact_board_before rows must all have the same width")
        rows.append(" ".join(str(cell) for cell in row))

    return "\n".join(rows)


def normalize_action(action: Any) -> dict[str, int | str]:
    if not isinstance(action, dict):
        raise ValueError("action must be an object")

    action_name = action.get("action")
    x = action.get("x")
    y = action.get("y")
    if action_name not in VALID_ACTIONS:
        raise ValueError("action.action must be one of: flag, reveal")
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("action.x and action.y must be integers")

    return {"action": action_name, "x": x, "y": y}


def build_user_message(record: dict[str, Any]) -> str:
    board = record.get("compact_board_before")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    board_text = format_compact_board(board)

    board_size = metadata.get("board_size")
    if isinstance(board_size, list) and len(board_size) == 2:
        size_text = f"{board_size[0]} rows x {board_size[1]} columns"
    else:
        size_text = "unknown"

    mine_count = metadata.get("mine_count", "unknown")
    step = metadata.get("step", "unknown")

    return (
        "Current Minesweeper board state:\n"
        f"Board size: {size_text}\n"
        f"Mine count: {mine_count}\n"
        f"Current step: {step}\n\n"
        "Compact board symbols:\n"
        ". = hidden, F = flagged, _ = revealed zero, 1-8 = revealed adjacent mine count.\n"
        "Coordinates are zero-based: x is the column index, y is the row index.\n\n"
        "Board:\n"
        f"{board_text}\n\n"
        "Choose the next best move. Return only a JSON object with action, x, and y."
    )


def transition_to_chat_example(
    record: dict[str, Any],
    system_prompt: str,
    skip_terminal_losses: bool = False,
) -> dict[str, list[dict[str, str]]]:
    if skip_terminal_losses:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if metadata.get("current_state") == "lost":
            raise ValueError("terminal loss transition skipped")

    action = normalize_action(record.get("action"))
    user_message = build_user_message(record)
    assistant_message = json.dumps(action, separators=(",", ":"))

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def validate_chat_example(example: dict[str, Any]) -> None:
    encoded = json.dumps(example)
    decoded = json.loads(encoded)
    messages = decoded.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("fine-tuning example must contain exactly three messages")

    expected_roles = ["system", "user", "assistant"]
    for message, role in zip(messages, expected_roles):
        if message.get("role") != role:
            raise ValueError(f"expected {role} message")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise ValueError(f"{role} message content must be a non-empty string")

    normalize_action(json.loads(messages[2]["content"]))


def convert_file(
    input_path: Path,
    output_path: Path,
    system_prompt_path: Path | None = None,
    skip_terminal_losses: bool = False,
) -> ConversionSummary:
    system_prompt = load_system_prompt(system_prompt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_records = 0
    output_examples = 0
    skipped_records = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            input_records += 1
            try:
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                example = transition_to_chat_example(
                    record,
                    system_prompt=system_prompt,
                    skip_terminal_losses=skip_terminal_losses,
                )
                validate_chat_example(example)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                skipped_records += 1
                print(f"Skipping line {line_number}: {exc}")
                continue

            target.write(json.dumps(example, ensure_ascii=False) + "\n")
            output_examples += 1

    return ConversionSummary(
        input_records=input_records,
        output_examples=output_examples,
        skipped_records=skipped_records,
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Minesweeper transition JSONL records into chat fine-tuning JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw transition JSONL path.")
    parser.add_argument("--output", required=True, type=Path, help="Chat fine-tuning JSONL output path.")
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="System prompt file to use for every chat example.",
    )
    parser.add_argument(
        "--skip-terminal-losses",
        action="store_true",
        help="Skip transitions whose action immediately led to a lost state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_file(
        input_path=args.input,
        output_path=args.output,
        system_prompt_path=args.system_prompt,
        skip_terminal_losses=args.skip_terminal_losses,
    )

    print("Fine-tuning conversion complete.")
    print(f"Input records: {summary.input_records}")
    print(f"Output examples: {summary.output_examples}")
    print(f"Skipped records: {summary.skipped_records}")
    print(f"Output path: {summary.output_path}")


if __name__ == "__main__":
    main()
