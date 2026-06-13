from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_ACTIONS = {"reveal", "flag"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_message(example: dict[str, Any], role: str) -> str:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("example is missing messages list")

    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
            raise ValueError(f"{role} message content must be a string")

    raise ValueError(f"example is missing {role} message")


def parse_assistant_action(example: dict[str, Any]) -> dict[str, int | str]:
    content = get_message(example, "assistant")
    try:
        action = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant content is not valid JSON: {exc}") from exc

    if not isinstance(action, dict):
        raise ValueError("assistant content must decode to a JSON object")
    action_name = action.get("action")
    x = action.get("x")
    y = action.get("y")
    if action_name not in VALID_ACTIONS:
        raise ValueError("assistant action must be reveal or flag")
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("assistant x and y must be integers")

    return {"action": action_name, "x": x, "y": y}


def validate_finetune_example(example: dict[str, Any]) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("example must contain exactly three messages")

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"]:
        raise ValueError("messages must be ordered as system, user, assistant")

    get_message(example, "system")
    get_message(example, "user")
    parse_assistant_action(example)
