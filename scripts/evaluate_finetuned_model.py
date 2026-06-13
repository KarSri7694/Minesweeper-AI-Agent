from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetune_utils import get_message, parse_assistant_action, read_jsonl, validate_finetune_example


DEFAULT_INPUT_PATH = Path("datasets/finetune/val.jsonl")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class EvaluationResult:
    total_examples: int
    exact_matches: int
    action_matches: int
    coordinate_matches: int
    invalid_json: int

    @property
    def exact_match_accuracy(self) -> float:
        return self.exact_matches / self.total_examples if self.total_examples else 0.0

    @property
    def action_accuracy(self) -> float:
        return self.action_matches / self.total_examples if self.total_examples else 0.0

    @property
    def coordinate_accuracy(self) -> float:
        return self.coordinate_matches / self.total_examples if self.total_examples else 0.0


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    match = JSON_FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def parse_model_action(text: str) -> dict[str, int | str]:
    parsed = json.loads(extract_json_text(text))
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")

    action = parsed.get("action")
    x = parsed.get("x")
    y = parsed.get("y")
    if action not in {"reveal", "flag"}:
        raise ValueError("model action must be reveal or flag")
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("model x and y must be integers")
    return {"action": action, "x": x, "y": y}


def completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("OpenAI response did not contain choices")
    message = choices[0].message
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("OpenAI response message content was not text")
    return content


def call_model(
    client: Any,
    model: str,
    system_message: str,
    user_message: str,
    max_completion_tokens: int,
    temperature: float | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    return completion_text(client.chat.completions.create(**kwargs))


def select_examples(examples: list[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    if max_examples < 1:
        raise ValueError("--max-examples must be at least 1")
    selected = list(examples)
    random.Random(seed).shuffle(selected)
    return selected[:max_examples]


def evaluate_examples(
    client: Any,
    model: str,
    examples: list[dict[str, Any]],
    max_completion_tokens: int = 32,
    temperature: float | None = None,
) -> EvaluationResult:
    exact_matches = 0
    action_matches = 0
    coordinate_matches = 0
    invalid_json = 0

    for example in examples:
        expected = parse_assistant_action(example)
        response_text = call_model(
            client=client,
            model=model,
            system_message=get_message(example, "system"),
            user_message=get_message(example, "user"),
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )

        try:
            predicted = parse_model_action(response_text)
        except (json.JSONDecodeError, ValueError):
            invalid_json += 1
            continue

        if predicted == expected:
            exact_matches += 1
        if predicted["action"] == expected["action"]:
            action_matches += 1
        if predicted["x"] == expected["x"] and predicted["y"] == expected["y"]:
            coordinate_matches += 1

    return EvaluationResult(
        total_examples=len(examples),
        exact_matches=exact_matches,
        action_matches=action_matches,
        coordinate_matches=coordinate_matches,
        invalid_json=invalid_json,
    )


def load_validation_examples(input_path: Path, max_examples: int, seed: int) -> list[dict[str, Any]]:
    records = read_jsonl(input_path)
    for index, record in enumerate(records, start=1):
        try:
            validate_finetune_example(record)
        except ValueError as exc:
            raise ValueError(f"{input_path}:{index} failed validation: {exc}") from exc
    return select_examples(records, max_examples=max_examples, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on validation examples.")
    parser.add_argument("--model", required=True, help="Fine-tuned model name to evaluate.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Validation JSONL file.")
    parser.add_argument("--max-examples", type=int, default=50, help="Maximum examples to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--max-completion-tokens", type=int, default=32, help="Completion token cap per example.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_api_key()

    from openai import OpenAI

    examples = load_validation_examples(args.input, max_examples=args.max_examples, seed=args.seed)
    result = evaluate_examples(
        client=OpenAI(),
        model=args.model,
        examples=examples,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    print("Fine-tuned model evaluation")
    print(f"Examples evaluated: {result.total_examples}")
    print(f"Exact match accuracy: {result.exact_match_accuracy:.2%} ({result.exact_matches}/{result.total_examples})")
    print(f"Action accuracy: {result.action_accuracy:.2%} ({result.action_matches}/{result.total_examples})")
    print(f"Coordinate accuracy: {result.coordinate_accuracy:.2%} ({result.coordinate_matches}/{result.total_examples})")
    print(f"Invalid JSON count: {result.invalid_json}")


if __name__ == "__main__":
    main()
