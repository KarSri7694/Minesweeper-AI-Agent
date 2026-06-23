"""Run a Minesweeper game using an OpenAI-compatible LLM endpoint.

The model is intentionally kept outside the game API.  This module validates every
model-proposed move against the visible compact board before making an API call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import requests
from openai import AsyncOpenAI


BASE_URL = "http://localhost:8080/"
API_URL_V1 = BASE_URL + "v1/"
GAME_URL = "http://localhost:8000/"
MODEL_NAME = "Qwen-3-4B-Instruct-2507-Minesweeper-Agent"
MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_GAME_SEED = 7

logger = logging.getLogger(__name__)


class LLMResponseError(ValueError):
    """Raised when a completion is not one valid Minesweeper move."""


class MoveValidationError(ValueError):
    """Raised when a move cannot be applied to the visible compact board."""


def get_system_prompt() -> str:
    system_prompt = Path(__file__).parent.parent / "system_prompt.md"
    return system_prompt.read_text(encoding="utf-8")


def _configured_seed() -> int | None:
    """Use a reproducible default, with an opt-out or override for live games."""
    value = os.getenv("MINESWEEPER_SEED", str(DEFAULT_GAME_SEED)).strip()
    if value.lower() in {"", "none", "random"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("MINESWEEPER_SEED must be an integer, 'random', or 'none'.") from exc


def create_game() -> str:
    payload: dict[str, Any] = {
        "width": 12,
        "height": 12,
        "mine_density": 0.15,
        "seed": _configured_seed(),
        "output_format": "compact",
    }
    response = requests.post(GAME_URL + "games/", json=payload)
    response.raise_for_status()
    game_id = response.json()["game_id"]
    logger.info("Created game %s (seed=%r).", game_id, payload["seed"])
    # Retain the original small startup delay for servers that initialize games lazily.
    time.sleep(5)
    return game_id


def get_game_state(game_id: str) -> dict[str, Any]:
    response = requests.post(GAME_URL + f"games/{game_id}/state", json={"output_format": "compact"})
    response.raise_for_status()
    return response.json()


def make_move(game_id: str, action: str, x: int, y: int) -> dict[str, Any]:
    payload = {"action": action, "x": x, "y": y, "output_format": "compact"}
    response = requests.post(GAME_URL + f"games/{game_id}/moves/", json=payload)
    response.raise_for_status()
    result = response.json()
    logger.info("API response for accepted move %s: %s", payload, result.get("last_move"))
    return result


def connect_to_llm() -> AsyncOpenAI:
    return AsyncOpenAI(api_key="your-api-key", base_url=API_URL_V1)


def build_user_prompt(game_state: dict[str, Any], invalid_feedback: str | None = None) -> str:
    """Build the training-compatible prompt, then add useful live-game metadata."""
    # Keep these first four lines in the exact order used by the fine-tuning data.
    prompt = (
        f"Board State: {game_state['board']}\n"
        f"Max Mines: {game_state['mine_count']}\n"
        f"Max Rows: {game_state['height']}\n"
        f"Max Columns: {game_state['width']}\n"
        f"Score: {game_state.get('score', 0)}\n"
        f"Status: {game_state.get('status', 'in_progress')}\n"
        f"Flagged Count: {game_state.get('flagged_count', 0)}\n"
        f"Move Count: {game_state.get('move_count', 0)}"
    )
    if invalid_feedback:
        prompt += f"\n\n{invalid_feedback}\nReturn one valid JSON move only."
    return prompt


async def get_llm_response(llm: AsyncOpenAI, system_prompt: str, user_prompt: str):
    return await llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )


async def _consume_stream(response: Any) -> str:
    assistant_text = ""
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            assistant_text += delta.content
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(f"\033[93m{reasoning}\033[0m", end="", flush=True)
    print()
    return assistant_text


def _extract_json_object(text: str) -> str:
    """Remove an optional Markdown fence and isolate the first JSON object."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline == -1:
            raise LLMResponseError("Markdown fence does not contain JSON.")
        stripped = stripped[first_newline + 1 :]
        closing_fence = stripped.rfind("```")
        if closing_fence != -1:
            stripped = stripped[:closing_fence]

    object_start = stripped.find("{")
    if object_start == -1:
        raise LLMResponseError("Response does not contain a JSON object.")
    decoder = json.JSONDecoder()
    try:
        _, object_end = decoder.raw_decode(stripped[object_start:])
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Invalid JSON move: {exc.msg}.") from exc
    return stripped[object_start : object_start + object_end]


def parse_llm_response(text: str) -> tuple[str, int, int]:
    """Parse strictly valid JSON, never Python-literal syntax, into a move."""
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Invalid JSON move: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise LLMResponseError("Move JSON must be an object.")

    action = payload.get("action")
    x = payload.get("x")
    y = payload.get("y")
    if action not in {"reveal", "flag"}:
        raise LLMResponseError("'action' must be either 'reveal' or 'flag'.")
    if isinstance(x, bool) or not isinstance(x, int):
        raise LLMResponseError("'x' must be an integer.")
    if isinstance(y, bool) or not isinstance(y, int):
        raise LLMResponseError("'y' must be an integer.")
    return action, x, y


def validate_move(game_state: dict[str, Any], action: str, x: int, y: int) -> None:
    """Ensure a model move targets one currently hidden (``'.'``) compact tile."""
    if action not in {"reveal", "flag"}:
        raise MoveValidationError("action must be 'reveal' or 'flag'.")
    if isinstance(x, bool) or not isinstance(x, int) or isinstance(y, bool) or not isinstance(y, int):
        raise MoveValidationError("x and y must be integers.")

    board = game_state.get("board")
    if not isinstance(board, list):
        raise MoveValidationError("Game state does not contain a compact board.")
    height = game_state.get("height", len(board))
    width = game_state.get("width", len(board[0]) if board else 0)
    if not isinstance(height, int) or not isinstance(width, int) or not (0 <= y < height and 0 <= x < width):
        raise MoveValidationError(f"({x}, {y}) is outside the {width}x{height} board.")
    if y >= len(board) or not isinstance(board[y], list) or x >= len(board[y]):
        raise MoveValidationError("Game state's compact board dimensions are invalid.")
    if board[y][x] != ".":
        raise MoveValidationError(
            f"board[{y}][{x}] is {board[y][x]!r}, not a hidden '.' cell."
        )


async def play_game(game_id: str, llm: AsyncOpenAI, *, max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES) -> dict[str, Any]:
    """Play until terminal state or too many invalid LLM responses occur."""
    game_state = get_game_state(game_id)
    system_prompt = get_system_prompt()
    failures = 0
    invalid_feedback: str | None = None

    while game_state.get("status") == "in_progress":
        user_prompt = build_user_prompt(game_state, invalid_feedback)
        response = await get_llm_response(llm, system_prompt, user_prompt)
        assistant_text = await _consume_stream(response)
        logger.info("Raw model response: %r", assistant_text)

        try:
            action, x, y = parse_llm_response(assistant_text)
            logger.info("Parsed model move: action=%s x=%s y=%s", action, x, y)
            validate_move(game_state, action, x, y)
        except (LLMResponseError, MoveValidationError) as exc:
            failures += 1
            invalid_feedback = (
                f"The previous move {assistant_text.strip()!r} was invalid because {exc} "
                "Choose only a hidden '.' cell."
            )
            logger.warning("Rejected model move (%s/%s): %s", failures, max_consecutive_failures, exc)
            if failures >= max_consecutive_failures:
                logger.error("Stopping after %s consecutive invalid model responses.", failures)
                break
            continue

        try:
            make_move(game_id, action, x, y)
        except requests.RequestException as exc:
            logger.error("API rejected an otherwise validated move: %s", exc)
            break

        failures = 0
        invalid_feedback = None
        game_state = get_game_state(game_id)

    logger.info("Game finished with status=%s score=%s", game_state.get("status"), game_state.get("score"))
    return game_state


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    game_id = create_game()
    game_state = await play_game(game_id, connect_to_llm())
    print(f"Game ended with status {game_state.get('status')}; final score: {game_state.get('score')}")


if __name__ == "__main__":
    asyncio.run(main())
